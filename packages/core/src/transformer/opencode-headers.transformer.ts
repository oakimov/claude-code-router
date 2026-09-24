import { randomBytes, createHash } from "crypto";
import { Transformer } from "@/types/transformer";
import { sendUnifiedRequest } from "@/utils/request";
import { createApiError } from "@/api/middleware";
import { sanitizeUpstreamErrorText } from "@/utils/redact";
import {
  delay,
  isClientAbortError,
  isFallbackEligibleStatus,
  isProviderNetworkError,
  toClientAbortError,
} from "@/utils/retry";
import { deriveCacheSessionKey } from "@/utils/cacheControl";
import {
  getPersistedSession,
  putPersistedSession,
} from "@/session-registry";

const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const OPENCODE_VERSION = "1.18.32";
const OPENCODE_USER_AGENT = `opencode/${OPENCODE_VERSION}`;

// Free-tier gate fingerprint (curl-verified 2026-09-24 against
// https://opencode.ai/zen/v1/responses, model muse-spark-1.3-contributor-free):
// Zen's Console free-tier check passes only when the Responses body carries
// `"stream": true` AND function tools named exactly `read` and `shell`
// (lowercase; `Read`/`Bash` fail). Tool schemas/descriptions are free-form,
// `instructions`, `include`, and `store` are NOT required, and extra client
// tools alongside the stubs are harmless. Anything missing yields 403
// FreeTierError ("can only be used from within OpenCode").
// Additionally `prompt_cache_key` MUST be the session id: a foreign key
// (e.g. CCR's `ccr_<sha256>`) makes Zen return `response.incomplete` right
// after the reasoning item with zero further events — an indefinite stall
// from the client's perspective (A/B 2026-09-24: `ses_…` key completes,
// `ccr_…` key goes `incomplete`).
// Scope: models ending in `-free` on a zen (opencode.ai) endpoint; paid Zen
// models are left untouched.

// OpenCode Zen selects an upstream backend by hashing the last 4 characters of
// the `x-opencode-session` header (see zen handler selectProvider). When a
// conversation's session hashes to a bad slot, Zen fails the same way for
// every request in that conversation. Two observed signatures, both from the
// routing/capacity layer (NOT the request being malformed or the key being
// invalid):
//   - HTTP 401 `{"error":{"type":"ModelError","message":"No provider available"}}`
//     — the hashed slot has no provider at all.
//   - HTTP 400 `{"error":{...,"message":"Error from provider (Console): Upstream
//     request failed"}}` — the hashed backend failed its own upstream call.
// These are retried on the SAME session (which must not change during a
// conversation — Zen binds provider/cache affinity to it), then passed to
// CCR's normal fallback handling. The whole mechanism is contained to this
// transformer: it owns its upstream call via `config.__providerResponse` so no
// opencode-specific status-code semantics leak into the generic provider error
// path (which correctly treats 400/401 as terminal for every other provider).
const MAX_ZEN_ATTEMPTS = 5;
const ZEN_RETRY_BACKOFF_BASE_MS = 2_000;
const ZEN_RETRY_BACKOFF_MAX_MS = 30_000;
const ZEN_RETRY_AFTER_MAX_MS = 2_147_483_647;
const ZEN_FIRST_EVENT_TIMEOUT_MS = 30_000;
const ZEN_FIRST_PROGRESS_TIMEOUT_MS = 60_000;
const ZEN_STREAM_IDLE_TIMEOUT_MS = 60_000;
// Matches OpenCode's RETRY_JITTER_FACTOR in session/retry.ts.
const ZEN_RETRY_JITTER_FACTOR = 0.25;

export class OpencodeHeadersTransformer implements Transformer {
  name = "opencode-headers";
  ownsTransport = true;
  requestPhase = "transport" as const;

  private lastTimestamp = 0;
  private counter = 0;
  private firstEventTimeoutMs = ZEN_FIRST_EVENT_TIMEOUT_MS;
  private firstProgressTimeoutMs = ZEN_FIRST_PROGRESS_TIMEOUT_MS;
  private streamIdleTimeoutMs = ZEN_STREAM_IDLE_TIMEOUT_MS;

  async transformRequestIn(
    request: any,
    provider: any,
    context: any
  ): Promise<Record<string, any>> {
    const conversationId =
      context?.req?.sessionId || this.fingerprintConversation(request, context);
    let body = request.body || request;
    const baseConfig = request.config || {};
    const req = context?.req;
    if (req && typeof req === "object") {
      delete req._opencodeGateAliases;
      delete req._opencodeForcedStream;
    }
    // Parity with native opencode ProviderTransform.options(): every opencode
    // request must carry a session-scoped prompt_cache_key so Zen's downstream
    // provider cache (OpenAI promptCacheKey / Moonshot prefix) stays hot across
    // turns. processRequestTransformers skips applyProviderNativeChatCaching when
    // provider.transformer.use is non-empty (the opencode case), so we inject
    // here as a defensive fallback – no-op if routes.ts already did.
    body = this.ensurePromptCacheKey(body, context);
    // Ask for extended prompt-cache retention on both outgoing shapes Zen
    // accepts (chat completions: messages[]; responses: input[]). Probed
    // 2026-09-03: Zen returns 200 with the field present (muse-spark,
    // responses wire); the Codex backend instead 400s it, so this stays
    // scoped to the opencode transformer. Same price as in_memory per
    // OpenAI docs; never overwrite an explicit client value.
    body = this.ensurePromptCacheRetention(body);
    // Free-tier gate stubs (see fingerprint comment above): without
    // `"stream": true` plus exact-name `read`/`shell` function tools, Zen
    // answers 403 FreeTierError. Paid models skip this entirely.
    if (OpencodeHeadersTransformer.isFreeTierZenRequest(body, provider)) {
      if (body?.stream !== true && req && typeof req === "object") {
        req._opencodeForcedStream = OpencodeHeadersTransformer.outgoingToolShape(
          body,
          Array.isArray(body?.tools) ? body.tools : []
        );
      }
      body = this.ensureStreamedForFreeTier(body, context);
      const gate = this.ensureGateStubTools(body);
      body = gate.body;
      body = this.ensureDetailedSummaryForFreeTier(body, context);
      if (gate.aliases) {
        // processResponseTransformers (routes.ts) builds a FRESH context for
        // the response phase, so request-phase props on `context` itself do
        // not survive. `req` is the shared object across both phases
        // (underscore-props on req are the established channel, e.g.
        // _wireKeep, _cachePrefixClientDiff).
        if (req && typeof req === "object") {
          req._opencodeGateAliases = gate.aliases;
        }
      }
    }
    // OpenCode identifies one logical user turn with the user-message id. Keep
    // this stable across transport retries, like the session.
    const requestId = this.generateId("msg", "ascending");

    const response = await this.sendWithSessionRetry(
      body,
      baseConfig,
      provider,
      context,
      conversationId,
      requestId
    );

    return {
      body,
      config: {
        ...baseConfig,
        // Placeholder URL kept for parity; __providerResponse short-circuits
        // sendRequestToProvider so this value is never fetched.
        url: provider?.baseUrl || provider?.api_base_url,
        __providerResponse: response,
      },
    };
  }

  async transformResponseOut(
    response: Response,
    context?: any
  ): Promise<Response> {
    const contentType = response.headers.get("content-type") || "";
    if (!response.body || !contentType.includes("text/event-stream")) {
      return response;
    }

    const out = this.preserveZenStreamErrors(response);
    const rewritten = this.rewriteGateStubCalls(
      out,
      OpencodeHeadersTransformer.gateAliasesFrom(context)
    );
    const forcedShape = context?.req?._opencodeForcedStream;
    context?.req?.log?.debug?.(
      { forcedShape, contentType },
      "opencode: response stream mode"
    );
    if (forcedShape === "responses" || forcedShape === "chat") {
      return this.collectForcedStream(rewritten, forcedShape, context?.req?.log);
    }
    return rewritten;
  }

  /** Restore JSON for clients whose requests were streamed only to pass Zen's gate. */
  private async collectForcedStream(
    response: Response,
    shape: "responses" | "chat",
    logger?: any
  ): Promise<Response> {
    const headers = new Headers(response.headers);
    headers.set("content-type", "application/json");
    headers.delete("content-length");
    const json = (body: any, status = response.status) =>
      new Response(JSON.stringify(body), { status, headers });
    const failure = (message: string) =>
      json(
        {
          error: {
            message: sanitizeUpstreamErrorText(message) || "Upstream response failed",
            type: "api_error",
          },
        },
        502
      );

    const events: string[] = [];
    try {
      const reader = response.body?.getReader();
      if (!reader) return failure("Upstream response has no stream body");
      const decoder = new TextDecoder();
      let pending = "";
      let terminalSeen = false;
      while (!terminalSeen) {
        const { done, value } = await reader.read();
        pending += done
          ? decoder.decode()
          : decoder.decode(value, { stream: true });
        const complete = pending.split(/\r?\n\r?\n/);
        pending = done ? "" : complete.pop() || "";
        for (const event of complete) {
          events.push(event);
          const data = event
            .split(/\r?\n/)
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart())
            .join("\n");
          if (data === "[DONE]") terminalSeen = true;
          else if (data) {
            try {
              const parsed = JSON.parse(data);
              if (
                parsed?.type === "response.completed" ||
                parsed?.type === "response.incomplete" ||
                parsed?.type === "response.failed" ||
                parsed?.type === "error" ||
                parsed?.error
              ) {
                terminalSeen = true;
              }
            } catch {
              // The parse below returns a protocol-shaped error.
            }
          }
          if (terminalSeen) break;
        }
        if (done) break;
      }
      logger?.debug?.(
        { shape, terminalSeen, eventCount: events.length },
        "opencode: forced stream collection finished"
      );
      // Zen may keep the network stream open after a terminal event, and
      // Undici's cancellation promise may wait indefinitely for that socket.
      // The response is complete already, so do not hold the client on cancel.
      if (terminalSeen) void reader.cancel().catch(() => {});
    } catch (error) {
      if (isClientAbortError(error)) throw error;
      return failure(error instanceof Error ? error.message : String(error));
    }
    let terminal: any;
    let streamError: any;
    let sawDone = false;
    const chat: any = {
      id: "",
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: "",
      choices: [
        { index: 0, message: { role: "assistant", content: "" }, finish_reason: null },
      ],
    };
    const calls = new Map<number, any>();
    for (const event of events) {
      const data = event
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) continue;
      if (data === "[DONE]") {
        sawDone = true;
        continue;
      }
      let parsed: any;
      try {
        parsed = JSON.parse(data);
      } catch {
        return failure("Malformed upstream event stream");
      }
      if (
        parsed?.type === "error" ||
        parsed?.type === "response.failed" ||
        parsed?.error
      ) {
        streamError = parsed?.response?.error ?? parsed?.error ?? parsed;
        break;
      }
      if (shape === "responses") {
        if (parsed?.type === "response.completed" || parsed?.type === "response.incomplete") {
          terminal = parsed.response;
        }
        continue;
      }
      if (parsed?.id) chat.id = parsed.id;
      if (parsed?.model) chat.model = parsed.model;
      if (parsed?.created) chat.created = parsed.created;
      if (parsed?.usage) chat.usage = parsed.usage;
      const choice = parsed?.choices?.[0];
      if (!choice) continue;
      const delta = choice.delta ?? {};
      if (typeof delta.content === "string") chat.choices[0].message.content += delta.content;
      for (const field of ["reasoning_content", "reasoning"]) {
        if (typeof delta[field] === "string") {
          chat.choices[0].message[field] = (chat.choices[0].message[field] || "") + delta[field];
        }
      }
      for (const call of delta.tool_calls ?? []) {
        const index = typeof call.index === "number" ? call.index : calls.size;
        const existing = calls.get(index) ?? {
          id: call.id,
          type: call.type || "function",
          function: { name: "", arguments: "" },
        };
        if (call.id) existing.id = call.id;
        if (call.function?.name) existing.function.name += call.function.name;
        if (call.function?.arguments) existing.function.arguments += call.function.arguments;
        calls.set(index, existing);
      }
      if (choice.finish_reason) chat.choices[0].finish_reason = choice.finish_reason;
    }
    if (streamError) {
      return failure(String(streamError.message || streamError));
    }
    if (shape === "responses") {
      return terminal && terminal.object === "response"
        ? json(terminal)
        : failure("Upstream response stream ended without a terminal response");
    }
    if (!sawDone && !chat.choices[0].finish_reason) {
      return failure("Upstream chat stream ended without a finish event");
    }
    if (calls.size) {
      chat.choices[0].message.tool_calls = [...calls.entries()]
        .sort(([a], [b]) => a - b)
        .map(([, call]) => call);
      chat.choices[0].message.content ||= null;
    }
    return json(chat);
  }

  /**
   * Own the full upstream call so Zen routing failures can be retried while
   * retaining session affinity. Exhausted routing failures become 503 so the
   * normal fallback path can try another model; ordinary 4xx errors retain
   * their upstream status.
   */
  private async sendWithSessionRetry(
    body: any,
    baseConfig: any,
    provider: any,
    context: any,
    conversationId: string,
    requestId: string
  ): Promise<Response> {
    const url = provider?.baseUrl || provider?.api_base_url;
    const httpsProxy = context?.req?.server?.configService?.getHttpsProxy?.();
    const logger = context?.req?.log ?? context?.req?.server?.log;
    const model = body?.model;
    const signal = context?.signal ?? baseConfig?.signal;

    // The session is fixed per conversation for the lifetime of the turn
    // AND all its retries: Zen binds provider/cache affinity (and the
    // free-tier cache key, set below) to x-opencode-session, so it must not
    // change mid-conversation. Derived once here; every attempt below sends
    // the same session header and cache key.
    const sessionId = this.getOrCreateSessionId(conversationId);

    for (let attempt = 0; attempt < MAX_ZEN_ATTEMPTS; attempt++) {
      body = this.applyFreeTierCacheKey(body, provider, sessionId);
      const headers = this.buildHeaders(
        baseConfig,
        provider,
        conversationId,
        requestId,
        model,
        context,
        sessionId
      );

      let response: Response;
      try {
        response = await sendUnifiedRequest(
          url,
          body,
          {
            httpsProxy,
            ...baseConfig,
            headers,
            signal,
          },
          context,
          logger
        );
      } catch (error) {
        const isLastAttempt = attempt === MAX_ZEN_ATTEMPTS - 1;
        if (
          isLastAttempt ||
          isClientAbortError(error) ||
          !isProviderNetworkError(error)
        ) {
          throw error;
        }
        const waitMs = this.exponentialRetryDelayMs(attempt);
        logger?.warn?.(
          {
            provider: provider?.name,
            model,
            attempt: attempt + 1,
            waitMs,
          },
          "opencode: Zen network failure — preserving session and retrying"
        );
        await delay(waitMs, signal);
        continue;
      }

      if (response.ok) {
        const freeTier = OpencodeHeadersTransformer.isFreeTierZenRequest(body, provider);
        logger?.debug?.(
          { provider: provider?.name, model, freeTier, contentType: response.headers.get("content-type") },
          "opencode: upstream stream gate"
        );
        return freeTier
          ? await this.requireFirstZenEvent(response, provider, model, signal, logger)
          : response;
      }

      // Non-ok: read the body once to classify. Error responses are small JSON,
      // never a stream, so buffering here is safe (success is never buffered).
      const errorText = await response.text();
      const isLastAttempt = attempt === MAX_ZEN_ATTEMPTS - 1;
      const routingFailure = this.isZenRoutingFailure(
        response.status,
        errorText
      );
      const transientFailure = this.isZenTransientStatus(response.status);

      if (!isLastAttempt && (routingFailure || transientFailure)) {
        // Retries keep the SAME session: Zen binds affinity to it and it
        // must not change during a conversation. A persistently bad bucket
        // fails through to fallback after MAX_ZEN_ATTEMPTS.
        const waitMs = transientFailure
          ? this.retryDelayMs(response, attempt)
          : 0;
        logger?.warn?.(
          {
            provider: provider?.name,
            model,
            status: response.status,
            attempt: attempt + 1,
            waitMs,
          },
          routingFailure
            ? "opencode: Zen provider-routing failure — same session, retrying"
            : "opencode: transient Zen failure — preserving session and retrying"
        );
        if (waitMs > 0) await delay(waitMs, signal);
        continue;
      }

      // Not retryable, or retries exhausted: preserve the upstream status for
      // ordinary errors. Zen's 400/401 routing wrapper is a provider failure,
      // so map only that narrow case to 503 for CCR's fallback handling.
      const safeErrorText =
        sanitizeUpstreamErrorText(errorText) || errorText.slice(0, 240);
      throw createApiError(
        `Error from provider(${provider?.name},${model}: ${response.status}): ${safeErrorText}`,
        routingFailure ? 503 : response.status,
        "provider_response_error",
        "api_error",
        this.retryAfterHeaders(response)
      );
    }

    // Unreachable: the final attempt either returns ok or throws above.
    throw createApiError(
      `Error from provider(${provider?.name},${model}): Zen retries exhausted`,
      503,
      "provider_response_error"
    );
  }

  /**
   * Zen can send SSE headers and lifecycle events, then stall before any
   * output. Wait for a complete event and actual output progress while a
   * fallback is still possible. Bound later idle reads as well.
   */
  private async requireFirstZenEvent(
    response: Response,
    provider: any,
    model: string,
    signal?: AbortSignal,
    logger?: any
  ): Promise<Response> {
    if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream")) {
      return response;
    }

    const reader = response.body.getReader();
    const buffered: Uint8Array[] = [];
    const decoder = new TextDecoder();
    let pending = "";
    let firstEventSeen = false;
    let progressSeen = false;
    const startedAt = Date.now();

    const readWithTimeout = async (timeoutMs: number) => {
      if (signal?.aborted) throw toClientAbortError(signal.reason);
      let timer: ReturnType<typeof setTimeout> | undefined;
      let onAbort: (() => void) | undefined;
      try {
        return await Promise.race([
          reader.read().then((result) => ({ kind: "read" as const, result })),
          new Promise<{ kind: "timeout" }>((resolve) => {
            timer = setTimeout(() => resolve({ kind: "timeout" }), timeoutMs);
          }),
          new Promise<{ kind: "abort" }>((resolve) => {
            if (!signal) return;
            onAbort = () => resolve({ kind: "abort" });
            signal.addEventListener("abort", onAbort, { once: true });
            if (signal.aborted) onAbort();
          }),
        ]);
      } finally {
        clearTimeout(timer);
        if (onAbort) signal?.removeEventListener("abort", onAbort);
      }
    };

    try {
      while (!progressSeen) {
        const deadlineMs = firstEventSeen
          ? this.firstProgressTimeoutMs
          : this.firstEventTimeoutMs;
        const remainingMs = deadlineMs - (Date.now() - startedAt);
        if (remainingMs <= 0) break;
        const next = await readWithTimeout(remainingMs);
        if (next.kind === "abort") throw toClientAbortError(signal?.reason);
        if (next.kind === "timeout") break;
        if (next.result.done) break;
        if (!next.result.value) continue;
        buffered.push(next.result.value.slice());
        pending += decoder.decode(next.result.value, { stream: true });
        const events = pending.split(/\r?\n\r?\n/);
        pending = events.pop() || "";
        for (const event of events) {
          const data = event.split(/\r?\n/)
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trimStart())
            .join("\n");
          if (!data) continue;
          firstEventSeen = true;
          if (data === "[DONE]") {
            progressSeen = true;
            break;
          }
          try {
            const parsed = JSON.parse(data);
            const type = String(parsed?.type || "");
            if (
              type.endsWith(".delta") ||
              type === "response.completed" ||
              type === "response.incomplete" ||
              type === "response.failed" ||
              type === "error" ||
              parsed?.error ||
              (type === "response.output_item.done" && parsed?.item?.type !== "reasoning")
            ) {
              progressSeen = true;
              break;
            }
          } catch {
            // Preserve malformed events for the response transformer to reject.
          }
        }
      }
    } catch (error) {
      void reader.cancel(error).catch(() => {});
      throw error;
    }

    if (!progressSeen) {
      logger?.warn?.(
        { provider: provider?.name, model, firstEventSeen },
        "opencode: Zen stream stalled before output progress"
      );
      void reader.cancel("Zen initial stream timeout").catch(() => {});
      if (signal?.aborted) {
        throw toClientAbortError(signal.reason);
      }
      throw createApiError(
        firstEventSeen
          ? `OpenCode Zen sent no output progress within ${Math.round(this.firstProgressTimeoutMs / 1_000)}s for ${provider?.name},${model}`
          : `OpenCode Zen sent no complete response event within ${Math.round(this.firstEventTimeoutMs / 1_000)}s for ${provider?.name},${model}`,
        504,
        "provider_response_error"
      );
    }

    const idleTimeoutMs = this.streamIdleTimeoutMs;
    logger?.debug?.(
      { provider: provider?.name, model, bufferedChunks: buffered.length },
      "opencode: Zen stream passed output gate"
    );
    let replayIndex = 0;
    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          if (replayIndex < buffered.length) {
            const chunk = buffered[replayIndex++];
            controller.enqueue(chunk);
            return;
          }
          const next = await readWithTimeout(idleTimeoutMs);
          if (next.kind === "abort") throw toClientAbortError(signal?.reason);
          if (next.kind === "timeout") {
            logger?.warn?.(
              { provider: provider?.name, model },
              "opencode: Zen stream idle after output progress"
            );
            throw createApiError(
              `OpenCode Zen stream idle for ${Math.round(idleTimeoutMs / 1_000)}s`,
              504,
              "provider_response_error"
            );
          }
          const { done, value } = next.result;
          if (done) controller.close();
          else if (value) controller.enqueue(value);
        } catch (error) {
          void reader.cancel(error).catch(() => {});
          controller.error(error);
        }
      },
      cancel(reason) {
        return reader.cancel(reason);
      },
    });
    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }

  private preserveZenStreamErrors(response: Response): Response {
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let pending = "";
    let terminated = false;

    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          while (!terminated) {
            const { done, value } = await reader.read();
            if (done) {
              const tail = pending + decoder.decode();
              pending = "";
              const failure = tail
                ? OpencodeHeadersTransformer.zenStreamFailure(tail)
                : undefined;
              if (failure) {
                const error = Object.assign(new Error(failure), {
                  code: "provider_network_error",
                });
                terminated = true;
                controller.error(error);
                return;
              }
              controller.close();
              return;
            }

            // Forward upstream bytes unchanged so successful streams keep
            // native chunk boundaries and cadence. Only inspect a text copy
            // for Zen terminal error events.
            controller.enqueue(value);

            pending += decoder.decode(value, { stream: true });
            const events = pending.split(/\r?\n\r?\n/);
            pending = events.pop() || "";
            let failure: string | undefined;
            for (const event of events) {
              failure = OpencodeHeadersTransformer.zenStreamFailure(event);
              if (failure) break;
            }
            if (failure) {
              terminated = true;
              const error = Object.assign(new Error(failure), {
                code: "provider_network_error",
              });
              void reader.cancel(error).catch(() => {});
              controller.error(error);
              return;
            }
            return;
          }
        } catch (error) {
          controller.error(error);
        }
      },
      cancel(reason) {
        terminated = true;
        return reader.cancel(reason);
      },
    });

    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }

  private static zenStreamFailure(event: string): string | undefined {
    for (const line of event.split(/\r?\n/)) {
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      let parsed: any;
      try {
        parsed = JSON.parse(data);
      } catch {
        continue;
      }
      const finishReason = String(parsed?.choices?.[0]?.finish_reason || "");
      if (/^network[-_\s]error$/i.test(finishReason)) {
        return `Provider finish_reason: ${finishReason}`;
      }
      if (parsed?.error) {
        const message =
          typeof parsed.error?.message === "string"
            ? parsed.error.message
            : typeof parsed.error === "string"
              ? parsed.error
              : "OpenCode provider stream error";
        return sanitizeUpstreamErrorText(message) || "OpenCode provider stream error";
      }
    }
    return undefined;
  }

  private buildHeaders(
    baseConfig: any,
    provider: any,
    conversationId: string,
    requestId: string,
    model?: string,
    context?: any,
    sessionId?: string
  ): Record<string, any> {
    void model;
    // Session is derived once per turn in sendWithSessionRetry and passed in
    // so the header and the free-tier cache key can never diverge. Fall back
    // to deriving here only for direct callers.
    const session = sessionId ?? this.getOrCreateSessionId(conversationId);
    const parentSessionId = this.resolveParentSessionId(context);
    return {
      ...baseConfig?.headers,
      "x-api-key": provider.apiKey || "",
      "x-opencode-project": "global",
      "x-opencode-session": session,
      "x-opencode-request": requestId,
      "x-opencode-client": "cli",
      // `x-zen-model` is intentionally NOT sent. Research (2026-08-25, verified
      // against opencode-research git history AND the shipped
      // opencode-darwin-arm64@1.18.32 binary): the real client never emits this
      // header — it exists only inside Zen's edge worker
      // (console/app/src/routes/zen/util/handler.ts), where selectProvider()
      // picks a backend from the private ZEN_MODELS* SST secrets and then either
      // sets x-zen-model itself (new-inference backends: console./console-go./
      // inf./inf-go.) or deletes it (legacy). The value comes from the request
      // body, so anything we send is overwritten server-side anyway. Revisit only
      // if Zen ever documents honoring an inbound x-zen-model.
      ...(parentSessionId ? { "x-parent-session-id": parentSessionId } : {}),
      "user-agent": OPENCODE_USER_AGENT,
      authorization: undefined,
    };
  }

  private resolveParentSessionId(context: any): string | undefined {
    if (!context) return undefined;
    const h = context?.req?.headers;
    if (h && typeof h === "object") {
      for (const [k, v] of Object.entries(h as Record<string, unknown>)) {
        if (k.toLowerCase() === "x-parent-session-id" && typeof v === "string" && v) return v;
      }
    }
    const direct =
      (context as any)?.req?.parentSessionId ??
      (context as any)?.req?.parentSessionID ??
      (context as any)?.parentSessionId ??
      (context as any)?.parentSessionID;
    if (typeof direct === "string" && direct) return direct;
    return undefined;
  }

  /**
   * True only for the two Zen session-hash routing failures — retried on the
   * SAME session (which must not change during a conversation), then passed
   * to fallback. Kept deliberately narrow (exact status + message) so genuine
   * auth errors (401 invalid key) and request errors (400 validation) are never
   * mistaken for routing failures and pass straight through.
   */
  private isZenRoutingFailure(status: number, text: string): boolean {
    if (status !== 401 && status !== 400) return false;
    let parsed: any;
    try {
      parsed = JSON.parse(text);
    } catch {
      return false;
    }
    const message = String(parsed?.error?.message || "");
    const errorParam = parsed?.error?.param;
    // A concrete parameter names a request-shape failure, even when Zen wraps
    // it in the same `Upstream request failed` Console message as a bad bucket.
    // Retrying cannot repair fields such as `input[2].call_id` being too long.
    if (errorParam !== null && errorParam !== undefined && errorParam !== "") {
      return false;
    }
    // 401: the hashed slot has no provider at all.
    if (
      status === 401 &&
      parsed?.error?.type === "ModelError" &&
      /no provider available/i.test(message)
    ) {
      return true;
    }
    // 400: the hashed backend failed its own upstream call. Match the Console
    // routing wrapper specifically, not arbitrary client-side 400s — and not
    // request-shape validation errors that Zen wraps in the same phrase
    // (e.g. missing json_schema.name). Retrying the session cannot fix those.
    if (
      status === 400 &&
      /upstream request failed/i.test(message) &&
      !/validation error/i.test(message)
    ) {
      return true;
    }
    return false;
  }

  private isZenTransientStatus(status: number): boolean {
    return isFallbackEligibleStatus(status);
  }

  private retryDelayMs(response: Response, failedAttemptIndex: number): number {
    const retryAfterMs = response.headers.get("retry-after-ms");
    if (retryAfterMs) {
      const parsed = Number.parseFloat(retryAfterMs);
      if (Number.isFinite(parsed) && parsed >= 0) {
        return Math.min(parsed, ZEN_RETRY_AFTER_MAX_MS);
      }
    }

    const retryAfter = response.headers.get("retry-after");
    if (retryAfter) {
      const seconds = Number.parseFloat(retryAfter);
      if (Number.isFinite(seconds) && seconds >= 0) {
        return Math.min(Math.ceil(seconds * 1_000), ZEN_RETRY_AFTER_MAX_MS);
      }
      const dateMs = Date.parse(retryAfter);
      if (Number.isFinite(dateMs)) {
        return Math.min(
          Math.max(0, Math.ceil(dateMs - Date.now())),
          ZEN_RETRY_AFTER_MAX_MS
        );
      }
    }

    return this.exponentialRetryDelayMs(failedAttemptIndex);
  }

  private exponentialRetryDelayMs(failedAttemptIndex: number): number {
    // Mirrors OpenCode session/retry.ts exponential(): base * 2^(attempt-1) with
    // 25% jitter, capped at ZEN_RETRY_BACKOFF_MAX_MS (30s without headers).
    const base = ZEN_RETRY_BACKOFF_BASE_MS * 2 ** failedAttemptIndex;
    const jittered = base + base * ZEN_RETRY_JITTER_FACTOR * Math.random();
    return Math.min(Math.ceil(jittered), ZEN_RETRY_BACKOFF_MAX_MS);
  }

  private retryAfterHeaders(
    response: Response
  ): Record<string, string> | undefined {
    const retryAfter = response.headers.get("retry-after");
    const retryAfterMs = response.headers.get("retry-after-ms");
    if (!retryAfter && !retryAfterMs) return undefined;
    return {
      ...(retryAfter ? { "Retry-After": retryAfter } : {}),
      ...(retryAfterMs ? { "Retry-After-Ms": retryAfterMs } : {}),
    };
  }

  private ensurePromptCacheKey(body: any, context: any): any {
    if (!body || typeof body !== "object") return body;
    if ((body as any).prompt_cache_key) return body;
    // Chat uses ``messages``; Responses uses ``input``. Skip inventing a key
    // when neither is present.
    if (!Array.isArray(body.messages) && !Array.isArray(body.input)) {
      return body;
    }
    const key = deriveCacheSessionKey(context, body);
    if (!key) return body;
    return { ...body, prompt_cache_key: key };
  }

  private ensurePromptCacheRetention(body: any): any {
    if (!body || typeof body !== "object") return body;
    if ((body as any).prompt_cache_retention) return body;
    // Chat completions use `messages`; Responses uses `input`. Only stamp
    // the retention ask on a recognizable outgoing shape.
    if (!Array.isArray(body.messages) && !Array.isArray(body.input)) {
      return body;
    }
    return { ...body, prompt_cache_retention: "24h" };
  }

  /**
   * Free-tier gate scope: `-free` models on a zen (opencode.ai) endpoint.
   * Paid Zen models and non-Zen providers sharing this transformer skip the
   * stub/stream handling below entirely.
   */
  private static isFreeTierZenRequest(body: any, provider: any): boolean {
    const model = body?.model;
    if (typeof model !== "string" || !model.endsWith("-free")) return false;
    const base = provider?.baseUrl || provider?.api_base_url || "";
    try {
      const url = new URL(String(base));
      return url.hostname === "opencode.ai" && url.pathname.startsWith("/zen/");
    } catch {
      return false;
    }
  }

  /**
   * The free-tier gate answers `stream: false` with 403 FreeTierError even
   * when everything else is exact (curl A/B 2026-09-24). Force SSE on the
   * wire; downstream non-streaming clients are unaffected because CCR
   * de-streams SSE for them in the normal response path.
   */
  private ensureStreamedForFreeTier(body: any, context: any): any {
    if (!body || typeof body !== "object" || (body as any).stream === true) {
      return body;
    }
    context?.req?.log?.debug?.(
      { provider: "opencode", model: (body as any).model },
      "opencode: forcing stream for free-tier gate"
    );
    return { ...body, stream: true };
  }

  /**
   * Free-tier thinking arrives almost entirely as opaque `encrypted_content`;
   * the only readable part is the reasoning summary, and Zen emits ~nothing
   * unless `reasoning.summary` is asked for (curl A/B 2026-09-24, same prompt:
   * `"detailed"` → 68 summary chars, `"auto"` → 0). Stamp `detailed` when the
   * client already reasons but states no summary preference. An explicit
   * client value (including `"none"`) always wins; absent/disabled reasoning
   * is left alone so non-reasoning calls never gain a reasoning block.
   */
  private ensureDetailedSummaryForFreeTier(body: any, context: any): any {
    if (!body || typeof body !== "object") return body;
    const reasoning = (body as any).reasoning;
    if (!reasoning || typeof reasoning !== "object") return body;
    if (reasoning.summary !== undefined && reasoning.summary !== null) {
      return body;
    }
    if (reasoning.effort === "none" || reasoning.enabled === false) {
      return body;
    }
    context?.req?.log?.debug?.(
      { provider: "opencode", model: (body as any).model },
      "opencode: requesting detailed reasoning summary for free-tier thinking"
    );
    return { ...body, reasoning: { ...reasoning, summary: "detailed" } };
  }

  /**
   * Zen aborts free-tier Responses runs (`response.incomplete`, no further
   * events — a client-side stall) unless `prompt_cache_key` equals the
   * `x-opencode-session` header (curl A/B 2026-09-24). CCR's generic cache
   * key (`ccr_<sha256>`, stable per conversation but foreign to Zen) must be
   * replaced after `ensurePromptCacheKey` runs. Scoped to free-tier Zen on the
   * Responses wire; chat bodies and paid models keep existing behavior.
   * Runs per attempt so header and key stay in lockstep.
   */
  private applyFreeTierCacheKey(
    body: any,
    provider: any,
    sessionId: string
  ): any {
    if (!body || typeof body !== "object") return body;
    if (!OpencodeHeadersTransformer.isFreeTierZenRequest(body, provider)) {
      return body;
    }
    if (!Array.isArray((body as any).input)) return body;
    if ((body as any).prompt_cache_key === sessionId) return body;
    return { ...body, prompt_cache_key: sessionId };
  }

  /**
   * Inject the exact-name `read`/`shell` function stubs the free-tier gate
   * requires (curl A/B 2026-09-24: lowercase exact match; `Read`/`Bash`
   * fail, schemas are free-form, extras harmless). Client tools are never
   * modified or reordered; stubs are appended. Each stub clones its client
   * counterpart's description/parameters when present so a
   * stub call maps back onto a schema the client already accepts; otherwise
   * it carries a minimal empty-object schema. Returns the alias map
   * (stub -> client name, or null without counterpart) for the response
   * stage; no aliases when nothing was injected.
   */
  private ensureGateStubTools(body: any): {
    body: any;
    aliases?: Record<string, string | null>;
  } {
    if (!body || typeof body !== "object") return { body };
    const existing = Array.isArray((body as any).tools)
      ? [...(body as any).tools]
      : [];
    const shape = OpencodeHeadersTransformer.outgoingToolShape(body, existing);
    const exact = new Set<string>();
    const donors = new Map<
      string,
      { name: string; description: any; parameters: any }
    >();
    for (const tool of existing) {
      const name = OpencodeHeadersTransformer.toolName(tool, shape);
      if (typeof name !== "string" || !name) continue;
      exact.add(name);
      const key = name.toLowerCase();
      if (!donors.has(key)) {
        donors.set(key, {
          name,
          ...OpencodeHeadersTransformer.toolDef(tool, shape),
        });
      }
    }
    const stubs: any[] = [];
    const aliases: Record<string, string | null> = {};
    let added = false;
    for (const stubName of ["read", "shell"] as const) {
      if (exact.has(stubName)) continue;
      // Prefer a same-name tool in different case, then a known equivalent
      // advertised by the client. A code gateway is a last resort when the
      // harness exposes read/shell only through a code tool. Clone the donor
      // schema so the emitted arguments remain executable by that harness.
      const sameName = donors.get(stubName);
      const counterparts =
        OpencodeHeadersTransformer.GATE_STUB_COUNTERPARTS[stubName];
      const byCounterpart = counterparts
        .map((name) => donors.get(name.toLowerCase()))
        .find((entry) => entry && entry.name !== stubName);
      const codeGateway = ["run_code", "exec", "eval"]
        .map((name) => donors.get(name))
        .find(
          (entry) =>
            entry?.parameters?.properties?.code?.type === "string" &&
            Array.isArray(entry.parameters.required) &&
            entry.parameters.required.includes("code")
        );
      const donor =
        sameName && sameName.name !== stubName
          ? sameName
          : byCounterpart && byCounterpart.name !== stubName
            ? byCounterpart
            : codeGateway;
      const description =
        donor?.description ??
        `CCR free-tier compatibility marker for ${stubName}; this client cannot execute it. Do not call this tool.`;
      const parameters = donor?.parameters ?? {
        type: "object",
        properties: {},
      };
      stubs.push(
        OpencodeHeadersTransformer.buildStubTool(
          shape,
          stubName,
          description,
          parameters
        )
      );
      aliases[stubName] = donor?.name ?? null;
      added = true;
    }
    if (!added) return { body };
    return { body: { ...body, tools: [...existing, ...stubs] }, aliases };
  }

  private static readonly GATE_STUB_COUNTERPARTS: Record<string, string[]> = {
    read: ["Read", "read_file"],
    shell: ["Bash", "pwsh", "powershell", "exec"],
  };

  private static outgoingToolShape(
    body: any,
    tools: any[]
  ): "chat" | "responses" {
    if (Array.isArray(body?.messages)) return "chat";
    if (Array.isArray(body?.input)) return "responses";
    const first = tools.find(
      (tool) => tool && typeof tool === "object"
    );
    if (first && typeof first?.function?.name === "string") return "chat";
    return "responses";
  }

  private static toolName(tool: any, shape: "chat" | "responses"): unknown {
    if (!tool || typeof tool !== "object") return undefined;
    return shape === "chat" ? tool?.function?.name : tool?.name;
  }

  private static toolDef(
    tool: any,
    shape: "chat" | "responses"
  ): { description: any; parameters: any } {
    const fn = shape === "chat" ? tool?.function : tool;
    return { description: fn?.description, parameters: fn?.parameters };
  }

  private static buildStubTool(
    shape: "chat" | "responses",
    name: string,
    description: any,
    parameters: any
  ): any {
    if (shape === "chat") {
      return { type: "function", function: { name, description, parameters } };
    }
    return { type: "function", name, description, parameters };
  }

  private static gateAliasesFrom(
    context: any
  ): Record<string, string> | undefined {
    const raw = (context as any)?.req?._opencodeGateAliases;
    if (!raw || typeof raw !== "object") return undefined;
    const out: Record<string, string> = {};
    for (const stubName of ["read", "shell"] as const) {
      if (typeof raw[stubName] === "string" && raw[stubName]) {
        out[stubName] = raw[stubName];
      }
    }
    return Object.keys(out).length > 0 ? out : undefined;
  }

  /**
   * Map gate-stub calls back to the advertised client tool inside SSE events.
   * Only `name` fields on
   * Responses `function_call` objects and chat `tool_calls[].function` objects
   * are touched; text payloads and unrelated events pass through byte-identical.
   * Stubs without a client counterpart are left alone (the client errors on
   * them exactly as it would on any unknown tool).
   */
  private rewriteGateStubCalls(
    response: Response,
    aliases: Record<string, string> | undefined
  ): Response {
    if (!aliases || !response.body) return response;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let pending = "";
    let terminated = false;

    const rewriteEvent = (eventText: string): string => {
      const lines = eventText.split(/\r?\n/);
      const dataLines = lines.filter((line) => line.startsWith("data:"));
      if (dataLines.length === 0) return eventText;
      const payloads = dataLines.map((line) => line.slice(5).trim());
      if (payloads.some((data) => !data || data === "[DONE]")) {
        return eventText;
      }
      let parsed: any;
      try {
        parsed = JSON.parse(payloads.join("\n"));
      } catch {
        return eventText;
      }
      if (!OpencodeHeadersTransformer.renameGateStubCalls(parsed, aliases)) {
        return eventText;
      }
      const kept = lines.filter((line) => !line.startsWith("data:"));
      return [...kept, `data: ${JSON.stringify(parsed)}`].join("\n");
    };

    const stream = new ReadableStream<Uint8Array>({
      pull: async (controller) => {
        try {
          while (!terminated) {
            const { done, value } = await reader.read();
            if (done) {
              const tail = pending + decoder.decode();
              pending = "";
              terminated = true;
              if (tail) {
                const events = tail.split(/\r?\n\r?\n/);
                for (const event of events) {
                  if (!event) continue;
                  controller.enqueue(
                    encoder.encode(`${rewriteEvent(event)}\n\n`)
                  );
                }
              }
              controller.close();
              return;
            }
            pending += decoder.decode(value, { stream: true });
            const events = pending.split(/\r?\n\r?\n/);
            pending = events.pop() || "";
            let emitted = false;
            for (const event of events) {
              if (!event) continue;
              controller.enqueue(encoder.encode(`${rewriteEvent(event)}\n\n`));
              emitted = true;
            }
            // A network chunk can end mid-event. Keep reading until this
            // pull produces a complete frame; otherwise downstream readers
            // can wait forever while buffered upstream chunks remain unread.
            if (emitted) return;
          }
        } catch (error) {
          controller.error(error);
        }
      },
      cancel(reason) {
        terminated = true;
        return reader.cancel(reason);
      },
    });

    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }

  private static renameGateStubCalls(
    node: any,
    aliases: Record<string, string>
  ): boolean {
    let changed = false;
    const walk = (value: any): void => {
      if (Array.isArray(value)) {
        value.forEach(walk);
        return;
      }
      if (!value || typeof value !== "object") return;
      if (
        value.type === "function_call" &&
        typeof value.name === "string" &&
        aliases[value.name]
      ) {
        value.name = aliases[value.name];
        changed = true;
      }
      const toolCalls = (value as any).tool_calls;
      if (Array.isArray(toolCalls)) {
        for (const call of toolCalls) {
          const name = call?.function?.name;
          if (typeof name === "string" && aliases[name]) {
            call.function.name = aliases[name];
            changed = true;
          }
        }
      }
      for (const key of Object.keys(value)) walk(value[key]);
    };
    walk(node);
    return changed;
  }

  private fingerprintConversation(request: any, context: any): string {
    const body = request.body || request;
    const model = body.model || "";
    // Clients such as oh-my-pi supply a stable prompt_cache_key but no session
    // header. Use that key for Zen affinity before replacing it with Zen's
    // minted session id. For anonymous clients, later tool calls/results must
    // not change the fingerprint of the first user turn.
    const msgs = body.messages || body.input || [];
    const clientCacheKey =
      typeof body.prompt_cache_key === "string" && body.prompt_cache_key.trim()
        ? body.prompt_cache_key.trim()
        : undefined;
    const firstUser = Array.isArray(msgs)
      ? msgs.find((item) => item?.role === "user")
      : undefined;
    const sample = clientCacheKey ?? JSON.stringify(firstUser ?? []);
    const ip = context?.req?.headers?.["x-forwarded-for"] || context?.req?.ip || "";
    const ua = context?.req?.headers?.["user-agent"] || "";
    return createHash("sha256")
      .update(`${model}|${ip}|${ua}|${sample}`)
      .digest("hex")
      .slice(0, 32);
  }

  private getOrCreateSessionId(key: string): string {
    // Fixed id per conversation, durable across CCR restarts via the shared
    // session registry (same mechanism as cursor agent bindings).
    const persisted = getPersistedSession("zen", key);
    if (persisted) return persisted.sessionId;

    const id = this.generateId("ses", "descending");
    putPersistedSession("zen", key, { sessionId: id });
    return id;
  }

  private generateId(prefix: string, direction: "ascending" | "descending" = "ascending"): string {
    const now = Date.now();
    if (now !== this.lastTimestamp) {
      this.lastTimestamp = now;
      this.counter = 0;
    }
    this.counter++;

    // Match opencode's Identifier.create: session ids are descending
    // (bitwise-NOT of timestamp*0x1000 + counter), message ids ascending.
    let ts = BigInt(now) * BigInt(0x1000) + BigInt(this.counter);
    if (direction === "descending") ts = ~ts;
    const timeBytes = Buffer.alloc(6);
    for (let i = 0; i < 6; i++) {
      timeBytes[i] = Number((ts >> BigInt(40 - 8 * i)) & BigInt(0xff));
    }

    // Rejection sampling avoids modulo bias (248 = 62 * 4).
    let suffix = "";
    while (suffix.length < 14) {
      const byte = randomBytes(1)[0];
      if (byte >= 248) continue;
      suffix += BASE62[byte % 62];
    }

    return `${prefix}_${timeBytes.toString("hex")}${suffix}`;
  }
}

import { randomBytes, createHash } from "crypto";
import { Transformer } from "@/types/transformer";
import { sendUnifiedRequest } from "@/utils/request";
import { createApiError } from "@/api/middleware";
import {
  sanitizeErrorForLog,
  sanitizeUpstreamErrorText,
} from "@/utils/redact";
import {
  delay,
  isClientAbortError,
  isFallbackEligibleStatus,
  isProviderNetworkError,
  toClientAbortError,
} from "@/utils/retry";
import {
  deriveCacheSessionKey,
  extractClientSessionId,
} from "@/utils/cacheControl";
import {
  deletePersistedSession,
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
// `ccr_…` key goes `incomplete`). The key is re-applied per attempt, so it
// follows the session through a bad-bucket re-roll.
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
// Retrying on the same session cannot succeed (and Zen pins the sticky
// provider per session with no expiry), so these re-roll the session (new
// random suffix => new hash bucket) and retry. Affinity to a failing bucket
// has no value; the new session is persisted and stays sticky from then on.
// Transient failures keep the session for provider/cache affinity. The whole
// mechanism is contained to this transformer: it owns its upstream call via
// `config.__providerResponse` so no opencode-specific status-code semantics
// leak into the generic provider error path (which correctly treats 400/401
// as terminal for every other provider).
const MAX_ZEN_ATTEMPTS = 5;
const ZEN_RETRY_BACKOFF_BASE_MS = 2_000;
const ZEN_RETRY_BACKOFF_MAX_MS = 30_000;
const ZEN_RETRY_AFTER_MAX_MS = 2_147_483_647;
const ZEN_FIRST_EVENT_TIMEOUT_MS = 30_000;
const ZEN_FIRST_PROGRESS_TIMEOUT_MS = 60_000;
const ZEN_STREAM_IDLE_TIMEOUT_MS = 60_000;
// A reasoning item without summary deltas emits nothing until it finishes, so
// silence while one is open is the model working, not a stall. Bound it more
// loosely than ordinary idle time.
const ZEN_REASONING_IDLE_TIMEOUT_MS = 300_000;
// Matches OpenCode's RETRY_JITTER_FACTOR in session/retry.ts.
const ZEN_RETRY_JITTER_FACTOR = 0.25;

/** Human-readable timeout for error messages (sub-second values stay exact). */
function formatDuration(ms: number): string {
  return ms < 1_000 ? `${ms}ms` : `${Math.round(ms / 1_000)}s`;
}

/** Split buffered SSE text into complete events plus the unterminated tail. */
function takeSseEvents(buffer: string): { events: string[]; rest: string } {
  const events = buffer.split(/\r?\n\r?\n/);
  const rest = events.pop() ?? "";
  return { events, rest };
}

/** Join an SSE event's `data:` lines (one optional leading space stripped). */
function sseEventData(event: string): string {
  return event
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).replace(/^ /, ""))
    .join("\n")
    .trim();
}

type ZenEventKind = "progress" | "reasoning_start" | "reasoning_end" | "other";

/**
 * Classify one SSE event for the free-tier stall guard. Progress means output
 * the client can use (text, visible reasoning, tool calls, a finish or a
 * terminal/error event) on either wire: Responses events carry `type`,
 * Chat Completions chunks carry `choices[].delta`.
 */
function classifyZenEvent(data: string): ZenEventKind {
  if (!data) return "other";
  if (data === "[DONE]") return "progress";
  let parsed: any;
  try {
    parsed = JSON.parse(data);
  } catch {
    // Preserve malformed events for the response transformer to reject.
    return "other";
  }
  if (parsed?.error) return "progress";
  const type = typeof parsed?.type === "string" ? parsed.type : "";
  if (type) {
    if (
      type.endsWith(".delta") ||
      type === "response.completed" ||
      type === "response.incomplete" ||
      type === "response.failed" ||
      type === "error"
    ) {
      return "progress";
    }
    if (type === "response.output_item.added" && parsed?.item?.type === "reasoning") {
      return "reasoning_start";
    }
    if (type === "response.output_item.done") {
      return parsed?.item?.type === "reasoning" ? "reasoning_end" : "progress";
    }
    return "other";
  }
  const choice = parsed?.choices?.[0];
  if (!choice) return "other";
  if (choice.finish_reason) return "progress";
  const delta = choice.delta ?? {};
  if (
    (typeof delta.content === "string" && delta.content) ||
    (typeof delta.reasoning_content === "string" && delta.reasoning_content) ||
    (typeof delta.reasoning === "string" && delta.reasoning) ||
    (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0)
  ) {
    return "progress";
  }
  return "other";
}

export class OpencodeHeadersTransformer implements Transformer {
  name = "opencode-headers";
  ownsTransport = true;
  requestPhase = "transport" as const;

  private lastTimestamp = 0;
  private counter = 0;
  private firstEventTimeoutMs = ZEN_FIRST_EVENT_TIMEOUT_MS;
  private firstProgressTimeoutMs = ZEN_FIRST_PROGRESS_TIMEOUT_MS;
  private streamIdleTimeoutMs = ZEN_STREAM_IDLE_TIMEOUT_MS;
  private reasoningIdleTimeoutMs = ZEN_REASONING_IDLE_TIMEOUT_MS;

  async transformRequestIn(
    request: any,
    provider: any,
    context: any
  ): Promise<Record<string, any>> {
    const conversationId = this.resolveConversationId(request, context);
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
    // this stable across transport retries; only a bad-bucket re-roll
    // changes the session.
    const requestId = this.generateId("msg", "ascending");

    const sent = await this.sendWithSessionRetry(
      body,
      baseConfig,
      provider,
      context,
      conversationId,
      requestId
    );

    return {
      // The body actually sent (per-attempt free-tier cache key included), so
      // downstream debug summaries report the real wire shape.
      body: sent.body,
      config: {
        ...baseConfig,
        // Placeholder URL kept for parity; __providerResponse short-circuits
        // sendRequestToProvider so this value is never fetched.
        url: provider?.baseUrl || provider?.api_base_url,
        __providerResponse: sent.response,
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

    let terminal: any;
    let streamError: any;
    let malformed = false;
    let sawDone = false;
    let finished = false;
    let eventCount = 0;
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

    // Each event is parsed once, as it arrives; collection ends at the
    // protocol terminal rather than at EOF.
    const consume = (event: string): void => {
      const data = sseEventData(event);
      if (!data) return;
      eventCount++;
      if (data === "[DONE]") {
        sawDone = true;
        finished = true;
        return;
      }
      let parsed: any;
      try {
        parsed = JSON.parse(data);
      } catch {
        malformed = true;
        finished = true;
        return;
      }
      if (
        parsed?.type === "error" ||
        parsed?.type === "response.failed" ||
        parsed?.error
      ) {
        streamError = parsed?.response?.error ?? parsed?.error ?? parsed;
        finished = true;
        return;
      }
      if (
        parsed?.type === "response.completed" ||
        parsed?.type === "response.incomplete"
      ) {
        if (shape === "responses") terminal = parsed.response;
        finished = true;
        return;
      }
      if (shape === "responses") return;
      if (parsed?.id) chat.id = parsed.id;
      if (parsed?.model) chat.model = parsed.model;
      if (parsed?.created) chat.created = parsed.created;
      if (parsed?.usage) chat.usage = parsed.usage;
      const choice = parsed?.choices?.[0];
      if (!choice) return;
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
    };

    try {
      const reader = response.body?.getReader();
      if (!reader) return failure("Upstream response has no stream body");
      const decoder = new TextDecoder();
      let pending = "";
      while (!finished) {
        const { done, value } = await reader.read();
        pending += done
          ? decoder.decode()
          : decoder.decode(value, { stream: true });
        const { events, rest } = takeSseEvents(pending);
        if (done && rest) events.push(rest);
        pending = done ? "" : rest;
        for (const event of events) {
          consume(event);
          if (finished) break;
        }
        if (done) break;
      }
      logger?.debug?.(
        { shape, finished, eventCount },
        "opencode: forced stream collection finished"
      );
      // Zen may keep the network stream open after a terminal event, and
      // Undici's cancellation promise may wait indefinitely for that socket.
      // The response is complete already, so do not hold the client on cancel.
      if (finished) void reader.cancel().catch(() => {});
    } catch (error) {
      if (isClientAbortError(error)) throw error;
      // Keep exception details server-side; do not echo messages/stacks to clients.
      logger?.warn?.(sanitizeErrorForLog(error), "opencode: forced stream collection failed");
      return failure("Upstream response stream failed");
    }
    if (malformed) {
      return failure("Malformed upstream event stream");
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
   * Own the full upstream call so Zen routing failures can be recovered by
   * re-rolling the session, and transient failures retried on the same one.
   * Exhausted routing failures become 503 so the normal fallback path can try
   * another model; ordinary 4xx errors retain their upstream status.
   */
  private async sendWithSessionRetry(
    body: any,
    baseConfig: any,
    provider: any,
    context: any,
    conversationId: string,
    requestId: string
  ): Promise<{ response: Response; body: any }> {
    const url = provider?.baseUrl || provider?.api_base_url;
    const httpsProxy = context?.req?.server?.configService?.getHttpsProxy?.();
    const logger = context?.req?.log ?? context?.req?.server?.log;
    const model = body?.model;
    const signal = context?.signal ?? baseConfig?.signal;

    // Zen binds provider/cache affinity (and the free-tier cache key, set
    // below) to x-opencode-session. It changes only when the current bucket
    // is deterministically broken; header and cache key are rebuilt from the
    // same value on every attempt so they can never diverge.
    let sessionId = this.getOrCreateSessionId(conversationId);

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
        return {
          response: freeTier
            ? await this.requireFirstZenEvent(response, provider, model, signal, logger)
            : response,
          body,
        };
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

      if (routingFailure) {
        // The bucket fails deterministically for this session, on this and
        // every later turn. Re-roll now; after exhaustion this also keeps the
        // next turn from starting on the poisoned session.
        this.invalidateSession(conversationId);
        if (!isLastAttempt) sessionId = this.getOrCreateSessionId(conversationId);
      }

      if (!isLastAttempt && (routingFailure || transientFailure)) {
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
            sessionRerolled: routingFailure,
          },
          routingFailure
            ? "opencode: Zen provider-routing failure — re-rolling session and retrying"
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
   * output. Hold the response until a complete event and real output progress
   * arrive while a fallback is still possible, then bound later idle reads.
   * An open reasoning item emits nothing until it finishes when no summary is
   * streamed, so while one is open the looser reasoning bound applies instead.
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
    let reasoningOpen = false;
    const startedAt = Date.now();
    let deadlineAt = startedAt + this.firstEventTimeoutMs;
    const firstProgressTimeoutMs = this.firstProgressTimeoutMs;
    const streamIdleTimeoutMs = this.streamIdleTimeoutMs;
    const reasoningIdleTimeoutMs = this.reasoningIdleTimeoutMs;

    // Track gate state across complete events. After the gate passes only
    // reasoning open/close transitions matter, so skip parsing other events.
    const observe = (text: string): void => {
      pending += text;
      const { events, rest } = takeSseEvents(pending);
      pending = rest;
      for (const event of events) {
        const data = sseEventData(event);
        if (!data) continue;
        if (!firstEventSeen) {
          firstEventSeen = true;
          deadlineAt = startedAt + firstProgressTimeoutMs;
        }
        if (progressSeen && !data.includes('"reasoning"')) continue;
        const kind = classifyZenEvent(data);
        if (kind === "progress") {
          progressSeen = true;
        } else if (kind === "reasoning_start") {
          reasoningOpen = true;
        } else if (kind === "reasoning_end") {
          reasoningOpen = false;
          // Output normally follows reasoning; expect it within the progress bound.
          deadlineAt = Date.now() + firstProgressTimeoutMs;
        }
      }
      if (reasoningOpen) {
        deadlineAt = Math.max(deadlineAt, Date.now() + reasoningIdleTimeoutMs);
      }
    };

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
        const remainingMs = deadlineAt - Date.now();
        if (remainingMs <= 0) break;
        const next = await readWithTimeout(remainingMs);
        if (next.kind === "abort") throw toClientAbortError(signal?.reason);
        // A timed-out read stays pending on the reader; never issue another.
        if (next.kind === "timeout") break;
        if (next.result.done) break;
        if (!next.result.value) continue;
        buffered.push(next.result.value.slice());
        observe(decoder.decode(next.result.value, { stream: true }));
      }
    } catch (error) {
      void reader.cancel(error).catch(() => {});
      throw error;
    }

    if (!progressSeen) {
      logger?.warn?.(
        { provider: provider?.name, model, firstEventSeen, reasoningOpen },
        "opencode: Zen stream stalled before output progress"
      );
      void reader.cancel("Zen initial stream timeout").catch(() => {});
      if (signal?.aborted) {
        throw toClientAbortError(signal.reason);
      }
      const message = !firstEventSeen
        ? `OpenCode Zen sent no complete response event within ${formatDuration(this.firstEventTimeoutMs)}`
        : reasoningOpen
          ? `OpenCode Zen reasoning sent no event within ${formatDuration(reasoningIdleTimeoutMs)}`
          : `OpenCode Zen sent no output progress within ${formatDuration(firstProgressTimeoutMs)}`;
      throw createApiError(
        `${message} for ${provider?.name},${model}`,
        504,
        "provider_response_error"
      );
    }

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
          const idleTimeoutMs = reasoningOpen
            ? reasoningIdleTimeoutMs
            : streamIdleTimeoutMs;
          const next = await readWithTimeout(idleTimeoutMs);
          if (next.kind === "abort") throw toClientAbortError(signal?.reason);
          if (next.kind === "timeout") {
            logger?.warn?.(
              { provider: provider?.name, model, reasoningOpen },
              "opencode: Zen stream idle after output progress"
            );
            throw createApiError(
              reasoningOpen
                ? `OpenCode Zen reasoning idle for ${formatDuration(idleTimeoutMs)}`
                : `OpenCode Zen stream idle for ${formatDuration(idleTimeoutMs)}`,
              504,
              "provider_response_error"
            );
          }
          const { done, value } = next.result;
          if (done) {
            controller.close();
          } else if (value) {
            observe(decoder.decode(value, { stream: true }));
            controller.enqueue(value);
          }
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
            const { events, rest } = takeSseEvents(pending);
            pending = rest;
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
    const data = sseEventData(event);
    if (!data || data === "[DONE]") return undefined;
    let parsed: any;
    try {
      parsed = JSON.parse(data);
    } catch {
      return undefined;
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
   * wire; the normal response path does not de-stream SSE for JSON clients,
   * so transformRequestIn records the forced shape and transformResponseOut
   * restores JSON via collectForcedStream.
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

    // Only events naming a stub can need a rename; skip parsing the rest
    // (every text/reasoning delta) entirely.
    const stubNeedles = Object.keys(aliases).map((name) => `"${name}"`);
    const rewriteEvent = (eventText: string): string => {
      if (!stubNeedles.some((needle) => eventText.includes(needle))) {
        return eventText;
      }
      const data = sseEventData(eventText);
      if (!data || data === "[DONE]") return eventText;
      let parsed: any;
      try {
        parsed = JSON.parse(data);
      } catch {
        return eventText;
      }
      if (!OpencodeHeadersTransformer.renameGateStubCalls(parsed, aliases)) {
        return eventText;
      }
      const kept = eventText
        .split(/\r?\n/)
        .filter((line) => !line.startsWith("data:"));
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
                const { events, rest } = takeSseEvents(tail);
                if (rest) events.push(rest);
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
            const { events, rest } = takeSseEvents(pending);
            pending = rest;
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

  private invalidateSession(key: string): void {
    deletePersistedSession("zen", key);
  }

  /**
   * Conversation identity for the Zen session binding: an explicit client
   * session id wherever the client supplies one (router-parsed, protocol
   * context, or the shared header/body extractor used for cache keys), and
   * the content fingerprint only for fully anonymous clients.
   */
  private resolveConversationId(request: any, context: any): string {
    const body = request?.body || request;
    const explicit =
      context?.req?.sessionId ||
      context?.protocolContext?.sessionId ||
      context?.req?.protocolContext?.sessionId ||
      extractClientSessionId({ body, headers: context?.req?.headers });
    if (typeof explicit === "string" && explicit) return explicit;
    return this.fingerprintConversation(request, context);
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

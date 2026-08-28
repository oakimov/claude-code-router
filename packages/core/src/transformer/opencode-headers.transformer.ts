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
} from "@/utils/retry";
import { deriveCacheSessionKey } from "@/utils/cacheControl";

const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const OPENCODE_VERSION = "1.18.25";
const OPENCODE_USER_AGENT = `opencode/${OPENCODE_VERSION}`;

// OpenCode Zen selects an upstream backend by hashing the last 4 characters of
// the `x-opencode-session` header (see zen handler selectProvider). When a
// conversation's generated session hashes to a bad slot, Zen fails the same way
// for every request in that conversation — a deterministic, self-inflicted dead
// end. Two observed signatures, both from the routing/capacity layer (NOT the
// request being malformed or the key being invalid):
//   - HTTP 401 `{"error":{"type":"ModelError","message":"No provider available"}}`
//     — the hashed slot has no provider at all.
//   - HTTP 400 `{"error":{...,"message":"Error from provider (Console): Upstream
//     request failed"}}` — the hashed backend failed its own upstream call.
// Both are recovered by re-rolling the session (new random suffix => new hash
// bucket) and retrying. The whole mechanism is contained to this transformer:
// it owns its upstream call via `config.__providerResponse` so no
// opencode-specific status-code semantics leak into the generic provider error
// path (which correctly treats 400/401 as terminal for every other provider).
const MAX_ZEN_ATTEMPTS = 5;
const ZEN_RETRY_BACKOFF_BASE_MS = 2_000;
const ZEN_RETRY_BACKOFF_MAX_MS = 30_000;
const ZEN_RETRY_AFTER_MAX_MS = 2_147_483_647;
// Matches OpenCode's RETRY_JITTER_FACTOR in session/retry.ts.
const ZEN_RETRY_JITTER_FACTOR = 0.25;

export class OpencodeHeadersTransformer implements Transformer {
  name = "opencode-headers";
  ownsTransport = true;
  requestPhase = "transport" as const;

  private sessionCache = new Map<string, string>();
  private readonly MAX_SESSIONS = 100;
  private lastTimestamp = 0;
  private counter = 0;

  async transformRequestIn(
    request: any,
    provider: any,
    context: any
  ): Promise<Record<string, any>> {
    const conversationId =
      context?.req?.sessionId || this.fingerprintConversation(request, context);
    let body = request.body || request;
    const baseConfig = request.config || {};
    // Parity with native opencode ProviderTransform.options(): every opencode
    // request must carry a session-scoped prompt_cache_key so Zen's downstream
    // provider cache (OpenAI promptCacheKey / Moonshot prefix) stays hot across
    // turns. processRequestTransformers skips applyProviderNativeChatCaching when
    // provider.transformer.use is non-empty (the opencode case), so we inject
    // here as a defensive fallback – no-op if routes.ts already did.
    body = this.ensurePromptCacheKey(body, context);
    // OpenCode identifies one logical user turn with the user-message id. Keep
    // this stable across transport retries; only the session changes when Zen's
    // deterministic provider bucket must be re-rolled.
    const requestId = this.generateId("msg");

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

  async transformResponseOut(response: Response): Promise<Response> {
    const contentType = response.headers.get("content-type") || "";
    if (!response.body || !contentType.includes("text/event-stream")) {
      return response;
    }

    return this.preserveZenStreamErrors(response);
  }

  /**
   * Own the full upstream call so a `No provider available` 401 can be retried
   * with a fresh session in isolation. Any other non-ok response is re-thrown in
   * the exact shape sendRequestToProvider would have produced, so genuine
   * auth/rate/server errors keep flowing through the normal error + fallback
   * path unchanged.
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

    for (let attempt = 0; attempt < MAX_ZEN_ATTEMPTS; attempt++) {
      const headers = this.buildHeaders(
        baseConfig,
        provider,
        conversationId,
        requestId,
        model,
        context
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
        return response;
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
        if (routingFailure) {
          // A deterministic bad Zen bucket needs a new sticky session. Transient
          // failures stay on the same session to preserve provider/cache affinity.
          this.invalidateSession(conversationId);
        }
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

      // Not retryable, or retries exhausted: reproduce the generic provider
      // error so auth, validation, permissions, and final transient failures
      // keep flowing through CCR's normal error and fallback handling.
      const safeErrorText =
        sanitizeUpstreamErrorText(errorText) || errorText.slice(0, 240);
      throw createApiError(
        `Error from provider(${provider?.name},${model}: ${response.status}): ${safeErrorText}`,
        response.status,
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
              await reader.cancel(error).catch(() => {});
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
    context?: any
  ): Record<string, any> {
    const sessionId = this.getOrCreateSessionId(conversationId);
    const parentSessionId = this.resolveParentSessionId(context);
    return {
      ...baseConfig?.headers,
      "x-api-key": provider.apiKey || "",
      "x-opencode-project": "global",
      "x-opencode-session": sessionId,
      "x-opencode-request": requestId,
      "x-opencode-client": "cli",
      // `x-zen-model` is intentionally NOT sent. Research (2026-08-25, verified
      // against opencode-research git history AND the shipped
      // opencode-darwin-arm64@1.18.25 binary): the real client never emits this
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
   * True only for the two Zen session-hash routing failures — a re-roll can
   * recover these. Kept deliberately narrow (exact status + message) so genuine
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
    // (e.g. missing json_schema.name). Re-rolling the session cannot fix those.
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
    if (!body || typeof body !== "object" || !Array.isArray(body.messages)) return body;
    if ((body as any).prompt_cache_key) return body;
    const key = deriveCacheSessionKey(context, body);
    if (!key) return body;
    return { ...body, prompt_cache_key: key };
  }

  private fingerprintConversation(request: any, context: any): string {
    const body = request.body || request;
    const model = body.model || "";
    const msgs = body.messages || [];
    const sample = JSON.stringify(msgs.slice(0, 3));
    const ip = context?.req?.headers?.["x-forwarded-for"] || context?.req?.ip || "";
    const ua = context?.req?.headers?.["user-agent"] || "";
    return createHash("sha256")
      .update(`${model}|${ip}|${ua}|${sample}`)
      .digest("hex")
      .slice(0, 32);
  }

  private getOrCreateSessionId(key: string): string {
    const existing = this.sessionCache.get(key);
    if (existing) return existing;

    if (this.sessionCache.size >= this.MAX_SESSIONS) {
      const oldest = this.sessionCache.keys().next().value;
      if (oldest !== undefined) this.sessionCache.delete(oldest);
    }

    const id = this.generateId("ses");
    this.sessionCache.set(key, id);
    return id;
  }

  private invalidateSession(key: string): void {
    this.sessionCache.delete(key);
  }

  private generateId(prefix: string): string {
    const now = Date.now();
    if (now !== this.lastTimestamp) {
      this.lastTimestamp = now;
      this.counter = 0;
    }
    this.counter++;

    const ts = BigInt(now) * BigInt(0x1000) + BigInt(this.counter);
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

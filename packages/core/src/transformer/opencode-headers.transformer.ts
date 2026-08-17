import { randomBytes, createHash } from "crypto";
import { Transformer } from "@/types/transformer";
import { sendUnifiedRequest } from "@/utils/request";
import { createApiError } from "@/api/middleware";
import { sanitizeUpstreamErrorText } from "@/utils/redact";

const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

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
const MAX_ZEN_ROUTING_RETRIES = 3;

export class OpencodeHeadersTransformer implements Transformer {
  name = "opencode-headers";

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
    const body = request.body || request;
    const baseConfig = request.config || {};

    const response = await this.sendWithSessionRetry(
      body,
      baseConfig,
      provider,
      context,
      conversationId
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
    conversationId: string
  ): Promise<Response> {
    const url = provider?.baseUrl || provider?.api_base_url;
    const httpsProxy = context?.req?.server?.configService?.getHttpsProxy?.();
    const logger = context?.req?.log ?? context?.req?.server?.log;
    const model = body?.model;

    for (let attempt = 0; attempt < MAX_ZEN_ROUTING_RETRIES; attempt++) {
      const headers = this.buildHeaders(baseConfig, provider, conversationId);

      const response = await sendUnifiedRequest(
        url,
        body,
        {
          httpsProxy,
          ...baseConfig,
          headers,
          signal: context?.signal ?? baseConfig?.signal,
        },
        context,
        logger
      );

      if (response.ok) {
        return response;
      }

      // Non-ok: read the body once to classify. Error responses are small JSON,
      // never a stream, so buffering here is safe (success is never buffered).
      const errorText = await response.text();
      const isLastAttempt = attempt === MAX_ZEN_ROUTING_RETRIES - 1;

      if (!isLastAttempt && this.isZenRoutingFailure(response.status, errorText)) {
        // Drop the pinned session so the next attempt generates a new suffix and
        // lands on a different Zen provider bucket.
        this.invalidateSession(conversationId);
        logger?.warn?.(
          { provider: provider?.name, model, status: response.status, attempt: attempt + 1 },
          "opencode: Zen provider-routing failure — re-rolling session and retrying"
        );
        continue;
      }

      // Not retryable, or retries exhausted: reproduce the generic provider
      // error so 401(bad key)/429/5xx behave exactly as before.
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
      `Error from provider(${provider?.name},${model}): Zen provider-routing retries exhausted`,
      503,
      "provider_response_error"
    );
  }

  private buildHeaders(
    baseConfig: any,
    provider: any,
    conversationId: string
  ): Record<string, any> {
    const sessionId = this.getOrCreateSessionId(conversationId);
    const requestId = this.generateId("msg");
    return {
      ...baseConfig?.headers,
      "x-api-key": provider.apiKey || "",
      "x-opencode-project": "global",
      "x-opencode-session": sessionId,
      "x-opencode-request": requestId,
      "x-opencode-client": "cli",
      "user-agent": "opencode/1.18.4",
      authorization: undefined,
    };
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

  private retryAfterHeaders(
    response: Response
  ): Record<string, string> | undefined {
    const retryAfter = response.headers.get("retry-after");
    return retryAfter ? { "Retry-After": retryAfter } : undefined;
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

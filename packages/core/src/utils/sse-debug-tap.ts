import { SSEParserTransform } from "./sse/SSEParser.transform";
import { preserveUpstreamResponseHeaders } from "./headers";
import {
  DEFAULT_LOG_BODY_MAX_BYTES,
  sanitizeBodyForLog,
  sanitizeUpstreamErrorBody,
} from "./redact";
import {
  attributeDivergenceStage,
  rememberAndDiffOutboundCachePrefix,
  type CacheAffinityHeaders,
  type CachePrefixDiff,
  type CachePrefixStage,
} from "./cache-prefix-debug";
import type { MessageDebugDirection } from "./message-debug";

export type UpstreamSSEDebugOptions = {
  logger?: any;
  reqId?: string | number;
  provider?: string;
  /** Routed model — snapshots are keyed per destination. */
  model?: string;
  /** Conversation / Claude session id used to pair consecutive cache snapshots. */
  conversationId?: string;
  /** Pipeline position this body was captured at. Defaults to `wire`. */
  stage?: CachePrefixStage;
  /** Codex (and similar) routing headers that pin prompt-cache affinity. */
  cacheAffinity?: CacheAffinityHeaders;
  /** Upstream status. Non-2xx bodies are diffed but never become the baseline. */
  responseStatus?: number;
  /** Client-leg diff, used to attribute a broken wire prefix to a stage. */
  clientStageDiff?: CachePrefixDiff | null;
  /** Outbound diff for this request, joined with the observed cache usage. */
  cacheDiff?: CachePrefixDiff | null;
  /** Cap for a single logged payload string (raw `data` field). */
  maxBytes?: number;
  /**
   * Opt-in raw per-event SSE logging. Independent of LOG_LEVEL=debug.
   * When false (default), the debug tap still computes terminal cache outcome
   * summaries but does not log every delta.
   */
  rawEvents?: boolean;
  /**
   * Which wire leg produced these bytes. Defaults to `provider→ccr` for the
   * upstream tap and `ccr→client` for the client tap.
   */
  direction?: MessageDebugDirection;
  /**
   * When true, only emit raw SSE/JSON event logs (no cache outcome summary).
   * Used for the client-bound leg where cache usage was already observed upstream.
   */
  eventsOnly?: boolean;
};

export type ClientSSEDebugOptions = {
  logger?: any;
  reqId?: string | number;
  provider?: string;
  model?: string;
  protocol?: string;
  maxBytes?: number;
  rawEvents?: boolean;
};

function isDebugEnabled(logger: any): boolean {
  if (!logger || typeof logger.debug !== "function") return false;
  if (typeof logger.isLevelEnabled === "function") {
    try {
      return logger.isLevelEnabled("debug") === true;
    } catch {
      return false;
    }
  }
  const level = typeof logger.level === "string" ? logger.level.toLowerCase() : "";
  if (level === "debug" || level === "trace") return true;
  // Pino numeric levels: trace=10, debug=20
  if (typeof logger.levelVal === "number") return logger.levelVal <= 20;
  // Minimal test loggers often expose only debug(); treat that as enabled.
  return !("level" in logger) && !("levelVal" in logger);
}

function isRawSSETraceEnabled(opts: UpstreamSSEDebugOptions): boolean {
  if (opts.rawEvents === true) return true;
  if (opts.rawEvents === false) return false;
  // Explicit config / env opt-in (not implied by LOG_LEVEL=debug).
  const env = process.env.LOG_SSE_EVENTS || process.env.CCR_LOG_SSE_EVENTS;
  if (env === "1" || env === "true") return true;
  return false;
}

function cloneResponseHeaders(
  headers: Headers,
  contentTypeFallback?: string
): Record<string, string> {
  const out = preserveUpstreamResponseHeaders(headers);
  const contentType = headers.get("Content-Type") || contentTypeFallback;
  if (contentType) out["Content-Type"] = contentType;
  return out;
}

function contentTypeOf(response: Response): string {
  return (response.headers.get("Content-Type") || "").toLowerCase();
}

function logReceived(
  opts: UpstreamSSEDebugOptions,
  data: string,
  parsed?: unknown
): void {
  if (!isRawSSETraceEnabled(opts)) return;
  const logger = opts.logger;
  if (!logger?.debug) return;
  const maxBytes = opts.maxBytes ?? DEFAULT_LOG_BODY_MAX_BYTES;
  const base = {
    reqId: opts.reqId,
    provider: opts.provider,
    model: opts.model,
    direction: opts.direction ?? "provider→ccr",
  };

  logger.debug({
    ...base,
    type: "received data",
    data: sanitizeBodyForLog(data, maxBytes),
  });

  if (parsed !== undefined) {
    logger.debug({
      ...base,
      response: sanitizeUpstreamErrorBody(parsed),
      // Keep the historical typo so existing log greps keep working.
      tppe: "Original Response",
    });
  }
}

type CacheUsage = {
  /** True when cached tokens are reported outside the prompt total (Anthropic). */
  cachedExcludedFromPrompt: boolean;
  promptTokens?: number;
  cachedTokens?: number;
  cacheWriteTokens?: number;
  outputTokens?: number;
};

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function pickMax(a: number | undefined, b: number | undefined) {
  if (a === undefined) return b;
  if (b === undefined) return a;
  return Math.max(a, b);
}

/**
 * Every provider reports prompt-cache hits under a different name. Collect the
 * usage objects each protocol can emit and normalize them onto one shape.
 */
function mergeCacheUsage(acc: CacheUsage, event: unknown): CacheUsage {
  if (!event || typeof event !== "object") return acc;
  const node = event as any;
  const candidates = [
    node.usage,
    node.message?.usage,
    node.response?.usage,
    node.usageMetadata,
  ].filter((value) => value && typeof value === "object");

  let next = acc;
  for (const usage of candidates) {
    const cached = pickMax(
      pickMax(
        num(usage.cache_read_input_tokens),
        num(usage.prompt_tokens_details?.cached_tokens)
      ),
      pickMax(
        pickMax(
          num(usage.input_tokens_details?.cached_tokens),
          num(usage.cachedContentTokenCount)
        ),
        num(usage.prompt_cache_hit_tokens)
      )
    );
    next = {
      // Anthropic's input_tokens counts only the uncached remainder; the
      // OpenAI/Gemini families fold cached tokens into the prompt total.
      cachedExcludedFromPrompt:
        next.cachedExcludedFromPrompt ||
        num(usage.cache_read_input_tokens) !== undefined ||
        num(usage.cache_creation_input_tokens) !== undefined,
      promptTokens: pickMax(
        next.promptTokens,
        pickMax(
          pickMax(num(usage.input_tokens), num(usage.prompt_tokens)),
          num(usage.promptTokenCount)
        )
      ),
      cachedTokens: pickMax(next.cachedTokens, cached),
      cacheWriteTokens: pickMax(
        next.cacheWriteTokens,
        num(usage.cache_creation_input_tokens)
      ),
      outputTokens: pickMax(
        next.outputTokens,
        pickMax(
          pickMax(num(usage.output_tokens), num(usage.completion_tokens)),
          num(usage.candidatesTokenCount)
        )
      ),
    };
  }
  return next;
}

/**
 * One-line triage verdict joining what we predicted against what upstream did.
 * `unexpected-miss` is the row worth chasing: the prefix we sent was intact and
 * the provider still charged full price.
 */
function cacheVerdict(
  diff: CachePrefixDiff | null | undefined,
  hitRatio: number | undefined
): string {
  if (hitRatio === undefined) return "unknown";
  if (!diff || diff.firstTurn) return hitRatio > 0 ? "warm-start" : "cold";
  if (diff.prefixIntact) return hitRatio > 0 ? "hit" : "unexpected-miss";
  return hitRatio > 0 ? "partial" : "expected-miss";
}

function logCacheOutcome(opts: UpstreamSSEDebugOptions, usage: CacheUsage): void {
  if (!isDebugEnabled(opts.logger)) return;
  const diff = opts.cacheDiff;
  const cached = usage.cachedTokens;
  const prompt = usage.promptTokens;
  // Anthropic reports the uncached remainder, so the billable prompt is the sum.
  const promptTotal =
    prompt === undefined
      ? undefined
      : usage.cachedExcludedFromPrompt
        ? prompt + (cached || 0) + (usage.cacheWriteTokens || 0)
        : prompt;
  const hitRatio =
    cached === undefined || !promptTotal
      ? undefined
      : Math.round((cached / promptTotal) * 1000) / 1000;

  if (cached === undefined && prompt === undefined && !diff) return;

  const divergenceStage = diff
    ? attributeDivergenceStage(opts.clientStageDiff, diff)
    : undefined;

  opts.logger.debug({
    reqId: opts.reqId,
    provider: opts.provider,
    type: "cache outcome",
    verdict: cacheVerdict(diff, hitRatio),
    ...(opts.model ? { model: opts.model } : {}),
    ...(opts.responseStatus ? { status: opts.responseStatus } : {}),
    ...(diff
      ? {
          conversationId: diff.conversationId,
          conversationIdSource: diff.conversationIdSource,
          predictedChange: diff.change,
          prefixIntact: diff.prefixIntact,
          ...(diff.firstDivergencePath
            ? { firstDivergencePath: diff.firstDivergencePath }
            : {}),
          approxPrefixTokensLost: diff.approxPrefixTokensLost,
          ...(diff.msSinceLastTurn !== undefined
            ? { msSinceLastTurn: diff.msSinceLastTurn }
            : {}),
          ...(divergenceStage ? { divergenceStage } : {}),
        }
      : {}),
    ...(promptTotal !== undefined ? { promptTokens: promptTotal } : {}),
    ...(cached !== undefined ? { cachedTokens: cached } : {}),
    ...(usage.cacheWriteTokens !== undefined
      ? { cacheWriteTokens: usage.cacheWriteTokens }
      : {}),
    ...(usage.outputTokens !== undefined
      ? { outputTokens: usage.outputTokens }
      : {}),
    ...(hitRatio !== undefined ? { cacheHitRatio: hitRatio } : {}),
  });
}

function consumeSSEDebugBranch(
  debugBranch: ReadableStream<Uint8Array>,
  opts: UpstreamSSEDebugOptions
): void {
  const run = async () => {
    // Cast avoids DOM lib ArrayBuffer/ArrayBufferLike mismatch on TextDecoderStream.
    const reader = (debugBranch as ReadableStream)
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new SSEParserTransform())
      .getReader();

    let usage: CacheUsage = { cachedExcludedFromPrompt: false };
    const rawEvents = isRawSSETraceEnabled(opts);

    // Yield so a sync-heavy logger.debug (pino file I/O) cannot monopolize the
    // event loop and starve the client TransformStream on the same thread.
    const yieldToClient = () =>
      new Promise<void>((resolve) => setImmediate(resolve));

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value || typeof value !== "object") continue;

        const event = value as { data?: unknown; event?: string };
        if (event.data == null) continue;

        if (
          typeof event.data === "object" &&
          (event.data as any).type === "done"
        ) {
          if (rawEvents) {
            logReceived(opts, "[DONE]");
            await yieldToClient();
          }
          continue;
        }

        if (typeof event.data === "object" && (event.data as any).raw != null) {
          if (rawEvents) {
            logReceived(opts, String((event.data as any).raw));
            await yieldToClient();
          }
          continue;
        }

        usage = mergeCacheUsage(usage, event.data);

        if (rawEvents) {
          let dataStr: string;
          try {
            dataStr = JSON.stringify(event.data);
          } catch {
            dataStr = String(event.data);
          }
          logReceived(opts, dataStr, event.data);
          await yieldToClient();
        }
      }
    } finally {
      if (!opts.eventsOnly) {
        logCacheOutcome(opts, usage);
      }
      try {
        reader.releaseLock();
      } catch {
        // already released
      }
    }
  };

  void run().catch(() => {});
}

async function tapJsonBody(
  response: Response,
  opts: UpstreamSSEDebugOptions
): Promise<Response> {
  const text = await response.text();
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = undefined;
  }
  logReceived(opts, text, parsed);
  if (!opts.eventsOnly) {
    logCacheOutcome(
      opts,
      mergeCacheUsage({ cachedExcludedFromPrompt: false }, parsed)
    );
  }
  return new Response(text, {
    status: response.status,
    statusText: response.statusText,
    headers: cloneResponseHeaders(response.headers, "application/json"),
  });
}

/**
 * Byte-preserving upstream response debug tap.
 *
 * For SSE: mirrors bytes to a background consumer that emits Codex-parity
 * `received data` / `Original Response` logs (including Anthropic usage /
 * cache fields on message_start / message_delta).
 *
 * Important: do **not** use `ReadableStream.tee()` here. Tee couples
 * backpressure across both branches — a slow debug logger (pino file I/O on
 * every thinking delta) stalls the client branch. Claude Code then idles long
 * enough to surface "Waiting for API response · check your network" and retry
 * while CCR eventually still finishes the upstream as HTTP 200.
 *
 * Instead, a TransformStream forwards each chunk to the client immediately and
 * copies into a debug tunnel whose writable side has an infinite high-water
 * mark, so debug I/O can never delay the client. The debug consumer drains the
 * buffered copy at its own pace (memory-bound to the stream size).
 *
 * Exact-wire passthrough never enters transformer stream loggers; this tap is
 * the single shared place that covers every outbound provider.
 */
export async function tapUpstreamSSEDebug(
  response: Response,
  opts: UpstreamSSEDebugOptions
): Promise<Response> {
  if (!isDebugEnabled(opts.logger) || !response.body) {
    return response;
  }

  const withDirection: UpstreamSSEDebugOptions = {
    ...opts,
    direction: opts.direction ?? "provider→ccr",
  };

  const contentType = contentTypeOf(response);

  if (contentType.includes("application/json")) {
    try {
      return await tapJsonBody(response, withDirection);
    } catch {
      return response;
    }
  }

  if (!contentType.includes("text/event-stream")) {
    // Unknown body shape: no usage to read, but still surface the prediction.
    if (!withDirection.eventsOnly) {
      logCacheOutcome(withDirection, { cachedExcludedFromPrompt: false });
    }
    return response;
  }

  try {
    const clientBranch = pipeSSEWithNonblockingDebugTap(
      response.body,
      withDirection
    );
    return new Response(clientBranch, {
      status: response.status,
      statusText: response.statusText,
      headers: cloneResponseHeaders(response.headers, "text/event-stream"),
    });
  } catch {
    return response;
  }
}

/**
 * Byte-preserving tap for the SSE body CCR sends back to the client after
 * response transformers. Same non-blocking tunnel as the upstream tap; events
 * are tagged `direction: "ccr→client"` so greps can split the two legs.
 */
export function tapClientSSEDebug(
  body: ReadableStream<Uint8Array>,
  opts: ClientSSEDebugOptions
): ReadableStream<Uint8Array> {
  if (!body || !opts.rawEvents || !isDebugEnabled(opts.logger)) {
    return body;
  }
  try {
    return pipeSSEWithNonblockingDebugTap(body, {
      logger: opts.logger,
      reqId: opts.reqId,
      provider: opts.provider,
      model: opts.model,
      maxBytes: opts.maxBytes,
      rawEvents: true,
      direction: "ccr→client",
      eventsOnly: true,
    });
  } catch {
    return body;
  }
}

/**
 * Forward upstream SSE bytes to the client without waiting on debug I/O.
 *
 * The debug tunnel writable uses `highWaterMark: Infinity` so `write()` always
 * accepts immediately and queues in memory. That preserves full debug logs
 * (including terminal usage frames for cache outcome) without the tee()
 * backpressure footgun and without dropping mid-stream chunks (which would
 * tear the SSE parser and lose usage anyway).
 */
function pipeSSEWithNonblockingDebugTap(
  body: ReadableStream<Uint8Array>,
  opts: UpstreamSSEDebugOptions
): ReadableStream<Uint8Array> {
  const debugTunnel = new TransformStream<Uint8Array, Uint8Array>(
    undefined,
    // Writable strategy: never apply backpressure toward the client tap.
    { highWaterMark: Infinity },
    // Readable strategy: let the debug consumer pull at its own pace.
    { highWaterMark: 1 }
  );
  const debugWriter = debugTunnel.writable.getWriter();
  consumeSSEDebugBranch(debugTunnel.readable, opts);

  let debugAlive = true;
  const abandonDebug = () => {
    if (!debugAlive) return;
    debugAlive = false;
    void debugWriter.close().catch(() => {});
  };

  return body.pipeThrough(
    // `cancel` is part of the Transformer contract at runtime but missing from
    // the bundled DOM lib types.
    new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, controller) {
        controller.enqueue(chunk);
        if (!debugAlive) return;
        // Infinite writable HWM: write() queues synchronously and resolves
        // without waiting on the debug consumer / pino I/O.
        void debugWriter.write(chunk.slice()).catch(() => {
          abandonDebug();
        });
      },
      flush() {
        abandonDebug();
      },
      cancel() {
        abandonDebug();
      },
    } as Transformer<Uint8Array, Uint8Array>)
  );
}

export type CacheStructureSummary = {
  systemBreakpoints: number;
  messageBreakpoints: number;
  toolBreakpoints: number;
  prompt_cache_key?: string;
  lastAssistantBlockOrder?: string[];
};

function countEphemeralOnValue(value: unknown): number {
  if (!value || typeof value !== "object") return 0;
  const cc = (value as any).cache_control;
  return cc && typeof cc === "object" && cc.type === "ephemeral" ? 1 : 0;
}

function countEphemeralInContent(content: unknown): number {
  if (!Array.isArray(content)) return 0;
  return content.reduce(
    (sum: number, part: unknown) => sum + countEphemeralOnValue(part),
    0
  );
}

function assistantBlockOrder(content: unknown): string[] | undefined {
  if (!Array.isArray(content)) return undefined;
  const order: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const type = String((part as any).type || "");
    if (!type) continue;
    if (order[order.length - 1] !== type) order.push(type);
  }
  return order.length ? order : undefined;
}

/**
 * Summarize outbound Anthropic/OpenAI cache-oriented request structure for
 * debug verification (breakpoints / prompt_cache_key / assistant block order).
 */
export function summarizeOutboundCacheStructure(
  body: Record<string, any> | null | undefined
): CacheStructureSummary | null {
  if (!body || typeof body !== "object") return null;

  let systemBreakpoints = 0;
  if (Array.isArray(body.system)) {
    for (const block of body.system) {
      systemBreakpoints += countEphemeralOnValue(block);
    }
  } else if (body.system && typeof body.system === "object") {
    systemBreakpoints += countEphemeralOnValue(body.system);
  }

  let messageBreakpoints = 0;
  let lastAssistantBlockOrder: string[] | undefined;
  if (Array.isArray(body.messages)) {
    for (const message of body.messages) {
      if (!message || typeof message !== "object") continue;
      messageBreakpoints += countEphemeralOnValue(message);
      messageBreakpoints += countEphemeralInContent(message.content);
      if (message.role === "assistant") {
        lastAssistantBlockOrder = assistantBlockOrder(message.content);
      }
    }
  }

  let toolBreakpoints = 0;
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      toolBreakpoints += countEphemeralOnValue(tool);
    }
  }

  const prompt_cache_key =
    typeof body.prompt_cache_key === "string" && body.prompt_cache_key
      ? body.prompt_cache_key
      : undefined;

  if (
    systemBreakpoints === 0 &&
    messageBreakpoints === 0 &&
    toolBreakpoints === 0 &&
    !prompt_cache_key &&
    !lastAssistantBlockOrder
  ) {
    return null;
  }

  return {
    systemBreakpoints,
    messageBreakpoints,
    toolBreakpoints,
    ...(prompt_cache_key ? { prompt_cache_key } : {}),
    ...(lastAssistantBlockOrder
      ? { lastAssistantBlockOrder }
      : {}),
  };
}

/**
 * Snapshot the outbound body, diff it against the previous turn, and log both.
 * Returns the diff so the caller can join it with the observed cache usage
 * once upstream responds.
 */
export function logOutboundCacheStructure(
  body: Record<string, any> | null | undefined,
  opts: UpstreamSSEDebugOptions
): CachePrefixDiff | null {
  if (!isDebugEnabled(opts.logger)) return null;
  const summary = summarizeOutboundCacheStructure(body);
  if (summary) {
    opts.logger.debug({
      reqId: opts.reqId,
      provider: opts.provider,
      type: "cache structure",
      stage: opts.stage ?? "wire",
      ...summary,
    });
  }

  const status = opts.responseStatus;
  const diff = rememberAndDiffOutboundCachePrefix(
    opts.conversationId,
    body,
    opts.cacheAffinity,
    {
      stage: opts.stage ?? "wire",
      provider: opts.provider,
      model: opts.model,
      // A rejected request was never cached upstream; keeping it as the
      // baseline would report the next turn as a phantom prefix break.
      commit: status === undefined || (status >= 200 && status < 300),
    }
  );
  if (!diff) return null;

  const divergenceStage = attributeDivergenceStage(opts.clientStageDiff, diff);
  opts.logger.debug({
    reqId: opts.reqId,
    provider: opts.provider,
    type: "cache prefix diff",
    ...diff,
    ...(divergenceStage ? { divergenceStage } : {}),
  });
  return diff;
}

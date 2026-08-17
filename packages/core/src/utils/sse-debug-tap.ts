import { SSEParserTransform } from "./sse/SSEParser.transform";
import { preserveUpstreamResponseHeaders } from "./headers";
import {
  DEFAULT_LOG_BODY_MAX_BYTES,
  sanitizeBodyForLog,
  sanitizeUpstreamErrorBody,
} from "./redact";

export type UpstreamSSEDebugOptions = {
  logger?: any;
  reqId?: string | number;
  provider?: string;
  /** Cap for a single logged payload string (raw `data` field). */
  maxBytes?: number;
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
  const logger = opts.logger;
  if (!logger?.debug) return;
  const maxBytes = opts.maxBytes ?? DEFAULT_LOG_BODY_MAX_BYTES;
  const base = {
    reqId: opts.reqId,
    provider: opts.provider,
  };

  logger.debug({
    ...base,
    type: "recieved data",
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
          logReceived(opts, "[DONE]");
          continue;
        }

        if (typeof event.data === "object" && (event.data as any).raw != null) {
          const raw = String((event.data as any).raw);
          logReceived(opts, raw);
          continue;
        }

        let dataStr: string;
        try {
          dataStr = JSON.stringify(event.data);
        } catch {
          dataStr = String(event.data);
        }
        logReceived(opts, dataStr, event.data);
      }
    } catch {
      // Debug branch must never affect the client stream.
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // ignore
      }
    }
  };

  void run();
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
  return new Response(text, {
    status: response.status,
    statusText: response.statusText,
    headers: cloneResponseHeaders(response.headers, "application/json"),
  });
}

/**
 * Byte-preserving upstream response debug tap.
 *
 * For SSE: tees the body so the client branch is unchanged while a background
 * consumer emits Codex-parity `recieved data` / `Original Response` logs
 * (including Anthropic usage / cache fields on message_start / message_delta).
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

  const contentType = contentTypeOf(response);

  if (contentType.includes("application/json")) {
    try {
      return await tapJsonBody(response, opts);
    } catch {
      return response;
    }
  }

  if (!contentType.includes("text/event-stream")) {
    return response;
  }

  try {
    const [clientBranch, debugBranch] = response.body.tee();
    consumeSSEDebugBranch(debugBranch, opts);
    return new Response(clientBranch, {
      status: response.status,
      statusText: response.statusText,
      headers: cloneResponseHeaders(response.headers, "text/event-stream"),
    });
  } catch {
    return response;
  }
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

export function logOutboundCacheStructure(
  body: Record<string, any> | null | undefined,
  opts: UpstreamSSEDebugOptions
): void {
  if (!isDebugEnabled(opts.logger)) return;
  const summary = summarizeOutboundCacheStructure(body);
  if (!summary) return;
  opts.logger.debug({
    reqId: opts.reqId,
    provider: opts.provider,
    type: "cache structure",
    ...summary,
  });
}

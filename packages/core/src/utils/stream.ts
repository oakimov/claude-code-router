import {
  IncrementalSSEParser,
  serializeSSEEvent,
  type ParsedSSEEvent,
} from "./sse/incremental-parser";
import { isClientAbortError, isProviderNetworkError } from "./retry";
import { preserveUpstreamResponseHeaders } from "./headers";
import { sanitizeErrorForLog, sanitizeUpstreamErrorText } from "./redact";

function logStreamError(
  logger: any,
  message: string,
  error: unknown,
  extra?: Record<string, unknown>
): void {
  const details = { ...extra, ...sanitizeErrorForLog(error) };
  if (logger?.error) {
    logger.error(details, message);
  } else {
    console.error(message, details);
  }
}

export interface StreamContext {
  controller: ReadableStreamDefaultController;
  encoder: TextEncoder;
}

export type SSEEvent = ParsedSSEEvent;

export type ProcessSSEEvent = (
  event: SSEEvent,
  context: StreamContext
) => void;

export type ProcessSSELine = (line: string, context: StreamContext) => void;

function normalizeStreamError(error: unknown): unknown {
  if (isClientAbortError(error)) return error;
  if (!isProviderNetworkError(error)) return error;

  const err = error as any;
  if (err && typeof err === "object" && err.code === "provider_network_error") {
    return err;
  }

  const detail = String(err?.cause?.message || err?.cause?.code || "").trim();
  const base = String(err?.message || "Upstream stream failed");
  const normalized = new Error(
    detail && !base.includes(detail)
      ? `Upstream stream terminated: ${base} (${detail})`
      : `Upstream stream terminated: ${base}`
  );
  return Object.assign(normalized, {
    type: "api_error",
    code: "provider_network_error",
    cause: err,
  });
}

/** Forward an event using its original raw bytes/string. */
export function forwardSSEEvent(
  event: SSEEvent,
  context: StreamContext
): void {
  context.controller.enqueue(context.encoder.encode(event.raw));
}

/** Serialize and enqueue a modified event (only when data changed). */
export function emitSSEEvent(
  event: {
    event?: string;
    id?: string;
    retry?: number;
    data?: unknown;
    dataRaw?: string;
  },
  context: StreamContext
): void {
  context.controller.enqueue(
    context.encoder.encode(serializeSSEEvent(event))
  );
}

type StreamReaderOptions = {
  bufferSize?: number;
  onComplete?: (context: StreamContext) => void;
  onError?: (error: unknown, context: StreamContext) => boolean | void;
  logger?: any;
  processEvent?: ProcessSSEEvent;
};

/**
 * Create a transformed SSE Response.
 *
 * Prefer `options.processEvent` for event-native handling (parse once, forward
 * raw when unchanged). The legacy `processLine` callback is still supported via
 * a shim that feeds reconstructed data lines.
 */
export function createSSEStreamReader(
  response: Response,
  processLineOrOptions?: ProcessSSELine | StreamReaderOptions,
  maybeOptions?: StreamReaderOptions
): Response {
  let processLine: ProcessSSELine | undefined;
  let options: StreamReaderOptions;

  if (typeof processLineOrOptions === "function") {
    processLine = processLineOrOptions;
    options = maybeOptions || {};
  } else {
    options = processLineOrOptions || {};
  }

  const processEvent = options.processEvent;
  const encoder = new TextEncoder();
  let streamFailed = false;
  let upstreamReader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  let cancelled = false;

  const stream = new ReadableStream({
    async start(controller) {
      if (!response.body) {
        controller.close();
        return;
      }

      const ctx: StreamContext = { controller, encoder };
      const parser = new IncrementalSSEParser();
      const decoder = new TextDecoder();

      const handleEvent = (event: SSEEvent) => {
        if (processEvent) {
          processEvent(event, ctx);
          return;
        }
        if (!processLine) {
          forwardSSEEvent(event, ctx);
          return;
        }
        // Legacy line shim: feed data/event/id/retry lines without a
        // serializer round-trip of the whole stream.
        if (event.event !== undefined) {
          processLine(`event: ${event.event}`, ctx);
        }
        if (event.id !== undefined) {
          processLine(`id: ${event.id}`, ctx);
        }
        if (event.retry !== undefined) {
          processLine(`retry: ${event.retry}`, ctx);
        }
        if (event.dataRaw !== undefined) {
          processLine(`data: ${event.dataRaw}`, ctx);
        } else if (event.data !== undefined) {
          if (
            event.data &&
            typeof event.data === "object" &&
            (event.data as { type?: string }).type === "done"
          ) {
            processLine("data: [DONE]", ctx);
          } else {
            processLine(`data: ${JSON.stringify(event.data)}`, ctx);
          }
        }
      };

      try {
        const reader = response.body.getReader();
        upstreamReader = reader;

        while (true) {
          if (cancelled) break;

          const { done, value } = await reader.read();
          if (done) break;
          if (!value) continue;

          const text = decoder.decode(value, { stream: true });
          for (const event of parser.push(text)) {
            if (cancelled) break;
            try {
              handleEvent(event);
            } catch (error) {
              logStreamError(options?.logger, "Error processing event", error, {
                line: sanitizeUpstreamErrorText(event.raw.slice(0, 240)),
              });
              forwardSSEEvent(event, ctx);
            }
          }
        }

        if (!cancelled) {
          const tail = decoder.decode();
          for (const event of parser.push(tail)) {
            handleEvent(event);
          }
          for (const event of parser.flush()) {
            handleEvent(event);
          }
        }
      } catch (error) {
        if (!cancelled) {
          const normalized = normalizeStreamError(error);
          logStreamError(options?.logger, "Stream error", normalized);
          let handled = false;
          try {
            handled = options?.onError?.(normalized, ctx) === true;
          } catch (handlerError) {
            logStreamError(
              options?.logger,
              "Stream error handler failed",
              handlerError
            );
          }
          streamFailed = !handled;
          if (!handled) {
            controller.error(normalized);
          }
        }
      } finally {
        upstreamReader = null;

        if (!cancelled) {
          try {
            options?.onComplete?.(ctx);
          } catch (error) {
            logStreamError(options?.logger, "Stream completion error", error);
            if (!streamFailed) {
              streamFailed = true;
              try {
                controller.error(error);
              } catch {
                // Controller already closed.
              }
            }
          }
        }

        if (!streamFailed && !cancelled) {
          try {
            controller.close();
          } catch {
            // Already closed.
          }
        }
      }
    },

    cancel: async (reason) => {
      cancelled = true;
      const pending = upstreamReader;
      upstreamReader = null;
      if (pending) {
        await pending.cancel(reason).catch(() => undefined);
      }
    },
  });

  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers: {
      ...preserveUpstreamResponseHeaders(response.headers),
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

export function encodeSSEData(data: string, encoder: TextEncoder): Uint8Array {
  return encoder.encode(`data: ${data}\n\n`);
}

export function encodeSSELine(line: string, encoder: TextEncoder): Uint8Array {
  return encoder.encode(line + "\n");
}

export { serializeSSEEvent } from "./sse/incremental-parser";

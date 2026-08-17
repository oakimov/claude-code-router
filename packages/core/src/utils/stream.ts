import { SSEParserTransform, SSESerializerTransform } from "./sse";
import { isClientAbortError, isProviderNetworkError } from "./retry";
import { preserveUpstreamResponseHeaders } from "./headers";
import { sanitizeErrorForLog, sanitizeUpstreamErrorText } from "./redact";

// pino's logger.error(msg, ...args) only interpolates %-style placeholders in
// msg; unconsumed extra args (like a raw Error) are silently dropped instead
// of appended the way console.error would. Always pass the error as the
// merging object so pino serializes it into the log record.
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

/**
 * Upstream closed the socket mid-stream (undici surfaces this as a bare
 * `TypeError: terminated` whose cause carries the real code, e.g.
 * UND_ERR_SOCKET / "other side closed").
 *
 * The bare message reaches the client as `{"message":"terminated"}`, which is
 * indistinguishable from a client-side abort. Tag it so downstream transformers
 * and the fallback logic can classify it as a retryable provider network error.
 */
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

export function createSSEStreamReader(
  response: Response,
  processLine: (line: string, context: StreamContext) => void,
  options?: {
    bufferSize?: number;
    onComplete?: (context: StreamContext) => void;
    /** Return true when the protocol adapter emitted its own terminal error. */
    onError?: (error: unknown, context: StreamContext) => boolean | void;
    logger?: any;
  }
): Response {
  const encoder = new TextEncoder();
  let streamFailed = false;
  // Shared with cancel(): a client disconnect must tear down the upstream
  // reader, otherwise the provider keeps streaming into a dead response.
  let upstreamReader: ReadableStreamDefaultReader<string> | null = null;
  let cancelled = false;

  const stream = new ReadableStream({
    async start(controller) {
      if (!response.body) {
        controller.close();
        return;
      }

      const ctx: StreamContext = { controller, encoder };

      try {
        const reader = response.body
          .pipeThrough(new TextDecoderStream())
          .pipeThrough(new SSEParserTransform())
          .pipeThrough(new SSESerializerTransform())
          .getReader();
        upstreamReader = reader;

        while (true) {
          if (cancelled) break;

          const { done, value } = await reader.read();
          if (done) {
            break;
          }

          if (!value) continue;

          // The SSESerializerTransform outputs clean string blocks (e.g., "data: ...\n\n")
          // Split into lines to maintain backward compatibility with the processLine callback
          const lines = value.split("\n");

          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              processLine(line, ctx);
            } catch (error) {
              logStreamError(options?.logger, "Error processing line", error, {
                line: sanitizeUpstreamErrorText(line),
              });
              controller.enqueue(encoder.encode(line + "\n"));
            }
          }
        }
      } catch (error) {
        // Our own cancel() rejects the pending read and already terminates the
        // controller — erroring it again would throw. Any other failure must
        // still reach the consumer, or it hangs waiting for bytes.
        if (!cancelled) {
          const normalized = normalizeStreamError(error);
          logStreamError(options?.logger, "Stream error", normalized);
          let handled = false;
          try {
            handled = options?.onError?.(normalized, ctx) === true;
          } catch (handlerError) {
            logStreamError(options?.logger, "Stream error handler failed", handlerError);
          }
          streamFailed = !handled;
          if (!handled) {
            controller.error(normalized);
          }
        }
      } finally {
        upstreamReader = null;

        // The client can disconnect (cancel() fires, see below) at the same
        // moment the upstream read loop is independently reaching its
        // natural end (done: true) — there's no ordering guarantee between
        // the two. When that race lands here, the controller is already
        // unusable from the platform's perspective even though we never
        // called close()/error() on it ourselves: any onComplete that
        // enqueues (e.g. flushing finalizeResponsesStream's terminal events)
        // throws "Invalid state: Controller is already closed", which this
        // catch turned into an unconditional controller.error() call that
        // could itself throw on an already-dead controller — an unhandled
        // rejection with a genuinely connected client still on the other
        // end, i.e. a request that looked fine over HTTP but silently
        // truncated. Skip onComplete entirely once cancelled: there is no
        // consumer left to receive anything it would enqueue.
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
                // Controller was already closed/errored by the platform
                // (e.g. the same disconnect race) — nothing left to signal.
              }
            }
          }
        }

        if (!streamFailed && !cancelled) {
          try {
            controller.close();
          } catch {
            // Same race as above: already closed by the platform.
          }
        }
      }
    },

    cancel: async (reason) => {
      // Client went away. Stop the loop and release the upstream connection so
      // the provider request does not keep running against a dead consumer.
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

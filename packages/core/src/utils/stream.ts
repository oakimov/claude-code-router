import { SSEParserTransform, SSESerializerTransform } from "./sse";
import { isClientAbortError, isProviderNetworkError } from "./retry";

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
              (options?.logger?.error ?? console.error)("Error processing line:", line, error);
              controller.enqueue(encoder.encode(line + "\n"));
            }
          }
        }
      } catch (error) {
        // Our own cancel() rejects the pending read and already terminates the
        // controller — erroring it again would throw. Any other failure must
        // still reach the consumer, or it hangs waiting for bytes.
        streamFailed = true;
        if (!cancelled) {
          const normalized = normalizeStreamError(error);
          (options?.logger?.error ?? console.error)("Stream error:", normalized);
          controller.error(normalized);
        }
      } finally {
        upstreamReader = null;

        try {
          options?.onComplete?.(ctx);
        } catch (error) {
          (options?.logger?.error ?? console.error)("Stream completion error:", error);
          if (!streamFailed) {
            streamFailed = true;
            controller.error(error);
          }
        }

        if (!streamFailed) {
          controller.close();
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

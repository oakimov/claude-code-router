export interface StreamContext {
  controller: ReadableStreamDefaultController;
  encoder: TextEncoder;
}

const MAX_BUFFER_SIZE = 1000000; // 1MB

export function createSSEStreamReader(
  response: Response,
  processLine: (line: string, context: StreamContext) => void,
  options?: { bufferSize?: number }
): Response {
  const maxBufferSize = options?.bufferSize ?? MAX_BUFFER_SIZE;

  const decoder = new TextDecoder();
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      if (!response.body) {
        controller.close();
        return;
      }

      const reader = response.body.getReader();
      let buffer = "";

      const ctx: StreamContext = { controller, encoder };

      const flushRemainingBuffer = () => {
        if (buffer.trim()) {
          processLine(buffer, ctx);
          buffer = "";
        }
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            flushRemainingBuffer();
            break;
          }

          if (!value || value.length === 0) continue;

          let chunk: string;
          try {
            chunk = decoder.decode(value, { stream: true });
          } catch {
            continue;
          }

          if (chunk.length === 0) continue;

          buffer += chunk;

          // Buffer overflow protection
          if (buffer.length > maxBufferSize) {
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";
            for (const line of lines) {
              if (line.trim()) {
                try {
                  processLine(line, ctx);
                } catch {
                  controller.enqueue(encoder.encode(line + "\n"));
                }
              }
            }
            continue;
          }

          // Process complete lines
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.trim()) continue;
            try {
              processLine(line, ctx);
            } catch (error) {
              console.error("Error processing line:", line, error);
              controller.enqueue(encoder.encode(line + "\n"));
            }
          }
        }
      } catch (error) {
        console.error("Stream error:", error);
        controller.error(error);
      } finally {
        try {
          reader.releaseLock();
        } catch {
          // Ignore release lock errors
        }
        controller.close();
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

import { Transformer } from "../types/transformer";

type ThinkState = "SEARCHING" | "THINKING" | "FINAL";

/**
 * Extracts <think>...</think> blocks from OpenAI-compatible responses into
 * thinking.content (and streaming thinking deltas).
 */
export class ExtraThinkTagTransformer implements Transformer {
  name = "extrathinktag";

  async transformRequestIn(request: any): Promise<any> {
    return request;
  }

  async transformResponseOut(response: Response): Promise<Response> {
    const thinkStart = "<think>";
    const thinkEnd = "</think>";

    if (response.headers.get("Content-Type")?.includes("application/json")) {
      try {
        const jsonResponse = await response.json();
        const content = jsonResponse?.choices?.[0]?.message?.content;
        if (typeof content === "string") {
          const match = content.match(/<think>([\s\S]*?)<\/think>/);
          if (match?.[1]) {
            jsonResponse.thinking = { content: match[1] };
          }
        }
        return new Response(JSON.stringify(jsonResponse), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      } catch {
        return response;
      }
    }

    if (!response.headers.get("Content-Type")?.includes("stream")) {
      return response;
    }

    if (!response.body) {
      return response;
    }

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let index = 0;

    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body!.getReader();
        let buffer = "";
        let state: ThinkState = "SEARCHING";
        let partial = "";
        let pendingWhitespace = "";

        const enqueue = (payload: any) => {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)
          );
        };

        const processDelta = (data: any, content: unknown) => {
          if (
            typeof content !== "string" &&
            data?.choices?.[0]?.delta &&
            Object.keys(data.choices[0].delta).length > 0 &&
            !data.choices[0].delta.content
          ) {
            data.choices[0].index = index;
            enqueue(data);
            return;
          }

          if (typeof content !== "string") {
            return;
          }

          let remaining = partial + content;
          partial = "";

          while (remaining.length > 0) {
            if (state === "SEARCHING") {
              const startIdx = remaining.indexOf(thinkStart);
              if (startIdx !== -1) {
                remaining = remaining.substring(startIdx + thinkStart.length);
                state = "THINKING";
                continue;
              }

              for (let i = thinkStart.length - 1; i > 0; i--) {
                if (remaining.endsWith(thinkStart.substring(0, i))) {
                  partial = remaining.substring(remaining.length - i);
                  break;
                }
              }
              remaining = "";
              continue;
            }

            if (state === "THINKING") {
              const endIdx = remaining.indexOf(thinkEnd);
              if (endIdx !== -1) {
                const thinkingChunk = remaining.substring(0, endIdx);
                if (thinkingChunk.length > 0) {
                  const delta = {
                    ...data.choices[0].delta,
                    thinking: { content: thinkingChunk },
                  };
                  delete delta.content;
                  enqueue({
                    ...data,
                    choices: [{ ...data.choices[0], delta, index }],
                  });
                }

                const signatureDelta = {
                  ...data.choices[0].delta,
                  thinking: { signature: Date.now().toString() },
                };
                delete signatureDelta.content;
                enqueue({
                  ...data,
                  choices: [
                    { ...data.choices[0], delta: signatureDelta, index },
                  ],
                });

                index++;
                remaining = remaining.substring(endIdx + thinkEnd.length);
                state = "FINAL";
                continue;
              }

              let emitChunk = remaining;
              for (let i = thinkEnd.length - 1; i > 0; i--) {
                if (remaining.endsWith(thinkEnd.substring(0, i))) {
                  partial = remaining.substring(remaining.length - i);
                  emitChunk = remaining.substring(0, remaining.length - i);
                  break;
                }
              }

              if (emitChunk.length > 0) {
                const delta = {
                  ...data.choices[0].delta,
                  thinking: { content: emitChunk },
                };
                delete delta.content;
                enqueue({
                  ...data,
                  choices: [{ ...data.choices[0], delta, index }],
                });
              }
              remaining = "";
              continue;
            }

            // FINAL
            if (remaining.length > 0) {
              if (/^\s*$/.test(remaining)) {
                pendingWhitespace += remaining;
              } else {
                const finalContent = pendingWhitespace + remaining;
                const delta = {
                  ...data.choices[0].delta,
                  content: finalContent,
                };
                if (delta.thinking) {
                  delete delta.thinking;
                }
                enqueue({
                  ...data,
                  choices: [{ ...data.choices[0], delta }],
                });
                pendingWhitespace = "";
              }
            }
            index++;
            remaining = "";
          }
        };

        try {
          for (;;) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.trim()) {
                continue;
              }

              if (line.trim() === "data: [DONE]") {
                controller.enqueue(encoder.encode(line + "\n\n"));
                continue;
              }

              if (line.startsWith("data:")) {
                try {
                  const data = JSON.parse(line.slice(5));
                  processDelta(data, data?.choices?.[0]?.delta?.content);
                } catch {
                  controller.enqueue(encoder.encode(line + "\n"));
                }
              } else {
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
          } catch (error) {
            console.error("Error releasing reader lock:", error);
          }

          if (state === "THINKING") {
            enqueue({
              choices: [
                {
                  delta: {
                    thinking: { signature: Date.now().toString() },
                  },
                },
              ],
            });
          }

          controller.close();
        }
      },
    });

    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        "Content-Type":
          response.headers.get("Content-Type") || "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  }
}

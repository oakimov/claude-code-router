/**
 * Sniff a response body as JSON vs SSE without Response.clone().
 *
 * Cloudflare (Codex) often strips Content-Type from SSE. The first
 * non-whitespace byte is `{`/`[` for a JSON object/array and anything
 * else (typically `d` from `data:`) for SSE. JSON drains into text;
 * SSE returns a new Response that replays the peeked chunk then
 * continues from the same reader.
 */
export type PeekedResponseBody =
  | { kind: "json"; firstChar: string; text: string }
  | { kind: "sse"; response: Response }
  | { kind: "empty" };

export async function peekResponseBody(
  response: Response
): Promise<PeekedResponseBody> {
  const body = response.body;
  if (!body) return { kind: "empty" };

  const reader = body.getReader();
  const decoder = new TextDecoder();
  const cancelReader = async () => {
    try {
      await reader.cancel();
    } catch {
      // ignore
    }
    try {
      reader.releaseLock();
    } catch {
      // already released
    }
  };

  let firstRead: { done: boolean; value?: Uint8Array };
  try {
    firstRead = await reader.read();
  } catch (err) {
    await cancelReader();
    throw err;
  }

  if (firstRead.done || !firstRead.value) {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
    return { kind: "empty" };
  }

  const firstChunk = firstRead.value;
  const firstText = decoder.decode(firstChunk, { stream: true });
  const firstChar = firstText.trimStart().charAt(0) || "";

  if (firstChar === "{" || firstChar === "[") {
    let buffer = firstText;
    try {
      while (true) {
        const r = await reader.read();
        if (r.done) break;
        buffer += decoder.decode(r.value, { stream: true });
      }
      buffer += decoder.decode();
    } catch (err) {
      await cancelReader();
      throw err;
    } finally {
      try {
        reader.releaseLock();
      } catch {
        // ignore
      }
    }
    return { kind: "json", firstChar, text: buffer };
  }

  const replayed = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(firstChunk);
    },
    async pull(controller) {
      try {
        const r = await reader.read();
        if (r.done) {
          controller.close();
          try {
            reader.releaseLock();
          } catch {
            // ignore
          }
          return;
        }
        controller.enqueue(r.value!);
      } catch (err) {
        try {
          await reader.cancel();
        } catch {
          // ignore
        }
        try {
          reader.releaseLock();
        } catch {
          // ignore
        }
        controller.error(err);
      }
    },
    async cancel(reason) {
      try {
        await reader.cancel(reason);
      } catch {
        // ignore
      }
      try {
        reader.releaseLock();
      } catch {
        // ignore
      }
    },
  });

  const headers = new Headers(response.headers);
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "text/event-stream");
  }

  return {
    kind: "sse",
    response: new Response(replayed, {
      status: response.status,
      statusText: response.statusText,
      headers,
    }),
  };
}

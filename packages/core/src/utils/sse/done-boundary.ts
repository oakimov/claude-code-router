/**
 * OpenCode Zen (DeepSeek flash, etc.) appends a cost trailer in the same SSE
 * event as the Chat Completions terminator:
 *
 *   data: [DONE]
 *   data: {"choices":[],"cost":"0"}
 *
 * or on one line:
 *
 *   data: [DONE] {"choices":[],"cost":"0"}
 *
 * EventSource parsers concatenate those into one payload
 * `[DONE]\n{"choices":[],"cost":"0"}` (or with a space), which JSON.parse
 * rejects. Close `[DONE]` as its own event and drop the trailer — it is not a
 * `chat.completion.chunk`, and Chat Completions streams end at `[DONE]`.
 */
export function splitChatCompletionsDoneLine(line: string): string[] {
  const idx = line.indexOf("data:");
  if (idx < 0) return [line];
  const prefix = line.slice(0, idx);
  const payload = line.slice(idx + 5).trim();
  if (!payload.startsWith("[DONE]")) return [line];
  return [`${prefix}data: [DONE]`, ""];
}

function isDoneLine(line: string): boolean {
  return /^data:\s*\[DONE\]\s*$/.test(line);
}

/**
 * Rewrite a Chat Completions SSE body so `[DONE]` always closes its event and
 * nothing follows it. Idempotent on an already-clean stream.
 */
export function withChatCompletionsDoneBoundary(
  body: ReadableStream<Uint8Array>
): ReadableStream<Uint8Array> {
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let sawDone = false;
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

  const emitText = (text: string, flush = false): string => {
    if (sawDone) return "";
    buffer += text;
    const lines = buffer.split(/\r?\n/);
    buffer = flush ? "" : lines.pop() || "";
    const out: string[] = [];
    const pushLine = (line: string) => {
      if (sawDone) return;
      for (const piece of splitChatCompletionsDoneLine(line)) {
        if (sawDone) return;
        if (isDoneLine(piece)) {
          out.push("data: [DONE]");
          out.push("");
          sawDone = true;
          return;
        }
        out.push(piece);
      }
    };
    for (const line of lines) pushLine(line);
    if (flush && buffer) {
      pushLine(buffer);
      buffer = "";
    }
    return out.length ? `${out.join("\n")}\n` : "";
  };

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      reader = body.getReader();
      try {
        while (!sawDone) {
          const { done, value } = await reader.read();
          if (done) {
            const tail = emitText(decoder.decode(), true);
            if (tail) controller.enqueue(encoder.encode(tail));
            controller.close();
            return;
          }
          const rewritten = emitText(decoder.decode(value, { stream: true }));
          if (rewritten) controller.enqueue(encoder.encode(rewritten));
        }
        await reader.cancel();
        controller.close();
      } catch (err) {
        try {
          controller.error(err);
        } catch {
          // Controller already closed.
        }
      } finally {
        try {
          reader?.releaseLock();
        } catch {
          // ignore
        }
      }
    },
    cancel(reason) {
      const r = reader;
      reader = null;
      if (r) return r.cancel(reason).catch(() => {});
      return body.cancel(reason).catch(() => {});
    },
  });
}

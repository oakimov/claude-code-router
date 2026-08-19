/**
 * OpenCode Zen (DeepSeek flash, etc.) glues extra SSE `data:` fields onto the
 * Chat Completions terminator without a blank event separator:
 *
 *   data: [DONE]
 *   data: {"choices":[],"cost":"0"}
 *
 *   data: [DONE] {"choices":[],"cost":"0"}
 *
 *   data: {"choices":[],"usage":{...}}
 *   data: [DONE]
 *
 * EventSource / AI SDK concatenate those fields with `\n` and JSON.parse the
 * result (`[DONE]\n{…}` or `{…}\n[DONE]`). Close `[DONE]` as its own event and
 * drop anything after it — Chat Completions streams end there.
 */
export function splitChatCompletionsDoneLine(line: string): string[] {
  if (/^\s*\[DONE\]/.test(line) && !line.includes("data:")) {
    return ["data: [DONE]", ""];
  }
  const idx = line.indexOf("data:");
  if (idx < 0) return [line];
  const prefix = line.slice(0, idx);
  const payload = line.slice(idx + 5).trim();
  if (!payload) return [line];
  if (payload.startsWith("[DONE]")) {
    return [`${prefix}data: [DONE]`, ""];
  }
  const doneAt = doneTrailerIndex(payload);
  if (doneAt < 0) return [line];
  const json = payload.slice(0, doneAt).trimEnd();
  if (!json) return [`${prefix}data: [DONE]`, ""];
  return [`${prefix}data: ${json}`, "", `${prefix}data: [DONE]`, ""];
}

function doneTrailerIndex(payload: string): number {
  const marker = "[DONE]";
  let from = 0;
  while (from < payload.length) {
    const i = payload.indexOf(marker, from);
    if (i < 0) return -1;
    const before = payload.slice(0, i).trimEnd();
    if (!before) return i;
    try {
      JSON.parse(before);
      return i;
    } catch {
      from = i + marker.length;
    }
  }
  return -1;
}

export function isChatCompletionsDoneLine(line: string): boolean {
  return /^data:\s*\[DONE\]\s*$/.test(line) || /^\s*\[DONE\]\s*$/.test(line);
}

/** Always close the previous SSE event before emitting the terminator. */
export function pushChatCompletionsDone(out: string[]): void {
  out.push("");
  out.push("data: [DONE]");
  out.push("");
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
        if (isChatCompletionsDoneLine(piece)) {
          pushChatCompletionsDone(out);
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

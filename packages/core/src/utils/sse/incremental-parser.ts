/**
 * Incremental SSE event parser that preserves each event's raw bytes/text so
 * unchanged events can be forwarded without re-serialization.
 */

export type ParsedSSEEvent = {
  event?: string;
  id?: string;
  retry?: number;
  /** Parsed JSON, `{ type: "done" }` for [DONE], or `{ raw, error }`. */
  data?: unknown;
  /** Original data field string (without the `data:` prefix), if any. */
  dataRaw?: string;
  /**
   * Exact event block as received, including the blank-line delimiter.
   * Forward this unchanged for byte-preserving passthrough.
   */
  raw: string;
};

const DELIM_RE = /\r?\n\r?\n/;

/**
 * Feed decoded SSE text and emit complete events. Tolerates `\n` and `\r\n`.
 */
export class IncrementalSSEParser {
  private buffer = "";

  push(chunk: string): ParsedSSEEvent[] {
    this.buffer += chunk;
    return this.takeComplete(false);
  }

  flush(): ParsedSSEEvent[] {
    return this.takeComplete(true);
  }

  private takeComplete(flush: boolean): ParsedSSEEvent[] {
    const out: ParsedSSEEvent[] = [];
    while (true) {
      const match = DELIM_RE.exec(this.buffer);
      if (!match) break;
      const end = match.index + match[0].length;
      const raw = this.buffer.slice(0, end);
      this.buffer = this.buffer.slice(end);
      const event = parseEventBlock(raw, true);
      if (event) out.push(event);
    }

    if (flush && this.buffer.length) {
      const event = parseEventBlock(this.buffer, false);
      this.buffer = "";
      if (event) out.push(event);
    }

    return out;
  }
}

function parseEventBlock(
  raw: string,
  hasDelimiter: boolean
): ParsedSSEEvent | null {
  // Strip trailing delimiter for field parsing only; keep `raw` exact.
  const block = hasDelimiter ? raw.replace(/\r?\n\r?\n$/, "") : raw;
  const lines = block.split(/\r?\n/);
  let eventName: string | undefined;
  let id: string | undefined;
  let retry: number | undefined;
  const dataLines: string[] = [];
  let sawField = false;

  for (const line of lines) {
    if (!line || line.startsWith(":")) continue;
    sawField = true;
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      // SSE spec: optional single leading space after the colon.
      const value = line.slice(5);
      dataLines.push(value.startsWith(" ") ? value.slice(1) : value);
    } else if (line.startsWith("id:")) {
      id = line.slice(3).trim();
    } else if (line.startsWith("retry:")) {
      const n = Number.parseInt(line.slice(6).trim(), 10);
      if (Number.isFinite(n)) retry = n;
    }
  }

  if (!sawField) return null;

  const dataRaw = dataLines.length ? dataLines.join("\n") : undefined;
  let data: unknown;
  if (dataRaw !== undefined) {
    if (dataRaw === "[DONE]") {
      data = { type: "done" };
    } else {
      try {
        data = JSON.parse(dataRaw);
      } catch {
        data = { raw: dataRaw, error: "JSON parse failed" };
      }
    }
  }

  return {
    event: eventName,
    id,
    retry,
    data,
    dataRaw,
    raw: hasDelimiter ? raw : raw.endsWith("\n") ? `${raw}\n` : `${raw}\n\n`,
  };
}

/** Serialize a (possibly modified) event. Prefer `event.raw` when unchanged. */
export function serializeSSEEvent(event: {
  event?: string;
  id?: string;
  retry?: number;
  data?: unknown;
  dataRaw?: string;
}): string {
  const lines: string[] = [];
  if (event.event !== undefined) lines.push(`event: ${event.event}`);
  if (event.id !== undefined) lines.push(`id: ${event.id}`);
  if (event.retry !== undefined) lines.push(`retry: ${event.retry}`);
  if (event.data !== undefined) {
    if (
      event.data &&
      typeof event.data === "object" &&
      (event.data as { type?: string }).type === "done"
    ) {
      lines.push("data: [DONE]");
    } else if (
      event.data &&
      typeof event.data === "object" &&
      "error" in (event.data as object) &&
      "raw" in (event.data as object) &&
      typeof (event.data as { raw: unknown }).raw === "string"
    ) {
      lines.push(`data: ${(event.data as { raw: string }).raw}`);
    } else {
      lines.push(`data: ${JSON.stringify(event.data)}`);
    }
  } else if (event.dataRaw !== undefined) {
    lines.push(`data: ${event.dataRaw}`);
  }
  return `${lines.join("\n")}\n\n`;
}

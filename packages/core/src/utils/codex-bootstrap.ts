/**
 * Codex stream bootstrap buffering (CLIProxyAPI `stream-bootstrap-buffering` port).
 *
 * The ChatGPT backend smuggles overload/quota rejections *inside* an HTTP 200
 * SSE stream — right after the handshake events — instead of returning a
 * retryable status on the wire. Once downstream headers are committed that
 * failure can only be delivered to the client; holding the pre-generation
 * bootstrap frames uncommitted keeps the window open for a transparent
 * fallback to the next model.
 *
 * Held (never released on their own): handshake frames
 * (`response.created`, `response.in_progress`, `response.queued`), `*.added`
 * announcements, empty `output_text.delta` heartbeats, SSE comments, `event:`
 * lines and blank separators. Anything else — text deltas, tool-call deltas,
 * `*.done` / `*.completed` / `*.failed` / `error` frames, or an unparsable
 * `data:` line — releases the buffer immediately. The hold is bounded by a
 * frame budget and a byte budget, never by nothing.
 */

export interface CodexBootstrapOptions {
  /** Max held `data:` lines before release. Default 48. */
  maxFrames?: number;
  /** Max held bytes before release. Default 1 MiB. */
  maxBytes?: number;
  /**
   * Max hold time in ms before release. Default 0 (unlimited — the budget
   * bounds what is held, not how long). Evaluated between reads; a peer that
   * stops mid-line is still bounded by the request context, not this timer.
   */
  timeoutMs?: number;
}

/**
 * Capacity failure classes, as Codex CLI distinguishes them: the server is
 * overloaded (503), the request rate is limited (429), or the account's quota
 * or plan does not cover it (429). Each is worth another model; a request
 * failure is not.
 */
export type CodexOverloadKind = "overload" | "rate_limit" | "quota";

export type CodexBootstrapRelease =
  | "generated"
  | "budget"
  | "timeout"
  | "ended"
  | "overload";

export interface CodexBootstrapResult {
  /** Replayable response: held bytes followed by the live remainder. */
  response: Response;
  overloaded: boolean;
  overloadKind?: CodexOverloadKind;
  /** Raw bootstrap text that carried the overload signal (truncated). */
  overloadText?: string;
  /** Structured error fields retained for internal request-scoped matching. */
  overloadErrorText?: string;
  /** Retry-After value (seconds or HTTP date) the failed event advised. */
  overloadRetryAfter?: string;
  heldBytes: number;
  heldFrames: number;
  releasedBy: CodexBootstrapRelease;
}

export const DEFAULT_BOOTSTRAP_MAX_FRAMES = 48;
export const DEFAULT_BOOTSTRAP_MAX_BYTES = 1024 * 1024;
const OVERLOAD_TEXT_SNIPPET_MAX = 2000;

const HOLD_EVENT_TYPES = new Set([
  "response.created",
  "response.in_progress",
  "response.queued",
  "response.output_item.added",
  "response.content_part.added",
]);

/*
 * Codex CLI classifies a failed Responses event by its exact error `code`
 * (plus `type` for plan and quota errors), never by message text
 * (codex-rs `codex-api/src/sse/responses_error.rs`, `api_bridge.rs`).
 * Request failures such as `context_length_exceeded` ("Please adjust your
 * input and try again.") or `invalid_prompt` fail the same way on every
 * model, so anything not listed here reaches the client unchanged.
 */
const OVERLOAD_CODES: ReadonlySet<string> = new Set(["server_is_overloaded"]);
const RATE_LIMIT_CODES: ReadonlySet<string> = new Set([
  "rate_limit_exceeded",
  "slow_down",
  "flex_unavailable",
]);
const QUOTA_CODES: ReadonlySet<string> = new Set([
  "insufficient_quota",
  "credit_balance_exhausted",
  "organization_spend_limit_exceeded",
  "project_spend_limit_exceeded",
  "organization_usage_limit_exceeded",
  "usage_not_included",
]);
const QUOTA_TYPES: ReadonlySet<string> = new Set([
  "usage_limit_reached",
  "usage_not_included",
  "insufficient_quota",
]);

/** Codex's rate-limit delay hint inside the message ("try again in 1.5s"). */
const RETRY_DELAY_RE = /try again in\s*(\d+(?:\.\d+)?)\s*(s|ms|seconds?)/i;

/** Capacity class of a structured Responses error; `message` is ignored. */
export function classifyCodexError(error: {
  code?: unknown;
  type?: unknown;
  message?: unknown;
}): CodexOverloadKind | undefined {
  const code = typeof error.code === "string" ? error.code : undefined;
  const type = typeof error.type === "string" ? error.type : undefined;
  if ((code && QUOTA_CODES.has(code)) || (type && QUOTA_TYPES.has(type))) {
    return "quota";
  }
  if (code && OVERLOAD_CODES.has(code)) return "overload";
  if (code && RATE_LIMIT_CODES.has(code)) return "rate_limit";
  return undefined;
}

/**
 * Retry-After for a failed event: the `retry-after` value carried in
 * `error.headers`, else (rate limits only) the delay in the message.
 */
function retryAfterOf(error: any, kind: CodexOverloadKind): string | undefined {
  const headers = error?.headers;
  if (headers && typeof headers === "object") {
    for (const [name, value] of Object.entries(headers)) {
      if (name.toLowerCase() !== "retry-after") continue;
      const text = Array.isArray(value) ? value[0] : value;
      if (typeof text === "string" && text.trim()) return text.trim();
      if (typeof text === "number" && Number.isFinite(text)) return String(text);
    }
  }
  if (kind !== "rate_limit" || typeof error?.message !== "string") {
    return undefined;
  }
  const match = RETRY_DELAY_RE.exec(error.message);
  if (!match) return undefined;
  const value = Number.parseFloat(match[1]);
  if (!Number.isFinite(value)) return undefined;
  const seconds = match[2].toLowerCase() === "ms" ? value / 1000 : value;
  return String(Math.max(1, Math.ceil(seconds)));
}

function classifyOverloadDataPayload(
  dataPayload: string
):
  | { kind: CodexOverloadKind; text: string; retryAfter?: string }
  | undefined {
  try {
    const event = JSON.parse(dataPayload);
    if (event?.type !== "response.failed" && event?.type !== "error") {
      return undefined;
    }
    const error =
      event.type === "response.failed" ? event.response?.error : event.error ?? event;
    if (!error || typeof error !== "object") return undefined;
    const kind = classifyCodexError(error);
    if (!kind) return undefined;
    const text = [error.code, error.type, error.message]
      .filter((value): value is string => typeof value === "string")
      .join("\n");
    const retryAfter = retryAfterOf(error, kind);
    return { kind, text, ...(retryAfter ? { retryAfter } : {}) };
  } catch {
    return undefined;
  }
}

function releaseReader(reader: ReadableStreamDefaultReader<Uint8Array>): void {
  try {
    reader.releaseLock();
  } catch {
    // Reader already released.
  }
}

async function cancelReader(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  reason?: unknown
): Promise<void> {
  try {
    await reader.cancel(reason);
  } catch {
    // Preserve the original failure even if upstream cleanup rejects.
  } finally {
    releaseReader(reader);
  }
}

function eventTypeOf(dataPayload: string): string | undefined {
  try {
    const parsed = JSON.parse(dataPayload);
    return typeof parsed?.type === "string" ? parsed.type : undefined;
  } catch {
    return undefined;
  }
}

/** True when this `data:` payload is generation (releases the hold). */
function isGeneratedDataPayload(dataPayload: string): boolean {
  const trimmed = dataPayload.trim();
  if (!trimmed || trimmed === "[DONE]") return true;
  const type = eventTypeOf(dataPayload);
  if (type === undefined) return true;
  if (HOLD_EVENT_TYPES.has(type)) return false;
  if (type === "response.output_text.delta") {
    try {
      const parsed = JSON.parse(dataPayload);
      const delta = parsed?.delta;
      if (typeof delta === "string") return delta.length > 0;
      if (delta && typeof delta === "object") {
        const text =
          typeof delta.text === "string"
            ? delta.text
            : typeof delta.delta === "string"
              ? delta.delta
              : "";
        return text.length > 0;
      }
      return false;
    } catch {
      return true;
    }
  }
  return true;
}

/**
 * Buffer the Codex SSE bootstrap. Never throws on upstream content: overload
 * is reported on the result so the caller can raise a fallback-eligible error
 * *before* downstream headers commit. The returned response replays held
 * bytes first, then the untouched remainder of the original stream.
 */
export async function bufferCodexBootstrapStream(
  response: Response,
  options: CodexBootstrapOptions = {},
  signal?: AbortSignal
): Promise<CodexBootstrapResult> {
  const maxFrames = options.maxFrames ?? DEFAULT_BOOTSTRAP_MAX_FRAMES;
  const maxBytes = options.maxBytes ?? DEFAULT_BOOTSTRAP_MAX_BYTES;
  const timeoutMs = options.timeoutMs ?? 0;
  const deadline = timeoutMs > 0 ? Date.now() + timeoutMs : 0;

  const finish = (
    held: Uint8Array[],
    reader: ReadableStreamDefaultReader<Uint8Array> | undefined,
    partial: Omit<CodexBootstrapResult, "response">
  ): CodexBootstrapResult => {
    const status = response.status;
    const statusText = response.statusText;
    const headers = new Headers(response.headers);
    if (!headers.get("Content-Type")) {
      headers.set("Content-Type", "text/event-stream");
    }
    const replay = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of held) controller.enqueue(chunk);
        if (!reader) controller.close();
      },
      async pull(controller) {
        if (!reader) {
          controller.close();
          return;
        }
        try {
          const next = await reader.read();
          if (next.done) {
            releaseReader(reader);
            controller.close();
          } else {
            controller.enqueue(next.value);
          }
        } catch (error) {
          await cancelReader(reader, error);
          controller.error(error);
        }
      },
      async cancel(reason) {
        if (reader) await cancelReader(reader, reason);
      },
    });
    return {
      ...partial,
      response: new Response(replay, { status, statusText, headers }),
    };
  };

  if (!response.body) {
    return finish([], undefined, {
      overloaded: false,
      heldBytes: 0,
      heldFrames: 0,
      releasedBy: "ended",
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const held: Uint8Array[] = [];
  let heldBytes = 0;
  let heldFrames = 0;
  let lineRemainder = "";

  const checkBudget = (): boolean =>
    heldFrames >= maxFrames || heldBytes >= maxBytes;

  const checkTimeout = (): boolean =>
    deadline > 0 && Date.now() >= deadline;

  let releasedBy: CodexBootstrapRelease = "generated";
  let overloadKind: CodexOverloadKind | undefined;
  let overloadText: string | undefined;
  let overloadErrorText: string | undefined;
  let overloadRetryAfter: string | undefined;
  let streamEnded = false;

  try {
    while (true) {
      if (signal?.aborted) {
        releasedBy = "ended";
        break;
      }
      if (checkTimeout()) {
        releasedBy = "timeout";
        break;
      }
      const next = await reader.read();
      if (next.done) {
        streamEnded = true;
        releaseReader(reader);
        releasedBy = "ended";
        break;
      }
      // Hold whole chunks: replay stays byte-exact even when the release
      // line shares a chunk with already-generated output. Parsing below
      // only decides *when* to release, never what is kept.
      held.push(next.value);
      heldBytes += next.value.length;
      const chunkText = decoder.decode(next.value, { stream: true });
      const text = lineRemainder + chunkText;
      const lines = text.split("\n");
      lineRemainder = lines.pop() ?? "";
      let release = false;
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        const overload = classifyOverloadDataPayload(payload);
        if (overload) {
          overloadKind = overload.kind;
          overloadErrorText = overload.text;
          overloadRetryAfter = overload.retryAfter;
          overloadText = line.slice(0, OVERLOAD_TEXT_SNIPPET_MAX);
          heldFrames += 1;
          releasedBy = "overload";
          release = true;
          break;
        }
        heldFrames += 1;
        if (payload === "[DONE]") {
          releasedBy = "ended";
          release = true;
          break;
        }
        if (isGeneratedDataPayload(payload)) {
          releasedBy = "generated";
          release = true;
          break;
        }
        if (checkBudget()) {
          releasedBy = "budget";
          release = true;
          break;
        }
      }
      if (checkTimeout()) releasedBy = "timeout";
      if (release || releasedBy === "timeout" || checkBudget()) {
        if (checkBudget() && !release) releasedBy = "budget";
        break;
      }
    }
  } catch (error) {
    await cancelReader(reader, error);
    throw error;
  }

  // Stream ended while still holding with a trailing partial line (no
  // terminating newline, so it never completed above): scan it for an
  // overload signal. Its bytes are already held — scan only, never re-hold.
  if (streamEnded && lineRemainder.startsWith("data:")) {
    const overload = classifyOverloadDataPayload(lineRemainder.slice(5).trim());
    if (overload && !overloadKind) {
      overloadKind = overload.kind;
      overloadErrorText = overload.text;
      overloadRetryAfter = overload.retryAfter;
      overloadText = lineRemainder.slice(0, OVERLOAD_TEXT_SNIPPET_MAX);
      releasedBy = "overload";
    }
  }

  return finish(held, streamEnded ? undefined : reader, {
    overloaded: overloadKind !== undefined,
    ...(overloadKind
      ? {
          overloadKind,
          overloadText,
          overloadErrorText,
          ...(overloadRetryAfter ? { overloadRetryAfter } : {}),
        }
      : {}),
    heldBytes,
    heldFrames,
    releasedBy,
  });
}

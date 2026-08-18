/**
 * Client-facing SSE keepalive.
 *
 * Claude Code shows "Waiting for API response · will retry in … · check your
 * network" when no bytes arrive on the response stream for 20s (advisor: 90s),
 * even before any retry has started — see Claude Code error docs. Anthropic's
 * own `ping` events often arrive only every ~25–30s during long thinking /
 * tool-argument streams, so a transparent proxy that only forwards upstream
 * bytes trips that spinner on every slow Opus turn.
 *
 * Inject SSE comment frames (`: …\n\n`) after `idleMs` of silence. Comments are
 * transport keepalives: EventSource / Anthropic SSE parsers ignore them, but
 * they reset Claude Code's byte-idle timer. Default 10s is half the 20s warning.
 */

const DEFAULT_IDLE_MS = 10_000;
const KEEPALIVE_BYTES = new TextEncoder().encode(": keepalive\n\n");

export type SSEClientKeepaliveOptions = {
  /** Silence before emitting a comment frame. Default 10_000. */
  idleMs?: number;
};

export function withSSEClientKeepalive(
  body: ReadableStream<Uint8Array>,
  options?: SSEClientKeepaliveOptions
): ReadableStream<Uint8Array> {
  const idleMs = options?.idleMs ?? DEFAULT_IDLE_MS;
  if (!(idleMs > 0)) return body;

  let timer: ReturnType<typeof setTimeout> | null = null;
  let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  let cancelled = false;

  const clear = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      reader = body.getReader();

      const arm = () => {
        clear();
        if (cancelled) return;
        timer = setTimeout(() => {
          timer = null;
          if (cancelled) return;
          try {
            controller.enqueue(KEEPALIVE_BYTES);
          } catch {
            // Controller already closed/errored — stop arming.
            cancelled = true;
            return;
          }
          arm();
        }, idleMs);
      };

      arm();

      try {
        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value && value.byteLength > 0) {
            controller.enqueue(value);
            arm();
          }
        }
        clear();
        if (!cancelled) controller.close();
      } catch (err) {
        clear();
        if (!cancelled) {
          try {
            controller.error(err);
          } catch {
            // ignore
          }
        }
      } finally {
        clear();
        try {
          reader?.releaseLock();
        } catch {
          // ignore
        }
      }
    },
    cancel(reason) {
      cancelled = true;
      clear();
      const r = reader;
      reader = null;
      if (r) {
        return r.cancel(reason).catch(() => {});
      }
      return body.cancel(reason).catch(() => {});
    },
  });
}

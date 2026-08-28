/**
 * Hermetic paced SSE upstream for latency / cadence benchmarks.
 * Emits fixed-cadence Chat Completions SSE events with configurable delays.
 */

export type PacedSSEOptions = {
  /** Delay before response headers resolve (ms). */
  headerDelayMs?: number;
  /** Delay before first SSE body byte after headers (ms). */
  firstByteDelayMs?: number;
  /** Gap between successive events (ms). */
  eventIntervalMs?: number;
  /** Number of content delta events. */
  eventCount?: number;
  /** Split each event across this many TCP-ish chunks (default 1). */
  fragmentChunks?: number;
  /** Abort tracking. */
  signal?: AbortSignal;
  /** Optional terminal Zen-style network_error finish_reason. */
  terminalNetworkError?: boolean;
  /** Fail with this HTTP status instead of streaming. */
  httpErrorStatus?: number;
  httpErrorBody?: string;
};

export type PacedSSEEventStamp = {
  index: number;
  enqueuedAt: number;
};

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
      return;
    }
    const timer = setTimeout(resolve, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(Object.assign(new Error("aborted"), { name: "AbortError" }));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function chatDeltaEvent(index: number, text: string): string {
  return `data: ${JSON.stringify({
    id: "chatcmpl-paced",
    object: "chat.completion.chunk",
    choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
  })}\n\n`;
}

function chatDoneEvent(networkError?: boolean): string {
  if (networkError) {
    return `data: ${JSON.stringify({
      id: "chatcmpl-paced",
      object: "chat.completion.chunk",
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: "network_error",
        },
      ],
    })}\n\n`;
  }
  return `data: ${JSON.stringify({
    id: "chatcmpl-paced",
    object: "chat.completion.chunk",
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  })}\n\ndata: [DONE]\n\n`;
}

/**
 * Build a Response whose body emits Chat Completions SSE at a fixed cadence.
 * `stamps` is filled as each logical event is enqueued (before fragmentation).
 */
export async function createPacedSSEResponse(
  options: PacedSSEOptions = {},
  stamps?: PacedSSEEventStamp[]
): Promise<Response> {
  const headerDelayMs = options.headerDelayMs ?? 0;
  const firstByteDelayMs = options.firstByteDelayMs ?? 0;
  const eventIntervalMs = options.eventIntervalMs ?? 10;
  const eventCount = options.eventCount ?? 20;
  const fragmentChunks = Math.max(1, options.fragmentChunks ?? 1);
  const signal = options.signal;
  const t0 = performance.now();

  await sleep(headerDelayMs, signal);

  if (options.httpErrorStatus) {
    return new Response(options.httpErrorBody || "error", {
      status: options.httpErrorStatus,
      headers: { "Content-Type": "application/json" },
    });
  }

  const encoder = new TextEncoder();
  let cancelled = false;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        await sleep(firstByteDelayMs, signal);
        for (let i = 0; i < eventCount; i++) {
          if (cancelled || signal?.aborted) break;
          if (i > 0) await sleep(eventIntervalMs, signal);
          const event = chatDeltaEvent(i, `t${i}`);
          stamps?.push({ index: i, enqueuedAt: performance.now() - t0 });
          const bytes = encoder.encode(event);
          if (fragmentChunks === 1) {
            controller.enqueue(bytes);
          } else {
            const size = Math.ceil(bytes.length / fragmentChunks);
            for (let f = 0; f < fragmentChunks; f++) {
              const slice = bytes.subarray(f * size, (f + 1) * size);
              if (slice.length) controller.enqueue(slice);
            }
          }
        }
        if (!cancelled && !signal?.aborted) {
          const done = chatDoneEvent(options.terminalNetworkError);
          stamps?.push({
            index: eventCount,
            enqueuedAt: performance.now() - t0,
          });
          controller.enqueue(encoder.encode(done));
          controller.close();
        }
      } catch (error) {
        if (!cancelled) controller.error(error);
      }
    },
    cancel() {
      cancelled = true;
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

/** Read a stream and record when each complete SSE event (blank-line delimited) arrives. */
export async function collectSSEEventTimings(
  body: ReadableStream<Uint8Array>,
  t0 = performance.now()
): Promise<{ events: string[]; arrivalsMs: number[] }> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  const events: string[] = [];
  const arrivalsMs: number[] = [];
  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      pending += decoder.decode();
      if (pending.trim()) {
        events.push(pending);
        arrivalsMs.push(performance.now() - t0);
      }
      break;
    }
    pending += decoder.decode(value, { stream: true });
    const parts = pending.split(/\r?\n\r?\n/);
    pending = parts.pop() || "";
    for (const part of parts) {
      if (!part.trim()) continue;
      events.push(part);
      arrivalsMs.push(performance.now() - t0);
    }
  }
  return { events, arrivalsMs };
}

export function interArrivalGaps(arrivalsMs: number[]): number[] {
  const gaps: number[] = [];
  for (let i = 1; i < arrivalsMs.length; i++) {
    gaps.push(arrivalsMs[i]! - arrivalsMs[i - 1]!);
  }
  return gaps;
}

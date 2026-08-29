/**
 * Request-stage latency timings. Allocation-light monotonic stamps; one
 * terminal structured record per request. No per-event logging, no tee.
 */

export type LatencyStage =
  | "received"
  | "bodyParsed"
  | "normalized"
  | "projectLookup"
  | "tokenizeStart"
  | "tokenizeEnd"
  | "routeSelected"
  | "destinationPolicy"
  | "requestTransformers"
  | "upstreamFetchStart"
  | "upstreamHeaders"
  | "upstreamFirstByte"
  | "responseTransformers"
  | "downstreamFirstByte"
  | "complete";

export type RequestLatency = {
  t0: number;
  stages: Partial<Record<LatencyStage, number>>;
  meta: {
    protocol?: string;
    provider?: string;
    model?: string;
    method?: string;
    url?: string;
    scenario?: string;
    bypass?: boolean;
    wireKeep?: boolean;
    tokenCount?: number;
    tokenCountSource?: "exact" | "estimate" | "skipped";
    inputBytes?: number;
    upstreamAttempts?: number;
    cancelled?: boolean;
    error?: string;
  };
  emitted?: boolean;
};

export function createRequestLatency(): RequestLatency {
  return {
    t0: performance.now(),
    stages: { received: 0 },
    meta: {},
  };
}

export function markLatency(
  latency: RequestLatency | undefined | null,
  stage: LatencyStage
): void {
  if (!latency || latency.emitted || latency.stages[stage] !== undefined) return;
  latency.stages[stage] = performance.now() - latency.t0;
}

export function tapResponseFirstByte(
  response: Response,
  onFirstByte: () => void
): Response {
  if (!response.body) return response;

  const reader = response.body.getReader();
  let firstByteSeen = false;
  let finished = false;
  const body = new ReadableStream<Uint8Array>({
    async pull(controller) {
      if (finished) return;
      try {
        const { done, value } = await reader.read();
        if (done) {
          finished = true;
          controller.close();
          return;
        }
        if (!firstByteSeen) {
          firstByteSeen = true;
          onFirstByte();
        }
        controller.enqueue(value);
      } catch (error) {
        finished = true;
        controller.error(error);
      }
    },
    async cancel(reason) {
      finished = true;
      await reader.cancel(reason).catch(() => undefined);
    },
  });

  return new Response(body, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

export function attachLatencyMeta(
  latency: RequestLatency | undefined | null,
  meta: Partial<RequestLatency["meta"]>
): void {
  if (!latency || latency.emitted) return;
  Object.assign(latency.meta, meta);
}

/** Emit one structured terminal latency record. Safe to call multiple times. */
export function emitLatencyRecord(
  logger: { info?: (obj: unknown, msg?: string) => void } | undefined | null,
  latency: RequestLatency | undefined | null
): void {
  if (!latency || latency.emitted) return;
  latency.emitted = true;
  if (!latency.stages.complete) {
    latency.stages.complete = performance.now() - latency.t0;
  }
  const s = latency.stages;
  const tokenizeMs =
    s.tokenizeStart !== undefined && s.tokenizeEnd !== undefined
      ? s.tokenizeEnd - s.tokenizeStart
      : undefined;
  const upstreamHeaderMs =
    s.upstreamFetchStart !== undefined && s.upstreamHeaders !== undefined
      ? s.upstreamHeaders - s.upstreamFetchStart
      : undefined;
  const ccrTtftMs =
    s.downstreamFirstByte !== undefined ? s.downstreamFirstByte : undefined;
  const upstreamTtftMs =
    s.upstreamFirstByte !== undefined && s.upstreamFetchStart !== undefined
      ? s.upstreamFirstByte - s.upstreamFetchStart
      : undefined;
  const conversionDelayMs =
    s.downstreamFirstByte !== undefined && s.upstreamFirstByte !== undefined
      ? s.downstreamFirstByte - s.upstreamFirstByte
      : undefined;

  logger?.info?.(
    {
      type: "latency",
      ...latency.meta,
      stagesMs: s,
      tokenizeMs,
      upstreamHeaderMs,
      upstreamTtftMs,
      ccrTtftMs,
      conversionDelayMs,
      totalMs: s.complete,
    },
    "request latency"
  );
}

export function ensureRequestLatency(req: {
  _latency?: RequestLatency;
}): RequestLatency {
  if (!req._latency) {
    req._latency = createRequestLatency();
  }
  return req._latency;
}

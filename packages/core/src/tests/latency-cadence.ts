import assert from "node:assert/strict";
import {
  createPacedSSEResponse,
  collectSSEEventTimings,
  interArrivalGaps,
} from "../utils/paced-sse";
import {
  createRequestLatency,
  emitLatencyRecord,
  markLatency,
  tapResponseFirstByte,
} from "../utils/request-latency";
import {
  createSSEStreamReader,
  forwardSSEEvent,
} from "../utils/stream";
import { OpencodeHeadersTransformer } from "../transformer/opencode-headers.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import {
  createResponsesStreamState,
  unifiedChunkToResponsesEvents,
} from "../utils/openai.responses.util";

async function eventNativePassthroughPreservesCadence() {
  const stamps: { index: number; enqueuedAt: number }[] = [];
  const intervalMs = 15;
  const eventCount = 12;
  const upstream = await createPacedSSEResponse(
    {
      headerDelayMs: 0,
      firstByteDelayMs: 5,
      eventIntervalMs: intervalMs,
      eventCount,
    },
    stamps
  );

  const t0 = performance.now();
  const passthrough = createSSEStreamReader(upstream, {
    processEvent(event, ctx) {
      forwardSSEEvent(event, ctx);
    },
  });

  const { events, arrivalsMs } = await collectSSEEventTimings(
    passthrough.body!,
    t0
  );
  // content deltas + terminal done event(s)
  assert.ok(events.length >= eventCount, `expected >= ${eventCount} events, got ${events.length}`);

  const gaps = interArrivalGaps(arrivalsMs.slice(0, eventCount));
  const avgGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  // Allow jitter but reject wholesale coalescing into one burst.
  assert.ok(
    avgGap >= intervalMs * 0.4,
    `avg inter-event gap ${avgGap.toFixed(1)}ms too small (coalesced?)`
  );
  assert.ok(
    avgGap <= intervalMs * 3,
    `avg inter-event gap ${avgGap.toFixed(1)}ms unexpectedly large`
  );
}

async function zenInspectorPreservesChunkBoundaries() {
  const encoder = new TextEncoder();
  const chunkA = encoder.encode(
    'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
  );
  const chunkB = encoder.encode(
    'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
  );
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(chunkA);
      controller.enqueue(chunkB);
      controller.close();
    },
  });
  const wrapped = await new OpencodeHeadersTransformer().transformResponseOut(
    new Response(upstream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const reader = wrapped.body!.getReader();
  const out: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.push(value!);
  }
  assert.equal(out.length, 2);
  assert.deepEqual(Buffer.from(out[0]!), Buffer.from(chunkA));
  assert.deepEqual(Buffer.from(out[1]!), Buffer.from(chunkB));
}

async function pacedMockDetectsCoalescing() {
  // Intentionally coalesce: buffer all events then emit once.
  const stamps: { index: number; enqueuedAt: number }[] = [];
  const upstream = await createPacedSSEResponse(
    { eventIntervalMs: 10, eventCount: 8, firstByteDelayMs: 0 },
    stamps
  );
  const reader = upstream.body!.getReader();
  const chunks: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value!);
  }
  const coalesced = new ReadableStream<Uint8Array>({
    start(controller) {
      const total = chunks.reduce((n, c) => n + c.length, 0);
      const merged = new Uint8Array(total);
      let offset = 0;
      for (const c of chunks) {
        merged.set(c, offset);
        offset += c.length;
      }
      controller.enqueue(merged);
      controller.close();
    },
  });
  const t0 = performance.now();
  const { arrivalsMs } = await collectSSEEventTimings(coalesced, t0);
  const gaps = interArrivalGaps(arrivalsMs);
  const maxGap = gaps.length ? Math.max(...gaps) : 0;
  // All events arrive in one pull → gaps near zero.
  assert.ok(maxGap < 5, `coalesced stream should have tiny gaps, got ${maxGap}`);
}

async function responsesCreatedImmediatelyBecomesUnifiedStart() {
  const encoder = new TextEncoder();
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({
            type: "response.created",
            response: {
              id: "resp-latency",
              object: "response",
              status: "in_progress",
              model: "test-model",
              output: [],
            },
          })}\n\n`
        )
      );
      setTimeout(() => {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({
              type: "response.output_text.delta",
              item_id: "msg-latency",
              delta: "late",
            })}\n\n`
          )
        );
        controller.close();
      }, 80);
    },
  });
  const response = await new OpenAIResponsesTransformer().transformResponseOut(
    new Response(upstream, {
      headers: { "Content-Type": "text/event-stream" },
    })
  );

  const t0 = performance.now();
  const reader = response.body!.getReader();
  const first = await reader.read();
  const firstMs = performance.now() - t0;
  assert.equal(first.done, false);
  assert.ok(firstMs < 40, `response.created was delayed ${firstMs.toFixed(1)}ms`);
  const payload = JSON.parse(
    new TextDecoder().decode(first.value).split("data: ")[1]!.trim()
  );
  assert.equal(payload.choices[0].delta.role, "assistant");
  for (;;) {
    if ((await reader.read()).done) break;
  }
}

async function firstByteTapFiresOnRawProviderBody() {
  const encoder = new TextEncoder();
  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode("data: hello\n\n"));
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } }
  );
  const latency = createRequestLatency();
  markLatency(latency, "upstreamFetchStart");
  const tapped = tapResponseFirstByte(upstream, () => {
    markLatency(latency, "upstreamFirstByte");
  });
  await tapped.text();
  assert.ok(
    latency.stages.upstreamFirstByte !== undefined,
    "raw provider first byte must stamp upstreamFirstByte"
  );
}

async function conversionDelayIsDownstreamMinusUpstream() {
  const records: any[] = [];
  const latency = createRequestLatency();
  latency.stages.upstreamFetchStart = 5;
  latency.stages.upstreamFirstByte = 15;
  latency.stages.downstreamFirstByte = 20;
  emitLatencyRecord({ info(obj) { records.push(obj); } }, latency);
  assert.equal(records[0].conversionDelayMs, 5);
  assert.equal(records[0].upstreamTtftMs, 10);
  assert.equal(records[0].ccrTtftMs, 20);
}

async function unifiedRoleImmediatelyBecomesResponsesCreated() {
  const events = unifiedChunkToResponsesEvents(
    {
      id: "chatcmpl-latency",
      object: "chat.completion.chunk",
      model: "test-model",
      choices: [
        {
          index: 0,
          delta: { role: "assistant" },
          finish_reason: null,
        },
      ],
    },
    createResponsesStreamState()
  );
  assert.equal(events[0]?.type, "response.created");
}

async function main() {
  await eventNativePassthroughPreservesCadence();
  await zenInspectorPreservesChunkBoundaries();
  await pacedMockDetectsCoalescing();
  await responsesCreatedImmediatelyBecomesUnifiedStart();
  await unifiedRoleImmediatelyBecomesResponsesCreated();
  await firstByteTapFiresOnRawProviderBody();
  await conversionDelayIsDownstreamMinusUpstream();
  console.log("latency-cadence: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

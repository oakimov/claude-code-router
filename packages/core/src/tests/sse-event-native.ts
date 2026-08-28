import assert from "node:assert/strict";
import {
  createSSEStreamReader,
  forwardSSEEvent,
} from "../utils/stream";
import { createPacedSSEResponse } from "../utils/paced-sse";

/**
 * Same-protocol / byte-forward path: when no event mutation is needed,
 * downstream bytes must match upstream event framing exactly.
 */
async function sameProtocolByteForward() {
  const upstreamText = [
    'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
    "data: [DONE]\n\n",
  ].join("");
  const encoder = new TextEncoder();
  const upstream = new Response(
    new ReadableStream({
      start(c) {
        // Fragment across chunk boundaries.
        const bytes = encoder.encode(upstreamText);
        c.enqueue(bytes.subarray(0, 17));
        c.enqueue(bytes.subarray(17));
        c.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const out = createSSEStreamReader(upstream, {
    processEvent(event, ctx) {
      forwardSSEEvent(event, ctx);
    },
  });
  const text = await out.text();
  assert.equal(text, upstreamText);
}

async function pacedThroughEventNative() {
  const response = await createPacedSSEResponse({
    eventCount: 3,
    eventIntervalMs: 1,
    firstByteDelayMs: 0,
  });
  const out = createSSEStreamReader(response, {
    processEvent(event, ctx) {
      forwardSSEEvent(event, ctx);
    },
  });
  const text = await out.text();
  assert.ok(text.includes("data: "));
  assert.ok(text.includes("[DONE]") || text.includes("finish_reason"));
}

async function main() {
  await sameProtocolByteForward();
  await pacedThroughEventNative();
  console.log("sse-event-native: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

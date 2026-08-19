/**
 * Chat Completions [DONE] must close its SSE event; OpenCode cost trailers
 * must not be concatenated into the terminator payload.
 */
import assert from "node:assert/strict";
import {
  splitChatCompletionsDoneLine,
  withChatCompletionsDoneBoundary,
} from "../utils/sse/done-boundary";

function encode(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

async function readText(stream: ReadableStream<Uint8Array>): Promise<string> {
  const text = await new Response(stream).text();
  return text;
}

function testSplitDropsSameLineTrailer() {
  assert.deepEqual(splitChatCompletionsDoneLine("data: [DONE]"), [
    "data: [DONE]",
    "",
  ]);
  assert.deepEqual(
    splitChatCompletionsDoneLine('data: [DONE] {"choices":[],"cost":"0"}'),
    ["data: [DONE]", ""]
  );
  assert.deepEqual(
    splitChatCompletionsDoneLine(
      'data: {"id":"c","choices":[{"delta":{"content":"hi"}}]}'
    ),
    ['data: {"id":"c","choices":[{"delta":{"content":"hi"}}]}']
  );
}

async function testBoundarySplitsSameLineAndDropsCost() {
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encode(
          'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
        )
      );
      controller.enqueue(encode('data: [DONE] {"choices":[],"cost":"0"}\n\n'));
      controller.close();
    },
  });
  const text = await readText(withChatCompletionsDoneBoundary(upstream));
  assert.ok(text.includes('"content":"hi"'));
  assert.ok(text.includes("data: [DONE]\n\n"));
  assert.equal(text.includes("[DONE] {"), false);
  assert.equal(text.includes('"cost":"0"'), false);
}

async function testBoundaryClosesTwoLineSameEvent() {
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encode(
          'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\ndata: {"choices":[],"cost":"0"}\n\n'
        )
      );
      controller.close();
    },
  });
  const text = await readText(withChatCompletionsDoneBoundary(upstream));
  assert.ok(text.includes("data: [DONE]\n\n"));
  assert.equal(text.includes('"cost":"0"'), false);
}

async function testBoundaryIsIdempotentOnCleanDone() {
  const payload =
    'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n';
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encode(payload));
      controller.close();
    },
  });
  const text = await readText(withChatCompletionsDoneBoundary(upstream));
  assert.equal((text.match(/data: \[DONE\]/g) || []).length, 1);
  assert.ok(text.includes("data: [DONE]\n\n"));
}

async function main() {
  testSplitDropsSameLineTrailer();
  await testBoundarySplitsSameLineAndDropsCost();
  await testBoundaryClosesTwoLineSameEvent();
  await testBoundaryIsIdempotentOnCleanDone();
  console.log("sse.done-boundary: PASS");
}

process.exitCode = 1;
main().then(
  () => {
    process.exitCode = 0;
  },
  (err) => {
    console.error(err);
    process.exitCode = 1;
  }
);

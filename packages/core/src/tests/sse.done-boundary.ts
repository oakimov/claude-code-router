/**
 * Chat Completions [DONE] must close its SSE event; OpenCode cost trailers
 * and usage chunks must not be concatenated into the terminator payload.
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

/** EventSource / AI SDK concatenate every `data:` field in one event with `\n`. */
function concatenatedDataPayloads(sse: string): string[] {
  return sse
    .split(/\r?\n\r?\n/)
    .map((event) =>
      event
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).replace(/^ /, ""))
        .join("\n")
    )
    .filter(Boolean);
}

function assertDoneIsOwnEvent(sse: string) {
  const payloads = concatenatedDataPayloads(sse);
  const done = payloads.filter((p) => p.includes("[DONE]"));
  assert.equal(done.length, 1, `expected one [DONE] event, got ${JSON.stringify(payloads)}`);
  assert.equal(done[0], "[DONE]");
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
  assert.deepEqual(
    splitChatCompletionsDoneLine(
      'data: {"id":"c","choices":[],"usage":{"prompt_tokens":1}} [DONE]'
    ),
    [
      'data: {"id":"c","choices":[],"usage":{"prompt_tokens":1}}',
      "",
      "data: [DONE]",
      "",
    ]
  );
  assert.deepEqual(splitChatCompletionsDoneLine("[DONE]"), [
    "data: [DONE]",
    "",
  ]);
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
  assertDoneIsOwnEvent(text);
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
  assertDoneIsOwnEvent(text);
  assert.equal(text.includes('"cost":"0"'), false);
}

async function testBoundarySeparatesUsageChunkFromDone() {
  const usage =
    '{"id":"f4c3037970f54888bdc6f7e95d7216a9","object":"chat.completion.chunk","created":1787139553,"model":"deepseek-v4-flash-free","choices":[],"usage":{"prompt_tokens":386,"completion_tokens":42,"total_tokens":428,"prompt_tokens_details":{}}}';
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encode(`data: ${usage}\ndata: [DONE]\n`));
      controller.close();
    },
  });
  const text = await readText(withChatCompletionsDoneBoundary(upstream));
  assert.ok(text.includes(usage));
  assertDoneIsOwnEvent(text);
  assert.equal(
    concatenatedDataPayloads(text).some((p) => p.includes("{") && p.includes("[DONE]")),
    false
  );
}

async function testBoundarySeparatesUsageChunkFromDoneAcrossChunks() {
  const usage = '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}';
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encode(`data: ${usage}\n`));
      controller.enqueue(encode("data: [DONE]\n"));
      controller.close();
    },
  });
  const text = await readText(withChatCompletionsDoneBoundary(upstream));
  assertDoneIsOwnEvent(text);
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
  assertDoneIsOwnEvent(text);
}

async function main() {
  testSplitDropsSameLineTrailer();
  await testBoundarySplitsSameLineAndDropsCost();
  await testBoundaryClosesTwoLineSameEvent();
  await testBoundarySeparatesUsageChunkFromDone();
  await testBoundarySeparatesUsageChunkFromDoneAcrossChunks();
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

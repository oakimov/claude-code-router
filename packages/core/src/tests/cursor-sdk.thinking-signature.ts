import assert from "node:assert/strict";
import {
  accumulateChatCompletion,
  createSseHelpers,
} from "../cursor-sdk/events-to-sse";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

function deltaOf(chunk: Record<string, unknown>) {
  return ((chunk.choices as any)?.[0]?.delta || {}) as Record<string, any>;
}

{
  const helpers = createSseHelpers("grok-4.5", new TextEncoder());
  const thinking = helpers.thinking("step one");
  const signature = helpers.thinkingSignature("ccr_cursor_test");
  const content = helpers.content("hello");

  assert.equal(deltaOf(thinking).thinking?.content, "step one");
  assert.equal(deltaOf(signature).thinking?.signature, "ccr_cursor_test");
  assert.equal(deltaOf(content).content, "hello");

  // AnthropicTransformer prefers signature branch when present; signature-only
  // chunks must not carry content or the thinking_delta would be skipped.
  assert.equal(deltaOf(signature).thinking?.content, undefined);
  assert.equal(deltaOf(signature).content, undefined);
}

{
  const helpers = createSseHelpers("grok-4.5", new TextEncoder());
  const completion = accumulateChatCompletion("grok-4.5", [
    helpers.thinking("plan A"),
    helpers.thinking(" then B"),
    helpers.thinkingSignature("ccr_cursor_fixed"),
    helpers.content("done"),
    helpers.finish("stop", {
      prompt_tokens: 10,
      completion_tokens: 2,
      total_tokens: 12,
    }),
  ]);

  const message = (completion.choices as any)[0].message;
  assert.equal(message.content, "done");
  assert.deepEqual(message.thinking, {
    content: "plan A then B",
    signature: "ccr_cursor_fixed",
  });
}

{
  // Streaming path without an explicit signature still needs a non-empty
  // fallback so Anthropic request replay keeps the thinking block.
  const helpers = createSseHelpers("grok-4.5", new TextEncoder());
  const completion = accumulateChatCompletion("grok-4.5", [
    helpers.thinking("unsigned reasoning"),
    helpers.content("answer"),
  ]);
  const message = (completion.choices as any)[0].message;
  assert.equal(message.thinking.content, "unsigned reasoning");
  assert.equal(typeof message.thinking.signature, "string");
  assert.ok(message.thinking.signature.length > 0);
  assert.match(message.thinking.signature, /^ccr_cursor_/);
}

{
  // Verify the actual Cursor SDK -> Unified SSE -> Anthropic SSE boundary.
  // Thinking bytes alone are insufficient: Claude Code requires the signature
  // delta before the thinking block closes and before the text block starts.
  const helpers = createSseHelpers("grok-4.5", new TextEncoder());
  const unifiedChunks = [
    helpers.thinking("inspect the stream"),
    helpers.thinkingSignature("ccr_cursor_pipeline"),
    helpers.content("final answer"),
    helpers.finish("stop", {
      prompt_tokens: 10,
      completion_tokens: 5,
      total_tokens: 15,
    }),
  ];
  const upstreamBody =
    unifiedChunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("") +
    "data: [DONE]\n\n";

  const transformer = new AnthropicTransformer();
  transformer.logger = { debug() {}, error() {} };
  const response = await transformer.transformResponseIn(
    new Response(upstreamBody, {
      headers: { "Content-Type": "text/event-stream" },
    }),
    { req: { id: "cursor-thinking-regression" } } as any
  );
  const output = await response.text();

  const thinkingStart = output.indexOf(
    '"content_block":{"type":"thinking","thinking":""}'
  );
  const thinkingDelta = output.indexOf(
    '"delta":{"type":"thinking_delta","thinking":"inspect the stream"}'
  );
  const signatureDelta = output.indexOf(
    '"delta":{"type":"signature_delta","signature":"ccr_cursor_pipeline"}'
  );
  const thinkingStop = output.indexOf(
    'event: content_block_stop',
    signatureDelta
  );
  const textStart = output.indexOf(
    '"content_block":{"type":"text","text":""}'
  );

  assert.ok(thinkingStart >= 0);
  assert.ok(thinkingDelta > thinkingStart);
  assert.ok(signatureDelta > thinkingDelta);
  assert.ok(thinkingStop > signatureDelta);
  assert.ok(textStart > thinkingStop);
  assert.match(output, /"delta":\{"type":"text_delta","text":"final answer"\}/);
}

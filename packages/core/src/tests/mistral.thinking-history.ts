/**
 * Mistral outbound thinking history: Anthropic thinking, Chat
 * reasoning_content, and native ThinkChunks all become Mistral content
 * arrays. Unified-only fields are stripped.
 */
import assert from "node:assert/strict";
import { buildRequestBody } from "../utils/mistral.util";

function assistantOf(body: any) {
  return body.messages.find((message: any) => message.role === "assistant");
}

function testAnthropicThinkingBecomesThinkChunk() {
  const body = buildRequestBody({
    model: "magistral-medium-latest",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "ok",
        thinking: { content: "plan first", signature: "sig" },
      },
    ],
    reasoning: { effort: "high" },
    thinking: { type: "enabled" },
    enable_thinking: true,
  } as any);

  const assistant = assistantOf(body);
  assert.deepEqual(assistant.content, [
    { type: "thinking", thinking: [{ type: "text", text: "plan first" }] },
    { type: "text", text: "ok" },
  ]);
  assert.equal(assistant.thinking, undefined);
  assert.equal(assistant.reasoning_content, undefined);
  assert.equal(body.reasoning, undefined);
  assert.equal(body.thinking, undefined);
  assert.equal(body.enable_thinking, undefined);
  assert.equal(body.reasoning_effort, "high");
}

function testChatReasoningContentBecomesThinkChunk() {
  const body = buildRequestBody({
    model: "mistral-large-latest",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "ok",
        reasoning_content: "plan first",
      },
    ],
  } as any);

  const assistant = assistantOf(body);
  assert.deepEqual(assistant.content[0], {
    type: "thinking",
    thinking: [{ type: "text", text: "plan first" }],
  });
  assert.equal(assistant.reasoning_content, undefined);
}

function testNativeThinkChunkIsPreserved() {
  const native = [
    { type: "thinking", thinking: [{ type: "text", text: "already mistral" }] },
    { type: "text", text: "answer" },
  ];
  const body = buildRequestBody({
    model: "magistral-medium-latest",
    messages: [
      { role: "user", content: "hi" },
      { role: "assistant", content: native },
    ],
  } as any);

  assert.deepEqual(assistantOf(body).content, native);
}

function testThinkingOnlyAssistantKeepsToolCalls() {
  const body = buildRequestBody({
    model: "magistral-medium-latest",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: null,
        thinking: { content: "need a tool" },
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "Read", arguments: "{}" },
          },
        ],
      },
    ],
  } as any);

  const assistant = assistantOf(body);
  assert.deepEqual(assistant.content, [
    { type: "thinking", thinking: [{ type: "text", text: "need a tool" }] },
  ]);
  assert.equal(assistant.tool_calls[0].function.name, "Read");
  assert.equal(assistant.thinking, undefined);
}

function testPlainUserContentStaysString() {
  const body = buildRequestBody({
    model: "mistral-large-latest",
    messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
  } as any);
  assert.equal(body.messages[0].content, "hi");
}

function main() {
  testAnthropicThinkingBecomesThinkChunk();
  testChatReasoningContentBecomesThinkChunk();
  testNativeThinkChunkIsPreserved();
  testThinkingOnlyAssistantKeepsToolCalls();
  testPlainUserContentStaysString();
  console.log("mistral.thinking-history: PASS");
}

main();

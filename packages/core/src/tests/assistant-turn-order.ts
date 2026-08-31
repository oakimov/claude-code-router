/**
 * Assistant / system turn order is fixed so cache prefixes stay stable:
 *   system/instructions first, then conversation;
 *   inside an assistant turn: thinking → text → images → tools.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { CodexTransformer } from "../transformer/codex.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { buildRequestBody as buildGeminiBody } from "../utils/gemini.util";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";
import { unifiedResponseToResponses } from "../utils/openai.responses.util";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

function mixedAssistant() {
  return {
    role: "assistant" as const,
    content: [
      { type: "text", text: "visible" },
      {
        type: "image_url",
        image_url: { url: "https://example.test/a.png" },
      },
    ],
    thinking: { content: "plan", signature: "sig" },
    tool_calls: [
      {
        id: "call_1",
        type: "function" as const,
        function: { name: "Read", arguments: "{}" },
      },
    ],
  };
}

async function testAnthropicOrder() {
  const body = AnthropicTransformer.buildAnthropicBody(
    {
      model: "claude-sonnet-4-20250514",
      system: [{ type: "text", text: "sys-top" }],
      messages: [
        { role: "system", content: "sys-residual" },
        { role: "user", content: "hi" },
        mixedAssistant() as any,
      ],
      stream: false,
    } as any,
    logger
  );
  assert.deepEqual(
    body.system.map((part: any) => part.text),
    ["sys-top", "sys-residual"]
  );
  const assistant = body.messages.find((m: any) => m.role === "assistant");
  assert.deepEqual(
    assistant.content.map((part: any) => part.type),
    ["thinking", "text", "image", "tool_use"]
  );
}

async function testResponsesOrder() {
  const tf = new OpenAIResponsesTransformer();
  const result = await tf.transformRequestIn(
    {
      model: "grok-4.6",
      messages: [
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
        mixedAssistant(),
        { role: "tool", tool_call_id: "call_1", content: "ok" },
      ],
    } as any,
    {},
    {}
  );
  const types = (result as any).input.map((item: any) => item.type || item.role);
  const assistantSlice = types.slice(
    types.indexOf("reasoning"),
    types.indexOf("function_call") + 1
  );
  assert.deepEqual(assistantSlice, ["reasoning", "message", "function_call"]);
}

async function testCodexKeepsTextWithTools() {
  const unified = await new OpenAIResponsesTransformer().transformRequestIn(
    {
      model: "gpt-5.4",
      messages: [
        { role: "system", content: "sys-a" },
        { role: "system", content: "sys-b" },
        { role: "user", content: "hi" },
        mixedAssistant(),
        { role: "tool", tool_call_id: "call_1", content: "ok" },
      ],
    } as any,
    {},
    {}
  );
  const tf = new CodexTransformer();
  (tf as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "t",
    accountId: "a",
    isFedramp: false,
  });
  const result = await tf.transformRequestIn(
    unified,
    { baseUrl: "https://example.test" },
    {}
  );
  const input = result.body.input;
  assert.equal(result.body.instructions, "sys-a\n\nsys-b");
  assert.ok(
    !input.some((item: any) => item.role === "system" || item.role === "developer"),
    "Codex input must not contain role:system/developer"
  );
  const types = input.map((item: any) => item.type);
  const start = types.indexOf("reasoning");
  assert.deepEqual(types.slice(start, start + 3), [
    "reasoning",
    "message",
    "function_call",
  ]);
  assert.equal(input[start + 1].content[0].text, "visible");
}

async function testMistralAndGeminiOrder() {
  const mistral = buildMistralBody({
    model: "magistral-medium-latest",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          { type: "text", text: "visible" },
          { type: "thinking", thinking: [{ type: "text", text: "late" }] },
        ],
      },
    ],
  } as any);
  assert.deepEqual(
    mistral.messages[1].content.map((part: any) => part.type),
    ["thinking", "text"]
  );

  const gemini = buildGeminiBody({
    model: "gemini-3-flash",
    messages: [
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
      mixedAssistant() as any,
    ],
  });
  assert.equal(gemini.systemInstruction.parts[0].text, "sys");
  const model = gemini.contents.find((c: any) => c.role === "model");
  const kinds = model.parts.map((part: any) =>
    part.thought ? "thought" : part.functionCall ? "functionCall" : part.inlineData || part.fileData ? "image" : "text"
  );
  assert.deepEqual(kinds[0], "thought");
  assert.ok(kinds.includes("text"));
  assert.ok(kinds.includes("functionCall"));
  assert.ok(kinds.indexOf("thought") < kinds.indexOf("text"));
  assert.ok(kinds.indexOf("text") < kinds.indexOf("functionCall"));
}

function testCompletedResponsesOrder() {
  const responses = unifiedResponseToResponses({
    id: "c1",
    choices: [
      {
        message: {
          role: "assistant",
          content: "visible",
          thinking: { content: "plan", signature: "sig" },
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "Read", arguments: "{}" },
            },
          ],
        },
      },
    ],
  });
  assert.deepEqual(
    responses.output.map((item: any) => item.type),
    ["reasoning", "message", "function_call"]
  );
}

async function main() {
  await testAnthropicOrder();
  await testResponsesOrder();
  await testCodexKeepsTextWithTools();
  await testMistralAndGeminiOrder();
  testCompletedResponsesOrder();
  console.log("assistant-turn-order: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

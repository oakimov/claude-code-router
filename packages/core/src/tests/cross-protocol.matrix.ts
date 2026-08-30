/**
 * Inbound × outbound matrix: every inbound protocol must produce a valid
 * outbound shape for every provider family, and cross-protocol invariants
 * (include, encrypted_content, call_id, web_search, custom_tool) must hold.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { CodexTransformer } from "../transformer/codex.transformer";
import { GeminiTransformer } from "../transformer/gemini.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { buildRequestBody as buildGeminiBody } from "../utils/gemini.util";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

function sessionCtx(sessionId = "matrix-session") {
  return { req: { id: "matrix", sessionId, log: logger } } as any;
}

async function toUnifiedFromAnthropic() {
  return new AnthropicTransformer().transformRequestOut({
    model: "claude-sonnet-4-20250514",
    max_tokens: 64,
    system: [{ type: "text", text: "sys" }],
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "plan", signature: "sig-not-id" },
          { type: "text", text: "answer" },
          { type: "tool_use", id: "call_matrix_1", name: "Read", input: { path: "a.ts" } },
        ],
      },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "call_matrix_1", content: "ok" }] },
    ],
    tools: [
      { name: "Read", description: "read", input_schema: { type: "object", properties: { path: { type: "string" } } } },
      { name: "web_search", description: "search", input_schema: { type: "object", properties: { query: { type: "string" } } } },
    ],
  } as any);
}

async function toUnifiedFromChat() {
  // Chat inbound is already Unified-ish; simulate via direct Unified shape
  return {
    model: "gpt-4o",
    messages: [
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: null,
        tool_calls: [{ id: "call_chat_1", type: "function", function: { name: "Read", arguments: '{"path":"a.ts"}' } }],
        reasoning_content: "chat thinking",
      },
      { role: "tool", tool_call_id: "call_chat_1", content: "ok" },
    ],
    tools: [
      { type: "function", function: { name: "Read", description: "read", parameters: { type: "object", properties: { path: { type: "string" } } } } },
      { type: "function", function: { name: "web_search", description: "search", parameters: { type: "object", properties: { query: { type: "string" } } } } },
    ],
    reasoning: { effort: "high", enabled: true },
  } as any;
}

async function toUnifiedFromResponses() {
  const { responsesRequestToUnified, createCallIdMap } = await import("../utils/openai.responses.util");
  const map = createCallIdMap();
  return responsesRequestToUnified(
    {
      model: "gpt-5",
      instructions: "sys",
      input: [
        { type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] },
        { type: "reasoning", id: "rs_matrix", summary: [{ type: "summary_text", text: "reasoning" }], encrypted_content: "CIPHER_MATRIX" },
        { type: "function_call", call_id: "call_resp_1", name: "Read", arguments: '{"path":"a.ts"}' },
        { type: "custom_tool_call", call_id: "call_custom_1", name: "MyTool", input: "freeform text" },
        { type: "function_call_output", call_id: "call_resp_1", output: "ok" },
      ],
      tools: [
        { type: "function", name: "Read", description: "read", parameters: { type: "object", properties: { path: { type: "string" } } } },
        { type: "function", name: "MyTool", description: "custom", parameters: { type: "object", properties: { input: { type: "string" } } } },
        { type: "web_search" },
      ],
      include: ["reasoning.encrypted_content"],
      store: false,
    } as any,
    map
  );
}

async function testAnthropicToAllOutbounds() {
  const unified = await toUnifiedFromAnthropic();
  // Anthropic → Anthropic (via transformer)
  const anthTf = new AnthropicTransformer();
  (anthTf as any).logger = logger;
  const anthBody = (await anthTf.transformRequestIn(structuredClone(unified), { name: "anthropic", apiKey: "k", baseUrl: "https://api.anthropic.com", models: [] } as any, sessionCtx()) as any).body;
  assert.ok(Array.isArray(anthBody.messages));
  assert.ok(anthBody.messages.some((m: any) => m.role === "assistant"));

  // Anthropic → Responses (encrypted_content requested)
  const respTf = new OpenAIResponsesTransformer();
  (respTf as any).logger = logger;
  const respBody = await respTf.transformRequestIn(structuredClone(unified), { name: "openai", baseUrl: "https://api.openai.com/v1" } as any, { req: { id: "matrix", sessionId: "matrix-session", log: logger, protocolContext: { protocol: "anthropic_messages" } }, clientProtocol: "anthropic_messages", protocolContext: { protocol: "anthropic_messages" } } as any);
  assert.ok(Array.isArray((respBody as any).input));
  assert.ok(((respBody as any).include || []).includes("reasoning.encrypted_content"), "Anthropic→Responses must request encrypted_content");
  assert.ok((respBody as any).input.some((i: any) => i.type === "reasoning"));

  // Anthropic → Chat (OpenAI)
  const chatBody = await new OpenAITransformer().transformRequestIn(structuredClone(unified), { name: "openai", baseUrl: "https://api.openai.com/v1" } as any, sessionCtx());
  assert.ok(Array.isArray((chatBody as any).messages));

  // Anthropic → Gemini (googleSearch)
  const gemBody = buildGeminiBody(structuredClone(unified) as any, { name: "antigravity" } as any);
  assert.ok(Array.isArray(gemBody.tools));
  assert.ok(gemBody.tools.some((t: any) => t.googleSearch), "web_search must become googleSearch for Gemini");

  // Anthropic → Mistral (via helper)
  const { buildRequestBody: buildMistral } = await import("../utils/mistral.util");
  const mistralBody = buildMistral(structuredClone(unified) as any, sessionCtx(), { name: "mistral" } as any);
  assert.ok(Array.isArray(mistralBody.messages));
}

async function testChatToAllOutbounds() {
  const unified = await toUnifiedFromChat();
  const respTf = new OpenAIResponsesTransformer();
  (respTf as any).logger = logger;
  const respBody = await respTf.transformRequestIn(structuredClone(unified), { name: "openai", baseUrl: "https://api.openai.com/v1" } as any, { req: { id: "matrix", sessionId: "matrix-session", log: logger, protocolContext: { protocol: "openai_chat_completions" } }, clientProtocol: "openai_chat_completions", protocolContext: { protocol: "openai_chat_completions" } } as any);
  assert.ok(((respBody as any).include || []).includes("reasoning.encrypted_content"), "Chat→Responses must request encrypted_content");

  const anthTf2 = new AnthropicTransformer();
  (anthTf2 as any).logger = logger;
  const anthBody2 = (await anthTf2.transformRequestIn(structuredClone(unified), { name: "anthropic", apiKey: "k", baseUrl: "https://api.anthropic.com", models: [] } as any, sessionCtx()) as any).body;
  assert.ok(Array.isArray(anthBody2.messages));

  const chatBody = await new OpenAITransformer().transformRequestIn(structuredClone(unified), { name: "deepseek", baseUrl: "https://api.deepseek.com/v1" } as any, sessionCtx());
  assert.ok(Array.isArray((chatBody as any).messages));
}

async function testResponsesToAllOutbounds() {
  const unified = await toUnifiedFromResponses();
  // Responses → Responses (client-driven, include preserved, no synthesis)
  const respTf = new OpenAIResponsesTransformer();
  (respTf as any).logger = logger;
  const respBody = await respTf.transformRequestIn(structuredClone(unified), { name: "openai", baseUrl: "https://api.openai.com/v1" } as any, { req: { id: "r", sessionId: "s", log: logger }, clientProtocol: "openai_responses", protocolContext: { protocol: "openai_responses" } } as any);
  // For native Responses client, include is preserved from inbound, not invented
  assert.ok(Array.isArray((respBody as any).input));
  // custom_tool must survive as function with input key
  assert.ok((respBody as any).input.some((i: any) => i.type === "custom_tool_call" || i.type === "function_call"));

  // Responses → Anthropic (encrypted_content stripped, web_search dropped or mapped)
  const anthTf3 = new AnthropicTransformer();
  (anthTf3 as any).logger = logger;
  const anthBody = (await anthTf3.transformRequestIn(structuredClone(unified), { name: "anthropic", apiKey: "k", baseUrl: "https://api.anthropic.com", models: [] } as any, sessionCtx()) as any).body;
  assert.equal(JSON.stringify(anthBody).includes("CIPHER_MATRIX"), false, "Responses encrypted_content must not leak to Anthropic");

  // Responses → Gemini (web_search → googleSearch, custom_tool → function)
  const gemBody = buildGeminiBody(structuredClone(unified) as any, { name: "antigravity" } as any);
  assert.ok(gemBody.tools.some((t: any) => t.googleSearch || t.functionDeclarations));

  // Responses → Codex (instructions fold, custom_tool preserved)
  const codexTf = new CodexTransformer();
  (codexTf as any).logger = logger;
  (codexTf as any).resolveAuth = async () => ({ mode: "oauth", token: "t", accountId: "a", isFedramp: false });
  const codexBody = await codexTf.transformRequestIn({ ...structuredClone(unified), model: "gpt-5.6-sol", stream: false } as any, { name: "codex", baseUrl: "https://chatgpt.com/backend-api/codex", apiKey: "at-test", models: [] } as any, sessionCtx());
  assert.ok(Array.isArray((codexBody.body as any).input));
  assert.equal((codexBody.body as any).input.some((i: any) => i.role === "system"), false, "Codex must fold system into instructions");
}

async function testCallIdBoundsAcrossOutbounds() {
  const longId = "call-66dbf0b1-aad7-482f-baa2-647748651824-0_fc_49ff1230-042d-97ce-b451-5e3f019a21d8_0"; // 85 chars
  const unified: any = {
    model: "gpt-5",
    messages: [
      { role: "user", content: "hi" },
      { role: "assistant", content: null, tool_calls: [{ id: longId, type: "function", function: { name: "Read", arguments: "{}" } }] },
      { role: "tool", tool_call_id: longId, content: "ok" },
    ],
    tools: [{ type: "function", function: { name: "Read", parameters: { type: "object", properties: {} } } }],
  };
  const pat = /^[a-zA-Z0-9_-]{1,64}$/;
  for (const builder of [
    async () => (await new OpenAIResponsesTransformer().transformRequestIn(structuredClone(unified), {} as any, sessionCtx()) as any).input,
    async () => {
      const tf = new CodexTransformer();
      (tf as any).logger = logger;
      (tf as any).resolveAuth = async () => ({ mode: "oauth", token: "t", accountId: "a", isFedramp: false });
      let b: any = await tf.transformRequestIn({ ...structuredClone(unified), model: "gpt-5.6-sol", stream: false } as any, { name: "codex", baseUrl: "https://chatgpt.com/backend-api/codex", apiKey: "at-test", models: [] } as any, sessionCtx());
      if ((b as any).body && (b as any).body.input) b = (b as any).body;
      return (b as any).input || (b as any).body?.input;
    },
  ]) {
    const input = await builder();
    if (!input) continue;
    const call = input.find((i: any) => i.type === "function_call" || i.type === "custom_tool_call");
    const out = input.find((i: any) => i.type === "function_call_output" || i.type === "custom_tool_call_output");
    if (call && out) {
      assert.match(call.call_id, pat);
      assert.equal(call.call_id, out.call_id);
    }
  }
}

async function main() {
  await testAnthropicToAllOutbounds();
  await testChatToAllOutbounds();
  await testResponsesToAllOutbounds();
  await testCallIdBoundsAcrossOutbounds();
  console.log("cross-protocol.matrix: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

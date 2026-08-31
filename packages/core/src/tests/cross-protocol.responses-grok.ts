/**
 * Cross-protocol matrix for the Responses/Grok incorporation:
 * inbound Anthropic | Chat | Responses → outbound Anthropic | Chat |
 * Responses | Codex | Gemini | Mistral, plus provider→client response legs.
 *
 * Catches thinking/ciphertext leaks, dropped tools, and duplicated reasoning
 * that unit tests scoped to a single transformer can miss.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { CodexTransformer } from "../transformer/codex.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { buildRequestBody as buildGeminiBody } from "../utils/gemini.util";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

const PLAN = "plan first";
const ANSWER = "visible answer";
const ANTHROPIC_SIG = "anth-sig-not-an-id";
const CIPHER = "enc-blob-not-an-id";
const REASONING_ID = "rs_abc123";

function sse(payloads: string[]): Response {
  const body = payloads.map((line) => `data: ${line}`).join("\n\n") + "\n\n";
  return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
}

function parseChatSse(text: string): any[] {
  const chunks: any[] = [];
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    chunks.push(JSON.parse(data));
  }
  return chunks;
}

type AnthropicSSE = { event: string; data: any };

function parseAnthropicSse(text: string): AnthropicSSE[] {
  const events: AnthropicSSE[] = [];
  for (const block of text.split("\n\n")) {
    const eventLine = block.split("\n").find((line) => line.startsWith("event: "));
    const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
    if (!eventLine || !dataLine) continue;
    const raw = dataLine.slice(6);
    if (raw === "[DONE]") continue;
    events.push({ event: eventLine.slice(7), data: JSON.parse(raw) });
  }
  return events;
}

async function inboundAnthropic() {
  const tf = new AnthropicTransformer();
  return tf.transformRequestOut({
    model: "claude-sonnet-4-20250514",
    max_tokens: 64,
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: PLAN, signature: ANTHROPIC_SIG },
          { type: "text", text: ANSWER },
          {
            type: "tool_use",
            id: "call_1",
            name: "Read",
            input: { path: "a.ts" },
          },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call_1", content: "ok" },
        ],
      },
    ],
    tools: [
      {
        name: "Read",
        description: "read",
        input_schema: { type: "object", properties: { path: { type: "string" } } },
      },
    ],
  });
}

async function inboundChat() {
  const tf = new OpenAITransformer();
  return tf.transformRequestOut({
    model: "gpt-4o",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: ANSWER,
        reasoning_content: PLAN,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "Read", arguments: '{"path":"a.ts"}' },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "ok" },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "Read",
          description: "read",
          parameters: { type: "object", properties: { path: { type: "string" } } },
        },
      },
    ],
  });
}

async function inboundResponses() {
  const tf = new OpenAIResponsesTransformer();
  return tf.transformRequestOut({
    model: "gpt-5.6-sol",
    input: [
      { role: "user", content: "hi" },
      {
        type: "reasoning",
        id: REASONING_ID,
        summary: [{ type: "summary_text", text: PLAN }],
        encrypted_content: CIPHER,
      },
      {
        type: "function_call",
        call_id: "call_1",
        name: "Read",
        arguments: '{"path":"a.ts"}',
      },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: "ok",
      },
    ],
  });
}

async function toAnthropic(unified: any) {
  return AnthropicTransformer.buildAnthropicBody(
    { ...unified, model: unified.model || "claude-sonnet-4-20250514", max_tokens: 64 },
    logger
  );
}

async function toChat(unified: any) {
  const tf = new OpenAITransformer();
  return tf.transformRequestIn(
    structuredClone(unified),
    { name: "openai" },
    {}
  );
}

async function toResponses(unified: any) {
  const tf = new OpenAIResponsesTransformer();
  return tf.transformRequestIn(structuredClone(unified), {}, {});
}

async function toCodex(unified: any) {
  const responses = await new OpenAIResponsesTransformer().transformRequestIn(
    structuredClone(unified),
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
    responses,
    { baseUrl: "https://example.test" },
    {}
  );
  return result.body;
}

function assistantAnthropic(body: any) {
  return body.messages.find((m: any) => m.role === "assistant");
}

function assistantChat(body: any) {
  return body.messages.find((m: any) => m.role === "assistant");
}

function reasoningItem(body: any) {
  return (body.input || []).find((item: any) => item.type === "reasoning");
}

function functionCall(body: any) {
  return (body.input || []).find(
    (item: any) => item.type === "function_call" || item.type === "custom_tool_call"
  );
}

async function testRequestMatrixThinkingAndTools() {
  const sources = [
    ["anthropic", await inboundAnthropic()],
    ["chat", await inboundChat()],
    ["responses", await inboundResponses()],
  ] as const;

  for (const [source, unified] of sources) {
    const assistant =
      unified.messages.find(
        (m: any) => m.role === "assistant" && (m.thinking || m.reasoning_content)
      ) || unified.messages.find((m: any) => m.role === "assistant");
    assert.ok(assistant, `${source} inbound produced an assistant turn`);
    assert.ok(
      assistant.thinking?.content === PLAN || assistant.reasoning_content === PLAN,
      `${source} inbound must carry thinking text`
    );
    assert.ok(assistant.tool_calls?.length, `${source} inbound must carry tool calls`);

    const anthropic = await toAnthropic(unified);
    const anthAssistant = assistantAnthropic(anthropic);
    const thinkingBlock = anthAssistant.content.find((b: any) => b.type === "thinking");
    assert.ok(thinkingBlock, `${source}→Anthropic keeps thinking`);
    assert.equal(thinkingBlock.thinking, PLAN, `${source}→Anthropic thinking text`);
    assert.ok(
      anthAssistant.content.some((b: any) => b.type === "tool_use"),
      `${source}→Anthropic keeps tool_use`
    );
    assert.notEqual(
      thinkingBlock.signature,
      CIPHER,
      `${source}→Anthropic must not put ciphertext on signature`
    );
    assert.notEqual(
      thinkingBlock.signature,
      REASONING_ID,
      `${source}→Anthropic must not put rs_ id on signature`
    );
    if (source === "anthropic") {
      assert.equal(thinkingBlock.signature, ANTHROPIC_SIG);
    } else {
      assert.equal(
        thinkingBlock.signature,
        undefined,
        `${source}→Anthropic must not invent a signature (official API may 400 on unsigned thinking)`
      );
    }

    const chat = await toChat(unified);
    const chatAssistant = assistantChat(chat);
    assert.equal(chatAssistant.reasoning_content, PLAN, `${source}→Chat reasoning_content`);
    assert.equal(chatAssistant.thinking, undefined, `${source}→Chat drops Unified thinking`);
    assert.ok(chatAssistant.tool_calls?.length, `${source}→Chat keeps tool_calls`);
    assert.equal(
      chatAssistant.encrypted_content,
      undefined,
      `${source}→Chat must not leak ciphertext onto the message`
    );

    const responses = await toResponses(unified);
    const reasoning = reasoningItem(responses);
    assert.ok(reasoning, `${source}→Responses reasoning item`);
    assert.equal(reasoning.summary[0].text, PLAN);
    assert.notEqual(reasoning.encrypted_content, ANTHROPIC_SIG);
    if (source === "responses") {
      assert.equal(reasoning.encrypted_content, CIPHER);
      assert.equal(reasoning.id, REASONING_ID);
    } else {
      assert.equal(
        reasoning.encrypted_content,
        undefined,
        `${source}→Responses must not mint ciphertext from a signature`
      );
    }
    assert.ok(functionCall(responses), `${source}→Responses keeps function_call`);

    const codex = await toCodex(unified);
    const codexReasoning = (codex.input || []).find((item: any) => item.type === "reasoning");
    assert.ok(codexReasoning, `${source}→Codex reasoning item`);
    assert.equal(codexReasoning.summary[0].text, PLAN);
    assert.notEqual(codexReasoning.encrypted_content, ANTHROPIC_SIG);
    if (source === "responses") {
      assert.equal(codexReasoning.encrypted_content, CIPHER);
    }

    const gemini = buildGeminiBody({
      ...unified,
      model: "gemini-3-flash",
    });
    const model = gemini.contents.find((c: any) => c.role === "model");
    assert.ok(model, `${source}→Gemini model turn`);
    assert.ok(
      model.parts.some((p: any) => p.functionCall),
      `${source}→Gemini keeps functionCall`
    );
    const thought = model.parts.find((p: any) => p.thought === true);
    if (source === "anthropic") {
      assert.ok(thought, "Anthropic signature authorizes a Gemini thought part");
      assert.equal(thought.text, PLAN);
      assert.equal(thought.thoughtSignature, ANTHROPIC_SIG);
    } else {
      assert.equal(
        thought,
        undefined,
        `${source}→Gemini must not invent an unsigned thought part`
      );
    }

    const mistral = buildMistralBody({
      ...unified,
      model: "magistral-medium-latest",
    });
    const mistralAssistant = mistral.messages.find((m: any) => m.role === "assistant");
    const mistralThinking = Array.isArray(mistralAssistant.content)
      ? mistralAssistant.content.find((p: any) => p.type === "thinking")
      : undefined;
    assert.ok(mistralThinking, `${source}→Mistral ThinkChunk`);
    assert.equal(mistralThinking.thinking[0].text, PLAN);
    assert.equal(mistralAssistant.thinking, undefined);
    assert.equal(mistralAssistant.reasoning_content, undefined);
  }
}

async function testChatProviderReasoningReachesAnthropicAndResponsesClients() {
  // Chat Completions providers have no transformResponseOut — Unified IS
  // Chat Completions. Anthropic and Responses clients consume the upstream
  // SSE (or JSON) directly, including native `reasoning_content`.
  const stream = sse([
    JSON.stringify({
      id: "c1",
      object: "chat.completion.chunk",
      choices: [{ index: 0, delta: { reasoning_content: PLAN }, finish_reason: null }],
    }),
    JSON.stringify({
      id: "c1",
      object: "chat.completion.chunk",
      choices: [{ index: 0, delta: { content: ANSWER }, finish_reason: "stop" }],
    }),
    "[DONE]",
  ]);

  const anthropic = new AnthropicTransformer();
  (anthropic as any).logger = logger;
  const anthOut = await anthropic.transformResponseIn(stream.clone(), {
    req: { id: "xproto-anth" },
  } as any);
  const anthEvents = parseAnthropicSse(await anthOut.text());
  const thinking = anthEvents
    .filter(
      (e) =>
        e.event === "content_block_delta" && e.data.delta.type === "thinking_delta"
    )
    .map((e) => e.data.delta.thinking)
    .join("");
  const text = anthEvents
    .filter(
      (e) => e.event === "content_block_delta" && e.data.delta.type === "text_delta"
    )
    .map((e) => e.data.delta.text)
    .join("");
  const blockTypes = anthEvents
    .filter((e) => e.event === "content_block_start")
    .map((e) => e.data.content_block.type);
  assert.equal(thinking, PLAN, "Chat reasoning_content must become Anthropic thinking");
  assert.equal(text, ANSWER);
  assert.deepEqual(blockTypes, ["thinking", "text"]);

  const responses = new OpenAIResponsesTransformer();
  (responses as any).logger = logger;
  const respOut = await responses.transformResponseIn(stream, {
    protocolContext: { originalModel: "gpt-4o" },
  } as any);
  const events = parseChatSse(await respOut.text()).filter(
    (event) => typeof event.type === "string"
  );
  const completed = events.find((event) => event.type === "response.completed");
  assert.ok(completed, "Responses client stream completes");
  const reasoning = completed.response.output.find((item: any) => item.type === "reasoning");
  assert.equal(reasoning.summary[0].text, PLAN);
  const message = completed.response.output.find((item: any) => item.type === "message");
  assert.equal(message.content[0].text, ANSWER);

  const jsonUpstream = new Response(
    JSON.stringify({
      id: "chatcmpl-1",
      object: "chat.completion",
      choices: [
        {
          message: {
            role: "assistant",
            content: ANSWER,
            reasoning_content: PLAN,
          },
          finish_reason: "stop",
        },
      ],
    }),
    { headers: { "Content-Type": "application/json" } }
  );
  const anthJson: any = await (
    await anthropic.transformResponseIn(jsonUpstream.clone(), {
      req: { id: "xproto-anth-json" },
    } as any)
  ).json();
  const thinkingBlock = anthJson.content.find((b: any) => b.type === "thinking");
  assert.equal(thinkingBlock?.thinking, PLAN);
  const respJson: any = await (
    await responses.transformResponseIn(jsonUpstream, {
      protocolContext: { originalModel: "gpt-4o" },
    } as any)
  ).json();
  const jsonReasoning = respJson.output.find((item: any) => item.type === "reasoning");
  assert.equal(jsonReasoning.summary[0].text, PLAN);
}

async function testResponsesProviderReasoningReachesChatClientOnce() {
  const provider = new OpenAIResponsesTransformer();
  (provider as any).logger = logger;
  const unified = await provider.transformResponseOut(
    sse([
      `{"type":"response.reasoning_summary_text.delta","item_id":"${REASONING_ID}","delta":"${PLAN}"}`,
      JSON.stringify({
        type: "response.output_item.done",
        item: {
          id: REASONING_ID,
          type: "reasoning",
          summary: [{ type: "summary_text", text: PLAN }],
          encrypted_content: CIPHER,
        },
      }),
      `{"type":"response.output_text.delta","item_id":"msg_1","delta":"${ANSWER}"}`,
      JSON.stringify({
        type: "response.completed",
        response: {
          id: "resp_1",
          model: "grok-4.6",
          output: [
            {
              type: "reasoning",
              id: REASONING_ID,
              summary: [{ type: "summary_text", text: PLAN }],
              encrypted_content: CIPHER,
            },
            {
              type: "message",
              id: "msg_1",
              content: [{ type: "output_text", text: ANSWER }],
            },
          ],
        },
      }),
    ])
  );

  const chat = new OpenAITransformer();
  const chatOut = await chat.transformResponseIn(unified, {} as any);
  const chunks = parseChatSse(await chatOut.text());
  const thinking = chunks
    .map((chunk) => chunk.choices?.[0]?.delta?.thinking?.content || "")
    .join("");
  const reasoning = chunks
    .map((chunk) => chunk.choices?.[0]?.delta?.reasoning_content || "")
    .join("");
  const text = chunks
    .map((chunk) => chunk.choices?.[0]?.delta?.content || "")
    .join("");
  assert.equal(
    thinking,
    "",
    "Chat Completions clients must not see Unified thinking"
  );
  assert.equal(reasoning, PLAN, "Chat client must see reasoning_content");
  assert.equal(text, ANSWER);
  const cipher = chunks
    .map((chunk) => chunk.choices?.[0]?.delta?.thinking?.encrypted_content)
    .find(Boolean);
  assert.equal(
    cipher,
    undefined,
    "Chat Completions clients must not see Responses ciphertext"
  );
}

async function testAnthropicProviderThinkingReachesChatAndResponsesClients() {
  const provider = new AnthropicTransformer();
  (provider as any).logger = logger;
  const unified = await provider.transformResponseOut(
    new Response(
      JSON.stringify({
        id: "msg_1",
        type: "message",
        role: "assistant",
        model: "claude-sonnet-4-20250514",
        content: [
          { type: "thinking", thinking: PLAN, signature: ANTHROPIC_SIG },
          { type: "text", text: ANSWER },
        ],
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
      { headers: { "Content-Type": "application/json" } }
    )
  );

  const chat = new OpenAITransformer();
  const chatJson: any = await (
    await chat.transformResponseIn(unified.clone(), {} as any)
  ).json();
  assert.equal(chatJson.choices[0].message.thinking, undefined);
  assert.equal(chatJson.choices[0].message.reasoning_content, PLAN);
  assert.equal(chatJson.choices[0].message.content, ANSWER);

  const responses = new OpenAIResponsesTransformer();
  (responses as any).logger = logger;
  const respJson: any = await (
    await responses.transformResponseIn(unified, {
      protocolContext: { originalModel: "claude-sonnet-4-20250514" },
    } as any)
  ).json();
  const reasoning = respJson.output.find((item: any) => item.type === "reasoning");
  assert.equal(reasoning.summary[0].text, PLAN);
  assert.equal(
    reasoning.encrypted_content,
    undefined,
    "Anthropic signature must not become Responses ciphertext"
  );
}

async function testResponsesJsonSchemaSurvivesChatAndAnthropic() {
  const inbound = new OpenAIResponsesTransformer();
  const unified = await inbound.transformRequestOut({
    model: "gpt-4o",
    input: "json please",
    text: {
      format: {
        type: "json_schema",
        name: "result",
        schema: { type: "object", properties: { ok: { type: "boolean" } } },
        strict: true,
      },
    },
  });

  const chat = (await toChat(unified)) as any;
  assert.equal(chat.response_format.type, "json_schema");
  assert.equal(chat.response_format.json_schema.name, "result");

  const anthropic = await toAnthropic(unified);
  assert.deepEqual(anthropic.output_config.format, {
    type: "json_schema",
    schema: { type: "object", properties: { ok: { type: "boolean" } } },
  });

  const responses = (await toResponses(unified)) as any;
  assert.deepEqual(responses.text.format, {
    type: "json_schema",
    name: "result",
    schema: { type: "object", properties: { ok: { type: "boolean" } } },
    strict: true,
  });

  const chatIn = await new OpenAITransformer().transformRequestOut({
    model: "gpt-4o",
    messages: [{ role: "user", content: "json please" }],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "result",
        schema: { type: "object", properties: { ok: { type: "boolean" } } },
        strict: true,
      },
    },
  });
  const chatToAnthropic = await toAnthropic(chatIn);
  assert.deepEqual(chatToAnthropic.output_config.format, {
    type: "json_schema",
    schema: { type: "object", properties: { ok: { type: "boolean" } } },
  });
  const chatToResponses = (await toResponses(chatIn)) as any;
  assert.equal(chatToResponses.text.format.name, "result");
  assert.equal(chatToResponses.text.format.strict, true);

  const anthIn = await new AnthropicTransformer().transformRequestOut({
    model: "claude-sonnet-4-20250514",
    max_tokens: 64,
    messages: [{ role: "user", content: "json please" }],
    output_config: {
      format: {
        type: "json_schema",
        name: "result",
        schema: { type: "object", properties: { ok: { type: "boolean" } } },
        strict: true,
      },
    },
  });
  const anthToChat = (await toChat(anthIn)) as any;
  assert.equal(anthToChat.response_format.type, "json_schema");
  assert.equal(anthToChat.response_format.json_schema.name, "result");
  const anthToResponses = (await toResponses(anthIn)) as any;
  assert.equal(anthToResponses.text.format.type, "json_schema");
  assert.deepEqual(anthToResponses.text.format.schema, {
    type: "object",
    properties: { ok: { type: "boolean" } },
  });
}

async function main() {
  await testRequestMatrixThinkingAndTools();
  await testChatProviderReasoningReachesAnthropicAndResponsesClients();
  await testResponsesProviderReasoningReachesChatClientOnce();
  await testAnthropicProviderThinkingReachesChatAndResponsesClients();
  await testResponsesJsonSchemaSurvivesChatAndAnthropic();
  console.log("cross-protocol.responses-grok: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

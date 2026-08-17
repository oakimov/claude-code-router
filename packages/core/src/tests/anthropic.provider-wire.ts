/**
 * Provider-side Anthropic ↔ Unified conversion.
 * Unified → Anthropic (transformRequestIn / buildAnthropicBody) and
 * Anthropic → Unified (transformResponseOut).
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import type { UnifiedChatRequest } from "../types/llm";
import { responsesRequestToUnified } from "../utils/openai.responses.util";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

function makeUnified(): UnifiedChatRequest {
  return {
    model: "claude-sonnet-4-20250514",
    messages: [
      { role: "system", content: "be helpful" },
      { role: "user", content: "hello" },
      {
        role: "assistant",
        content: "hi",
        tool_calls: [
          {
            id: "call_abc",
            type: "function",
            function: { name: "Read", arguments: '{"path":"a.ts"}' },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_abc",
        content: "file contents",
      },
      {
        role: "user",
        content: [
          { type: "text", text: "look" },
          {
            type: "image_url",
            image_url: {
              url: "data:image/png;base64,aaaa",
            },
          },
        ],
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "Read",
          description: "read a file",
          parameters: { type: "object", properties: { path: { type: "string" } } },
        },
      },
    ],
    tool_choice: "auto",
    temperature: 0.2,
    max_tokens: 1024,
    stream: false,
    reasoning: { enabled: true, effort: "high" },
  } as UnifiedChatRequest;
}

async function testUnifiedToAnthropicRequest() {
  const tf = new AnthropicTransformer();
  tf.logger = logger;

  const result = await tf.transformRequestIn(makeUnified(), {
    name: "anthropic",
    apiKey: "sk-test",
    baseUrl: "https://api.anthropic.com",
  } as any);

  assert.ok(result.body);
  assert.ok(result.config?.url?.includes("/v1/messages"));
  assert.equal(result.config.headers["x-api-key"], "sk-test");
  assert.equal(result.config.headers["anthropic-version"], "2023-06-01");

  const body = result.body;
  assert.equal(body.model, "claude-sonnet-4-20250514");
  assert.equal(body.system, "be helpful");
  assert.ok(Array.isArray(body.messages));
  assert.ok(body.messages.some((m: any) => m.role === "user"));
  assert.ok(
    body.messages.some(
      (m: any) =>
        m.role === "assistant" &&
        Array.isArray(m.content) &&
        m.content.some((c: any) => c.type === "tool_use")
    )
  );
  assert.ok(
    body.messages.some(
      (m: any) =>
        m.role === "user" &&
        Array.isArray(m.content) &&
        m.content.some((c: any) => c.type === "tool_result")
    )
  );
  assert.ok(
    body.messages.some(
      (m: any) =>
        m.role === "user" &&
        Array.isArray(m.content) &&
        m.content.some((c: any) => c.type === "image")
    )
  );
  assert.ok(Array.isArray(body.tools));
  assert.equal(body.tools[0].name, "Read");
  assert.equal(body.tool_choice.type, "auto");
  assert.equal(body.max_tokens, 1024);
  assert.equal(body.temperature, 0.2);
}

async function testBuildAnthropicBodyRoundTripViaClientOut() {
  const tf = new AnthropicTransformer();
  tf.logger = logger;

  const anthropicWire = {
    model: "claude-sonnet-4-20250514",
    system: "sys",
    max_tokens: 100,
    messages: [{ role: "user", content: "hi" }],
    stream: false,
  };

  const unified = await tf.transformRequestOut(anthropicWire, {
    req: { id: "t1" },
  } as any);

  const rebuilt = AnthropicTransformer.buildAnthropicBody(unified, logger);
  assert.equal(rebuilt.model, "claude-sonnet-4-20250514");
  assert.ok(
    rebuilt.system === "sys" ||
      (Array.isArray(rebuilt.system) &&
        rebuilt.system.some((s: any) => s.text === "sys"))
  );
  assert.ok(rebuilt.messages.some((m: any) => m.role === "user"));
}

async function testAnthropicJsonResponseToUnified() {
  const tf = new AnthropicTransformer();
  tf.logger = logger;

  const anthropicResponse = {
    id: "msg_test",
    type: "message",
    role: "assistant",
    model: "claude-sonnet-4-20250514",
    content: [
      { type: "text", text: "hello world" },
      {
        type: "tool_use",
        id: "toolu_1",
        name: "Read",
        input: { path: "a.ts" },
      },
    ],
    stop_reason: "tool_use",
    stop_sequence: null,
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_read_input_tokens: 2,
      cache_creation_input_tokens: 1,
    },
  };

  const unified = await tf.transformResponseOut(
    new Response(JSON.stringify(anthropicResponse), {
      headers: { "Content-Type": "application/json" },
    }),
    { req: { id: "t2" } } as any
  );

  const json = await unified.json();
  assert.equal(json.object, "chat.completion");
  assert.equal(json.choices[0].message.content, "hello world");
  assert.ok(json.choices[0].message.tool_calls?.length >= 1);
  assert.equal(json.choices[0].message.tool_calls[0].function.name, "Read");
  assert.equal(json.choices[0].finish_reason, "tool_calls");
  assert.ok(json.usage.prompt_tokens >= 10);
  assert.equal(json.usage.completion_tokens, 5);
}

async function testAnthropicSseResponseToUnified() {
  const tf = new AnthropicTransformer();
  tf.logger = logger;

  const sse = [
    'event: message_start',
    'data: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":3,"output_tokens":0}}}',
    "",
    'event: content_block_start',
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "",
    'event: content_block_delta',
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
    "",
    'event: content_block_stop',
    'data: {"type":"content_block_stop","index":0}',
    "",
    'event: message_delta',
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}',
    "",
    'event: message_stop',
    'data: {"type":"message_stop"}',
    "",
  ].join("\n");

  const unified = await tf.transformResponseOut(
    new Response(sse, {
      headers: { "Content-Type": "text/event-stream" },
    }),
    { req: { id: "t3" } } as any
  );

  assert.ok(
    (unified.headers.get("Content-Type") || "").includes("text/event-stream")
  );
  const text = await unified.text();
  assert.ok(text.includes("chat.completion.chunk") || text.includes("delta"));
  assert.ok(text.includes("[DONE]") || text.includes("hi"));
  const chunks = text
    .split("\n")
    .filter((line) => line.startsWith("data: {") && line !== "data: [DONE]")
    .map((line) => JSON.parse(line.slice(6)));
  assert.equal(chunks[0].choices[0].delta.role, "assistant");
  assert.ok(chunks.every((chunk) => chunk.id === "msg_s"));
  assert.equal(chunks.at(-1).choices[0].finish_reason, "stop");
  assert.ok(
    chunks
      .filter((chunk) => chunk.choices[0].delta.content)
      .every((chunk) => chunk.usage === undefined)
  );
}

async function testStopSequenceMapsToStop() {
  const tf = new AnthropicTransformer();
  const unified = await tf.transformResponseOut(
    new Response(
      JSON.stringify({
        id: "msg_stop",
        type: "message",
        role: "assistant",
        model: "claude-test",
        content: [{ type: "text", text: "done" }],
        stop_reason: "stop_sequence",
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
      { headers: { "Content-Type": "application/json" } }
    )
  );
  assert.equal((await unified.json()).choices[0].finish_reason, "stop");
}

async function testThinkingPreserved() {
  const unified: UnifiedChatRequest = {
    model: "claude-sonnet-4-20250514",
    messages: [
      {
        role: "assistant",
        content: "answer",
        thinking: { content: "reason", signature: "sig" },
      } as any,
    ],
    max_tokens: 64,
    anthropic_thinking: { type: "enabled", budget_tokens: 1024 },
  } as any;

  const body = AnthropicTransformer.buildAnthropicBody(unified, logger);
  assert.ok(body.thinking);
  const assistant = body.messages.find((m: any) => m.role === "assistant");
  assert.ok(
    assistant.content.some(
      (c: any) => c.type === "thinking" && c.thinking === "reason"
    )
  );
}

async function testMultipleSystemMessagesAndToolContent() {
  const body = AnthropicTransformer.buildAnthropicBody({
    model: "claude-test",
    messages: [
      { role: "system", content: "first" },
      { role: "system", content: "second" },
      {
        role: "tool",
        tool_call_id: "call_1",
        content: [{ type: "text", text: "structured result" }],
      },
    ],
    stream: false,
  } as any);
  assert.deepEqual(
    body.system.map((part: any) => part.text),
    ["first", "second"]
  );
  assert.equal(body.messages[0].content[0].type, "tool_result");
  assert.equal(
    body.messages[0].content[0].content[0].text,
    "structured result"
  );
}

async function testAnthropicSourcePreservesExactCacheAndFields() {
  const tf = new AnthropicTransformer();
  const wire = {
    model: "claude-opus-5",
    max_tokens: 128,
    stream: false,
    system: [
      {
        type: "text",
        text: "billing",
      },
      {
        type: "text",
        text: "cached system",
        cache_control: { type: "ephemeral", ttl: "1h" },
      },
    ],
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "hello",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
    ],
    metadata: { user_id: "source-user" },
    thinking: { type: "adaptive" },
    output_config: { effort: "high" },
    stop_sequences: ["STOP"],
  };
  const protocolContext: any = { protocol: "anthropic_messages" };
  const unified = await tf.transformRequestOut(wire, { protocolContext } as any);
  const rebuilt = AnthropicTransformer.buildAnthropicBody(
    unified,
    logger,
    { protocolContext } as any
  );

  assert.deepEqual(rebuilt.system, wire.system);
  assert.deepEqual(rebuilt.metadata, wire.metadata);
  assert.deepEqual(rebuilt.thinking, wire.thinking);
  assert.deepEqual(rebuilt.output_config, wire.output_config);
  assert.deepEqual(rebuilt.stop_sequences, wire.stop_sequences);
  assert.equal(rebuilt.cache_control, undefined);
  assert.deepEqual(rebuilt.messages[0].content[0].cache_control, {
    type: "ephemeral",
  });
}

async function testAnthropicBodyBuilderDoesNotInventCaching() {
  const body = AnthropicTransformer.buildAnthropicBody(makeUnified(), logger, {
    protocolContext: {
      protocol: "openai_chat_completions",
    },
  } as any);
  assert.equal(body.cache_control, undefined);
}

async function testExactAuthPreservesBodyAndNormalizesUrl() {
  const tf = new AnthropicTransformer();
  const request = {
    model: "claude-opus-5",
    max_tokens: 64,
    messages: [{ role: "user", content: "hi" }],
    cache_control: undefined,
  };
  const result = await tf.auth(
    request,
    {
      name: "anthropic",
      apiKey: "sk-test",
      baseUrl: "https://api.anthropic.com/root?existing=1",
    } as any,
    {
      protocolContext: {
        protocol: "anthropic_messages",
      },
    } as any
  );
  assert.equal(result.body, request);
  const url = new URL(result.config.url);
  assert.equal(url.pathname, "/root/v1/messages");
  assert.equal(url.searchParams.get("existing"), "1");
  assert.equal(url.searchParams.get("beta"), "true");
  assert.equal(result.config.headers["x-api-key"], "sk-test");
}

async function testClaudeAuthChainOwnsAuthNotWireBuild() {
  const tf = new AnthropicTransformer();
  const request = makeUnified();
  const provider = {
    name: "subscription",
    transformer: {
      use: [{ name: "claude-auth" }, tf],
    },
  } as any;

  // This stage still owns building the wire body and URL regardless of who
  // owns auth — only claude-auth's non-Claude-Code branch previously did
  // this work, and Step 8 relocated it here unconditionally.
  const requestResult = await tf.transformRequestIn(request, provider);
  assert.notEqual(requestResult.body, request);
  assert.equal(requestResult.body.model, "claude-sonnet-4-20250514");
  assert.equal(
    requestResult.config.url,
    "https://api.anthropic.com/v1/messages?beta=true"
  );
  // claude-auth owns Content-Type/anthropic-version/auth for this chain;
  // this stage must not introduce or clobber them.
  assert.equal(requestResult.config.headers["Authorization"], undefined);
  assert.equal(requestResult.config.headers["x-api-key"], undefined);

  const response = new Response('{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}', {
    headers: { "Content-Type": "application/json" },
  });
  const responseResult = await tf.transformResponseOut(response, {
    provider,
  } as any);
  assert.notEqual(responseResult, response);
  const converted = await responseResult.json();
  assert.equal(converted.model, "claude-sonnet-4-20250514");
}

async function testTopLevelSystemPlusResidualSystemMessagesMerge() {
  // Fix 3: a populated request.system must not cause residual
  // system/developer messages to be dropped — everything merges in order.
  const body = AnthropicTransformer.buildAnthropicBody(
    {
      model: "claude-test",
      system: [{ type: "text", text: "top-level" }],
      messages: [
        { role: "system", content: "residual-system" },
        {
          role: "developer",
          content: [
            {
              type: "text",
              text: "residual-developer",
              cache_control: { type: "ephemeral" },
            },
          ],
        },
        { role: "user", content: "hi" },
      ],
      stream: false,
    } as any,
    logger
  );
  assert.ok(Array.isArray(body.system));
  assert.deepEqual(
    body.system.map((part: any) => part.text),
    ["top-level", "residual-system", "residual-developer"]
  );
  assert.deepEqual(body.system[2].cache_control, { type: "ephemeral" });
  assert.equal(body.messages.length, 1);
}

async function testResponsesJsonSchemaMapsToOutputConfigFormat() {
  const unified = responsesRequestToUnified({
    model: "claude-sonnet-4-20250514",
    input: "give me json",
    text: {
      format: {
        type: "json_schema",
        name: "result",
        schema: { type: "object", properties: { ok: { type: "boolean" } } },
        strict: true,
      },
    },
    reasoning: { effort: "high" },
  });
  const body = AnthropicTransformer.buildAnthropicBody(unified, logger);
  assert.equal(body.output_config.effort, "high");
  assert.deepEqual(body.output_config.format, {
    type: "json_schema",
    schema: { type: "object", properties: { ok: { type: "boolean" } } },
  });
}

async function testResponsesJsonObjectOmitsOutputFormat() {
  const unified = responsesRequestToUnified({
    model: "claude-sonnet-4-20250514",
    input: "give me json",
    text: { format: { type: "json_object" } },
  });
  const body = AnthropicTransformer.buildAnthropicBody(unified, logger);
  assert.equal(body.output_config, undefined);
}

async function testResponsesParallelToolCallsDisableOnAnthropic() {
  const unified = responsesRequestToUnified({
    model: "claude-sonnet-4-20250514",
    input: "use a tool",
    parallel_tool_calls: false,
    tools: [
      {
        type: "function",
        name: "Read",
        description: "read",
        parameters: { type: "object", properties: {} },
      },
    ],
  });
  const body = AnthropicTransformer.buildAnthropicBody(unified, logger);
  assert.equal(body.tool_choice.type, "auto");
  assert.equal(body.tool_choice.disable_parallel_tool_use, true);
}

async function testAnthropicThinkingHistoryMapsToResponsesReasoning() {
  const anthropic = new AnthropicTransformer();
  const unified = await anthropic.transformRequestOut({
    model: "claude-sonnet-4-20250514",
    max_tokens: 64,
    messages: [
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "plan it", signature: "anth-sig" },
          { type: "text", text: "ok" },
        ],
      },
    ],
  });
  const responses = new OpenAIResponsesTransformer();
  const result = await responses.transformRequestIn(unified, {}, {});
  const reasoning = (result as any).input.find((item: any) => item.type === "reasoning");
  assert.ok(reasoning);
  assert.equal(reasoning.summary[0].text, "plan it");
  // Anthropic signatures are not Codex ciphertext — omit encrypted_content.
  assert.equal(reasoning.encrypted_content, undefined);
  assert.ok((result as any).input.every((item: any) => !item.thinking));
}

async function testAnthropicStructuredOutputMapsToUnifiedResponseFormat() {
  const tf = new AnthropicTransformer();
  const unified = await tf.transformRequestOut({
    model: "claude-sonnet-4-20250514",
    max_tokens: 64,
    messages: [{ role: "user", content: "json please" }],
    output_config: {
      effort: "high",
      format: {
        type: "json_schema",
        name: "result",
        schema: { type: "object", properties: { ok: { type: "boolean" } } },
        strict: true,
      },
    },
    tool_choice: { type: "any", disable_parallel_tool_use: true },
    tools: [
      {
        name: "Read",
        description: "read",
        input_schema: { type: "object", properties: {} },
      },
    ],
  });
  assert.equal(unified.tool_choice, "required");
  assert.equal(unified.parallel_tool_calls, false);
  assert.deepEqual((unified as any).response_format, {
    type: "json_schema",
    json_schema: {
      name: "result",
      schema: { type: "object", properties: { ok: { type: "boolean" } } },
      strict: true,
    },
  });
}

async function testResponsesCustomToolsAreAnthropicInputSchema() {
  const unified = responsesRequestToUnified({
    model: "claude-sonnet-4-20250514",
    input: [
      { role: "user", content: "patch it" },
      {
        type: "custom_tool_call",
        call_id: "ct_1",
        name: "apply_patch",
        input: "*** Begin Patch\n*** End Patch",
      },
      {
        type: "custom_tool_call_output",
        call_id: "ct_1",
        output: "ok",
      },
    ],
    tools: [
      {
        type: "custom",
        name: "apply_patch",
        description: "Apply a patch.",
      },
      {
        type: "custom",
        name: "exec",
        description: "Run JavaScript.",
      },
    ],
  });
  const body = AnthropicTransformer.buildAnthropicBody(unified, logger);
  const patch = body.tools.find((t: any) => t.name === "apply_patch");
  const exec = body.tools.find((t: any) => t.name === "exec");
  assert.ok(patch.input_schema.properties.input);
  assert.ok(exec.input_schema.properties.input);
  assert.ok(!exec.description.includes("await tools."));
  const assistant = body.messages.find((m: any) => m.role === "assistant");
  const toolUse = assistant.content.find((p: any) => p.type === "tool_use");
  assert.equal(toolUse.name, "apply_patch");
  assert.equal(toolUse.input.input, "*** Begin Patch\n*** End Patch");
}

async function main() {
  await testUnifiedToAnthropicRequest();
  await testBuildAnthropicBodyRoundTripViaClientOut();
  await testAnthropicJsonResponseToUnified();
  await testAnthropicSseResponseToUnified();
  await testStopSequenceMapsToStop();
  await testThinkingPreserved();
  await testMultipleSystemMessagesAndToolContent();
  await testAnthropicSourcePreservesExactCacheAndFields();
  await testAnthropicBodyBuilderDoesNotInventCaching();
  await testExactAuthPreservesBodyAndNormalizesUrl();
  await testClaudeAuthChainOwnsAuthNotWireBuild();
  await testTopLevelSystemPlusResidualSystemMessagesMerge();
  await testResponsesJsonSchemaMapsToOutputConfigFormat();
  await testResponsesJsonObjectOmitsOutputFormat();
  await testResponsesParallelToolCallsDisableOnAnthropic();
  await testResponsesCustomToolsAreAnthropicInputSchema();
  await testAnthropicStructuredOutputMapsToUnifiedResponseFormat();
  await testAnthropicThinkingHistoryMapsToResponsesReasoning();
  console.log("anthropic.provider-wire: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

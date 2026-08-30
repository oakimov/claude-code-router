import assert from "node:assert/strict";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { UnifiedChatRequest } from "../types/llm";
import { buildEncryptedReasoningCacheNamespace } from "../utils/responses.encrypted-content-cache";

const provider = {
  name: "opencode-responses",
  baseUrl: "https://opencode.ai/zen/v1/responses",
};

const CIPHER = "gAAAAABlencrypted-muse-reasoning-ciphertext-for-replay";
const LONG_CURSOR_CALL_ID =
  "call-66dbf0b1-aad7-482f-baa2-647748651824-0_fc_49ff1230-042d-97ce-b451-5e3f019a21d8_0";
const RESPONSES_CALL_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;

function anthropicContext(
  req: Record<string, unknown> = {},
  sessionId?: string
) {
  const protocolContext = {
    protocol: "anthropic_messages",
    ...(sessionId ? { sessionId } : {}),
  };
  return {
    req: { ...req, protocolContext, ...(sessionId ? { sessionId } : {}) },
    clientProtocol: "anthropic_messages",
    protocolContext,
  };
}

function chatContext(req: Record<string, unknown> = {}, sessionId?: string) {
  const protocolContext = {
    protocol: "openai_chat_completions",
    ...(sessionId ? { sessionId } : {}),
  };
  return {
    req: { ...req, protocolContext, ...(sessionId ? { sessionId } : {}) },
    clientProtocol: "openai_chat_completions",
    protocolContext,
  };
}

function responsesContext(req: Record<string, unknown> = {}) {
  return {
    req,
    clientProtocol: "openai_responses",
    protocolContext: { protocol: "openai_responses" },
  };
}

function buildStreamResponse(events: Array<Record<string, unknown>>): Response {
  const payload = events
    .map((event) => `data: ${JSON.stringify(event)}\n\n`)
    .join("");
  return new Response(payload, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function drainResponse(response: Response): Promise<string> {
  return await response.text();
}

function toolAssistantWithoutCipher(): UnifiedChatRequest["messages"][number] {
  return {
    role: "assistant",
    content: null,
    tool_calls: [
      {
        id: "call_cached",
        type: "function",
        function: {
          name: "Bash",
          arguments: '{"command":"ls -la"}',
        },
      },
    ],
  };
}

async function testAnthropicRequestsEncryptedInclude() {
  const transformer = new OpenAIResponsesTransformer();
  const context = anthropicContext();
  const request: UnifiedChatRequest = {
    model: "muse-spark-1.2-contributor-free",
    stream: false,
    messages: [{ role: "user", content: "hi" }],
    reasoning: { effort: "high", enabled: true, summary: "auto" },
  };

  const transformed = await transformer.transformRequestIn(
    request,
    provider as any,
    context as any
  );

  assert.deepEqual((transformed as any).include, [
    "reasoning.encrypted_content",
  ]);
}

async function testResponsesClientDoesNotInventInclude() {
  const transformer = new OpenAIResponsesTransformer();
  const context = responsesContext();
  const request: UnifiedChatRequest = {
    model: "muse-spark-1.2-contributor-free",
    stream: false,
    messages: [{ role: "user", content: "hi" }],
    reasoning: { effort: "high", enabled: true, summary: "auto" },
  };

  const transformed = await transformer.transformRequestIn(
    request,
    provider as any,
    context as any
  );

  assert.equal((transformed as any).include, undefined);
}

async function testCachedEncryptedReplayAfterJson() {
  const transformer = new OpenAIResponsesTransformer();
  const initialContext = anthropicContext();
  const initialRequest: UnifiedChatRequest = {
    model: "muse-spark-1.2-contributor-free",
    stream: false,
    messages: [{ role: "user", content: "what's in this folder" }],
    reasoning: { effort: "high", enabled: true, summary: "auto" },
  };

  await transformer.transformRequestIn(
    initialRequest,
    provider as any,
    initialContext as any
  );

  const jsonResponse = new Response(
    JSON.stringify({
      id: "resp_test",
      object: "response",
      model: "muse-spark-1.2-contributor-free",
      created_at: 1,
      output: [
        {
          type: "reasoning",
          id: "rs_cached",
          summary: [{ type: "summary_text", text: "Need to list files." }],
          encrypted_content: CIPHER,
        },
        {
          type: "function_call",
          call_id: "call_cached",
          name: "Bash",
          arguments: '{"command":"ls -la"}',
        },
      ],
    }),
    { headers: { "Content-Type": "application/json" } }
  );

  await drainResponse(
    await transformer.transformResponseOut(jsonResponse, initialContext as any)
  );

  const followupContext = anthropicContext();
  const followupRequest: UnifiedChatRequest = {
    model: "muse-spark-1.2-contributor-free",
    stream: false,
    messages: [
      { role: "user", content: "what's in this folder" },
      toolAssistantWithoutCipher(),
      {
        role: "tool",
        tool_call_id: "call_cached",
        content: "total 144",
      },
      { role: "user", content: "tell me more" },
    ],
    reasoning: { effort: "high", enabled: true, summary: "auto" },
  };

  const transformedFollowup = await transformer.transformRequestIn(
    followupRequest,
    provider as any,
    followupContext as any
  );

  const reasoning = ((transformedFollowup as any).input || []).find(
    (item: any) => item?.type === "reasoning"
  );
  assert.ok(reasoning, "expected restored reasoning item on tool turn");
  assert.equal(reasoning.encrypted_content, CIPHER);
  assert.equal(
    reasoning.summary?.[0]?.text || reasoning.summary?.[0],
    "Need to list files."
  );
}

async function testEncryptedReplayIsScopedToClientSession() {
  const transformer = new OpenAIResponsesTransformer();
  const prompt = "session-isolated encrypted replay";
  const callId = "call_session_isolated";
  const argumentsJson = '{"command":"printf isolated"}';
  const seedContext = anthropicContext({}, "client-session-alpha-sensitive");
  const seedRequest: UnifiedChatRequest = {
    model: "muse-spark-1.2-contributor-free",
    stream: false,
    messages: [{ role: "user", content: prompt }],
    reasoning: { effort: "high", enabled: true },
  };

  const namespace = buildEncryptedReasoningCacheNamespace(
    seedRequest,
    provider as any,
    seedContext as any
  );
  assert.match(namespace, /^[a-f0-9]{64}$/);
  assert.ok(!namespace.includes("client-session-alpha-sensitive"));
  assert.notEqual(
    namespace,
    buildEncryptedReasoningCacheNamespace(
      seedRequest,
      provider as any,
      anthropicContext({}, "client-session-beta-sensitive") as any
    )
  );

  await transformer.transformRequestIn(
    seedRequest,
    provider as any,
    seedContext as any
  );
  await drainResponse(
    await transformer.transformResponseOut(
      new Response(
        JSON.stringify({
          id: "resp_session_isolated",
          object: "response",
          model: "muse-spark-1.2-contributor-free",
          created_at: 1,
          output: [
            {
              type: "reasoning",
              id: "rs_session_isolated",
              summary: [],
              encrypted_content: CIPHER + "-session-alpha",
            },
            {
              type: "function_call",
              call_id: callId,
              name: "Bash",
              arguments: argumentsJson,
            },
          ],
        }),
        { headers: { "Content-Type": "application/json" } }
      ),
      seedContext as any
    )
  );

  const followup = (): UnifiedChatRequest => ({
    model: "muse-spark-1.2-contributor-free",
    stream: false,
    messages: [
      { role: "user", content: prompt },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: callId,
            type: "function",
            function: { name: "Bash", arguments: argumentsJson },
          },
        ],
      },
      { role: "tool", tool_call_id: callId, content: "isolated" },
    ],
    reasoning: { effort: "high", enabled: true },
  });

  const otherSession = await transformer.transformRequestIn(
    followup(),
    provider as any,
    anthropicContext({}, "client-session-beta-sensitive") as any
  );
  assert.equal(
    ((otherSession as any).input || []).find(
      (item: any) => item?.type === "reasoning"
    ),
    undefined
  );

  const sameSession = await transformer.transformRequestIn(
    followup(),
    provider as any,
    anthropicContext({}, "client-session-alpha-sensitive") as any
  );
  assert.equal(
    ((sameSession as any).input || []).find(
      (item: any) => item?.type === "reasoning"
    )?.encrypted_content,
    CIPHER + "-session-alpha"
  );
}

async function testCachedEncryptedReplayAfterStream() {
  const transformer = new OpenAIResponsesTransformer();
  const initialContext = chatContext();
  const initialRequest: UnifiedChatRequest = {
    model: "muse-spark-1.2-contributor-free",
    stream: true,
    messages: [{ role: "user", content: "inspect" }],
    reasoning: { effort: "high", enabled: true, summary: "auto" },
  };

  await transformer.transformRequestIn(
    initialRequest,
    provider as any,
    initialContext as any
  );

  const streamed = buildStreamResponse([
    {
      type: "response.created",
      response: {
        id: "resp_stream",
        model: "muse-spark-1.2-contributor-free",
      },
    },
    {
      type: "response.completed",
      response: {
        id: "resp_stream",
        model: "muse-spark-1.2-contributor-free",
        output: [
          {
            type: "reasoning",
            id: "rs_stream",
            summary: [{ type: "summary_text", text: "Check first." }],
            encrypted_content: CIPHER + "-stream",
          },
          {
            type: "function_call",
            call_id: "call_cached",
            name: "Bash",
            arguments: '{"command":"pwd"}',
          },
        ],
      },
    },
  ]);

  await drainResponse(
    await transformer.transformResponseOut(streamed, initialContext as any)
  );

  const followupContext = chatContext();
  const followupRequest: UnifiedChatRequest = {
    model: "muse-spark-1.2-contributor-free",
    stream: false,
    messages: [
      { role: "user", content: "inspect" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_cached",
            type: "function",
            function: {
              name: "Bash",
              arguments: '{"command":"pwd"}',
            },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_cached",
        content: "/tmp",
      },
    ],
    reasoning: { effort: "high", enabled: true, summary: "auto" },
  };

  const transformedFollowup = await transformer.transformRequestIn(
    followupRequest,
    provider as any,
    followupContext as any
  );

  const reasoning = ((transformedFollowup as any).input || []).find(
    (item: any) => item?.type === "reasoning"
  );
  assert.ok(reasoning, "expected restored reasoning item after stream");
  assert.equal(reasoning.encrypted_content, CIPHER + "-stream");
}

async function testStreamRecorderRestoresWithoutTerminalOutput() {
  const transformer = new OpenAIResponsesTransformer();
  const sessionId = "stream-recorder-session";
  const prompt = "stream without terminal output";
  const argumentsJson = '{"command":"pwd"}';
  const initialContext = chatContext({}, sessionId);
  await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: true,
      messages: [{ role: "user", content: prompt }],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    initialContext as any
  );

  const streamed = buildStreamResponse([
    {
      type: "response.created",
      response: { id: "resp_partial_terminal", model: "muse" },
    },
    {
      type: "response.reasoning_summary_text.delta",
      output_index: 0,
      item_id: "rs_partial_terminal",
      delta: "Need the current directory.",
    },
    {
      type: "response.output_item.done",
      output_index: 0,
      item_id: "rs_partial_terminal",
      item: {
        id: "rs_partial_terminal",
        type: "reasoning",
        summary: [
          { type: "summary_text", text: "Need the current directory." },
        ],
        encrypted_content: CIPHER + "-partial-terminal",
      },
    },
    {
      type: "response.output_item.added",
      output_index: 1,
      item: {
        id: "fc_partial_terminal",
        type: "function_call",
        call_id: LONG_CURSOR_CALL_ID,
        name: "Bash",
      },
    },
    {
      type: "response.function_call_arguments.delta",
      output_index: 1,
      item_id: "fc_partial_terminal",
      delta: '{"command":',
    },
    {
      type: "response.function_call_arguments.done",
      output_index: 1,
      item_id: "fc_partial_terminal",
      arguments: argumentsJson,
    },
    {
      type: "response.completed",
      response: {
        id: "resp_partial_terminal",
        model: "muse-spark-1.2-contributor-free",
        output: [],
      },
    },
  ]);

  const converted = await transformer.transformResponseOut(
    streamed,
    initialContext as any
  );
  const clientChunks = (await converted.text())
    .split("\n")
    .filter((line) => line.startsWith("data: {") )
    .map((line) => JSON.parse(line.slice(6)));
  const clientToolCall = clientChunks
    .flatMap((chunk) => chunk.choices?.[0]?.delta?.tool_calls || [])
    .find((call: any) => call?.id);
  assert.ok(clientToolCall);
  assert.match(clientToolCall.id, RESPONSES_CALL_ID_PATTERN);
  assert.notEqual(clientToolCall.id, LONG_CURSOR_CALL_ID);

  const followup = await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: false,
      messages: [
        { role: "user", content: prompt },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: clientToolCall.id,
              type: "function",
              function: { name: "Bash", arguments: argumentsJson },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: clientToolCall.id,
          content: "/tmp",
        },
      ],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    chatContext({}, sessionId) as any
  );

  const reasoning = ((followup as any).input || []).find(
    (item: any) => item?.type === "reasoning"
  );
  const call = ((followup as any).input || []).find(
    (item: any) => item?.type === "function_call"
  );
  const output = ((followup as any).input || []).find(
    (item: any) => item?.type === "function_call_output"
  );
  assert.equal(
    reasoning?.encrypted_content,
    CIPHER + "-partial-terminal",
    JSON.stringify((followup as any).input)
  );
  assert.equal(reasoning?.summary?.[0]?.text, "Need the current directory.");
  assert.match(call.call_id, RESPONSES_CALL_ID_PATTERN);
  assert.equal(call.call_id, output.call_id);
  assert.equal(call.call_id, clientToolCall.id);
}

async function testDiscardedRecorderDoesNotFallBackToTerminalOutput() {
  const transformer = new OpenAIResponsesTransformer();
  const sessionId = "discarded-recorder-session";
  const prompt = "discard oversized stream recorder";
  const initialContext = anthropicContext({}, sessionId);
  await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: true,
      messages: [{ role: "user", content: prompt }],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    initialContext as any
  );

  const oversizedEvents: Array<Record<string, unknown>> = [];
  for (let index = 0; index <= 256; index++) {
    oversizedEvents.push({
      type: "response.output_item.added",
      output_index: index,
      item: {
        id: `msg_oversized_${index}`,
        type: "message",
        content: [],
      },
    });
  }
  oversizedEvents.push({
    type: "response.completed",
    response: {
      id: "resp_oversized",
      model: "muse-spark-1.2-contributor-free",
      output: [
        {
          type: "reasoning",
          id: "rs_oversized",
          summary: [],
          encrypted_content: CIPHER + "-oversized",
        },
        {
          type: "function_call",
          call_id: "call_oversized",
          name: "Bash",
          arguments: "{}",
        },
      ],
    },
  });
  await drainResponse(
    await transformer.transformResponseOut(
      buildStreamResponse(oversizedEvents),
      initialContext as any
    )
  );

  const followup = await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: false,
      messages: [
        { role: "user", content: prompt },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_oversized",
              type: "function",
              function: { name: "Bash", arguments: "{}" },
            },
          ],
        },
        {
          role: "tool",
          tool_call_id: "call_oversized",
          content: "ignored",
        },
      ],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    anthropicContext({}, sessionId) as any
  );
  assert.equal(
    ((followup as any).input || []).find(
      (item: any) => item?.type === "reasoning"
    ),
    undefined
  );
}

async function testFailedStreamDoesNotPopulateCache() {
  const transformer = new OpenAIResponsesTransformer();
  const sessionId = "failed-stream-session";
  const prompt = "failed stream must not cache";
  const initialContext = anthropicContext({}, sessionId);
  await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: true,
      messages: [{ role: "user", content: prompt }],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    initialContext as any
  );

  const failed = buildStreamResponse([
    {
      type: "response.output_item.done",
      output_index: 0,
      item_id: "rs_failed",
      item: {
        id: "rs_failed",
        type: "reasoning",
        summary: [],
        encrypted_content: CIPHER + "-failed",
      },
    },
    {
      type: "response.output_item.done",
      output_index: 1,
      item_id: "fc_failed",
      item: {
        id: "fc_failed",
        type: "function_call",
        call_id: "call_failed",
        name: "Bash",
        arguments: "{}",
      },
    },
    {
      type: "response.failed",
      response: {
        id: "resp_failed",
        error: { message: "upstream failed", type: "api_error" },
      },
    },
  ]);
  await drainResponse(
    await transformer.transformResponseOut(failed, initialContext as any)
  );

  const followup = await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: false,
      messages: [
        { role: "user", content: prompt },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_failed",
              type: "function",
              function: { name: "Bash", arguments: "{}" },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_failed", content: "ignored" },
      ],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    anthropicContext({}, sessionId) as any
  );
  assert.equal(
    ((followup as any).input || []).find(
      (item: any) => item?.type === "reasoning"
    ),
    undefined
  );
}

async function testResponsesClientDoesNotRestoreFromCache() {
  const transformer = new OpenAIResponsesTransformer();
  // Seed cache via Anthropic path.
  const seedContext = anthropicContext();
  await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: false,
      messages: [{ role: "user", content: "seed" }],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    seedContext as any
  );
  await drainResponse(
    await transformer.transformResponseOut(
      new Response(
        JSON.stringify({
          id: "resp_seed",
          object: "response",
          model: "muse-spark-1.2-contributor-free",
          created_at: 1,
          output: [
            {
              type: "reasoning",
              id: "rs_seed",
              summary: [],
              encrypted_content: CIPHER + "-seed",
            },
            {
              type: "function_call",
              call_id: "call_cached",
              name: "Bash",
              arguments: "{}",
            },
          ],
        }),
        { headers: { "Content-Type": "application/json" } }
      ),
      seedContext as any
    )
  );

  // Same-protocol Responses client must not invent include or restore ciphertext.
  const responsesCtx = responsesContext();
  const transformed = await transformer.transformRequestIn(
    {
      model: "muse-spark-1.2-contributor-free",
      stream: false,
      messages: [
        { role: "user", content: "seed" },
        toolAssistantWithoutCipher(),
        {
          role: "tool",
          tool_call_id: "call_cached",
          content: "ok",
        },
      ],
      reasoning: { effort: "high", enabled: true },
    },
    provider as any,
    responsesCtx as any
  );

  assert.equal((transformed as any).include, undefined);
  const reasoning = ((transformed as any).input || []).find(
    (item: any) => item?.type === "reasoning"
  );
  assert.equal(reasoning, undefined);
}

async function main() {
  await testAnthropicRequestsEncryptedInclude();
  await testResponsesClientDoesNotInventInclude();
  await testCachedEncryptedReplayAfterJson();
  await testEncryptedReplayIsScopedToClientSession();
  await testCachedEncryptedReplayAfterStream();
  await testStreamRecorderRestoresWithoutTerminalOutput();
  await testDiscardedRecorderDoesNotFallBackToTerminalOutput();
  await testFailedStreamDoesNotPopulateCache();
  await testResponsesClientDoesNotRestoreFromCache();
  console.log("responses.encrypted-content-cache: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

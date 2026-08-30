import assert from "node:assert/strict";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { UnifiedChatRequest } from "../types/llm";

const provider = {
  name: "opencode-responses",
  baseUrl: "https://opencode.ai/zen/v1/responses",
};

const CIPHER = "gAAAAABlencrypted-muse-reasoning-ciphertext-for-replay";

function anthropicContext(req: Record<string, unknown> = {}) {
  return {
    req,
    clientProtocol: "anthropic_messages",
    protocolContext: { protocol: "anthropic_messages" },
  };
}

function chatContext(req: Record<string, unknown> = {}) {
  return {
    req,
    clientProtocol: "openai_chat_completions",
    protocolContext: { protocol: "openai_chat_completions" },
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
  await testCachedEncryptedReplayAfterStream();
  await testResponsesClientDoesNotRestoreFromCache();
  console.log("responses.encrypted-content-cache: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { CerebrasTransformer } from "../transformer/cerebras.transformer";
import { ChromeOnDeviceTransformer } from "../transformer/chrome-on-device.transformer";
import { CodexTransformer } from "../transformer/codex.transformer";
import { DeepseekTransformer } from "../transformer/deepseek.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { OpenrouterTransformer } from "../transformer/openrouter.transformer";
import { VercelTransformer } from "../transformer/vercel.transformer";
import { buildSessionKey } from "../cursor-sdk/session";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";
import {
  applyQwenPromptCaching,
  applyRawAnthropicPromptCaching,
} from "../utils/cacheControl";
import {
  applyProviderNativeChatCaching,
} from "../utils/openai.util";
import {
  buildRequestBody as buildVertexClaudeBody,
  transformResponseOut as transformVertexClaudeResponse,
} from "../utils/vertex-claude.util";
import type { UnifiedChatRequest } from "../types/llm";

function request(model: string): UnifiedChatRequest {
  return {
    model,
    messages: [
      {
        role: "system",
        content: [
          {
            type: "text",
            text: "stable system",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
      { role: "user", content: "hello" },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "Read",
          description: "read a file",
          parameters: { type: "object", properties: {} },
        },
        cache_control: { type: "ephemeral" },
      },
    ],
  };
}

async function toAnthropic(response: Response): Promise<any> {
  const transformer = new AnthropicTransformer();
  transformer.logger = { debug() {} };
  const converted = await transformer.transformResponseIn(response, {
    req: { id: "cache-path-test" },
  });
  return converted.json();
}

async function toAnthropicEvents(response: Response): Promise<any[]> {
  const transformer = new AnthropicTransformer();
  transformer.logger = { debug() {} };
  const converted = await transformer.transformResponseIn(response, {
    req: { id: "cache-path-stream-test" },
  });
  const text = await converted.text();
  return text
    .split("\n")
    .filter((line) => line.startsWith("data: ") && line !== "data: [DONE]")
    .map((line) => JSON.parse(line.slice(6)));
}

async function testOpenAIChatRoundTrip() {
  const outbound = applyProviderNativeChatCaching(
    request("gpt-5.6"),
    { name: "openai", baseUrl: "https://api.openai.com/v1/chat/completions" },
    { req: { sessionId: "openai-session" } }
  ) as any;
  assert.ok(outbound.prompt_cache_key.startsWith("ccr_"));
  assert.deepEqual(
    outbound.messages[0].content[0].prompt_cache_breakpoint,
    { mode: "explicit" }
  );
  assert.equal(outbound.messages[0].content[0].cache_control, undefined);
  assert.equal(outbound.tools[0].cache_control, undefined);

  const inbound = await toAnthropic(
    new Response(
      JSON.stringify({
        id: "chatcmpl-cache",
        model: "gpt-5.6",
        choices: [
          {
            finish_reason: "stop",
            message: { role: "assistant", content: "done" },
          },
        ],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 5,
          total_tokens: 105,
          prompt_tokens_details: {
            cached_tokens: 60,
            cache_write_tokens: 20,
          },
        },
      }),
      { headers: { "Content-Type": "application/json" } }
    )
  );
  assert.deepEqual(inbound.usage, {
    input_tokens: 20,
    output_tokens: 5,
    cache_creation_input_tokens: 20,
    cache_read_input_tokens: 60,
  });
}

async function testResponsesRoundTrip() {
  const transformer = new OpenAIResponsesTransformer();
  transformer.logger = { debug() {} };
  const outbound = (await transformer.transformRequestIn(
    request("gpt-5.6"),
    { baseUrl: "https://api.openai.com/v1/responses" },
    { req: { sessionId: "responses-session" } }
  )) as any;
  assert.ok(outbound.prompt_cache_key.startsWith("ccr_"));
  assert.deepEqual(
    outbound.input[0].content[0].prompt_cache_breakpoint,
    { mode: "explicit" }
  );

  const unified = await transformer.transformResponseOut(
    new Response(
      JSON.stringify({
        id: "resp-cache",
        object: "response",
        model: "gpt-5.6",
        output: [
          {
            type: "message",
            content: [{ type: "output_text", text: "done" }],
          },
        ],
        usage: {
          input_tokens: 100,
          output_tokens: 5,
          total_tokens: 105,
          input_tokens_details: {
            cached_tokens: 60,
            cache_write_tokens: 20,
          },
        },
      }),
      { headers: { "Content-Type": "application/json" } }
    )
  );
  const inbound = await toAnthropic(unified);
  assert.equal(inbound.usage.cache_read_input_tokens, 60);
  assert.equal(inbound.usage.cache_creation_input_tokens, 20);
}

async function testCodexRoundTrip() {
  const transformer = new CodexTransformer();
  transformer.logger = { debug() {} };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(
      JSON.stringify({
        chatgpt_account_id: "account-test",
        chatgpt_account_is_fedramp: false,
        chatgpt_user_id: "user-test",
        chatgpt_plan_type: "pro",
      }),
      { headers: { "Content-Type": "application/json" } }
    );

  const codexModels = [
    "gpt-5.4",
    "codex-auto-review",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4-mini",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
  ];
  let outbound: any;
  try {
    for (const model of codexModels) {
      const cacheRequest = request(model);
      cacheRequest.messages[1].content = [
        {
          type: "text",
          text: "hello",
          cache_control: { type: "ephemeral" },
        },
      ];
      const current = await transformer.transformRequestIn(
        { ...cacheRequest, stream: false },
        {
          name: "codex",
          baseUrl: "https://chatgpt.com/backend-api/codex",
          apiKey: "at-test",
          models: [],
        },
        {
          req: {
            id:
              model === "gpt-5.6-luna"
                ? "codex-cache-path"
                : `codex-cache-path-${model}`,
            sessionId: "codex-session",
          },
        }
      );
      assert.equal(
        JSON.stringify(current.body).includes("prompt_cache_breakpoint"),
        false,
        `${model} must use Codex key-based caching without explicit breakpoints`
      );
      if (model === "gpt-5.6-luna") {
        outbound = current;
      }
    }
  } finally {
    globalThis.fetch = originalFetch;
  }

  assert.ok(outbound.body.prompt_cache_key.startsWith("ccr_"));
  assert.equal(
    outbound.config.headers["session-id"],
    outbound.body.prompt_cache_key
  );
  assert.equal(
    outbound.config.headers["thread-id"],
    outbound.body.prompt_cache_key
  );
  assert.equal(
    outbound.config.headers["x-client-request-id"],
    outbound.body.prompt_cache_key
  );
  assert.equal(
    outbound.body.client_metadata.session_id,
    outbound.body.prompt_cache_key
  );
  assert.equal(
    outbound.body.input[0].content[0].prompt_cache_breakpoint,
    undefined
  );
  assert.equal(outbound.body.input[0].content[0].cache_control, undefined);

  const responsePayload = {
    id: "codex-cache",
    object: "response",
    model: "gpt-5.6-luna",
    created_at: 1,
    output: [
      {
        type: "message",
        content: [{ type: "output_text", text: "done" }],
      },
    ],
    usage: {
      input_tokens: 100,
      output_tokens: 5,
      total_tokens: 105,
      input_tokens_details: {
        cached_tokens: 60,
        cache_write_tokens: 20,
      },
    },
  };
  const unified = await transformer.transformResponseOut(
    new Response(
      JSON.stringify(responsePayload),
      { headers: { "Content-Type": "text/event-stream" } }
    ),
    { req: { id: "codex-cache-path" } }
  );
  const inbound = await toAnthropic(unified);
  assert.equal(inbound.usage.input_tokens, 20);
  assert.equal(inbound.usage.cache_read_input_tokens, 60);
  assert.equal(inbound.usage.cache_creation_input_tokens, 20);

  const streamingUnified = await transformer.transformResponseOut(
    new Response(JSON.stringify(responsePayload), {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const events = await toAnthropicEvents(streamingUnified);
  const usageEvent = events.find((event) => event.type === "message_delta");
  assert.equal(usageEvent.usage.input_tokens, 20);
  assert.equal(usageEvent.usage.cache_read_input_tokens, 60);
  assert.equal(usageEvent.usage.cache_creation_input_tokens, 20);
}

async function testGatewayAndRequestLevelPaths() {
  const openrouter = (
    await new OpenrouterTransformer().transformRequestIn(
      request("anthropic/claude-sonnet-4-6"),
      {},
      { req: { sessionId: "openrouter-session" } }
    )
  ).body as any;
  assert.ok(openrouter.session_id.startsWith("ccr_"));
  assert.deepEqual(openrouter.cache_control, { type: "ephemeral" });

  const vercel = (await new VercelTransformer().transformRequestIn(
    request("anthropic/claude-sonnet-4-6")
  )) as any;
  assert.equal(vercel.providerOptions.gateway.caching, "auto");
  assert.equal(vercel.messages[0].content[0].cache_control, undefined);

  const mistral = buildMistralBody(
    request("mistral-large-latest"),
    { req: { sessionId: "mistral-session" } },
    {}
  ) as any;
  assert.ok(mistral.prompt_cache_key.startsWith("ccr_"));
  assert.equal(mistral.tools[0].cache_control, undefined);

  const cerebras = (await new CerebrasTransformer().transformRequestIn(
    request("qwen-3-235b-a22b-instruct-2507"),
    {
      name: "cerebras",
      baseUrl: "https://api.cerebras.ai/v1/chat/completions",
      apiKey: "test",
      models: [],
    },
    { req: { sessionId: "cerebras-session" } }
  )) as any;
  assert.ok(cerebras.body.prompt_cache_key.startsWith("ccr_"));
  assert.equal(cerebras.body.tools[0].cache_control, undefined);
}

async function testDeepSeekUsageAndAutomaticRequest() {
  const transformer = new DeepseekTransformer();
  const outbound = await transformer.transformRequestIn(
    request("deepseek-chat")
  );
  assert.equal((outbound.tools?.[0] as any).cache_control, undefined);

  const unified = await transformer.transformResponseOut(
    new Response(
      JSON.stringify({
        id: "deepseek-cache",
        model: "deepseek-chat",
        choices: [
          {
            finish_reason: "stop",
            message: { role: "assistant", content: "done" },
          },
        ],
        usage: {
          prompt_tokens: 100,
          completion_tokens: 5,
          total_tokens: 105,
          prompt_cache_hit_tokens: 80,
          prompt_cache_miss_tokens: 20,
        },
      }),
      { headers: { "Content-Type": "application/json" } }
    )
  );
  const inbound = await toAnthropic(unified);
  assert.equal(inbound.usage.cache_read_input_tokens, 80);
  assert.equal(inbound.usage.input_tokens, 20);
}

async function testAnthropicAndVertexClaudePaths() {
  const direct = applyRawAnthropicPromptCaching({
    model: "claude-sonnet-4-6",
    messages: [{ role: "user", content: "hello" }],
  });
  assert.deepEqual((direct as any).cache_control, { type: "ephemeral" });

  const vertexBody = buildVertexClaudeBody(
    request("claude-sonnet-4@20250514")
  ) as any;
  assert.deepEqual(vertexBody.cache_control, { type: "ephemeral" });

  const unified = await transformVertexClaudeResponse(
    new Response(
      JSON.stringify({
        id: "vertex-cache",
        model: "claude-sonnet-4@20250514",
        content: [{ type: "text", text: "done" }],
        stop_reason: "end_turn",
        usage: {
          input_tokens: 20,
          output_tokens: 5,
          cache_creation_input_tokens: 10,
          cache_read_input_tokens: 70,
        },
      }),
      { headers: { "Content-Type": "application/json" } }
    ),
    "vertex-claude"
  );
  const inbound = await toAnthropic(unified);
  assert.equal(inbound.usage.input_tokens, 20);
  assert.equal(inbound.usage.cache_creation_input_tokens, 10);
  assert.equal(inbound.usage.cache_read_input_tokens, 70);
}

async function testQwenChromeAndCursorSessions() {
  const qwen = applyQwenPromptCaching(request("qwen-plus")) as any;
  assert.deepEqual(qwen.messages[1].content[0].cache_control, {
    type: "ephemeral",
  });

  const chrome = (await new ChromeOnDeviceTransformer().transformRequestIn(
    request("gemini-nano"),
    { baseUrl: "http://127.0.0.1:3457" },
    { req: { sessionId: "chrome-session", body: {} } }
  )) as any;
  assert.ok(chrome.config.headers["x-ccr-session-id"].startsWith("ccr_"));
  assert.equal(chrome.body.tools[0].cache_control, undefined);

  const cursorA = buildSessionKey({
    metadataUserId: "user_session_cursor-1",
    model: "cursor-model",
  });
  const cursorB = buildSessionKey({
    metadataUserId: "user_session_cursor-1",
    model: "cursor-model",
  });
  assert.equal(cursorA, cursorB);
}

async function main() {
  await testOpenAIChatRoundTrip();
  await testResponsesRoundTrip();
  await testCodexRoundTrip();
  await testGatewayAndRequestLevelPaths();
  await testDeepSeekUsageAndAutomaticRequest();
  await testAnthropicAndVertexClaudePaths();
  await testQwenChromeAndCursorSessions();
  console.log("cache-paths.integration: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

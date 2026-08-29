import assert from "node:assert/strict";
import {
  applyAnthropicPromptCaching,
  applyQwenPromptCaching,
  applyRawAnthropicPromptCaching,
  deriveCacheSessionKey,
  extractClientSessionId,
  stripToolsCacheControl,
  stripMessagesCacheControl,
} from "../utils/cacheControl";
import { GroqTransformer } from "../transformer/groq.transformer";
import { OpenrouterTransformer } from "../transformer/openrouter.transformer";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";
import { buildRequestBody as buildVertexClaudeBody } from "../utils/vertex-claude.util";
import { OpenAITransformer } from "../transformer/openai.transformer";
import {
  attachGeminiCachedContent,
  clearGeminiCachedContentForTests,
} from "../utils/gemini-cache";
import type { UnifiedChatRequest, UnifiedTool } from "../types/llm";

function toolWithCache(): UnifiedTool {
  return {
    type: "function",
    function: {
      name: "Bash",
      description: "run",
      parameters: { type: "object", properties: {} },
    },
    cache_control: { type: "ephemeral" },
  };
}

function testStripToolsCacheControl() {
  const tools = [toolWithCache()];
  const stripped = stripToolsCacheControl(tools);
  assert.equal((stripped?.[0] as any).cache_control, undefined);
  // Non-cache fields survive.
  assert.equal(stripped?.[0].function.name, "Bash");
  // Original is not mutated (helper returns clones).
  assert.deepEqual(tools[0].cache_control, { type: "ephemeral" });
  // Undefined passes through untouched.
  assert.equal(stripToolsCacheControl(undefined), undefined);
}

async function testGroqStripsToolCacheControl() {
  const transformer = new GroqTransformer();
  const request = {
    model: "llama-3.3-70b",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest;

  const out = await transformer.transformRequestIn(request);
  assert.equal((out.tools?.[0] as any).cache_control, undefined);
}

async function testOpenrouterAddsNativeSessionAndCleansOpenAIFields() {
  const transformer = new OpenrouterTransformer();
  const request = {
    model: "openai/gpt-4o",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest;

  const out = (
    await transformer.transformRequestIn(
      request,
      {},
      { req: { sessionId: "session-123" } }
    )
  ).body;
  assert.ok((out as any).session_id?.startsWith("ccr_"));
  assert.ok((out as any).prompt_cache_key?.startsWith("ccr_"));
  assert.equal((out.tools?.[0] as any).cache_control, undefined);
}

async function testOpenrouterKeepsToolCacheControlForClaude() {
  const transformer = new OpenrouterTransformer();
  const request = {
    model: "anthropic/claude-sonnet-4-6",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest;

  const out = (await transformer.transformRequestIn(request)).body;
  assert.deepEqual((out.tools?.[0] as any).cache_control, { type: "ephemeral" });
  assert.deepEqual((out as any).cache_control, { type: "ephemeral" });
}

function testMistralAddsPromptCacheKeyAndStripsToolCacheControl() {
  const body = buildMistralBody({
    model: "mistral-medium-2505",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest, { req: { sessionId: "session-123" } });

  assert.ok(body.prompt_cache_key?.startsWith("ccr_"));
  assert.equal(body.tools?.[0]?.cache_control, undefined);
  assert.equal(body.tools?.[0]?.function?.name, "Bash");
}

function testAnthropicCachingIsBoundedAndImmutable() {
  const request = {
    model: "claude-sonnet-4-6",
    messages: [
      { role: "system", content: "sys1" },
      { role: "system", content: "sys2" },
      { role: "user", content: "u1" },
      { role: "assistant", content: "a1" },
      { role: "user", content: "u2" },
    ],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest;

  const out = applyAnthropicPromptCaching(request, {
    maxBreakpoints: 4,
    includeTools: true,
  });
  const markedMessages = out.messages.filter((msg: any) => {
    if (msg.cache_control) return true;
    return Array.isArray(msg.content) && msg.content.some((part: any) => part.cache_control);
  });
  const markedTools = (out.tools || []).filter((tool: any) => tool.cache_control);
  assert.equal(markedMessages.length + markedTools.length, 4);
  assert.equal(typeof request.messages[0].content, "string");
}

function testAnthropicCachingTrimsExistingOverBudget() {
  const request = {
    model: "claude-sonnet-4-6",
    messages: Array.from({ length: 6 }, (_, i) => ({
      role: i === 0 ? "system" : "user",
      content: [
        {
          type: "text",
          text: `m${i}`,
          cache_control: { type: "ephemeral" },
        },
      ],
    })),
  } as unknown as UnifiedChatRequest;

  const out = applyAnthropicPromptCaching(request, { maxBreakpoints: 4 });
  const count = out.messages.reduce((sum, msg: any) => {
    return (
      sum +
      (Array.isArray(msg.content)
        ? msg.content.filter((part: any) => part.cache_control).length
        : 0)
    );
  }, 0);
  assert.equal(count, 4);
}

function testAnthropicAutomaticCachingUsesAvailableSlot() {
  const out = applyRawAnthropicPromptCaching({
    model: "claude-sonnet-4-6",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.deepEqual((out as any).cache_control, { type: "ephemeral" });

  const fourExplicit = applyRawAnthropicPromptCaching({
    messages: Array.from({ length: 4 }, (_, i) => ({
      role: "user",
      content: [
        {
          type: "text",
          text: `m${i}`,
          cache_control: { type: "ephemeral" },
        },
      ],
    })),
  });
  assert.equal((fourExplicit as any).cache_control, undefined);
}

function testAnthropicCachingDoesNotTouchToolSchemas() {
  const out = applyRawAnthropicPromptCaching({
    model: "claude-sonnet-4-6",
    tools: [
      {
        name: "inspect",
        input_schema: {
          type: "object",
          properties: {
            cache_control: { type: "string" },
          },
        },
      },
    ],
    messages: [{ role: "user", content: "hi" }],
  }) as any;

  assert.deepEqual(
    out.tools[0].input_schema.properties.cache_control,
    { type: "string" }
  );
  assert.deepEqual(out.cache_control, { type: "ephemeral" });
}

function testQwenUsesLastMessageBreakpoint() {
  const out = applyQwenPromptCaching({
    model: "qwen-plus",
    messages: [
      {
        role: "system",
        content: [{ type: "text", text: "sys", cache_control: { type: "ephemeral" } }],
      },
      { role: "user", content: "latest" },
    ],
    tools: [toolWithCache()],
  } as any);
  assert.equal((out.tools?.[0] as any).cache_control, undefined);
  assert.deepEqual((out.messages[1].content as any[])[0].cache_control, {
    type: "ephemeral",
  });
}

function testCacheSessionKeyIsHashed() {
  const key = deriveCacheSessionKey(
    { req: { sessionId: "raw-session-secret" } },
    { model: "gpt-5.6", messages: [{ role: "user", content: "hi" }] } as any
  );
  assert.ok(key?.startsWith("ccr_"));
  assert.equal(key?.includes("raw-session-secret"), false);
}

function testCacheSessionKeyIgnoresSystemTextWhenSessionPresent() {
  const session = "32c43daa-888d-4573-a563-ee88b833801d";
  const context = { protocolContext: { sessionId: session } };
  const a = deriveCacheSessionKey(context, {
    model: "grok-4.6",
    messages: [
      {
        role: "system",
        content: "x-anthropic-billing-header: cc_version=2.1.251.86c",
      },
      { role: "user", content: "hi" },
    ],
  } as any);
  const b = deriveCacheSessionKey(context, {
    model: "grok-4.6",
    messages: [
      {
        role: "system",
        content: "x-anthropic-billing-header: cc_version=2.1.251.e59; longer",
      },
      { role: "user", content: "hi" },
    ],
  } as any);
  assert.equal(a, b);
  assert.ok(a?.startsWith("ccr_"));
}

function testCacheSessionKeyFromAnthropicMetadataJson() {
  const session = extractClientSessionId({
    body: {
      metadata: {
        user_id: JSON.stringify({
          device_id: "dev",
          session_id: "32c43daa-888d-4573-a563-ee88b833801d",
        }),
      },
      system: "x-anthropic-billing-header: cc_version=2.1.251.e59",
    },
  });
  assert.equal(session, "32c43daa-888d-4573-a563-ee88b833801d");
}

async function testOpenAITransformerAddsNativeCacheFields() {
  const transformer = new OpenAITransformer();
  const out = await transformer.transformRequestIn(
    {
      model: "gpt-5.6",
      messages: [
        {
          role: "system",
          content: [
            {
              type: "text",
              text: "stable",
              cache_control: { type: "ephemeral" },
            },
          ],
        },
        { role: "user", content: "hi" },
      ],
      tools: [toolWithCache()],
    } as any,
    { name: "openai", baseUrl: "https://api.openai.com/v1/chat/completions" },
    { req: { sessionId: "session-123" } }
  );
  assert.ok((out as any).prompt_cache_key?.startsWith("ccr_"));
  assert.equal((out as any).prompt_cache_retention, undefined);
  assert.deepEqual(
    ((out.messages[0].content as any[])[0] as any).prompt_cache_breakpoint,
    { mode: "explicit" }
  );
  assert.equal(((out.messages[0].content as any[])[0] as any).cache_control, undefined);
  assert.equal((out.tools?.[0] as any).cache_control, undefined);
}

async function testOpenAITransformsImageCacheBreakpoint() {
  const transformer = new OpenAITransformer();
  const out = await transformer.transformRequestIn(
    {
      model: "gpt-5.6",
      messages: [
        {
          role: "user",
          content: [
            {
              type: "image_url",
              image_url: { url: "https://example.com/image.png" },
              media_type: "image/png",
              cache_control: { type: "ephemeral" },
            },
          ],
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "Read", arguments: "{}" },
              cache_control: { type: "ephemeral" },
            },
          ],
        },
      ],
    } as any,
    { name: "openai", baseUrl: "https://api.openai.com/v1/chat/completions" },
    { req: { sessionId: "session-123" } }
  );

  assert.deepEqual(
    ((out.messages[0].content as any[])[0] as any).prompt_cache_breakpoint,
    { mode: "explicit" }
  );
  assert.equal(
    ((out.messages[0].tool_calls as any[])[0] as any).cache_control,
    undefined
  );
}

async function testOpenAICompatibleProviderDoesNotReceiveOpenAIFields() {
  const transformer = new OpenAITransformer();
  const out = await transformer.transformRequestIn(
    {
      model: "deepseek-chat",
      messages: [
        {
          role: "system",
          content: [
            {
              type: "text",
              text: "stable",
              cache_control: { type: "ephemeral" },
            },
          ],
        },
      ],
      tools: [toolWithCache()],
    } as any,
    {
      name: "deepseek",
      baseUrl: "https://api.deepseek.com/chat/completions",
    },
    { req: { sessionId: "session-123" } }
  );

  assert.equal((out as any).prompt_cache_key, undefined);
  assert.equal(
    ((out.messages[0].content as any[])[0] as any).cache_control,
    undefined
  );
  assert.equal((out.tools?.[0] as any).cache_control, undefined);
}

function testVertexClaudePreservesCacheControl() {
  const body = buildVertexClaudeBody({
    model: "claude-sonnet-4@20250514",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest);
  const text = body.messages[0].content.find((part: any) => part.type === "text") as any;
  assert.equal(text.cache_control, undefined);
  assert.equal((body.tools?.[0] as any)?.cache_control?.type, "ephemeral");
  assert.equal((body as any).cache_control?.type, "ephemeral");
}

async function testGeminiExplicitCachedContentAttachment() {
  clearGeminiCachedContentForTests();
  const originalFetch = globalThis.fetch;
  let createCalls = 0;
  globalThis.fetch = (async (_url: any, init: any) => {
    createCalls += 1;
    const body = JSON.parse(init.body);
    assert.equal(body.model, "models/gemini-2.5-pro");
    assert.ok(body.systemInstruction.parts[0].text.startsWith("stable context"));
    assert.equal(body.tools[0].functionDeclarations[0].name, "Bash");
    return new Response(JSON.stringify({ name: "cachedContents/abc" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }) as any;

  try {
    const longSystem = "stable context ".repeat(900);
    const body = {
      systemInstruction: { parts: [{ text: longSystem }] },
      contents: [{ role: "user", parts: [{ text: "hi" }] }],
      tools: [{ functionDeclarations: [{ name: "Bash" }] }],
    };
    const cacheOptions = {
      body,
      modelResource: "models/gemini-2.5-pro",
      createUrl: "https://generativelanguage.googleapis.com/v1beta/cachedContents",
      headers: { "x-goog-api-key": "key" },
    };
    const [out, concurrentOut] = await Promise.all([
      attachGeminiCachedContent(cacheOptions),
      attachGeminiCachedContent(cacheOptions),
    ]);
    assert.equal(out.cachedContent, "cachedContents/abc");
    assert.equal(concurrentOut.cachedContent, "cachedContents/abc");
    assert.equal(out.systemInstruction, undefined);
    assert.equal(out.tools, undefined);
    assert.equal(createCalls, 1);

    const reused = await attachGeminiCachedContent(cacheOptions);
    assert.equal(reused.cachedContent, "cachedContents/abc");
    assert.equal(createCalls, 1);

    clearGeminiCachedContentForTests();
    globalThis.fetch = (async () => new Response("", { status: 500 })) as any;
    const fallback = await attachGeminiCachedContent(cacheOptions);
    assert.equal(fallback.cachedContent, undefined);
    assert.deepEqual(fallback.systemInstruction, body.systemInstruction);
    assert.deepEqual(fallback.tools, body.tools);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

function testStripMessagesStillWorks() {
  const messages = [
    {
      role: "user" as const,
      content: [
        { type: "text" as const, text: "hi", cache_control: { type: "ephemeral" } },
      ],
    },
  ];
  const stripped = stripMessagesCacheControl(messages as any);
  assert.equal((stripped[0].content as any)[0].cache_control, undefined);
}

async function main() {
  testStripToolsCacheControl();
  await testGroqStripsToolCacheControl();
  await testOpenrouterAddsNativeSessionAndCleansOpenAIFields();
  await testOpenrouterKeepsToolCacheControlForClaude();
  testMistralAddsPromptCacheKeyAndStripsToolCacheControl();
  testStripMessagesStillWorks();
  testAnthropicCachingIsBoundedAndImmutable();
  testAnthropicCachingTrimsExistingOverBudget();
  testAnthropicAutomaticCachingUsesAvailableSlot();
  testAnthropicCachingDoesNotTouchToolSchemas();
  testQwenUsesLastMessageBreakpoint();
  testCacheSessionKeyIsHashed();
  testCacheSessionKeyIgnoresSystemTextWhenSessionPresent();
  testCacheSessionKeyFromAnthropicMetadataJson();
  await testOpenAITransformerAddsNativeCacheFields();
  await testOpenAITransformsImageCacheBreakpoint();
  await testOpenAICompatibleProviderDoesNotReceiveOpenAIFields();
  testVertexClaudePreservesCacheControl();
  await testGeminiExplicitCachedContentAttachment();
  console.log("cache-control.strip: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

import assert from "node:assert/strict";
import {
  stripToolsCacheControl,
  stripMessagesCacheControl,
} from "../utils/cacheControl";
import { GroqTransformer } from "../transformer/groq.transformer";
import { OpenrouterTransformer } from "../transformer/openrouter.transformer";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";
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

async function testOpenrouterStripsToolCacheControlForNonClaude() {
  const transformer = new OpenrouterTransformer();
  const request = {
    model: "openai/gpt-4o",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest;

  const out = await transformer.transformRequestIn(request);
  assert.equal((out.tools?.[0] as any).cache_control, undefined);
}

async function testOpenrouterKeepsToolCacheControlForClaude() {
  const transformer = new OpenrouterTransformer();
  const request = {
    model: "anthropic/claude-sonnet-4-6",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest;

  const out = await transformer.transformRequestIn(request);
  // Claude models on OpenRouter support prompt caching — keep the hint.
  assert.deepEqual((out.tools?.[0] as any).cache_control, { type: "ephemeral" });
}

function testMistralStripsToolCacheControl() {
  const body = buildMistralBody({
    model: "mistral-medium-2505",
    messages: [{ role: "user", content: "hi" }],
    tools: [toolWithCache()],
  } as unknown as UnifiedChatRequest);

  assert.equal(body.tools?.[0]?.cache_control, undefined);
  assert.equal(body.tools?.[0]?.function?.name, "Bash");
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
  await testOpenrouterStripsToolCacheControlForNonClaude();
  await testOpenrouterKeepsToolCacheControlForClaude();
  testMistralStripsToolCacheControl();
  testStripMessagesStillWorks();
  console.log("cache-control.strip: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

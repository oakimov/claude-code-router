/**
 * Cache-optimal outbound message shape: a stable conversation prefix must
 * not be rewritten when a later user turn is appended. Breakpoints and
 * session keys follow each destination's native cache contract.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { ChromeOnDeviceTransformer } from "../transformer/chrome-on-device.transformer";
import { CodexTransformer } from "../transformer/codex.transformer";
import { GeminiTransformer } from "../transformer/gemini.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { OpenrouterTransformer } from "../transformer/openrouter.transformer";
import { TooluseTransformer } from "../transformer/tooluse.transformer";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

const PLAN = "plan first";
const ANSWER = "visible answer";
const ANTHROPIC_SIG = "anth-sig-not-an-id";

function sessionCtx() {
  return { req: { id: "cache-prefix", sessionId: "stable-session", log: logger } } as any;
}

function json(value: unknown): string {
  return JSON.stringify(value);
}

function stripCacheMarkers(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripCacheMarkers);
  if (!value || typeof value !== "object") return value;
  const out: Record<string, unknown> = {};
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (
      key === "cache_control" ||
      key === "prompt_cache_breakpoint" ||
      key === "prompt_cache_key" ||
      key === "session_id"
    ) {
      continue;
    }
    out[key] = stripCacheMarkers(child);
  }
  return out;
}

function assertStablePrefix(
  label: string,
  shortItems: unknown[],
  longItems: unknown[]
) {
  assert.ok(
    longItems.length >= shortItems.length,
    `${label}: longer request must grow`
  );
  assert.equal(
    json(stripCacheMarkers(longItems.slice(0, shortItems.length))),
    json(stripCacheMarkers(shortItems)),
    `${label}: earlier turns must be a byte-stable prefix`
  );
}

async function inboundConversation(extraUser?: string) {
  const messages: any[] = [
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
        {
          type: "tool_result",
          tool_use_id: "call_1",
          content: "ok",
          cache_control: { type: "ephemeral" },
        },
      ],
    },
  ];
  if (extraUser) {
    messages.push({ role: "user", content: extraUser });
  }
  return new AnthropicTransformer().transformRequestOut({
    model: "claude-sonnet-4-20250514",
    max_tokens: 64,
    system: [
      {
        type: "text",
        text: "stable system",
        cache_control: { type: "ephemeral" },
      },
    ],
    messages,
    tools: [
      {
        name: "Read",
        description: "read",
        input_schema: { type: "object", properties: { path: { type: "string" } } },
        cache_control: { type: "ephemeral" },
      },
    ],
  });
}

async function toAnthropic(unified: any) {
  const tf = new AnthropicTransformer();
  (tf as any).logger = logger;
  const result = await tf.transformRequestIn(
    structuredClone(unified),
    { name: "claude", apiKey: "k", baseUrl: "https://api.anthropic.com", models: [] },
    sessionCtx()
  );
  return result.body;
}

async function toResponses(unified: any, model = "grok-4.6") {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = logger;
  return tf.transformRequestIn(
    { ...structuredClone(unified), model },
    { name: "xai-supergrok", baseUrl: "https://api.x.ai/v1" },
    sessionCtx()
  );
}

async function toChat(unified: any, provider: { name: string; baseUrl: string }) {
  return new OpenAITransformer().transformRequestIn(
    structuredClone(unified),
    provider,
    sessionCtx()
  );
}

async function toCodex(unified: any) {
  const tf = new CodexTransformer();
  (tf as any).logger = logger;
  (tf as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "t",
    accountId: "a",
    isFedramp: false,
  });
  const result = await tf.transformRequestIn(
    { ...structuredClone(unified), model: "gpt-5.6-sol", stream: false },
    { name: "codex", baseUrl: "https://chatgpt.com/backend-api/codex", apiKey: "at-test", models: [] },
    sessionCtx()
  );
  return result.body;
}

async function toGemini(unified: any) {
  const tf = new GeminiTransformer({
    cachedContent: false,
    thoughtSignatureFallback: "skip",
  } as any);
  (tf as any).logger = logger;
  const result = await tf.transformRequestIn(
    { ...structuredClone(unified), model: "gemini-3-flash" },
    {
      name: "antigravity",
      apiKey: "oauth",
      baseUrl: "https://daily-cloudcode-pa.sandbox.googleapis.com",
      models: [],
    },
    sessionCtx()
  );
  return result.body;
}

async function applyChromeTooluse(unified: any) {
  let requestBody = structuredClone(unified);
  let config: any = {};
  const chrome = await new ChromeOnDeviceTransformer().transformRequestIn(
    requestBody,
    { name: "chrome-nano", baseUrl: "http://127.0.0.1:3457" },
    sessionCtx()
  );
  requestBody = chrome.body;
  config = chrome.config;
  requestBody = await new TooluseTransformer().transformRequestIn(
    requestBody,
    {},
    sessionCtx()
  );
  return { body: requestBody, config };
}

function hasEphemeral(value: unknown): boolean {
  return json(value).includes('"type":"ephemeral"');
}

async function testAnthropicPrefixAndBreakpoints() {
  const short = await toAnthropic(await inboundConversation());
  const long = await toAnthropic(await inboundConversation("continue"));
  assert.deepEqual(stripCacheMarkers(short.system), stripCacheMarkers(long.system));
  assert.ok(hasEphemeral(short.system), "Anthropic keeps system cache_control");
  assert.ok(hasEphemeral(short.tools), "Anthropic keeps tool cache_control");
  assertStablePrefix("Anthropic messages", short.messages, long.messages);
  const assistant = short.messages.find((m: any) => m.role === "assistant");
  assert.deepEqual(
    assistant.content.map((b: any) => b.type),
    ["thinking", "text", "tool_use"]
  );
}

async function testResponsesAndCodexPrefix() {
  const short = (await toResponses(await inboundConversation())) as any;
  const long = (await toResponses(await inboundConversation("continue"))) as any;
  assert.equal(short.prompt_cache_key, long.prompt_cache_key);
  assert.ok(String(short.prompt_cache_key).startsWith("ccr_"));
  assertStablePrefix("Responses input", short.input, long.input);
  assert.equal(json(short).includes("cache_control"), false);
  const start = short.input.findIndex((item: any) => item.type === "reasoning");
  assert.deepEqual(
    short.input.slice(start, start + 3).map((item: any) => item.type || item.role),
    ["reasoning", "assistant", "function_call"]
  );

  const gpt = (await toResponses(await inboundConversation(), "gpt-5.6")) as any;
  assert.ok(String(gpt.prompt_cache_key).startsWith("ccr_"));

  const shortCodex = await toCodex(await inboundConversation());
  const longCodex = await toCodex(await inboundConversation("continue"));
  assert.equal(shortCodex.prompt_cache_key, longCodex.prompt_cache_key);
  assert.equal(json(shortCodex).includes("prompt_cache_breakpoint"), false);
  assertStablePrefix("Codex input", shortCodex.input, longCodex.input);
}

async function testChatCompatAndOpenRouter() {
  const nvidia = {
    name: "nvidia",
    baseUrl: "https://integrate.api.nvidia.com/v1/chat/completions",
  };
  const short = (await toChat(await inboundConversation(), nvidia)) as any;
  const long = (await toChat(
    await inboundConversation("continue"),
    nvidia
  )) as any;
  assertStablePrefix("NVIDIA Chat messages", short.messages, long.messages);
  assert.equal(json(short).includes("cache_control"), false);
  const assistant = short.messages.find((m: any) => m.role === "assistant");
  assert.equal(assistant.thinking, undefined);
  assert.equal(assistant.reasoning_content, PLAN);

  const openrouter = (
    await new OpenrouterTransformer().transformRequestIn(
      {
        ...(await inboundConversation()),
        model: "google/gemma-4-26b-a4b-it:free",
      } as any,
      {},
      sessionCtx()
    )
  ).body;
  assert.ok(String((openrouter as any).session_id).startsWith("ccr_"));
  assert.ok(String((openrouter as any).prompt_cache_key).startsWith("ccr_"));
  assert.equal(json(openrouter).includes("cache_control"), false);

  const openrouterClaude = (
    await new OpenrouterTransformer().transformRequestIn(
      {
        ...(await inboundConversation()),
        model: "anthropic/claude-sonnet-4-6",
      } as any,
      {},
      sessionCtx()
    )
  ).body;
  assert.ok(
    hasEphemeral(openrouterClaude),
    "OpenRouter Claude keeps Anthropic-format cache_control"
  );
}

async function testMistralAndGeminiPrefix() {
  const shortM = buildMistralBody(
    { ...(await inboundConversation()), model: "codestral-latest" } as any,
    sessionCtx(),
    { name: "codestral" }
  );
  const longM = buildMistralBody(
    {
      ...(await inboundConversation("continue")),
      model: "codestral-latest",
    } as any,
    sessionCtx(),
    { name: "codestral" }
  );
  assert.equal(shortM.prompt_cache_key, longM.prompt_cache_key);
  assertStablePrefix("Mistral messages", shortM.messages, longM.messages);
  assert.equal(json(shortM).includes("cache_control"), false);

  const shortG = await toGemini(await inboundConversation());
  const longG = await toGemini(await inboundConversation("continue"));
  assert.deepEqual(shortG.systemInstruction, longG.systemInstruction);
  // Gemini consolidates adjacent user turns, so a follow-up user may merge
  // into the tool-result user. Committed turns before that open user stay
  // stable — that is the cacheable prefix.
  assertStablePrefix(
    "Gemini contents",
    shortG.contents.slice(0, -1),
    longG.contents.slice(0, -1)
  );
  assert.ok(
    JSON.stringify(shortG.contents).includes("functionResponse"),
    "Gemini keeps functionResponse"
  );
  const model = shortG.contents.find((c: any) => c.role === "model");
  const kinds = model.parts.map((p: any) =>
    p.thought ? "thought" : p.functionCall ? "functionCall" : "text"
  );
  assert.ok(kinds.indexOf("thought") < kinds.indexOf("text") || !kinds.includes("thought"));
  if (kinds.includes("text") && kinds.includes("functionCall")) {
    assert.ok(kinds.indexOf("text") < kinds.indexOf("functionCall"));
  }
}

async function testChromeTooluseStableSession() {
  const short = await applyChromeTooluse(await inboundConversation());
  const long = await applyChromeTooluse(await inboundConversation("continue"));
  assert.equal(
    short.config.headers["x-ccr-session-id"],
    long.config.headers["x-ccr-session-id"]
  );
  assert.equal(json(short.body).includes("cache_control"), false);
  const reminderCount = (messages: any[]) =>
    messages.filter(
      (m: any) =>
        m.role === "system" &&
        typeof m.content === "string" &&
        m.content.includes("Tool mode is active")
    ).length;
  assert.equal(reminderCount(short.body.messages), 1);
  assert.equal(reminderCount(long.body.messages), 1);
}

async function main() {
  await testAnthropicPrefixAndBreakpoints();
  await testResponsesAndCodexPrefix();
  await testChatCompatAndOpenRouter();
  await testMistralAndGeminiPrefix();
  await testChromeTooluseStableSession();
  console.log("cross-protocol.cache-prefix: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

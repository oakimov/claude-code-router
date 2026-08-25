/**
 * REASONING_AUTO_SUMMARY opt-in: stamp Unified reasoning.summary once so every
 * destination (Responses, Codex, Anthropic, Gemini) can request readable
 * thinking when the client only sent effort.
 */
import assert from "node:assert/strict";
import {
  applyReasoningAutoSummary,
  resolveOutboundReasoningSummary,
  resolveReasoningAutoSummary,
} from "../utils/reasoning-effort";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { buildGeminiThinkingConfig } from "../utils/gemini-thinking";
import type { UnifiedChatRequest } from "../types/llm";

function effortOnly(): UnifiedChatRequest {
  return {
    model: "test",
    messages: [{ role: "user", content: "hi" }],
    reasoning: { effort: "high", enabled: true },
  };
}

async function configParsing(): Promise<void> {
  assert.equal(resolveReasoningAutoSummary(true), "detailed");
  assert.equal(resolveReasoningAutoSummary("true"), "detailed");
  assert.equal(resolveReasoningAutoSummary("auto"), "auto");
  assert.equal(resolveReasoningAutoSummary("detailed"), "detailed");
  assert.equal(resolveReasoningAutoSummary("concise"), "concise");
  assert.equal(resolveReasoningAutoSummary(false), undefined);
  assert.equal(resolveReasoningAutoSummary("none"), undefined);
  assert.equal(resolveReasoningAutoSummary(undefined), undefined);
}

async function applyStampsUnifiedSummary(): Promise<void> {
  const req = effortOnly();
  applyReasoningAutoSummary(req, true);
  assert.equal(req.reasoning?.summary, "detailed");

  const already = effortOnly();
  already.reasoning!.summary = "auto";
  applyReasoningAutoSummary(already, true);
  assert.equal(already.reasoning?.summary, "auto", "explicit client wins");

  const optedOut = effortOnly();
  optedOut.reasoning!.summary = "none";
  applyReasoningAutoSummary(optedOut, true);
  assert.equal(optedOut.reasoning?.summary, "none", "explicit none wins");

  const off = effortOnly();
  off.reasoning = { effort: "none", enabled: false };
  applyReasoningAutoSummary(off, true);
  assert.equal(off.reasoning?.summary, undefined);

  const idle = effortOnly();
  applyReasoningAutoSummary(idle, false);
  assert.equal(idle.reasoning?.summary, undefined);
}

async function outboundPrecedence(): Promise<void> {
  assert.equal(
    resolveOutboundReasoningSummary({
      reasoning: { effort: "high", summary: "auto" },
    }),
    "auto"
  );
  assert.equal(
    resolveOutboundReasoningSummary(
      { reasoning: { effort: "high" } },
      { reasoningSummary: "detailed" }
    ),
    "detailed"
  );
  assert.equal(
    resolveOutboundReasoningSummary(
      { reasoning: { effort: "high", summary: "none" } },
      { reasoningSummary: "detailed" }
    ),
    undefined,
    "explicit none beats provider"
  );
  assert.equal(
    resolveOutboundReasoningSummary({ reasoning: { effort: "high" } }),
    undefined
  );
}

async function responsesOutboundUsesSummary(): Promise<void> {
  const tf = new OpenAIResponsesTransformer();
  const withFlag = await tf.transformRequestIn(
    applyReasoningAutoSummary(effortOnly(), true),
    {},
    {}
  );
  assert.equal((withFlag as any).reasoning?.summary, "detailed");

  const withProvider = await tf.transformRequestIn(effortOnly(), {
    reasoningSummary: "auto",
  }, {});
  assert.equal((withProvider as any).reasoning?.summary, "auto");

  const bare = await tf.transformRequestIn(effortOnly(), {}, {});
  assert.equal((bare as any).reasoning?.summary, undefined);
}

async function anthropicOutboundUsesDisplay(): Promise<void> {
  const tf = new AnthropicTransformer();
  const provider = {
    apiKey: "test-key",
    baseUrl: "https://api.anthropic.test",
    transformer: { use: [] },
  } as any;

  const withSummary = await tf.transformRequestIn(
    applyReasoningAutoSummary(effortOnly(), true),
    provider,
    {}
  );
  assert.equal((withSummary as any).body.thinking.type, "adaptive");
  assert.equal((withSummary as any).body.thinking.display, "summarized");

  const bare = await tf.transformRequestIn(effortOnly(), provider, {});
  assert.equal((bare as any).body.thinking.type, "adaptive");
  assert.equal((bare as any).body.thinking.display, undefined);
}

async function geminiRespectsSummaryNone(): Promise<void> {
  const on = buildGeminiThinkingConfig({
    model: "gemini-3-flash-preview",
    reasoning: { effort: "high", enabled: true },
  });
  assert.equal(on?.includeThoughts, true);

  const auto = buildGeminiThinkingConfig({
    model: "gemini-3-flash-preview",
    reasoning: { effort: "high", enabled: true, summary: "detailed" },
  });
  assert.equal(auto?.includeThoughts, true);

  const hidden = buildGeminiThinkingConfig({
    model: "gemini-3-flash-preview",
    reasoning: { effort: "high", enabled: true, summary: "none" },
  });
  assert.equal(hidden?.includeThoughts, false);
}

async function main(): Promise<void> {
  await configParsing();
  await applyStampsUnifiedSummary();
  await outboundPrecedence();
  await responsesOutboundUsesSummary();
  await anthropicOutboundUsesDisplay();
  await geminiRespectsSummaryNone();
  console.log("reasoning.auto-summary: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

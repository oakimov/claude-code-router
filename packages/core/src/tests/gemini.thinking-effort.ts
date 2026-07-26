/**
 * Effort translation for every Gemini/Antigravity model type.
 *
 * Claude Code is authoritative: the effort it sends decides how much the model
 * thinks, and the configured model id is never rewritten. Each family gets the
 * dialect it accepts (thinkingLevel vs thinkingBudget), never both.
 *
 * The end-to-end case below replays the exact request shape Claude Code sends,
 * captured from ~/.claude-code-router/logs:
 *   { model, max_tokens: 16384, thinking: {type:"adaptive", display:"summarized"},
 *     output_config: {effort:"high"}, stream: true }
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { buildRequestBody } from "../utils/gemini.util";
import {
  buildGeminiThinkingConfig,
  resolveThinkingDialect,
  translateThinkingLevel,
} from "../utils/gemini-thinking";
import type { UnifiedChatRequest } from "../types/llm";

function thinkingConfigFor(
  model: string,
  reasoning: UnifiedChatRequest["reasoning"],
  max_tokens?: number
) {
  return buildGeminiThinkingConfig({ model, reasoning, max_tokens });
}

/** The shape Claude Code actually sends, through the full request pipeline. */
async function testClaudeCodeAdaptiveHighEndToEnd() {
  const transformer = new AnthropicTransformer();
  const unified = await transformer.transformRequestOut({
    model: "gemini-3.6-flash-tiered",
    max_tokens: 16384,
    thinking: { type: "adaptive", display: "summarized" },
    output_config: { effort: "high" },
    stream: true,
    messages: [{ role: "user", content: "what's in this folder" }],
  } as any);

  assert.equal(unified.reasoning?.effort, "high");
  assert.equal(unified.reasoning?.enabled, true);
  assert.equal(unified.reasoning?.max_tokens, undefined, "no budget was sent");

  const body = buildRequestBody(unified);
  assert.deepEqual(body.generationConfig.thinkingConfig, {
    includeThoughts: true,
    thinkingLevel: "high",
  });
  // Never both dialects in one request.
  assert.equal(body.generationConfig.thinkingConfig.thinkingBudget, undefined);
  // The picked model is the model we talk to.
  assert.equal(unified.model, "gemini-3.6-flash-tiered");
}

/** Tier-suffixed ids keep their id; the level comes from Claude Code. */
async function testTierSuffixNeverOverridesEffortAndIdIsStable() {
  const req: UnifiedChatRequest = {
    model: "gemini-3-pro-low",
    max_tokens: 16384,
    messages: [{ role: "user", content: "hi" }],
    reasoning: { effort: "high", enabled: true },
  };
  const body = buildRequestBody(req);
  assert.equal(body.generationConfig.thinkingConfig.thinkingLevel, "high");
  assert.equal(req.model, "gemini-3-pro-low", "model id must not be rewritten");
  assert.equal(body.model, undefined, "request body carries no model override");
}

/** Per-family level sets, with out-of-range efforts rounded up. */
async function testLevelSetsPerFamily() {
  const cases: Array<[string, string, string]> = [
    // Gemini 3 Pro: low|high only — medium rounds up rather than degrading.
    ["gemini-3-pro-low", "medium", "high"],
    ["gemini-3-pro-high", "low", "low"],
    ["gemini-3-pro-preview", "minimal", "low"],
    // Claude effort levels beyond Gemini's range collapse to the ceiling.
    ["gemini-3-pro-low", "xhigh", "high"],
    ["gemini-3.6-flash-tiered", "max", "high"],
    // Later Pro minors add medium.
    ["gemini-3.1-pro-low", "medium", "medium"],
    ["gemini-pro-agent", "medium", "medium"],
    // Flash families additionally accept minimal.
    ["gemini-3-flash", "minimal", "minimal"],
    ["gemini-3.1-flash-lite", "minimal", "minimal"],
    ["gemini-3.6-flash-tiered", "medium", "medium"],
  ];

  for (const [model, effort, expected] of cases) {
    const config = thinkingConfigFor(model, { effort: effort as any, enabled: true });
    assert.deepEqual(
      config,
      { includeThoughts: true, thinkingLevel: expected },
      `${model} + ${effort}`
    );
  }
}

/** Thinking enabled but no effort: keep the model default, still ask for thoughts. */
async function testNoEffortStillRequestsThoughts() {
  const config = thinkingConfigFor("gemini-3.6-flash-tiered", { enabled: true });
  assert.deepEqual(config, { includeThoughts: true });
}

/** effort "none": Gemini 3 cannot disable thinking, so ask for the floor. */
async function testEffortNoneUsesFamilyFloor() {
  assert.deepEqual(
    thinkingConfigFor("gemini-3-pro-low", { effort: "none", enabled: true }),
    { includeThoughts: false, thinkingLevel: "low" }
  );
  assert.deepEqual(
    thinkingConfigFor("gemini-3-flash", { effort: "none", enabled: true }),
    { includeThoughts: false, thinkingLevel: "minimal" }
  );
  // Budget families can be switched off outright when 0 is allowed.
  assert.deepEqual(
    thinkingConfigFor("gemini-2.5-flash", { effort: "none", enabled: true }),
    { includeThoughts: false, thinkingBudget: 0 }
  );
  // Claude's floor is 1024, so the config is dropped instead.
  assert.equal(
    thinkingConfigFor("claude-sonnet-4-6", { effort: "none", enabled: true }),
    undefined
  );
}

/** Claude and Gemini 2.5 on Antigravity speak token budgets, never levels. */
async function testBudgetDialectFamilies() {
  const claudeHigh = thinkingConfigFor(
    "claude-opus-4-6-thinking",
    { effort: "high", enabled: true },
    64000
  );
  assert.equal(claudeHigh?.thinkingLevel, undefined, "no level for Claude");
  assert.ok(claudeHigh?.thinkingBudget && claudeHigh.thinkingBudget >= 1024);
  assert.ok(claudeHigh!.thinkingBudget! < 64000, "must leave room for the answer");

  // Effort scales the budget.
  const low = thinkingConfigFor("claude-sonnet-4-6", { effort: "low", enabled: true });
  const medium = thinkingConfigFor("claude-sonnet-4-6", { effort: "medium", enabled: true });
  assert.ok(low!.thinkingBudget! < medium!.thinkingBudget!);

  // An explicit client budget wins and is clamped to the family range.
  assert.equal(
    thinkingConfigFor("gemini-2.5-pro", { effort: "high", max_tokens: 16384 })
      ?.thinkingBudget,
    16384
  );
  assert.equal(
    thinkingConfigFor("gemini-2.5-pro", { effort: "high", max_tokens: 999999 })
      ?.thinkingBudget,
    32768
  );

  // Answer budget too small to satisfy Anthropic's floor: drop thinking.
  assert.equal(
    thinkingConfigFor("claude-sonnet-4-6", { effort: "high", enabled: true }, 512),
    undefined
  );
}

/** Claude Code's budget_tokens reaches the budget families when a client sends it. */
async function testExplicitBudgetTokensRoundTrip() {
  const transformer = new AnthropicTransformer();
  const unified = await transformer.transformRequestOut({
    model: "claude-sonnet-4-6",
    max_tokens: 16384,
    thinking: { type: "enabled", budget_tokens: 4096 },
    messages: [{ role: "user", content: "hi" }],
  } as any);

  assert.equal(unified.reasoning?.max_tokens, 4096);
  const config = buildGeminiThinkingConfig(unified);
  assert.deepEqual(config, { includeThoughts: true, thinkingBudget: 4096 });
}

/** Image models must not carry thinking config; unknown ids stay conservative. */
async function testImageAndUnknownModels() {
  assert.equal(
    thinkingConfigFor("gemini-3-pro-image", { effort: "high", enabled: true }),
    undefined
  );
  assert.deepEqual(
    thinkingConfigFor("gpt-oss-120b-medium", { effort: "high", enabled: true }),
    { includeThoughts: true },
    "unknown family: request thoughts, never guess a level or budget"
  );
  // No reasoning at all leaves the model on its own default.
  assert.equal(thinkingConfigFor("gemini-3-flash", undefined), undefined);
}

/** Dialect resolution is by family, so suffixes never change the dialect. */
async function testDialectResolution() {
  assert.equal(resolveThinkingDialect("models/gemini-3.6-flash-tiered").kind, "level");
  assert.equal(resolveThinkingDialect("GEMINI-3-PRO-LOW").kind, "level");
  assert.equal(resolveThinkingDialect("claude-sonnet-4-6").kind, "budget");
  assert.equal(resolveThinkingDialect("gemini-2.5-flash").kind, "budget");
  assert.equal(resolveThinkingDialect("gemini-3.1-flash-image").kind, "none");
  assert.equal(translateThinkingLevel("medium", ["low", "high"]), "high");
  assert.equal(translateThinkingLevel("max", ["minimal", "low"]), "low");
}

async function main() {
  await testClaudeCodeAdaptiveHighEndToEnd();
  await testTierSuffixNeverOverridesEffortAndIdIsStable();
  await testLevelSetsPerFamily();
  await testNoEffortStillRequestsThoughts();
  await testEffortNoneUsesFamilyFloor();
  await testBudgetDialectFamilies();
  await testExplicitBudgetTokensRoundTrip();
  await testImageAndUnknownModels();
  await testDialectResolution();
  console.log("gemini.thinking-effort: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

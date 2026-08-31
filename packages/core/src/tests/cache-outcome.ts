import assert from "node:assert/strict";
import {
  __resetGeminiCachedContentNamesForTests,
  buildCachePrediction,
  classifyCacheOutcome,
  predictAnthropicEphemeral,
  predictCursorConversation,
  predictGeminiCachedContent,
  predictOpenAiPrefix,
  resolveCacheFamily,
} from "../utils/cache-outcome";
import type { CachePrefixDiff } from "../utils/cache-prefix-debug";

function intactDiff(overrides: Partial<CachePrefixDiff> = {}): CachePrefixDiff {
  return {
    conversationId: "c",
    conversationIdSource: "session",
    stage: "wire",
    firstTurn: false,
    prefixIntact: true,
    change: "appended",
    unchangedPrefixCount: 2,
    previousSegmentCount: 2,
    currentSegmentCount: 3,
    unchangedPrefixApproxTokens: 100,
    approxPrefixTokensLost: 0,
    prompt_cache_keyChanged: false,
    affinityChanged: false,
    lastAssistantBlockOrderChanged: false,
    modelChanged: false,
    systemHashChanged: false,
    toolsHashChanged: false,
    breakpointsMoved: false,
    ...overrides,
  };
}

function brokenDiff(): CachePrefixDiff {
  return intactDiff({
    prefixIntact: false,
    change: "modified",
    firstDivergencePath: "messages[0]",
    approxPrefixTokensLost: 4200,
  });
}

// --- classifyCacheOutcome ---
{
  assert.equal(classifyCacheOutcome(null, undefined), "unknown");
  assert.equal(
    classifyCacheOutcome({ family: "openai_prefix", firstTurn: true, predictedHit: false, reason: "first" }, 0),
    "cold"
  );
  assert.equal(
    classifyCacheOutcome({ family: "openai_prefix", firstTurn: true, predictedHit: false, reason: "first" }, 0.5),
    "warm-start"
  );
  const hitPred = { family: "openai_prefix" as const, firstTurn: false, predictedHit: true, reason: "prefix-intact" };
  assert.equal(classifyCacheOutcome(hitPred, 0.9), "hit");
  assert.equal(classifyCacheOutcome(hitPred, 0), "unexpected-miss");
  const missPred = { family: "openai_prefix" as const, firstTurn: false, predictedHit: false, reason: "broken" };
  assert.equal(classifyCacheOutcome(missPred, 0), "expected-miss");
  assert.equal(classifyCacheOutcome(missPred, 0.4), "partial");
}

// --- resolveCacheFamily ---
{
  assert.equal(resolveCacheFamily({ provider: "cursor" }), "cursor_conversation");
  assert.equal(
    resolveCacheFamily({ provider: "openai", cursorLifecycle: { action: "send-incremental" } }),
    "cursor_conversation"
  );
  assert.equal(resolveCacheFamily({ provider: "deepseek" }), "deepseek_prefix");
  assert.equal(resolveCacheFamily({ provider: "anthropic" }), "anthropic_ephemeral");
  assert.equal(resolveCacheFamily({ provider: "gemini" }), "gemini_cached_content");
  assert.equal(
    resolveCacheFamily({
      provider: "xai",
      body: { messages: [{ role: "user", content: "hi" }], prompt_cache_key: "ccr_x" },
    }),
    "openai_prefix"
  );
  assert.equal(resolveCacheFamily({ provider: "mystery" }), "unknown");
}

// --- OpenAI prefix prediction ---
{
  const hit = predictOpenAiPrefix(intactDiff());
  assert.equal(hit.predictedHit, true);
  assert.equal(classifyCacheOutcome(hit, 0.8), "hit");

  const miss = predictOpenAiPrefix(brokenDiff());
  assert.equal(miss.predictedHit, false);
  assert.equal(classifyCacheOutcome(miss, 0), "expected-miss");

  const unexpected = predictOpenAiPrefix(intactDiff());
  assert.equal(classifyCacheOutcome(unexpected, 0), "unexpected-miss");
}

// --- Anthropic ephemeral ---
{
  const withBp = {
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: "hi", cache_control: { type: "ephemeral" } }],
      },
    ],
  };
  const hit = predictAnthropicEphemeral(intactDiff(), withBp);
  assert.equal(hit.predictedHit, true);
  assert.equal(classifyCacheOutcome(hit, 0.9), "hit");

  const noBp = predictAnthropicEphemeral(intactDiff(), { messages: [{ role: "user", content: "hi" }] });
  assert.equal(noBp.predictedHit, false);
  assert.equal(noBp.reason, "no-ephemeral-breakpoints");
  assert.equal(classifyCacheOutcome(noBp, 0), "expected-miss");
}

// --- Gemini cachedContent ---
{
  __resetGeminiCachedContentNamesForTests();
  const first = predictGeminiCachedContent({
    body: { cachedContent: "projects/x/cachedContents/a" },
    conversationId: "g1",
  });
  assert.equal(first.firstTurn, true);
  assert.equal(classifyCacheOutcome(first, 0), "cold");

  const reused = predictGeminiCachedContent({
    body: { cachedContent: "projects/x/cachedContents/a" },
    conversationId: "g1",
    diff: intactDiff(),
  });
  assert.equal(reused.predictedHit, true);
  assert.equal(classifyCacheOutcome(reused, 0.7), "hit");

  const rotated = predictGeminiCachedContent({
    body: { cachedContent: "projects/x/cachedContents/b" },
    conversationId: "g1",
  });
  assert.equal(rotated.predictedHit, false);
  assert.equal(rotated.reason, "cached-content-rotated");
  assert.equal(classifyCacheOutcome(rotated, 0), "expected-miss");
}

// --- Cursor lifecycle ---
{
  const incremental = predictCursorConversation({
    lifecycle: { sessionKey: "sk", action: "send-incremental", reason: "strictly-aligned-idle-session" },
    diff: brokenDiff(), // host prefix broken must NOT flip prediction
  });
  assert.equal(incremental.predictedHit, true);
  assert.equal(incremental.hostPrefixIntact, false);
  assert.equal(classifyCacheOutcome(incremental, 0.6), "hit");

  const resume = predictCursorConversation({
    lifecycle: { sessionKey: "sk", action: "resume-parked", reason: "exact-parked-tool-results" },
  });
  assert.equal(resume.predictedHit, true);

  const softDivergent = predictCursorConversation({
    lifecycle: {
      sessionKey: "sk",
      action: "send-incremental",
      reason: "divergent-context-alignment",
    },
    diff: brokenDiff(),
  });
  assert.equal(softDivergent.predictedHit, true);
  assert.equal(softDivergent.reason, "divergent-context-alignment");
  assert.equal(classifyCacheOutcome(softDivergent, 0.3), "hit");

  const retire = predictCursorConversation({
    lifecycle: {
      sessionKey: "sk",
      action: "retire-and-replay-full",
      reason: "poisoned-session",
    },
    diff: brokenDiff(),
  });
  assert.equal(retire.predictedHit, false);
  assert.equal(retire.reason, "poisoned-session");
  assert.equal(classifyCacheOutcome(retire, 0), "expected-miss");
  assert.equal(classifyCacheOutcome(retire, 0.3), "partial");

  const fresh = predictCursorConversation({
    lifecycle: { sessionKey: "sk", action: "send-full", reason: "unused-session" },
  });
  assert.equal(fresh.firstTurn, true);
  assert.equal(classifyCacheOutcome(fresh, 0), "cold");
}

// --- buildCachePrediction dispatch ---
{
  const cursor = buildCachePrediction({
    provider: "cursor",
    cursorLifecycle: { sessionKey: "sk", action: "send-incremental" },
    diff: brokenDiff(),
  });
  assert.equal(cursor.family, "cursor_conversation");
  assert.equal(cursor.predictedHit, true);

  const openai = buildCachePrediction({
    provider: "opencode",
    body: { messages: [], prompt_cache_key: "ccr_x" },
    diff: intactDiff(),
  });
  assert.equal(openai.family, "openai_prefix");
  assert.equal(openai.predictedHit, true);
}

console.log("cache-outcome: ok");

import assert from "node:assert/strict";
import {
  buildAccurateUsageFromSdk,
  cacheReadFromSdkDelta,
  requestUsageFromEstimate,
  usageFromSdk,
} from "../cursor-sdk/usage";

const current = {
  prompt_tokens: 2000,
  completion_tokens: 0,
  total_tokens: 2000,
  prompt_tokens_details: { cached_tokens: 1000 },
};

const previous = {
  prompt_tokens: 1000,
  completion_tokens: 0,
  total_tokens: 1000,
  prompt_tokens_details: { cached_tokens: 100 },
};

// Legacy delta helper now delegates to accurateCacheReadForPrompt: prefix wins.
assert.equal(cacheReadFromSdkDelta(current, previous, 100), 100);

const resetUsage = {
  prompt_tokens: 100,
  completion_tokens: 0,
  total_tokens: 100,
  prompt_tokens_details: { cached_tokens: 25 },
};
assert.equal(cacheReadFromSdkDelta(resetUsage, current, 80), 20);

const clamped = requestUsageFromEstimate(10, 16, 99);
assert.deepEqual(clamped, {
  prompt_tokens: 10,
  completion_tokens: 4,
  total_tokens: 14,
  prompt_tokens_details: { cached_tokens: 10 },
});

// --- buildAccurateUsageFromSdk (finishUsage path) ---

// Cold: runtime reported no usage → estimate, no cache field ("unknown" verdict).
{
  const cold = buildAccurateUsageFromSdk(undefined, 400, 40);
  assert.deepEqual(cold, {
    prompt_tokens: 400,
    completion_tokens: 10,
    total_tokens: 410,
  });
}

// Proportional: no prior → cacheRead = round(prompt * rawCacheRead / rawInput).
{
  const sdk = {
    prompt_tokens: 2000,
    completion_tokens: 50,
    total_tokens: 2050,
    prompt_tokens_details: { cached_tokens: 1000 },
  };
  const usage = buildAccurateUsageFromSdk(sdk, 800, 20);
  assert.equal(usage.prompt_tokens, 800);
  assert.equal(usage.prompt_tokens_details?.cached_tokens, 400); // 800 * 1000/2000
  assert.equal(usage.completion_tokens, 5); // min(chars/4=5, sdk output=50)
  assert.equal(usage.total_tokens, 805);
  assert.ok(
    (usage.prompt_tokens_details?.cached_tokens ?? 0) <= usage.prompt_tokens
  );
}

// Prefix wins: prior input covered by cacheRead → do not dilute with TurnEnded aggregate.
{
  const sdk = {
    prompt_tokens: 2000,
    completion_tokens: 10,
    total_tokens: 2010,
    prompt_tokens_details: { cached_tokens: 1000 },
  };
  const prior = {
    prompt_tokens: 1000,
    completion_tokens: 0,
    total_tokens: 1000,
    prompt_tokens_details: { cached_tokens: 100 },
  };
  const usage = buildAccurateUsageFromSdk(sdk, 100, 0, prior);
  assert.equal(usage.prompt_tokens_details?.cached_tokens, 100); // min(prompt, priorInput)
}

// cacheWrite scales with prompt; never exceeds remaining uncached prompt.
{
  const sdk = usageFromSdk({
    usage: {
      inputTokens: 2000,
      outputTokens: 40,
      cacheRead: 500,
      cacheWrite: 1500,
    },
  });
  assert.ok(sdk);
  const usage = buildAccurateUsageFromSdk(sdk, 400, 16);
  // proportional cacheRead = round(400 * 500/2000) = 100
  assert.equal(usage.prompt_tokens_details?.cached_tokens, 100);
  // cacheWrite = round(400 * 1500/2000) = 300, clamped to prompt - cacheRead
  assert.equal((usage as any)._cacheWriteTokens, 300);
  assert.ok(
    (usage as any)._cacheWriteTokens +
      (usage.prompt_tokens_details?.cached_tokens ?? 0) <=
      usage.prompt_tokens
  );
}

// Sequential turns: turn B uses turn A's sdk raw as prior (finishUsage session path).
{
  const turnA = {
    prompt_tokens: 1000,
    completion_tokens: 20,
    total_tokens: 1020,
    prompt_tokens_details: { cached_tokens: 0 },
    _cacheWriteTokens: 1000,
  } as any;
  const turnB = {
    prompt_tokens: 1500,
    completion_tokens: 30,
    total_tokens: 1530,
    prompt_tokens_details: { cached_tokens: 1000 },
  };
  const a = buildAccurateUsageFromSdk(turnA, 200, 80);
  assert.equal(a.prompt_tokens_details?.cached_tokens, 0);
  const b = buildAccurateUsageFromSdk(turnB, 120, 40, turnA);
  // prefixRead = prior.input (1000) >= cacheRead? cacheRead=1000 → prefix = min(120,1000)=120
  assert.equal(b.prompt_tokens_details?.cached_tokens, 120);
}

// Native SDK TokenUsage shape (inputTokens/cacheRead) via usageFromSdk → accurate path.
{
  const mapped = usageFromSdk({
    usage: {
      inputTokens: 4000,
      outputTokens: 100,
      cacheRead: 3000,
      cacheWrite: 200,
      reasoningTokens: 40,
    },
  });
  assert.ok(mapped);
  assert.equal(mapped.prompt_tokens, 4000);
  assert.equal(mapped.prompt_tokens_details?.cached_tokens, 3000);
  assert.equal((mapped as any)._cacheWriteTokens, 200);
  const usage = buildAccurateUsageFromSdk(mapped, 500, 0);
  assert.equal(usage.prompt_tokens_details?.cached_tokens, 375); // 500 * 3000/4000
}

// Turn-end witness: a reported usage message carries billed cache proportions;
// an unreported turn omits prompt_tokens_details so the outcome tap says
// "unknown" instead of a bogus miss.
{
  const reported = buildAccurateUsageFromSdk(
    {
      prompt_tokens: 69231,
      completion_tokens: 1709,
      total_tokens: 135740,
      prompt_tokens_details: { cached_tokens: 64800 },
    },
    44286,
    6836
  );
  assert.ok(reported.prompt_tokens_details);
  assert.equal(
    reported.prompt_tokens_details?.cached_tokens,
    Math.min(44286, Math.round((44286 * 64800) / 69231))
  );
}
{
  const unreported = buildAccurateUsageFromSdk(undefined, 36983, 2740);
  assert.equal(unreported.prompt_tokens, 36983);
  assert.equal(unreported.completion_tokens, Math.ceil(2740 / 4));
  assert.ok(!("prompt_tokens_details" in unreported));
}

console.log("cursor-sdk.usage: ok");

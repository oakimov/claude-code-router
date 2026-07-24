import assert from "node:assert/strict";
import {
  cacheReadFromSdkDelta,
  requestUsageFromEstimate,
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

assert.equal(cacheReadFromSdkDelta(current, previous, 100), 90);

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

console.log("cursor-sdk.usage: ok");

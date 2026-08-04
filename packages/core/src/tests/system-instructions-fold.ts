/**
 * System/developer instruction preservation (Fix 3).
 * normalizeSystemToArray folds every system source exactly once, in source
 * order; buildAnthropicBody merges residual system messages instead of
 * dropping them. Billing stays at system[0], identity at system[1].
 */
import assert from "node:assert/strict";
import {
  __resetClaudeBillingStateForTests,
  applyClaudeBillingSystemBlock,
  applyClaudeSystemIdentity,
  normalizeSystemToArray,
  SYSTEM_IDENTITY,
} from "../utils/claude-billing";
import { CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX } from "../utils/router";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import type { UnifiedChatRequest } from "../types/llm";

function testOrderedFoldAcrossSources() {
  const request = {
    model: "claude",
    system: "top-level",
    messages: [
      { role: "system", content: "first-message" },
      { role: "user", content: "hello" },
      {
        role: "developer",
        content: [
          { type: "text", text: "dev-one" },
          {
            type: "text",
            text: "dev-two",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
      {
        role: "system",
        content: [{ type: "text", text: "second-message" }],
      },
    ],
  } as unknown as UnifiedChatRequest;

  const blocks = normalizeSystemToArray(request);
  assert.deepEqual(
    blocks.map((block) => block.text),
    ["top-level", "first-message", "dev-one", "dev-two", "second-message"]
  );
  assert.deepEqual(blocks[3].cache_control, { type: "ephemeral" });
  // Consumed system/developer messages leave only the user message behind.
  assert.deepEqual(
    request.messages.map((msg) => msg.role),
    ["user"]
  );
  assert.equal(request.system, blocks);
}

function testFoldWithoutTopLevelSystem() {
  const request = {
    model: "claude",
    messages: [
      { role: "system", content: "s1" },
      { role: "system", content: "s2" },
      { role: "user", content: "hi" },
    ],
  } as unknown as UnifiedChatRequest;
  const blocks = normalizeSystemToArray(request);
  assert.deepEqual(
    blocks.map((block) => block.text),
    ["s1", "s2"]
  );
  assert.equal(request.messages.length, 1);
}

function testBuildAnthropicBodyMergesResidualSystemMessages() {
  const body = AnthropicTransformer.buildAnthropicBody({
    model: "claude",
    system: [{ type: "text", text: "from-system-field" }],
    messages: [
      { role: "system", content: "residual-one" },
      {
        role: "developer",
        content: [{ type: "text", text: "residual-two" }],
      },
      { role: "user", content: "hi" },
    ],
  } as any);
  assert.ok(Array.isArray(body.system));
  assert.deepEqual(
    body.system.map((block: any) => block.text),
    ["from-system-field", "residual-one", "residual-two"]
  );
  assert.equal(body.messages.length, 1);
}

function testBuildAnthropicBodySinglePlainBlockStaysString() {
  const body = AnthropicTransformer.buildAnthropicBody({
    model: "claude",
    messages: [
      { role: "system", content: "be helpful" },
      { role: "user", content: "hi" },
    ],
  } as any);
  assert.equal(body.system, "be helpful");
}

function testBuildAnthropicBodyMultipleBlocksStayArray() {
  const body = AnthropicTransformer.buildAnthropicBody({
    model: "claude",
    system: "top",
    messages: [
      { role: "system", content: "extra" },
      { role: "user", content: "hi" },
    ],
  } as any);
  assert.ok(Array.isArray(body.system));
  assert.deepEqual(
    body.system.map((block: any) => block.text),
    ["top", "extra"]
  );
}

function testBillingIdentityThenCallerInstructions() {
  __resetClaudeBillingStateForTests();
  const request = {
    model: "claude",
    messages: [
      { role: "system", content: "instruction-one" },
      { role: "developer", content: "instruction-two" },
      {
        role: "system",
        content: [{ type: "text", text: "instruction-three" }],
      },
      { role: "user", content: "do the thing" },
    ],
  } as unknown as UnifiedChatRequest;

  const system = normalizeSystemToArray(request);
  applyClaudeBillingSystemBlock(system, request.messages);
  applyClaudeSystemIdentity(system);

  assert.equal(system.length, 5);
  assert.ok(system[0].text.startsWith(CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX));
  assert.equal(system[1].text, SYSTEM_IDENTITY);
  assert.deepEqual(
    system.slice(2).map((block) => block.text),
    ["instruction-one", "instruction-two", "instruction-three"]
  );
  // Every marker occurs exactly once.
  const all = system.map((block) => block.text);
  for (const marker of [
    SYSTEM_IDENTITY,
    "instruction-one",
    "instruction-two",
    "instruction-three",
  ]) {
    assert.equal(
      all.filter((text) => text === marker).length,
      1,
      `${marker} must occur exactly once`
    );
  }
  assert.equal(
    all.filter((text) =>
      text.startsWith(CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX)
    ).length,
    1
  );
}

function main() {
  testOrderedFoldAcrossSources();
  testFoldWithoutTopLevelSystem();
  testBuildAnthropicBodyMergesResidualSystemMessages();
  testBuildAnthropicBodySinglePlainBlockStaysString();
  testBuildAnthropicBodyMultipleBlocksStayArray();
  testBillingIdentityThenCallerInstructions();
  console.log("system-instructions-fold: all tests passed");
}

main();

import assert from "node:assert/strict";
import {
  CLAUDE_OAUTH_REQUIRED_BETA,
  mergeAnthropicBetaValues,
  readHeaderValue,
  resolveClaudeAuthAnthropicBeta,
} from "../transformer/claude-auth.transformer";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import type { UnifiedChatRequest } from "../types/llm";

function testMergeAnthropicBetaValues() {
  assert.equal(
    mergeAnthropicBetaValues(
      "context-management-2025-06-27,effort-2025-11-24",
      CLAUDE_OAUTH_REQUIRED_BETA
    ),
    "context-management-2025-06-27,effort-2025-11-24,oauth-2025-04-20"
  );
  assert.equal(
    mergeAnthropicBetaValues(
      "oauth-2025-04-20,effort-2025-11-24",
      CLAUDE_OAUTH_REQUIRED_BETA
    ),
    "oauth-2025-04-20,effort-2025-11-24"
  );
  assert.equal(
    mergeAnthropicBetaValues("Effort-2025-11-24", "effort-2025-11-24"),
    "Effort-2025-11-24"
  );
}

function testReadHeaderValue() {
  assert.equal(
    readHeaderValue(
      { "anthropic-beta": "context-management-2025-06-27,effort-2025-11-24" },
      "anthropic-beta"
    ),
    "context-management-2025-06-27,effort-2025-11-24"
  );
  assert.equal(
    readHeaderValue({ "Anthropic-Beta": "effort-2025-11-24" }, "anthropic-beta"),
    "effort-2025-11-24"
  );
  assert.equal(readHeaderValue({}, "anthropic-beta"), undefined);
  assert.equal(readHeaderValue(undefined, "anthropic-beta"), undefined);
  // Array values joined with ", "
  assert.equal(
    readHeaderValue({ "anthropic-beta": ["a", "b"] }, "anthropic-beta"),
    "a, b"
  );
}

function testResolvePrefersClientBetas() {
  const beta = resolveClaudeAuthAnthropicBeta({
    clientBeta: "context-management-2025-06-27,effort-2025-11-24",
  });

  assert.equal(
    beta,
    "context-management-2025-06-27,effort-2025-11-24,oauth-2025-04-20"
  );
}

function testResolveOnlyOauthWithoutClientBeta() {
  assert.equal(
    resolveClaudeAuthAnthropicBeta({}),
    CLAUDE_OAUTH_REQUIRED_BETA
  );
  assert.equal(
    resolveClaudeAuthAnthropicBeta({ clientBeta: undefined }),
    CLAUDE_OAUTH_REQUIRED_BETA
  );
  assert.equal(
    resolveClaudeAuthAnthropicBeta({ clientBeta: "" }),
    CLAUDE_OAUTH_REQUIRED_BETA
  );
}

async function testCacheControlRoundTrip() {
  const transformer = new AnthropicTransformer();
  const inbound = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    system: [
      { type: "text", text: "sys", cache_control: { type: "ephemeral" } },
    ],
    messages: [
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "toolu_1",
            content: "ok",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
    ],
    tools: [
      {
        name: "Bash",
        description: "run",
        input_schema: { type: "object", properties: {} },
        cache_control: { type: "ephemeral" },
      },
    ],
  };

  const unified = await transformer.transformRequestOut(inbound, {
    provider: { transformer: { use: [{ name: "claude-auth" }] } },
  });

  const rebuilt = AnthropicTransformer.buildAnthropicBody(unified);
  assert.deepEqual(rebuilt.system?.[0]?.cache_control, { type: "ephemeral" });
  assert.equal(rebuilt.tools?.[0]?.cache_control?.type, "ephemeral");

  const toolResult = rebuilt.messages?.[0]?.content?.find(
    (c: any) => c.type === "tool_result"
  );
  assert.deepEqual(toolResult?.cache_control, { type: "ephemeral" });
}

async function testMediaAndToolUseCacheControlRoundTrip() {
  const transformer = new AnthropicTransformer();
  const unified = await transformer.transformRequestOut({
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "image",
            source: {
              type: "base64",
              media_type: "image/png",
              data: "aGVsbG8=",
            },
            cache_control: { type: "ephemeral" },
          },
        ],
      },
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "toolu_2",
            name: "Read",
            input: { file_path: "README.md" },
            cache_control: { type: "ephemeral" },
          },
        ],
      },
    ],
  });

  const rebuilt = AnthropicTransformer.buildAnthropicBody(unified);
  assert.deepEqual(rebuilt.messages[0].content[0].cache_control, {
    type: "ephemeral",
  });
  assert.deepEqual(rebuilt.messages[1].content[0].cache_control, {
    type: "ephemeral",
  });
}

async function main() {
  testMergeAnthropicBetaValues();
  testReadHeaderValue();
  testResolvePrefersClientBetas();
  testResolveOnlyOauthWithoutClientBeta();
  await testCacheControlRoundTrip();
  await testMediaAndToolUseCacheControlRoundTrip();
  console.log("claude-auth.beta-headers: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

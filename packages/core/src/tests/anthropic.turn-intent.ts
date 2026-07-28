import assert from "node:assert/strict";
import { buildCursorSdkRunnerOptions } from "../transformer/cursor-sdk.transformer";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import type { TransformerContext } from "../types/transformer";
import {
  classifyAnthropicTurnIntent,
  extractAnthropicSourceSessionIdentity,
} from "../types/turn-intent";

function trailingUser(content: any[]) {
  return [{ role: "user", content }];
}

async function testAnthropicContextPropagation() {
  const context: TransformerContext = {
    signal: AbortSignal.timeout(10_000),
  };
  const transformer = new AnthropicTransformer();
  const unified = await transformer.transformRequestOut(
    {
      model: "composer-2.5",
      max_tokens: 1024,
      stream: true,
      metadata: {
        user_id: JSON.stringify({
          device_id: "device-private",
          session_id: "session-structured-intent",
        }),
      },
      messages: [
        {
          role: "assistant",
          content: [
            {
              type: "tool_use",
              id: "tool-build",
              name: "Bash",
              input: { command: "npm run build" },
            },
          ],
        },
        {
          role: "user",
          content: [
            {
              type: "tool_result",
              tool_use_id: "tool-build",
              is_error: true,
              content: "The user rejected this tool.",
            },
            {
              type: "tool_result",
              tool_use_id: "tool-read",
              is_error: false,
              content: [{ type: "text", text: "read result" }],
            },
            {
              type: "text",
              text: "  [Request interrupted by user for tool use]\n",
            },
            {
              type: "text",
              text: "Use only current dependency versions.",
            },
          ],
        },
      ],
    },
    context
  );

  assert.deepEqual(context.unifiedRequest, {
    source: "anthropic",
    sourceSessionIdentity: "session-structured-intent",
    turnIntent: {
      source: "anthropic",
      trailingToolResults: [
        {
          toolCallId: "tool-build",
          content: "The user rejected this tool.",
          isError: true,
        },
        {
          toolCallId: "tool-read",
          content: "read result",
          isError: false,
        },
      ],
      interruption: "synthetic_client_interrupt",
      steering: "meaningful",
    },
  });
  assert.deepEqual(
    unified.messages.map((message) => message.role),
    ["assistant", "tool", "tool", "user"]
  );

  const runnerOptions = buildCursorSdkRunnerOptions({}, context);
  assert.equal(runnerOptions.turnIntent, context.unifiedRequest?.turnIntent);
  assert.equal(
    runnerOptions.sourceSessionIdentity,
    "session-structured-intent"
  );
  assert.equal(runnerOptions.abortSignal, context.signal);

  const serialized = JSON.stringify(unified);
  assert.equal(serialized.includes("turnIntent"), false);
  assert.equal(serialized.includes("sourceSessionIdentity"), false);
  assert.equal(serialized.includes("device-private"), false);
  assert.equal(serialized.includes("session-structured-intent"), false);
}

function testExactSyntheticMarkerAllowlist() {
  for (const marker of [
    "[Request interrupted by user]",
    "[Request interrupted by user for tool use]",
  ]) {
    const intent = classifyAnthropicTurnIntent(
      trailingUser([
        {
          type: "tool_result",
          tool_use_id: "tool-1",
          content: "cancelled",
        },
        { type: "text", text: ` \n${marker}\n` },
      ])
    );
    assert.equal(intent.interruption, "synthetic_client_interrupt");
    assert.equal(intent.steering, "none");
  }

  const standaloneMarker = classifyAnthropicTurnIntent(
    trailingUser([{ type: "text", text: "[Request interrupted by user]" }])
  );
  assert.equal(standaloneMarker.interruption, "none");
  assert.equal(standaloneMarker.steering, "meaningful");

  for (const meaningfulText of [
    "Before [Request interrupted by user] after",
    "[request interrupted by user]",
    "[Request interrupted by user for tools use]",
  ]) {
    const intent = classifyAnthropicTurnIntent(
      trailingUser([
        {
          type: "tool_result",
          tool_use_id: "tool-1",
          content: "cancelled",
        },
        { type: "text", text: meaningfulText },
      ])
    );
    assert.equal(intent.interruption, "none");
    assert.equal(intent.steering, "meaningful");
  }

  const imageIntent = classifyAnthropicTurnIntent(
    trailingUser([
      {
        type: "tool_result",
        tool_use_id: "tool-1",
        content: "cancelled",
      },
      { type: "text", text: "[Request interrupted by user]" },
      {
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "AA==" },
      },
    ])
  );
  assert.equal(imageIntent.interruption, "synthetic_client_interrupt");
  assert.equal(imageIntent.steering, "meaningful");
}

function testSourceSessionIdentityFallbacks() {
  assert.equal(
    extractAnthropicSourceSessionIdentity({
      user_id: "user_session_session-suffix",
    }),
    "session-suffix"
  );
  assert.equal(
    extractAnthropicSourceSessionIdentity({ user_id: "opaque-user-id" }),
    "opaque-user-id"
  );
  assert.equal(extractAnthropicSourceSessionIdentity(undefined), undefined);
}

async function main() {
  await testAnthropicContextPropagation();
  testExactSyntheticMarkerAllowlist();
  testSourceSessionIdentityFallbacks();
  console.log("anthropic.turn-intent: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

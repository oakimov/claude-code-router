import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import {
  isValidAnthropicToolCallId,
  sanitizeToolCallId,
} from "../utils/toolCallId";

/** The exact shape the Cursor SDK hands back via SDKCustomToolContext.toolCallId. */
const CURSOR_CONCATENATED_ID =
  "call-901b1ddc-d889-4a6e-8c58-564ad17bc095-3\nfc_b466705e-df33-9395-8d4a-21a95066affe_0";

const ANTHROPIC_PATTERN = /^[a-zA-Z0-9_-]+$/;

function sanitizerContract() {
  const clean = sanitizeToolCallId(CURSOR_CONCATENATED_ID)!;
  assert.match(clean, ANTHROPIC_PATTERN);
  assert.ok(isValidAnthropicToolCallId(clean));

  // Idempotent: the id is sanitized on the way out AND on the way back in.
  // If the two passes disagreed, tool_use / tool_result would stop pairing.
  assert.equal(sanitizeToolCallId(clean), clean);

  // Valid ids must pass through byte-identical, or every existing non-Cursor
  // conversation would have its pairing rewritten for no reason.
  const valid = "toolu_01A09q90qw90lq917835lq9";
  assert.equal(sanitizeToolCallId(valid), valid);

  // Invalid runs collapse to a single "_" instead of being deleted: deletion
  // maps two distinct ids onto one, and a duplicate tool_use.id mispairs
  // results silently, which is worse than the 400 we are fixing.
  assert.notEqual(sanitizeToolCallId("a\nb"), sanitizeToolCallId("ab"));

  assert.equal(sanitizeToolCallId(""), undefined);
  assert.equal(sanitizeToolCallId(undefined), undefined);
  assert.equal(sanitizeToolCallId("\n\n"), undefined);

  // Over-long ids are truncated without leaving a trailing separator.
  const long = sanitizeToolCallId("x".repeat(300) + "\n" + "y".repeat(50))!;
  assert.ok(long.length <= 256);
  assert.match(long, ANTHROPIC_PATTERN);

  // Trailing-underscore runs trim without the backtracking `/_+$/` regex
  // (ReDoS-safe; same fix as sanitizeResponsesCallId).
  const underscored = sanitizeToolCallId(`${"a".repeat(200)}${"_".repeat(500)}`)!;
  assert.equal(underscored, "a".repeat(200));
  assert.equal(
    sanitizeToolCallId(`${"a".repeat(200)}${"_".repeat(500)}`),
    underscored
  );
}

/**
 * Conversations poisoned before the fix are replayed in full by Claude Code on
 * every subsequent turn, so a source-only fix in tools.ts would leave existing
 * sessions permanently broken. The request direction must repair them.
 */
async function replayedTranscriptIsRepaired() {
  const body = {
    model: "claude-opus-5",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: CURSOR_CONCATENATED_ID,
            name: "Bash",
            input: { command: "ls" },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: CURSOR_CONCATENATED_ID,
            content: "ok",
          },
        ],
      },
    ],
  };

  const transformer = new AnthropicTransformer();
  (transformer as any).logger = { debug() {}, error() {} };

  const unified = await transformer.transformRequestOut(body as any, {
    req: { id: "sanitize" },
  } as any);

  const outbound = AnthropicTransformer.buildAnthropicBody(unified as any);
  const serialized = JSON.stringify(outbound);

  assert.doesNotMatch(
    serialized,
    /\\n/,
    "no id may carry a newline into the Anthropic request"
  );

  const assistant = outbound.messages.find((m: any) => m.role === "assistant");
  const toolUse = assistant.content.find((c: any) => c.type === "tool_use");
  const resultTurn = outbound.messages.find((m: any) =>
    (m.content || []).some?.((c: any) => c.type === "tool_result")
  );
  const toolResult = resultTurn.content.find(
    (c: any) => c.type === "tool_result"
  );

  assert.match(toolUse.id, ANTHROPIC_PATTERN);
  assert.match(toolResult.tool_use_id, ANTHROPIC_PATTERN);

  // The pair must still reference each other; Anthropic 400s on an orphan.
  assert.equal(
    toolUse.id,
    toolResult.tool_use_id,
    "tool_use and tool_result must stay paired after sanitization"
  );
}

/** A bad id must never enter the transcript from the response direction. */
async function streamedToolUseIsSanitized() {
  const chunk = {
    id: "chatcmpl-cursor-1",
    object: "chat.completion.chunk",
    created: 1,
    model: "default",
    choices: [
      {
        index: 0,
        delta: {
          role: "assistant",
          tool_calls: [
            {
              index: 0,
              id: CURSOR_CONCATENATED_ID,
              type: "function",
              function: { name: "Bash", arguments: '{"command":"ls"}' },
            },
          ],
        },
        finish_reason: null,
      },
    ],
  };

  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        const encode = (o: unknown) =>
          new TextEncoder().encode(`data: ${JSON.stringify(o)}\n\n`);
        controller.enqueue(encode(chunk));
        controller.enqueue(
          encode({
            ...chunk,
            choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
          })
        );
        controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const transformer = new AnthropicTransformer();
  (transformer as any).logger = { debug() {}, error() {} };

  const response = await transformer.transformResponseIn(upstream, {
    req: { id: "sanitize-stream" },
  } as any);
  const body = await response.text();

  const started = [...body.matchAll(/"type":"tool_use","id":"([^"]*)"/g)];
  assert.ok(started.length > 0, "expected a tool_use content block");
  for (const [, id] of started) {
    assert.match(id, ANTHROPIC_PATTERN);
  }
}

async function main() {
  sanitizerContract();
  await replayedTranscriptIsRepaired();
  await streamedToolUseIsSanitized();

  console.log("anthropic.tool-call-id-sanitize: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

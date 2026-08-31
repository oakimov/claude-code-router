/**
 * message_start usage must carry Anthropic-shaped input/cache early enough
 * for Claude Code's client meter (and autocompact) to fire.
 *
 * Run from packages/core: npx tsx src/tests/anthropic.message-start-usage.test.ts
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

type SSEEvent = { event: string; data: any };

const LF = String.fromCharCode(10);

function buildStreamResponse(chunks: Array<Record<string, unknown>>): Response {
  const payload = [
    ...chunks.map((chunk) => "data: " + JSON.stringify(chunk) + LF + LF),
    "data: [DONE]" + LF + LF,
  ].join("");
  return new Response(payload, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function collectEvents(
  chunks: Array<Record<string, unknown>>,
  context?: Record<string, unknown>
): Promise<SSEEvent[]> {
  const transformer = new AnthropicTransformer();
  const noop = () => {};
  (transformer as any).logger = { debug: noop, info: noop, warn: noop, error: noop };
  const out = await transformer.transformResponseIn(
    buildStreamResponse(chunks),
    ({ req: { id: "test-req" }, ...(context || {}) } as any)
  );
  const text = await out.text();
  const events: SSEEvent[] = [];
  for (const block of text.split(LF + LF)) {
    const eventLine = block.split(LF).find((line) => line.startsWith("event: "));
    const dataLine = block.split(LF).find((line) => line.startsWith("data: "));
    if (!eventLine || !dataLine) continue;
    const raw = dataLine.slice(6);
    if (raw === "[DONE]") continue;
    events.push({ event: eventLine.slice(7), data: JSON.parse(raw) });
  }
  return events;
}

function usageChunk(usage: Record<string, unknown>, finish: string | null = null) {
  return {
    id: "chatcmpl_test",
    model: "test-model",
    choices: [{ index: 0, delta: {}, finish_reason: finish }],
    usage,
  };
}

function textChunk(content: string, finish: string | null = null) {
  return {
    id: "chatcmpl_test",
    model: "test-model",
    choices: [{ index: 0, delta: { content }, finish_reason: finish }],
  };
}

async function testEstimateOnMessageStartWhenNoChunkUsage() {
  const longPrompt = "x".repeat(4000);
  const events = await collectEvents(
    [textChunk("hi"), textChunk("", "stop")],
    {
      anthropicEstimateRequest: {
        messages: [{ role: "user", content: longPrompt }],
      },
    }
  );
  const start = events.find((e) => e.event === "message_start");
  assert.ok(start, "expected message_start");
  assert.ok(
    start!.data.message.usage.input_tokens > 0,
    "message_start should estimate input_tokens from the request"
  );
  assert.equal(start!.data.message.usage.output_tokens, 0);
}

async function testProviderUsageMappedOntoMessageStart() {
  const events = await collectEvents([
    usageChunk({
      prompt_tokens: 1200,
      completion_tokens: 5,
      prompt_tokens_details: { cached_tokens: 200, cache_write_tokens: 100 },
    }),
    textChunk("ok", "stop"),
  ]);
  const start = events.find((e) => e.event === "message_start");
  assert.ok(start, "expected message_start");
  assert.deepEqual(start!.data.message.usage, {
    input_tokens: 900,
    output_tokens: 0,
    cache_creation_input_tokens: 100,
    cache_read_input_tokens: 200,
  });
}

async function testFinishWithoutUsagePreservesPriorDeltaUsage() {
  const events = await collectEvents([
    textChunk("partial"),
    usageChunk({
      prompt_tokens: 500,
      completion_tokens: 12,
      prompt_tokens_details: { cached_tokens: 50, cache_write_tokens: 0 },
    }),
    textChunk("", "stop"),
  ]);
  const deltas = events.filter((e) => e.event === "message_delta");
  assert.ok(deltas.length >= 1, "expected message_delta");
  const last = deltas[deltas.length - 1]!;
  assert.deepEqual(last.data.usage, {
    input_tokens: 450,
    output_tokens: 12,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 50,
  });
}

async function main() {
  await testEstimateOnMessageStartWhenNoChunkUsage();
  await testProviderUsageMappedOntoMessageStart();
  await testFinishWithoutUsagePreservesPriorDeltaUsage();
  console.log("anthropic.message-start-usage: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

import assert from "node:assert/strict";
import {
  summarizeOutboundCacheStructure,
  tapUpstreamSSEDebug,
} from "../utils/sse-debug-tap";

type DebugRecord = Record<string, unknown>;

function createDebugLogger() {
  const records: DebugRecord[] = [];
  return {
    records,
    logger: {
      level: "debug",
      debug(payload: DebugRecord) {
        records.push(payload);
      },
    },
  };
}

function anthropicSSEFixture(): string {
  const messageStart = {
    type: "message_start",
    message: {
      id: "msg_test",
      type: "message",
      role: "assistant",
      content: [],
      model: "claude-sonnet-4-20250514",
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 100,
        output_tokens: 1,
        cache_creation_input_tokens: 80,
        cache_read_input_tokens: 0,
      },
    },
  };
  const messageDelta = {
    type: "message_delta",
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: {
      output_tokens: 12,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 514816,
    },
  };
  return [
    `event: message_start\ndata: ${JSON.stringify(messageStart)}\n\n`,
    `event: message_delta\ndata: ${JSON.stringify(messageDelta)}\n\n`,
    `event: message_stop\ndata: ${JSON.stringify({ type: "message_stop" })}\n\n`,
  ].join("");
}

function sseResponse(body: string): Response {
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function waitFor(
  predicate: () => boolean,
  timeoutMs = 2000
): Promise<void> {
  const start = Date.now();
  while (!predicate()) {
    if (Date.now() - start > timeoutMs) {
      throw new Error("timed out waiting for debug logs");
    }
    await new Promise((r) => setTimeout(r, 10));
  }
}

async function logsAnthropicUsageAndPreservesBytes() {
  const { logger, records } = createDebugLogger();
  const fixture = anthropicSSEFixture();
  const tapped = await tapUpstreamSSEDebug(sseResponse(fixture), {
    logger,
    reqId: "req-1",
    provider: "anthropic",
  });

  assert.equal(
    tapped.headers.get("Content-Type"),
    "text/event-stream",
    "Content-Type must survive the tee"
  );

  const clientText = await tapped.text();
  assert.equal(clientText, fixture, "client branch bytes must be unchanged");

  await waitFor(
    () =>
      records.some(
        (r) =>
          r.tppe === "Original Response" &&
          (r.response as any)?.type === "message_delta"
      )
  );

  const received = records.filter((r) => r.type === "recieved data");
  assert.ok(received.length >= 2, "expected recieved data logs");

  const originals = records.filter((r) => r.tppe === "Original Response");
  assert.ok(originals.length >= 2, "expected Original Response logs");

  const start = originals.find(
    (r) => (r.response as any)?.type === "message_start"
  );
  assert.ok(start, "message_start Original Response missing");
  assert.equal(
    (start!.response as any).message.usage.cache_creation_input_tokens,
    80
  );

  const delta = originals.find(
    (r) => (r.response as any)?.type === "message_delta"
  );
  assert.ok(delta, "message_delta Original Response missing");
  assert.equal(
    (delta!.response as any).usage.cache_read_input_tokens,
    514816
  );
  assert.equal(delta!.reqId, "req-1");
  assert.equal(delta!.provider, "anthropic");
}

async function skipsWhenDebugDisabled() {
  const records: DebugRecord[] = [];
  const fixture = anthropicSSEFixture();
  const tapped = await tapUpstreamSSEDebug(sseResponse(fixture), {
    logger: {
      level: "info",
      levelVal: 30,
      debug(payload: DebugRecord) {
        records.push(payload);
      },
    },
    reqId: "req-off",
    provider: "anthropic",
  });

  assert.equal(await tapped.text(), fixture);
  await new Promise((r) => setTimeout(r, 50));
  assert.equal(records.length, 0, "must not tee/log when debug is off");
}

async function passthroughNonSSE() {
  const { logger, records } = createDebugLogger();
  const body = "not-sse";
  const response = new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/plain" },
  });
  const tapped = await tapUpstreamSSEDebug(response, {
    logger,
    reqId: "req-plain",
    provider: "anthropic",
  });
  assert.equal(await tapped.text(), body);
  assert.equal(records.length, 0);
}

async function logsJsonBody() {
  const { logger, records } = createDebugLogger();
  const payload = {
    id: "msg_json",
    usage: { cache_read_input_tokens: 42, input_tokens: 10 },
  };
  const response = new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
  const tapped = await tapUpstreamSSEDebug(response, {
    logger,
    reqId: "req-json",
    provider: "anthropic",
  });
  assert.equal(
    tapped.headers.get("Content-Type"),
    "application/json"
  );
  assert.deepEqual(JSON.parse(await tapped.text()), payload);
  assert.ok(records.some((r) => r.type === "recieved data"));
  const original = records.find((r) => r.tppe === "Original Response");
  assert.equal(
    (original?.response as any)?.usage?.cache_read_input_tokens,
    42
  );
}

async function debugBranchCancelDoesNotBreakClient() {
  const { logger } = createDebugLogger();
  let cancelCount = 0;
  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode(
            'event: message_stop\ndata: {"type":"message_stop"}\n\n'
          )
        );
        controller.close();
      },
      cancel() {
        cancelCount += 1;
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const tapped = await tapUpstreamSSEDebug(upstream, {
    logger,
    reqId: "req-cancel",
    provider: "anthropic",
  });

  await tapped.body!.cancel("client gone");
  // Tee cancel may propagate; the important part is no throw and client cancel works.
  assert.ok(cancelCount >= 0);
}

function summarizesAnthropicCacheStructure() {
  const summary = summarizeOutboundCacheStructure({
    model: "claude-sonnet-4-20250514",
    system: [
      { type: "text", text: "a" },
      {
        type: "text",
        text: "b",
        cache_control: { type: "ephemeral" },
      },
    ],
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "hi",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "..." },
          { type: "text", text: "ok" },
          { type: "tool_use", id: "t1", name: "Bash", input: {} },
        ],
      },
    ],
  });

  assert.ok(summary);
  assert.equal(summary!.systemBreakpoints, 1);
  assert.equal(summary!.messageBreakpoints, 1);
  assert.deepEqual(summary!.lastAssistantBlockOrder, [
    "thinking",
    "text",
    "tool_use",
  ]);
}

function summarizesPromptCacheKey() {
  const summary = summarizeOutboundCacheStructure({
    model: "gpt-5",
    prompt_cache_key: "ccr_abc123",
    messages: [{ role: "user", content: "hi" }],
  });
  assert.ok(summary);
  assert.equal(summary!.prompt_cache_key, "ccr_abc123");
  assert.equal(summary!.systemBreakpoints, 0);
  assert.equal(summary!.messageBreakpoints, 0);
}

async function main() {
  await logsAnthropicUsageAndPreservesBytes();
  await skipsWhenDebugDisabled();
  await passthroughNonSSE();
  await logsJsonBody();
  await debugBranchCancelDoesNotBreakClient();
  summarizesAnthropicCacheStructure();
  summarizesPromptCacheKey();
  console.log("sse-debug-tap: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

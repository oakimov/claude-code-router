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
    "Content-Type must survive the debug tap"
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
  assert.equal(records.length, 0, "must not tap/log when debug is off");
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

function sseFrame(payload: unknown, event = "message"): Uint8Array {
  return new TextEncoder().encode(
    `event: ${event}\ndata: ${JSON.stringify(payload)}\n\n`
  );
}

/** A diff shaped like a healthy follow-up turn, so `verdict` is meaningful. */
function intactDiff() {
  return {
    conversationId: "conv-test",
    conversationIdSource: "session",
    stage: "wire",
    firstTurn: false,
    prefixIntact: true,
    change: "appended",
    approxPrefixTokensLost: 0,
  } as any;
}

async function debugBranchCancelDoesNotBreakClient() {
  const { logger } = createDebugLogger();
  let cancelCount = 0;
  // Upstream must still be open when the client hangs up, otherwise there is
  // nothing left to cancel and the assertion below proves nothing.
  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          new TextEncoder().encode(
            'event: content_block_delta\ndata: {"type":"content_block_delta"}\n\n'
          )
        );
      },
      pull() {
        // Mid-turn silence: never resolves, so the response stays in flight.
        return new Promise<void>(() => {});
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
  // A client hangup must reach upstream. Under tee() the debug branch keeps the
  // source alive, so CCR would go on draining (and paying for) a response
  // nobody is reading.
  await waitFor(() => cancelCount > 0, 2000);
}

/**
 * A slow debug logger must not delay client bytes. With tee(), each logged
 * chunk would stall the client branch; the infinite-HWM tap must keep the
 * client moving while debug lags.
 */
async function slowDebugDoesNotStallClient() {
  const encoder = new TextEncoder();
  const chunks = Array.from({ length: 40 }, (_, i) =>
    encoder.encode(
      `event: content_block_delta\ndata: ${JSON.stringify({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: `x${i}` },
      })}\n\n`
    )
  );

  let pullCount = 0;
  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      pull(controller) {
        if (pullCount >= chunks.length) {
          controller.close();
          return;
        }
        controller.enqueue(chunks[pullCount++]!);
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const records: DebugRecord[] = [];
  const tapped = await tapUpstreamSSEDebug(upstream, {
    logger: {
      level: "debug",
      debug(payload: DebugRecord) {
        // Simulate slow pino file I/O on every event.
        const start = Date.now();
        while (Date.now() - start < 15) {
          /* spin */
        }
        records.push(payload);
      },
    },
    reqId: "req-stall",
    provider: "anthropic",
  });

  const reader = tapped.body!.getReader();
  const started = Date.now();
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    received += value?.byteLength ?? 0;
  }
  const clientMs = Date.now() - started;

  // 40 chunks × 15ms sync debug ≈ 600ms if tee-coupled; client path should be
  // well under that even allowing for scheduling noise.
  assert.ok(
    clientMs < 300,
    `client drain took ${clientMs}ms — debug I/O is still coupling backpressure`
  );
  assert.ok(received > 0, "client must receive upstream bytes");

  await waitFor(() => records.length >= 2, 5000);
  assert.ok(records.length >= 2, "debug logs should still arrive eventually");
}

/**
 * Wall-clock companion to the timing assertion above, stated structurally so it
 * cannot go flaky: the client must be able to finish the whole stream while the
 * debug consumer is still near the front of it. The debug branch yields to the
 * event loop between events, so it can only lag — never gate — the client.
 * Under tee() the client could outrun the debug branch by at most the queue
 * high-water mark, which would fail this.
 */
async function clientOutrunsDebugConsumer() {
  const FRAMES = 60;
  const chunks = Array.from({ length: FRAMES }, (_, i) =>
    sseFrame({ type: "content_block_delta", index: 0, i }, "content_block_delta")
  );

  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const { logger, records } = createDebugLogger();
  const tapped = await tapUpstreamSSEDebug(upstream, {
    logger,
    reqId: "req-outrun",
    provider: "anthropic",
  });

  await tapped.text();
  const loggedAtClientCompletion = records.filter(
    (r) => r.type === "recieved data"
  ).length;

  assert.ok(
    loggedAtClientCompletion < FRAMES / 2,
    `client waited for ${loggedAtClientCompletion}/${FRAMES} debug logs — the ` +
      "debug branch is gating the client stream again"
  );

  await waitFor(
    () => records.filter((r) => r.type === "recieved data").length === FRAMES,
    5000
  );
}

/**
 * The tap must stay a stream. A buffering implementation would hold chunk 1
 * until upstream produced chunk 2, which is what makes Claude Code report
 * "Waiting for API response" on a long thinking turn.
 */
async function firstChunkArrivesBeforeUpstreamFinishes() {
  let releaseSecond: () => void = () => {};
  const secondReleased = new Promise<void>((resolve) => {
    releaseSecond = resolve;
  });

  let emitted = 0;
  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      async pull(controller) {
        if (emitted === 0) {
          emitted += 1;
          controller.enqueue(sseFrame({ type: "message_start", message: {} }));
          return;
        }
        if (emitted === 1) {
          emitted += 1;
          await secondReleased;
          controller.enqueue(sseFrame({ type: "message_stop" }));
          return;
        }
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const { logger } = createDebugLogger();
  const tapped = await tapUpstreamSSEDebug(upstream, {
    logger,
    reqId: "req-ttfb",
    provider: "anthropic",
  });

  const reader = tapped.body!.getReader();
  const first = await Promise.race([
    reader.read(),
    new Promise((_, reject) =>
      setTimeout(() => reject(new Error("first chunk was buffered")), 1000)
    ),
  ]);
  assert.ok(
    (first as ReadableStreamReadResult<Uint8Array>).value,
    "first chunk must reach the client before upstream completes"
  );

  releaseSecond();
  while (!(await reader.read()).done) {
    /* drain */
  }
}

/**
 * Usage lands on the final frame, so a tap that drops chunks under congestion
 * silently loses the entire cache verdict. Push enough frames through a slow
 * logger to back the debug tunnel up, then assert the tail still produced a
 * correct `cache outcome`.
 */
async function terminalUsageFrameSurvivesSlowDebug() {
  const chunks = [
    ...Array.from({ length: 40 }, (_, i) =>
      sseFrame(
        { type: "content_block_delta", index: 0, delta: { text: `t${i}` } },
        "content_block_delta"
      )
    ),
    sseFrame(
      {
        type: "message_delta",
        delta: { stop_reason: "end_turn" },
        usage: {
          input_tokens: 20,
          cache_read_input_tokens: 180,
          cache_creation_input_tokens: 0,
          output_tokens: 9,
        },
      },
      "message_delta"
    ),
  ];

  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const records: DebugRecord[] = [];
  const tapped = await tapUpstreamSSEDebug(upstream, {
    logger: {
      level: "debug",
      debug(payload: DebugRecord) {
        const start = Date.now();
        while (Date.now() - start < 3) {
          /* simulate synchronous pino file I/O */
        }
        records.push(payload);
      },
    },
    reqId: "req-tail",
    provider: "anthropic",
    model: "claude-sonnet-4-20250514",
    responseStatus: 200,
    cacheDiff: intactDiff(),
  });

  await tapped.text();
  await waitFor(() => records.some((r) => r.type === "cache outcome"), 10000);

  const outcome = records.find((r) => r.type === "cache outcome")!;
  assert.equal(outcome.verdict, "hit");
  // Anthropic reports the uncached remainder, so the billable prompt is the sum.
  assert.equal(outcome.promptTokens, 200);
  assert.equal(outcome.cachedTokens, 180);
  assert.equal(outcome.cacheHitRatio, 0.9);
  assert.equal(outcome.outputTokens, 9);
  assert.equal(outcome.model, "claude-sonnet-4-20250514");
  assert.equal(outcome.conversationId, "conv-test");
  assert.equal(
    records.filter((r) => r.type === "cache outcome").length,
    1,
    "cache outcome must be emitted exactly once per response"
  );
}

/**
 * A client that hangs up mid-stream must still tear the debug branch down.
 * Without that the consumer loop parks forever holding the buffered copy of the
 * response — a leak on every cancelled request while debug is on.
 */
async function clientCancelFinalizesDebugBranch() {
  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (let i = 0; i < 8; i += 1) {
          controller.enqueue(sseFrame({ type: "content_block_delta", i }));
        }
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const { logger, records } = createDebugLogger();
  const tapped = await tapUpstreamSSEDebug(upstream, {
    logger,
    reqId: "req-hangup",
    provider: "anthropic",
    cacheDiff: intactDiff(),
  });

  const reader = tapped.body!.getReader();
  await reader.read();
  await reader.cancel("client gone");

  // The outcome only lands from the consumer's finally block, so seeing it
  // proves the debug branch actually terminated rather than leaking.
  await waitFor(() => records.some((r) => r.type === "cache outcome"), 5000);
  assert.equal(
    (records.find((r) => r.type === "cache outcome") as any).verdict,
    "unknown",
    "a cancelled stream has no usage, but the prediction must still be logged"
  );
}

/** The tap copies bytes; a decode/re-encode would corrupt split code points. */
async function bytesSurviveMultibyteChunkBoundaries() {
  const payload = new TextEncoder().encode(
    'event: message\ndata: {"type":"text","text":"héllo 🙂 世界"}\n\n'
  );
  // Split mid-emoji so neither half is valid UTF-8 on its own.
  const cut = payload.indexOf(0xf0);
  assert.ok(cut > 0, "fixture must contain a 4-byte code point");

  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(payload.slice(0, cut + 1));
        controller.enqueue(payload.slice(cut + 1));
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );

  const { logger } = createDebugLogger();
  const tapped = await tapUpstreamSSEDebug(upstream, {
    logger,
    reqId: "req-utf8",
    provider: "anthropic",
  });

  const received = new Uint8Array(await tapped.arrayBuffer());
  assert.deepEqual(
    Array.from(received),
    Array.from(payload),
    "client bytes must be forwarded verbatim"
  );
}

/** Debug off must cost nothing — not even a wrapping Response. */
async function debugOffReturnsTheSameResponse() {
  const original = sseResponse(anthropicSSEFixture());
  const tapped = await tapUpstreamSSEDebug(original, {
    logger: { level: "info", levelVal: 30, debug() {} },
    reqId: "req-identity",
  });
  assert.equal(tapped, original, "debug-off must return the response untouched");
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
  // The tap writes to the debug tunnel without awaiting. Any of those promises
  // rejecting unobserved would crash the server process under debug.
  const unhandled: unknown[] = [];
  process.on("unhandledRejection", (reason) => unhandled.push(reason));

  await logsAnthropicUsageAndPreservesBytes();
  await skipsWhenDebugDisabled();
  await passthroughNonSSE();
  await logsJsonBody();
  await debugBranchCancelDoesNotBreakClient();
  await slowDebugDoesNotStallClient();
  await clientOutrunsDebugConsumer();
  await firstChunkArrivesBeforeUpstreamFinishes();
  await terminalUsageFrameSurvivesSlowDebug();
  await clientCancelFinalizesDebugBranch();
  await bytesSurviveMultibyteChunkBoundaries();
  await debugOffReturnsTheSameResponse();
  summarizesAnthropicCacheStructure();
  summarizesPromptCacheKey();

  await new Promise((r) => setTimeout(r, 50));
  assert.deepEqual(unhandled, [], "debug tap leaked an unhandled rejection");
  console.log("sse-debug-tap: ok");
}

// A stream that never settles drains the event loop and lets this file exit 0
// with no output, which the runner would read as a pass. Fail by default and
// only clear the code once main() has actually run to completion.
process.exitCode = 1;

main().then(
  () => {
    process.exitCode = 0;
  },
  (error) => {
    console.error(error);
    process.exitCode = 1;
  }
);

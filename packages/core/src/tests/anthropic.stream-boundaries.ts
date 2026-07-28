/**
 * Anthropic response invariants that every provider depends on — block
 * boundaries while streaming, and block order in the non-streaming reply.
 *
 * Streaming:
 *
 * 1. Content blocks are closed before a different block type opens. Reasoning
 *    providers (DeepSeek, Qwen, GLM, Gemini) interleave thinking and text, so a
 *    second thinking run must open a fresh thinking block instead of emitting
 *    thinking_delta against the text block index.
 * 2. stop_reason only becomes tool_use when a tool_use block was actually
 *    streamed — an empty `tool_calls: []` delta must not flip it, or Claude Code
 *    waits forever for tool results that cannot arrive.
 *
 * Non-streaming:
 *
 * 3. Blocks come back as thinking → server tool use → text → tool_use, matching
 *    the streaming order and `buildAnthropicBody`. A trailing thinking block
 *    means a client replaying the turn has one that does not lead the message.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

type SSEEvent = { event: string; data: any };

function buildStreamResponse(chunks: Array<Record<string, unknown>>): Response {
  const payload = [
    ...chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`),
    "data: [DONE]\n\n",
  ].join("");

  return new Response(payload, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function collectEvents(chunks: Array<Record<string, unknown>>): Promise<SSEEvent[]> {
  const transformer = new AnthropicTransformer();
  const noop = () => {};
  (transformer as any).logger = {
    debug: noop,
    info: noop,
    warn: noop,
    error: noop,
  };
  const out = await transformer.transformResponseIn(
    buildStreamResponse(chunks),
    { req: { id: "test-req" } } as any
  );
  const text = await out.text();

  const events: SSEEvent[] = [];
  for (const block of text.split("\n\n")) {
    const eventLine = block
      .split("\n")
      .find((line) => line.startsWith("event: "));
    const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
    if (!eventLine || !dataLine) continue;
    const raw = dataLine.slice(6);
    if (raw === "[DONE]") continue;
    events.push({ event: eventLine.slice(7), data: JSON.parse(raw) });
  }
  return events;
}

function chunk(delta: Record<string, unknown>, finish: string | null = null) {
  return {
    id: "chatcmpl_test",
    model: "test-model",
    choices: [{ index: 0, delta, finish_reason: finish }],
  };
}

/** thinking → text → thinking must not stream thinking_delta into a text block. */
async function testThinkingAfterTextOpensNewThinkingBlock() {
  const events = await collectEvents([
    chunk({ thinking: { content: "first thought" } }),
    chunk({ content: "partial answer" }),
    chunk({ thinking: { content: "second thought" } }),
    chunk({ content: "final answer" }),
    chunk({}, "stop"),
  ]);

  // Track the block type per index from content_block_start.
  const blockTypes = new Map<number, string>();
  for (const e of events) {
    if (e.event === "content_block_start") {
      blockTypes.set(e.data.index, e.data.content_block.type);
    }
  }

  for (const e of events) {
    if (e.event !== "content_block_delta") continue;
    const type = blockTypes.get(e.data.index);
    if (e.data.delta.type === "thinking_delta") {
      assert.equal(
        type,
        "thinking",
        `thinking_delta landed on a ${type} block (index ${e.data.index})`
      );
    }
    if (e.data.delta.type === "text_delta") {
      assert.equal(
        type,
        "text",
        `text_delta landed on a ${type} block (index ${e.data.index})`
      );
    }
  }

  // Two thinking runs → two thinking blocks, each opened and closed cleanly.
  const thinkingBlocks = [...blockTypes.entries()].filter(
    ([, type]) => type === "thinking"
  );
  assert.equal(thinkingBlocks.length, 2, "expected a second thinking block");

  const starts = events.filter((e) => e.event === "content_block_start").length;
  const stops = events.filter((e) => e.event === "content_block_stop").length;
  assert.equal(starts, stops, "every content block must be closed exactly once");
}

/**
 * Annotations must not strand the block they interrupt. The annotation loop
 * opens and closes its own blocks, so clearing currentContentBlockIndex there
 * orphans a tool_use that is still streaming — every later close path is gated
 * on that index, so the block would never be closed at all.
 *
 * The close before annotations is deliberately limited to text blocks: a
 * tool_use may still be streaming argument JSON, and closing it early would
 * emit input_json_delta after content_block_stop. So the tool_use has to stay
 * open across the annotation and be closed afterwards, not before.
 */
async function testAnnotationsDoNotStrandOpenBlocks() {
  const cases: Array<{ name: string; chunks: Array<Record<string, unknown>> }> = [
    {
      name: "tool_use → annotations → text",
      chunks: [
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "Bash", arguments: '{"command":"ls"}' },
            },
          ],
        }),
        chunk({
          annotations: [
            { url_citation: { url: "https://example.test", title: "Example" } },
          ],
        }),
        chunk({ content: "after citations" }),
        chunk({}, "stop"),
      ],
    },
    {
      name: "annotations arriving mid tool_use arguments",
      chunks: [
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "Bash", arguments: '{"command":' },
            },
          ],
        }),
        chunk({
          annotations: [
            { url_citation: { url: "https://example.test", title: "Example" } },
          ],
        }),
        chunk({ tool_calls: [{ index: 0, function: { arguments: '"ls"}' } }] }),
        chunk({}, "stop"),
      ],
    },
  ];

  for (const { name, chunks: streamChunks } of cases) {
    const events = await collectEvents(streamChunks);

    const opened = new Set<number>();
    const closed = new Set<number>();
    for (const e of events) {
      if (e.event === "content_block_start") opened.add(e.data.index);
      if (e.event === "content_block_stop") closed.add(e.data.index);
    }

    const leaked = [...opened].filter((i) => !closed.has(i));
    assert.deepEqual(
      leaked,
      [],
      `${name}: block(s) ${leaked.join(",")} opened but never closed`
    );

    // A delta after its block closed is a protocol violation — the client has
    // already committed the block and cannot accept more content for it.
    const stopped = new Set<number>();
    for (const e of events) {
      if (e.event === "content_block_stop") stopped.add(e.data.index);
      if (e.event === "content_block_delta") {
        assert.equal(
          stopped.has(e.data.index),
          false,
          `${name}: ${e.data.delta.type} arrived after content_block_stop on index ${e.data.index}`
        );
      }
    }

    for (const index of closed) {
      const stops = events.filter(
        (e) => e.event === "content_block_stop" && e.data.index === index
      ).length;
      assert.equal(stops, 1, `${name}: index ${index} closed ${stops} times`);
    }
  }
}

/**
 * text → tool_use → text must open a fresh text block. If hasTextContentStarted
 * stays sticky after tool_use opens, text_delta lands on the tool_use index and
 * Claude Code drops the turn with "Content block is not a text block".
 */
async function testTextAfterToolUseOpensNewTextBlock() {
  const events = await collectEvents([
    chunk({ content: "hello" }),
    chunk({
      tool_calls: [
        {
          index: 0,
          id: "call_1",
          type: "function",
          function: { name: "Bash", arguments: '{"command":"ls"}' },
        },
      ],
    }),
    chunk({ content: "after tools" }),
    chunk({}, "stop"),
  ]);

  const blockTypes = new Map<number, string>();
  for (const e of events) {
    if (e.event === "content_block_start") {
      blockTypes.set(e.data.index, e.data.content_block.type);
    }
  }

  for (const e of events) {
    if (e.event !== "content_block_delta") continue;
    const type = blockTypes.get(e.data.index);
    if (e.data.delta.type === "text_delta") {
      assert.equal(
        type,
        "text",
        `text_delta landed on a ${type} block (index ${e.data.index})`
      );
    }
    if (e.data.delta.type === "input_json_delta") {
      assert.equal(
        type,
        "tool_use",
        `input_json_delta landed on a ${type} block (index ${e.data.index})`
      );
    }
  }

  const textBlocks = [...blockTypes.entries()].filter(([, type]) => type === "text");
  assert.equal(
    textBlocks.length,
    2,
    "expected a second text block after tool_use"
  );

  const starts = events.filter((e) => e.event === "content_block_start").length;
  const stops = events.filter((e) => e.event === "content_block_stop").length;
  assert.equal(starts, stops, "every content block must be closed exactly once");
}

/** An empty tool_calls delta must not claim tool_use. */
async function testEmptyToolCallsDeltaKeepsEndTurn() {
  const events = await collectEvents([
    chunk({ content: "no tools needed" }),
    chunk({ tool_calls: [] }),
    chunk({}, "stop"),
  ]);

  const delta = events.find(
    (e) => e.event === "message_delta" && e.data.delta?.stop_reason
  );
  assert.ok(delta, "expected a message_delta carrying stop_reason");
  assert.equal(delta.data.delta.stop_reason, "end_turn");
  assert.equal(
    events.some(
      (e) =>
        e.event === "content_block_start" &&
        e.data.content_block?.type === "tool_use"
    ),
    false,
    "no tool_use block should exist"
  );
}

/** A real tool call still upgrades finish_reason "stop" to tool_use. */
async function testStreamedToolUseUpgradesStopReason() {
  const events = await collectEvents([
    chunk({
      tool_calls: [
        {
          index: 0,
          id: "call_1",
          type: "function",
          function: { name: "Bash", arguments: '{"command":"ls"}' },
        },
      ],
    }),
    chunk({}, "stop"),
  ]);

  assert.equal(
    events.some(
      (e) =>
        e.event === "content_block_start" &&
        e.data.content_block?.type === "tool_use"
    ),
    true,
    "expected a tool_use block"
  );

  const delta = events.find(
    (e) => e.event === "message_delta" && e.data.delta?.stop_reason
  );
  assert.ok(delta);
  assert.equal(delta.data.delta.stop_reason, "tool_use");
}

/** Non-streaming replies must lead with thinking, not trail with it. */
async function testNonStreamingBlockOrder() {
  const transformer = new AnthropicTransformer();
  const noop = () => {};
  (transformer as any).logger = { debug: noop, info: noop, warn: noop, error: noop };

  const upstream = new Response(
    JSON.stringify({
      id: "chatcmpl_nonstream",
      model: "test-model",
      choices: [
        {
          index: 0,
          finish_reason: "tool_calls",
          message: {
            role: "assistant",
            content: "Here is what I found.",
            thinking: { content: "weighing options", signature: "sig_ns" },
            annotations: [
              { url_citation: { url: "https://example.test", title: "Example" } },
            ],
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "Bash", arguments: '{"command":"ls"}' },
              },
            ],
          },
        },
      ],
    }),
    { headers: { "Content-Type": "application/json" } }
  );

  const out = await transformer.transformResponseIn(upstream, {
    req: { id: "test-req" },
  } as any);
  const body: any = await out.json();

  assert.deepEqual(
    body.content.map((b: any) => b.type),
    ["thinking", "server_tool_use", "web_search_tool_result", "text", "tool_use"]
  );
  assert.equal(body.content[0].thinking, "weighing options");
  assert.equal(body.content[0].signature, "sig_ns");
  assert.equal(body.stop_reason, "tool_use");

  // Without thinking the remaining order is unchanged.
  const plain = new Response(
    JSON.stringify({
      id: "chatcmpl_plain",
      model: "test-model",
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: { role: "assistant", content: "just text" },
        },
      ],
    }),
    { headers: { "Content-Type": "application/json" } }
  );
  const plainBody: any = await (
    await transformer.transformResponseIn(plain, { req: { id: "r2" } } as any)
  ).json();
  assert.deepEqual(
    plainBody.content.map((b: any) => b.type),
    ["text"]
  );
}

async function main() {
  await testThinkingAfterTextOpensNewThinkingBlock();
  await testTextAfterToolUseOpensNewTextBlock();
  await testAnnotationsDoNotStrandOpenBlocks();
  await testEmptyToolCallsDeltaKeepsEndTurn();
  await testStreamedToolUseUpgradesStopReason();
  await testNonStreamingBlockOrder();
  console.log("anthropic.stream-boundaries: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

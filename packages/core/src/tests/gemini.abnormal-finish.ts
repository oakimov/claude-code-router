/**
 * Abnormal Gemini/Antigravity finish reasons must not look like a finished turn.
 *
 * Upstream ends a turn with MALFORMED_FUNCTION_CALL, MAX_TOKENS, SAFETY, OTHER…
 * Previously every one of them was lowercased into a value Anthropic did not
 * recognise and fell back to `end_turn`, so:
 *   - a MALFORMED_FUNCTION_CALL turn (thinking only, no reply) reached Claude
 *     Code as a successful, silent turn — the user had to prompt again;
 *   - a MAX_TOKENS truncation was indistinguishable from a complete answer.
 */
import assert from "node:assert/strict";
import { buildRequestBody, transformResponseOut } from "../utils/gemini.util";
import { UPSTREAM_STOP_NOTICE } from "../utils/google.util";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

type Delta = {
  content?: string | null;
  thinking?: any;
  tools?: string[];
  finish?: string | null;
};

async function streamDeltas(chunks: any[]): Promise<{
  deltas: Delta[];
  raw: string;
}> {
  const sse = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("");
  const out = await transformResponseOut(
    new Response(sse, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    }),
    "antigravity"
  );
  const raw = await out.text();
  const deltas = raw
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => JSON.parse(l.slice(6)))
    .map((d) => ({
      content: d.choices?.[0]?.delta?.content,
      thinking: d.choices?.[0]?.delta?.thinking,
      tools: d.choices?.[0]?.delta?.tool_calls?.map(
        (t: any) => t.function?.name
      ),
      finish: d.choices?.[0]?.finish_reason,
    }));
  return { deltas, raw };
}

async function anthropicEvents(unified: string): Promise<any[]> {
  const transformer = new AnthropicTransformer();
  const noop = () => {};
  (transformer as any).logger = {
    debug: noop,
    info: noop,
    warn: noop,
    error: noop,
  };
  const out = await transformer.transformResponseIn(
    new Response(unified + "data: [DONE]\n\n", {
      headers: { "Content-Type": "text/event-stream" },
    }),
    { req: { id: "abnormal-finish" } } as any
  );
  return (await out.text())
    .split("\n\n")
    .map((block) => {
      const line = block.split("\n").find((l) => l.startsWith("data: "));
      if (!line || line.slice(6) === "[DONE]") return null;
      return JSON.parse(line.slice(6));
    })
    .filter(Boolean) as any[];
}

/**
 * Real capture: the model emitted an empty functionCall, upstream refused to
 * parse it and returned thinking + signature + empty text.
 */
async function testMalformedFunctionCallIsReported() {
  const { deltas, raw } = await streamDeltas([
    {
      modelVersion: "gemini-3.6-flash-tiered",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ thought: true, text: "**Analyzing File Structure**" }],
          },
        },
      ],
    },
    {
      modelVersion: "gemini-3.6-flash-tiered",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ thoughtSignature: "sig_abc", text: "" }],
          },
          finishReason: "MALFORMED_FUNCTION_CALL",
          finishMessage:
            "Malformed function call: Failed to parse function call: Function call is empty - no input to parse.",
        },
      ],
    },
  ]);

  const contents = deltas
    .map((d) => d.content)
    .filter((c): c is string => typeof c === "string" && c.length > 0);
  assert.equal(
    contents.some((c) => c.includes("(no content)")),
    false,
    "an upstream failure must not be dressed up as an empty turn"
  );
  assert.equal(contents.length, 1, `expected one visible chunk, got ${contents.length}`);
  assert.ok(
    contents[0].startsWith(UPSTREAM_STOP_NOTICE),
    `notice must carry the marker, got ${JSON.stringify(contents[0])}`
  );
  assert.ok(
    contents[0].includes("MALFORMED_FUNCTION_CALL"),
    "notice must name the upstream reason"
  );
  assert.ok(
    contents[0].includes("Function call is empty"),
    "notice must carry the upstream finishMessage"
  );

  // Anthropic side: exactly one text block, after the thinking block.
  const events = await anthropicEvents(raw);
  const blocks = events
    .filter((e) => e.type === "content_block_start")
    .map((e) => e.content_block?.type);
  assert.deepEqual(blocks, ["thinking", "text"]);
}

/** Truncation must reach Anthropic as max_tokens, not end_turn. */
async function testMaxTokensMapsToLength() {
  const { deltas, raw } = await streamDeltas([
    {
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: { role: "model", parts: [{ text: "half an ans" }] },
          finishReason: "MAX_TOKENS",
        },
      ],
    },
  ]);

  const finishes = deltas.map((d) => d.finish).filter((f) => f != null);
  assert.deepEqual(finishes, ["length"]);

  const events = await anthropicEvents(raw);
  const messageDelta = events.find((e) => e.type === "message_delta");
  assert.equal(messageDelta?.delta?.stop_reason, "max_tokens");

  // Real text is untouched: no notice is added when there is something to show.
  const contents = deltas
    .map((d) => d.content)
    .filter((c): c is string => typeof c === "string" && c.length > 0);
  assert.deepEqual(contents, ["half an ans"]);
}

/** Safety stops map to content_filter rather than a clean end_turn. */
async function testSafetyMapsToContentFilter() {
  const { deltas } = await streamDeltas([
    {
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: { role: "model", parts: [{ text: "" }] },
          finishReason: "SAFETY",
        },
      ],
    },
  ]);
  const finishes = deltas.map((d) => d.finish).filter((f) => f != null);
  assert.deepEqual(finishes, ["content_filter"]);
  const contents = deltas
    .map((d) => d.content)
    .filter((c): c is string => typeof c === "string" && c.length > 0);
  assert.equal(contents.length, 1);
  assert.ok(contents[0].includes("SAFETY"));
}

/**
 * A tool turn that ends abnormally must still not grow text after tool_use —
 * the notice is suppressed once tools have been emitted.
 */
async function testNoNoticeAfterToolsWereEmitted() {
  const { deltas, raw } = await streamDeltas([
    {
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                thoughtSignature: "sig_tool",
                functionCall: { id: "c1", name: "Bash", args: { command: "ls" } },
              },
            ],
          },
        },
      ],
    },
    {
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: { role: "model", parts: [{ text: "" }] },
          finishReason: "MALFORMED_FUNCTION_CALL",
          finishMessage: "Malformed function call",
        },
      ],
    },
  ]);

  const contents = deltas
    .map((d) => d.content)
    .filter((c): c is string => typeof c === "string" && c.length > 0);
  assert.deepEqual(contents, [], "no visible text may follow tool_use");

  const events = await anthropicEvents(raw);
  const blocks = events
    .filter((e) => e.type === "content_block_start")
    .map((e) => e.content_block?.type);
  assert.deepEqual(blocks, ["tool_use"]);
  const messageDelta = events.find((e) => e.type === "message_delta");
  assert.equal(messageDelta?.delta?.stop_reason, "tool_use");
}

/** Non-streaming replies carry the same notice. */
async function testNonStreamingCarriesNotice() {
  const body = {
    responseId: "r1",
    modelVersion: "gemini-3.6-flash",
    candidates: [
      {
        content: { role: "model", parts: [{ text: "" }] },
        finishReason: "MALFORMED_FUNCTION_CALL",
        finishMessage: "Function call is empty",
      },
    ],
  };
  const out = await transformResponseOut(
    new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
    "antigravity"
  );
  const json: any = await out.json();
  assert.equal(json.choices[0].finish_reason, "malformed_function_call");
  assert.ok(json.choices[0].message.content.startsWith(UPSTREAM_STOP_NOTICE));
}

/** The notice is CCR's own text: strip it from replayed model turns. */
async function testNoticeStrippedFromReplayedModelTurns() {
  const notice = `${UPSTREAM_STOP_NOTICE} (MALFORMED_FUNCTION_CALL): boom`;
  const body = buildRequestBody({
    model: "gemini-3-flash",
    messages: [
      { role: "user", content: notice },
      { role: "assistant", content: notice },
      { role: "user", content: "carry on" },
    ],
  });

  const modelMsg = body.contents.find((c: any) => c.role === "model");
  assert.equal(
    modelMsg === undefined ||
      !modelMsg.parts.some((p: any) => p.text?.startsWith(UPSTREAM_STOP_NOTICE)),
    true,
    "model turns must not replay CCR's notice"
  );

  const userMsg = body.contents.find(
    (c: any) =>
      c.role === "user" &&
      (c.parts || []).some((p: any) => p.text?.startsWith(UPSTREAM_STOP_NOTICE))
  );
  assert.ok(userMsg, "user text is replayed verbatim, notice-looking or not");
}

async function main() {
  await testMalformedFunctionCallIsReported();
  await testMaxTokensMapsToLength();
  await testSafetyMapsToContentFilter();
  await testNoNoticeAfterToolsWereEmitted();
  await testNonStreamingCarriesNotice();
  await testNoticeStrippedFromReplayedModelTurns();
  console.log("gemini.abnormal-finish: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

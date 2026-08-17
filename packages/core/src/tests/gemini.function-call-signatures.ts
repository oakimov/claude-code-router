/**
 * Thought signatures on functionCall parts — Gemini/Antigravity only.
 *
 * Anthropic / Claude Code / claude-auth / enhancetool must NOT grow new
 * thought_signature fields on tool_use. When Claude Code replays tools
 * without a per-tool signature, buildRequestBody stamps
 * skip_thought_signature_validator on the first functionCall.
 */
import assert from "node:assert/strict";
import {
  buildRequestBody,
  SKIP_THOUGHT_SIGNATURE,
  transformResponseOut,
} from "../utils/gemini.util";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { EnhanceToolTransformer } from "../transformer/enhancetool.transformer";
import type { UnifiedChatRequest } from "../types/llm";

function assertFunctionCallHasSignature(body: any, expected: string) {
  const modelMsg = body.contents.find((c: any) => c.role === "model");
  assert.ok(modelMsg, "expected model content");
  const fcPart = modelMsg.parts.find((p: any) => p.functionCall);
  assert.ok(fcPart, "expected functionCall part");
  assert.equal(fcPart.thoughtSignature, expected);
}

async function testToolTurnNoPlaceholderAndToolCallsFinish() {
  const geminiChunks = [
    {
      responseId: "resp_tool_turn",
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                thoughtSignature: "sig_tool",
                functionCall: {
                  id: "call_1",
                  name: "Read",
                  args: { path: "/tmp/x" },
                },
              },
            ],
          },
          finishReason: "STOP",
        },
      ],
    },
  ];

  const sse = geminiChunks
    .map((c) => `data: ${JSON.stringify(c)}\n\n`)
    .join("");
  const mockResp = new Response(sse, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
  const out = await transformResponseOut(mockResp, "antigravity");
  const text = await out.text();
  const deltas = text
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => JSON.parse(l.slice(6)));

  const contents = deltas
    .map((d) => d.choices?.[0]?.delta?.content)
    .filter((c) => c != null);
  assert.equal(
    contents.some((c) => String(c).includes("no content")),
    false,
    `should not emit (no content) text on tool turns, got ${JSON.stringify(contents)}`
  );

  const thinking = deltas.filter((d) => d.choices?.[0]?.delta?.thinking);
  assert.equal(
    thinking.length,
    0,
    "tool-only signature must not open empty thinking"
  );

  const toolDelta = deltas.find((d) => d.choices?.[0]?.delta?.tool_calls);
  assert.ok(toolDelta);
  assert.equal(toolDelta.choices[0].finish_reason, "tool_calls");
}

/**
 * Antigravity splits a tool turn across two chunks: functionCall + sibling
 * thoughtSignature first, then a bare `text: ""` + STOP trailer. The trailer
 * must not invent a "(no content)" placeholder — that emits Anthropic text
 * after tool_use, which is illegal for clients and pollutes turn history.
 */
async function testCrossChunkToolThenStopTrailer() {
  const geminiChunks = [
    {
      responseId: "resp_cross_chunk",
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                thoughtSignature: "sig_tool",
                functionCall: {
                  id: "call_1",
                  name: "Bash",
                  args: { command: "ls" },
                },
              },
            ],
          },
          // No finishReason: the STOP arrives on the next chunk.
        },
      ],
    },
    {
      responseId: "resp_cross_chunk",
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: { role: "model", parts: [{ text: "" }] },
          finishReason: "STOP",
        },
      ],
    },
  ];

  const sse = geminiChunks
    .map((c) => `data: ${JSON.stringify(c)}\n\n`)
    .join("");
  const out = await transformResponseOut(
    new Response(sse, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    }),
    "antigravity"
  );
  const unified = await out.text();
  const deltas = unified
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => JSON.parse(l.slice(6)));

  const contents = deltas
    .map((d) => d.choices?.[0]?.delta?.content)
    .filter((c) => c != null);
  assert.equal(
    contents.some((c) => String(c).includes("no content")),
    false,
    `STOP trailer must not invent a placeholder, got ${JSON.stringify(contents)}`
  );

  const toolDelta = deltas.find((d) => d.choices?.[0]?.delta?.tool_calls);
  assert.ok(toolDelta, "expected a tool_calls delta");
  const finishReasons = deltas
    .map((d) => d.choices?.[0]?.finish_reason)
    .filter((r) => r != null);
  assert.deepEqual(
    finishReasons,
    ["tool_calls"],
    "the STOP trailer must finish as tool_calls once tools were emitted"
  );

  // Anthropic side: no text block may follow tool_use, and the turn stops as
  // tool_use so Claude Code's agent loop keeps running the tools.
  const transformer = new AnthropicTransformer();
  const noop = () => {};
  (transformer as any).logger = {
    debug: noop,
    info: noop,
    warn: noop,
    error: noop,
  };
  const anthropicResp = await transformer.transformResponseIn(
    new Response(unified + "data: [DONE]\n\n", {
      headers: { "Content-Type": "text/event-stream" },
    }),
    { req: { id: "test-cross-chunk" } } as any
  );
  const anthropicText = await anthropicResp.text();
  const events = anthropicText
    .split("\n\n")
    .map((block) => {
      const dataLine = block.split("\n").find((l) => l.startsWith("data: "));
      if (!dataLine || dataLine.slice(6) === "[DONE]") return null;
      return JSON.parse(dataLine.slice(6));
    })
    .filter(Boolean) as any[];

  const blockTypes = events
    .filter((e) => e.type === "content_block_start")
    .map((e) => e.content_block?.type);
  assert.deepEqual(
    blockTypes,
    ["tool_use"],
    `expected only a tool_use block, got ${JSON.stringify(blockTypes)}`
  );

  const messageDelta = events.find((e) => e.type === "message_delta");
  assert.equal(messageDelta?.delta?.stop_reason, "tool_use");
}

async function testBuildRequestBodyStripsNoContentPlaceholder() {
  const body = buildRequestBody({
    model: "gemini-3-flash",
    messages: [
      { role: "user", content: "go" },
      {
        role: "assistant",
        content: "(no content)",
        tool_calls: [
          {
            id: "c1",
            type: "function",
            function: { name: "Read", arguments: "{}" },
          },
        ],
      },
      { role: "tool", content: "ok", tool_call_id: "c1" },
    ],
  });
  const model = body.contents.find((c: any) => c.role === "model");
  assert.ok(model);
  assert.equal(
    model.parts.some((p: any) => p.text === "(no content)"),
    false
  );
  assert.ok(model.parts.some((p: any) => p.functionCall));
}

async function testSkipFallbackOnMissingToolSignature() {
  const req: UnifiedChatRequest = {
    model: "gemini-3.6-flash-low",
    messages: [
      { role: "user", content: "List files" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_bash",
            type: "function",
            function: {
              name: "Bash",
              arguments: '{"command":"ls -la"}',
            },
          },
        ],
      },
      { role: "tool", content: "ok", tool_call_id: "call_bash" },
    ],
  };

  assertFunctionCallHasSignature(buildRequestBody(req), SKIP_THOUGHT_SIGNATURE);
}

async function testPerToolSignaturePreferredWhenPresentOnUnified() {
  const req: UnifiedChatRequest = {
    model: "gemini-3-flash",
    messages: [
      { role: "user", content: "Search" },
      {
        role: "assistant",
        content: null,
        thinking: { content: "plan", signature: "sig_message" },
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "web_search", arguments: '{"q":"x"}' },
            thought_signature: "sig_per_tool",
          },
        ],
      },
      { role: "tool", content: "hits", tool_call_id: "call_1" },
    ],
  };
  assertFunctionCallHasSignature(buildRequestBody(req), "sig_per_tool");
}

async function testAntigravitySiblingSignatureOnStream() {
  const geminiChunks = [
    {
      responseId: "resp_ag",
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [{ text: "I'll list the directory." }],
          },
        },
      ],
    },
    {
      responseId: "resp_ag",
      modelVersion: "gemini-3.6-flash",
      candidates: [
        {
          content: {
            role: "model",
            parts: [
              {
                thoughtSignature: "sig_ag_tool_sibling",
                functionCall: {
                  id: "G3UHpzGQ",
                  name: "Bash",
                  args: { command: "ls -la" },
                },
              },
            ],
          },
          finishReason: "STOP",
        },
      ],
    },
  ];

  const sse = geminiChunks
    .map((c) => `data: ${JSON.stringify(c)}\n\n`)
    .join("");
  const mockResp = new Response(sse, {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
  const out = await transformResponseOut(mockResp, "antigravity");
  const text = await out.text();
  const deltas = text
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => JSON.parse(l.slice(6)));

  const toolDelta = deltas.find((d) => d.choices?.[0]?.delta?.tool_calls);
  assert.ok(toolDelta, "expected tool_calls delta");
  assert.equal(
    toolDelta.choices[0].delta.tool_calls[0].thought_signature,
    "sig_ag_tool_sibling"
  );

  const thinkingAfterContent = deltas.some(
    (d, i) =>
      i > 0 &&
      d.choices?.[0]?.delta?.thinking &&
      deltas
        .slice(0, i)
        .some((p) => typeof p.choices?.[0]?.delta?.content === "string")
  );
  assert.equal(thinkingAfterContent, false);
}

/** Claude Code / Anthropic wire format must stay free of thought_signature. */
async function testAnthropicDoesNotRoundTripToolSignature() {
  const transformer = new AnthropicTransformer();

  // Outbound to Claude Code (Unified → Anthropic JSON)
  const anthropicOut = AnthropicTransformer.buildAnthropicBody(
    {
      model: "claude-opus-4",
      max_tokens: 1024,
      messages: [
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "G3UHpzGQ",
              type: "function",
              function: {
                name: "Bash",
                arguments: '{"command":"ls -la"}',
              },
              thought_signature: "sig_must_not_leak",
            },
          ],
        },
      ],
    } as UnifiedChatRequest,
    undefined
  );
  const toolUse = anthropicOut.messages[0].content.find(
    (b: any) => b.type === "tool_use"
  );
  assert.ok(toolUse);
  assert.equal(toolUse.thought_signature, undefined);
  assert.equal(toolUse.thoughtSignature, undefined);

  // Inbound from Claude Code (Anthropic → Unified): ignore if somehow present
  const unified = await transformer.transformRequestOut({
    model: "claude-opus-4",
    max_tokens: 1024,
    messages: [
      { role: "user", content: "List files" },
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "G3UHpzGQ",
            name: "Bash",
            input: { command: "ls -la" },
            thought_signature: "sig_client_should_be_ignored",
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "G3UHpzGQ",
            content: "total 0",
          },
        ],
      },
    ],
  } as any);

  const toolCall = unified.messages.find((m) => m.role === "assistant")
    ?.tool_calls?.[0] as any;
  assert.equal(toolCall?.thought_signature, undefined);

  // Gemini rebuild then uses the skip sentinel (Claude Code path).
  assertFunctionCallHasSignature(
    buildRequestBody(unified),
    SKIP_THOUGHT_SIGNATURE
  );
}

/** enhancetool stays unchanged: no special thought_signature handling. */
async function testEnhanceToolUnchangedRegardingThoughtSignature() {
  const enhancer = new EnhanceToolTransformer();

  // Streaming: first tool chunk is passed through as-is (historical);
  // the completed rebuild omits thought_signature (also historical).
  const streamUpstream = new Response(
    [
      `data: ${JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index: 0,
                  id: "call_1",
                  type: "function",
                  function: { name: "Bash", arguments: "" },
                  thought_signature: "sig_stream",
                },
              ],
            },
            finish_reason: null,
          },
        ],
      })}\n\n`,
      `data: ${JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: '{"command":"ls"}' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      })}\n\n`,
      `data: ${JSON.stringify({
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "tool_calls",
          },
        ],
      })}\n\n`,
      "data: [DONE]\n\n",
    ].join(""),
    {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }
  );

  const streamOut = await enhancer.transformResponseOut(streamUpstream);
  const streamText = await streamOut.text();
  const dataLines = streamText
    .split("\n")
    .filter((l) => l.startsWith("data: ") && l.trim() !== "data: [DONE]")
    .map((l) => JSON.parse(l.slice(6)));

  // First chunk still passthrough (unchanged enhancetool behavior).
  assert.equal(
    dataLines[0]?.choices?.[0]?.delta?.tool_calls?.[0]?.thought_signature,
    "sig_stream"
  );

  // Final rebuilt tool_calls chunk has no thought_signature (unchanged).
  const rebuilt = dataLines.find(
    (d) => d.choices?.[0]?.finish_reason === "tool_calls"
  );
  assert.ok(rebuilt, "expected rebuilt finish chunk");
  assert.equal(
    rebuilt.choices[0].delta.tool_calls[0].thought_signature,
    undefined
  );
}

/**
 * Only the first functionCall part of a step is validated upstream, so the
 * fallback signature must not be duplicated onto sibling parallel calls, and the
 * sentinel must never appear on a thought part (it is not validated there and
 * costs model quality).
 */
async function testFallbackSignatureStaysOnFirstFunctionCallOnly() {
  const req: UnifiedChatRequest = {
    model: "gemini-3.6-flash",
    messages: [
      { role: "user", content: "Run both" },
      {
        role: "assistant",
        content: null,
        // CCR placeholder signature: never replayed upstream.
        thinking: { content: "plan", signature: "ccr_1234567890" },
        tool_calls: [
          {
            id: "call_a",
            type: "function",
            function: { name: "Bash", arguments: '{"command":"ls"}' },
          },
          {
            id: "call_b",
            type: "function",
            function: { name: "Bash", arguments: '{"command":"pwd"}' },
          },
        ],
      },
      { role: "tool", content: "a", tool_call_id: "call_a" },
      { role: "tool", content: "b", tool_call_id: "call_b" },
    ],
  };

  const modelMsg = buildRequestBody(req).contents.find(
    (c: any) => c.role === "model"
  );
  assert.ok(modelMsg);
  const fcParts = modelMsg.parts.filter((p: any) => p.functionCall);
  assert.equal(fcParts.length, 2);
  assert.equal(fcParts[0].thoughtSignature, SKIP_THOUGHT_SIGNATURE);
  assert.equal(fcParts[1].thoughtSignature, undefined);
  assert.equal(
    modelMsg.parts.some((p: any) => p.thought === true),
    false,
    "ccr_ placeholder must not produce a thought part"
  );

  // Real turn-level signature: replayed on the first call, still not duplicated.
  const withReal: UnifiedChatRequest = JSON.parse(JSON.stringify(req));
  (withReal.messages[1] as any).thinking = {
    content: "plan",
    signature: "sig_turn",
  };
  const realParts = buildRequestBody(withReal)
    .contents.find((c: any) => c.role === "model")
    .parts.filter((p: any) => p.functionCall);
  assert.equal(realParts[0].thoughtSignature, "sig_turn");
  assert.equal(realParts[1].thoughtSignature, undefined);
}

/** thoughtSignatureFallback: "none" opts out of the sentinel entirely. */
async function testNoneFallbackDisablesSentinel() {
  const req: UnifiedChatRequest = {
    model: "gemini-3.6-flash",
    messages: [
      { role: "user", content: "List files" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_bash",
            type: "function",
            function: { name: "Bash", arguments: "{}" },
          },
        ],
      },
      { role: "tool", content: "ok", tool_call_id: "call_bash" },
    ],
  };

  const body = buildRequestBody(req, { thoughtSignatureFallback: "none" });
  const fcPart = body.contents
    .find((c: any) => c.role === "model")
    .parts.find((p: any) => p.functionCall);
  assert.ok(fcPart);
  assert.equal(fcPart.thoughtSignature, undefined);

  // Default (no opts) still stamps it — Gemini 3 / Antigravity need it.
  assertFunctionCallHasSignature(buildRequestBody(req), SKIP_THOUGHT_SIGNATURE);
}

/** The "(no content)" filter targets CCR's model placeholder, not user text. */
async function testPlaceholderFilterIsScopedToModelTurns() {
  const body = buildRequestBody({
    model: "gemini-3-flash",
    messages: [
      { role: "user", content: "(no content)" },
      { role: "assistant", content: "(no content)" },
    ],
  });

  const userMsg = body.contents.find((c: any) => c.role === "user");
  assert.ok(userMsg);
  assert.equal(
    userMsg.parts.some((p: any) => p.text === "(no content)"),
    true,
    "user text must be replayed verbatim"
  );

  const modelMsg = body.contents.find((c: any) => c.role === "model");
  assert.ok(modelMsg);
  assert.equal(
    modelMsg.parts.some((p: any) => p.text === "(no content)"),
    false,
    "model placeholder must be stripped"
  );
}

/**
 * functionResponse.id must equal functionCall.id. Gemini uses that match for
 * parallel tools; Claude-on-Antigravity remaps it to Anthropic
 * tool_result.tool_use_id and 400s with "Field required" when it is absent.
 */
async function testFunctionResponseIdMatchesFunctionCall() {
  const body = buildRequestBody({
    model: "claude-opus-4-6-thinking",
    messages: [
      { role: "user", content: "list" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "toolu_abc",
            type: "function",
            function: {
              name: "Bash",
              arguments: '{"command":"ls"}',
            },
          },
          {
            id: "toolu_def",
            type: "function",
            function: {
              name: "Read",
              arguments: '{"path":"a.ts"}',
            },
          },
        ],
      },
      { role: "tool", content: "ok", tool_call_id: "toolu_abc" },
      { role: "tool", content: "src", tool_call_id: "toolu_def" },
    ],
  });

  const model = body.contents.find((c: any) => c.role === "model");
  assert.ok(model);
  const callIds = model.parts
    .filter((p: any) => p.functionCall)
    .map((p: any) => p.functionCall.id);
  assert.deepEqual(callIds, ["toolu_abc", "toolu_def"]);

  const userWithResponses = body.contents.find(
    (c: any) =>
      c.role === "user" &&
      (c.parts || []).some((p: any) => p.functionResponse)
  );
  assert.ok(userWithResponses, "expected functionResponse turn");
  const responseIds = userWithResponses.parts.map(
    (p: any) => p.functionResponse.id
  );
  assert.deepEqual(responseIds, callIds);
  assert.equal(userWithResponses.parts[0].functionResponse.name, "Bash");
  assert.equal(userWithResponses.parts[1].functionResponse.name, "Read");
}

async function testChatReasoningContentHistoryBecomesThoughtPart() {
  const body = buildRequestBody({
    model: "gemini-3-flash",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "ok",
        // Signature-only thinking: the text lives in Chat's reasoning_content,
        // which thinkingFromUnifiedAssistant falls back to.
        reasoning_content: "plan first",
        thinking: { content: "", signature: "sig_chat" },
      },
    ],
  });
  const model = body.contents.find((c: any) => c.role === "model");
  const thought = model.parts.find((p: any) => p.thought === true);
  assert.ok(thought);
  assert.equal(thought.text, "plan first");
  assert.equal(thought.thoughtSignature, "sig_chat");
}

async function testUnsignedReasoningContentDoesNotInventThoughtPart() {
  const body = buildRequestBody({
    model: "gemini-3-flash",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "ok",
        reasoning_content: "plan first",
      },
    ],
  });
  const model = body.contents.find((c: any) => c.role === "model");
  assert.equal(
    model.parts.some((p: any) => p.thought === true),
    false,
    "unsigned Chat reasoning must not become a Gemini thought part"
  );
  assert.ok(model.parts.some((p: any) => p.text === "ok"));
}

async function main() {
  await testToolTurnNoPlaceholderAndToolCallsFinish();
  await testCrossChunkToolThenStopTrailer();
  await testBuildRequestBodyStripsNoContentPlaceholder();
  await testSkipFallbackOnMissingToolSignature();
  await testFallbackSignatureStaysOnFirstFunctionCallOnly();
  await testNoneFallbackDisablesSentinel();
  await testPlaceholderFilterIsScopedToModelTurns();
  await testPerToolSignaturePreferredWhenPresentOnUnified();
  await testAntigravitySiblingSignatureOnStream();
  await testAnthropicDoesNotRoundTripToolSignature();
  await testEnhanceToolUnchangedRegardingThoughtSignature();
  await testFunctionResponseIdMatchesFunctionCall();
  await testChatReasoningContentHistoryBecomesThoughtPart();
  await testUnsignedReasoningContentDoesNotInventThoughtPart();
  console.log("gemini.function-call-signatures: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

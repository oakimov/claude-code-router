/**
 * Live thinking round trip: CCR → Anthropic for all three chat protocols.
 *
 * Each protocol runs a two-turn tool loop: turn 1 must return thinking plus a
 * tool call; turn 2 replays the assistant turn exactly as that protocol's
 * client would (Anthropic: signed thinking blocks; Responses: reasoning item
 * without a signature; Chat: reasoning_content without a signature) and must
 * succeed. Guards the unsigned-thinking 400 (`thinking.signature: Field
 * required`) and the manual-thinking budget 400 (`budget_tokens: Field
 * required`).
 *
 * Opt-in (never in CI): `pnpm test:live`. Environment:
 *   CCR_URL         default http://localhost:3456
 *   CCR_API_KEY     required: the CCR server's API key
 *   CCR_LIVE_MODEL  default "claude,claude-haiku-4-5-20251001". The prompt is a
 *                   multi-step problem so adaptive models (Opus) also think.
 */
import assert from "node:assert/strict";

const CCR_URL = process.env.CCR_URL || "http://localhost:3456";
const API_KEY = requireApiKey();

function requireApiKey(): string {
  const key = process.env.CCR_API_KEY;
  if (!key) {
    console.error("ccr-live.thinking: set CCR_API_KEY to the CCR server's API key");
    process.exit(1);
  }
  return key;
}
const MODEL = process.env.CCR_LIVE_MODEL || "claude,claude-haiku-4-5-20251001";
// Hard enough that adaptive models (Opus) always think: a multi-step search
// with no shortcut answer. The tool call comes right after the reasoning, so
// turn 1 still ends on tool_use; the answer is only given on turn 2.
const PROMPT = [
  "Work this out carefully before doing anything else.",
  "For every integer n from 1 to 50, decide whether n^2 + n + 41 is prime;",
  "for each n where it is composite, give its prime factorization and explain why.",
  "Then check whether any n in that range makes n^2 + n + 41 divisible by 43.",
  "Only after finishing that reasoning, call get_time.",
  "Once you have the time, reply with the time, the composite cases, and the divisibility result.",
].join(" ");
/** Room for long reasoning plus the tool call on turn 1. */
const MAX_TOKENS = 16_000;

async function post(path: string, body: unknown): Promise<{ status: number; json: any }> {
  const response = await fetch(CCR_URL + path, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": API_KEY,
      authorization: `Bearer ${API_KEY}`,
      // A non-Claude-Code client, so CCR's third-party emulation applies.
      "user-agent": "ccr-live-test/1.0",
    },
    body: JSON.stringify(body),
  });
  return { status: response.status, json: await response.json() };
}

function describe(result: { status: number; json: any }): string {
  return `${result.status} ${JSON.stringify(result.json).slice(0, 400)}`;
}

async function anthropicMessages(): Promise<void> {
  const base = {
    model: MODEL,
    max_tokens: MAX_TOKENS,
    thinking: { type: "enabled", budget_tokens: 8_000 },
    tools: [
      {
        name: "get_time",
        description: "Get the current time",
        input_schema: { type: "object", properties: {} },
      },
    ],
  };
  const question = { role: "user", content: PROMPT };
  const t1 = await post("/v1/messages", { ...base, messages: [question] });
  assert.equal(t1.status, 200, `anthropic T1: ${describe(t1)}`);
  const content: any[] = t1.json.content || [];
  const thinking = content.find((block) => block.type === "thinking");
  assert.ok(thinking?.thinking, `anthropic T1 thinking: ${describe(t1)}`);
  assert.ok(thinking.signature, "anthropic T1 thinking must be signed");
  const toolUse = content.find((block) => block.type === "tool_use");
  assert.ok(toolUse, `anthropic T1 tool_use: ${describe(t1)}`);

  const t2 = await post("/v1/messages", {
    ...base,
    messages: [
      question,
      { role: "assistant", content },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: toolUse.id, content: "12:00" }],
      },
    ],
  });
  assert.equal(t2.status, 200, `anthropic T2 (signed replay): ${describe(t2)}`);
  console.log(
    `  anthropic: T1 thinking ${thinking.thinking.length} chars (signed) + tool_use; T2 200 ${
      (t2.json.content || []).map((block: any) => block.type).join(",")
    }`
  );
}

async function responses(): Promise<void> {
  const base = {
    model: MODEL,
    stream: false,
    max_output_tokens: MAX_TOKENS,
    reasoning: { effort: "high", summary: "auto" },
    tools: [
      {
        type: "function",
        name: "get_time",
        description: "Get the current time",
        parameters: { type: "object", properties: {} },
      },
    ],
  };
  const question = {
    role: "user",
    content: [{ type: "input_text", text: PROMPT }],
  };
  const t1 = await post("/v1/responses", { ...base, input: [question] });
  assert.equal(t1.status, 200, `responses T1: ${describe(t1)}`);
  const output: any[] = t1.json.output || [];
  const reasoning = output.find((item) => item.type === "reasoning");
  const summary = (reasoning?.summary || []).map((part: any) => part.text).join("");
  assert.ok(summary, `responses T1 reasoning summary: ${describe(t1)}`);
  const call = output.find((item) => item.type === "function_call");
  assert.ok(call, `responses T1 function_call: ${describe(t1)}`);

  const t2 = await post("/v1/responses", {
    ...base,
    input: [
      question,
      ...output,
      { type: "function_call_output", call_id: call.call_id, output: "12:00" },
    ],
  });
  assert.equal(t2.status, 200, `responses T2 (reasoning replay): ${describe(t2)}`);
  console.log(
    `  responses: T1 reasoning ${summary.length} chars + function_call; T2 200 ${
      (t2.json.output || []).map((item: any) => item.type).join(",")
    }`
  );
}

async function chatCompletions(): Promise<void> {
  const base = {
    model: MODEL,
    stream: false,
    max_tokens: MAX_TOKENS,
    reasoning_effort: "high",
    tools: [
      {
        type: "function",
        function: {
          name: "get_time",
          description: "Get the current time",
          parameters: { type: "object", properties: {} },
        },
      },
    ],
  };
  const question = { role: "user", content: PROMPT };
  const t1 = await post("/v1/chat/completions", { ...base, messages: [question] });
  assert.equal(t1.status, 200, `chat T1: ${describe(t1)}`);
  const message = t1.json.choices?.[0]?.message || {};
  const reasoning: string = message.reasoning_content || message.reasoning || "";
  assert.ok(reasoning, `chat T1 reasoning: ${describe(t1)}`);
  const call = message.tool_calls?.[0];
  assert.ok(call, `chat T1 tool_calls: ${describe(t1)}`);

  const t2 = await post("/v1/chat/completions", {
    ...base,
    messages: [
      question,
      {
        role: "assistant",
        content: message.content ?? null,
        tool_calls: message.tool_calls,
        reasoning_content: reasoning,
      },
      { role: "tool", tool_call_id: call.id, content: "12:00" },
    ],
  });
  assert.equal(t2.status, 200, `chat T2 (reasoning_content replay): ${describe(t2)}`);
  console.log(
    `  chat: T1 reasoning ${reasoning.length} chars + tool_call; T2 200 finish=${
      t2.json.choices?.[0]?.finish_reason
    }`
  );
}

async function main(): Promise<void> {
  console.log(`ccr-live.thinking: ${MODEL} via ${CCR_URL}`);
  await anthropicMessages();
  await responses();
  await chatCompletions();
  console.log("ccr-live.thinking: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

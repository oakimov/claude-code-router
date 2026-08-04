/**
 * Responses → Chat provider-side conversion (Fixes 4–6).
 * Fix 4: streamed parallel tool calls keep stable per-item tool indexes.
 * Fix 5: non-streamed parallel tool calls all survive JSON conversion.
 * Fix 6: response.failed terminates the stream as an error, never as a
 * successful empty completion.
 */
import assert from "node:assert/strict";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

function sseResponse(lines: string[]): Response {
  const body = lines.map((line) => `data: ${line}`).join("\n\n") + "\n\n";
  return new Response(body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

interface ParsedStream {
  chunks: any[];
  doneCount: number;
  raw: string;
}

async function convertStream(lines: string[]): Promise<ParsedStream> {
  const tf = new OpenAIResponsesTransformer();
  tf.logger = logger;
  const out = await tf.transformResponseOut(sseResponse(lines));
  const raw = await out.text();
  const chunks: any[] = [];
  let doneCount = 0;
  for (const line of raw.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(5).trim();
    if (data === "[DONE]") {
      doneCount += 1;
      continue;
    }
    chunks.push(JSON.parse(data));
  }
  return { chunks, doneCount, raw };
}

function toolIndexesOf(chunks: any[]): number[] {
  const indexes: number[] = [];
  for (const chunk of chunks) {
    for (const call of chunk.choices?.[0]?.delta?.tool_calls ?? []) {
      indexes.push(call.index);
    }
  }
  return indexes;
}

async function testInterleavedParallelCalls() {
  const { chunks, doneCount } = await convertStream([
    '{"type":"response.output_item.added","item":{"id":"fc_A","type":"function_call","call_id":"call_A","name":"Read"}}',
    '{"type":"response.output_item.added","item":{"id":"fc_B","type":"function_call","call_id":"call_B","name":"Grep"}}',
    '{"type":"response.function_call_arguments.delta","item_id":"fc_A","delta":"{\\"pa"}',
    '{"type":"response.function_call_arguments.delta","item_id":"fc_B","delta":"{\\"qu"}',
    '{"type":"response.function_call_arguments.delta","item_id":"fc_A","delta":"th\\":\\"a.ts\\"}"}',
    '{"type":"response.function_call_arguments.delta","item_id":"fc_B","delta":"ery\\":\\"foo\\"}"}',
    '{"type":"response.completed","response":{"id":"resp_1","model":"gpt","output":[{"type":"function_call"},{"type":"function_call"}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}',
  ]);

  // Two start chunks with distinct, stable tool indexes.
  const starts = chunks.filter(
    (chunk) => chunk.choices?.[0]?.delta?.tool_calls?.[0]?.id
  );
  assert.equal(starts.length, 2);
  assert.equal(starts[0].choices[0].delta.tool_calls[0].index, 0);
  assert.equal(starts[1].choices[0].delta.tool_calls[0].index, 1);
  assert.equal(starts[0].choices[0].delta.tool_calls[0].function.name, "Read");
  assert.equal(starts[1].choices[0].delta.tool_calls[0].function.name, "Grep");

  // Interleaved deltas attach to the right call: A, B, A, B → 0, 1, 0, 1.
  assert.deepEqual(toolIndexesOf(chunks), [0, 1, 0, 1, 0, 1]);

  // Choice index is always 0; identity lives in the tool call index.
  assert.ok(chunks.every((chunk) => chunk.choices?.[0]?.index === 0));

  // Exactly one terminal finish chunk with tool_calls reason and one [DONE].
  const finishes = chunks.filter(
    (chunk) => chunk.choices?.[0]?.finish_reason
  );
  assert.equal(finishes.length, 1);
  assert.equal(finishes[0].choices[0].finish_reason, "tool_calls");
  assert.equal(doneCount, 1);
}

async function testNonContiguousOutputIndexFallback() {
  const { chunks } = await convertStream([
    '{"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","call_id":"call_X","name":"F"}}',
    '{"type":"response.output_item.added","output_index":7,"item":{"type":"function_call","call_id":"call_Y","name":"G"}}',
    '{"type":"response.function_call_arguments.delta","output_index":7,"delta":"g1"}',
    '{"type":"response.function_call_arguments.delta","output_index":3,"delta":"f1"}',
    '{"type":"response.function_call_arguments.delta","output_index":7,"delta":"g2"}',
    '{"type":"response.completed","response":{"id":"resp_2","model":"gpt","output":[{"type":"function_call"}]}}',
  ]);
  // output_index 3 → tool 0, output_index 7 → tool 1, stable across deltas.
  assert.deepEqual(toolIndexesOf(chunks), [0, 1, 1, 0, 1]);
}

async function testSanitizeCollisionProneIdsKeepDistinctIndexes() {
  const { chunks } = await convertStream([
    '{"type":"response.output_item.added","item":{"id":"item_1","type":"function_call","call_id":"call a:b","name":"F"}}',
    '{"type":"response.output_item.added","item":{"id":"item_2","type":"function_call","call_id":"call a.b","name":"G"}}',
    '{"type":"response.function_call_arguments.delta","item_id":"item_2","delta":"x"}',
    '{"type":"response.function_call_arguments.delta","item_id":"item_1","delta":"y"}',
    '{"type":"response.completed","response":{"id":"resp_3","model":"gpt","output":[{"type":"function_call"}]}}',
  ]);
  // Item identity (not the sanitized call id) keeps the indexes distinct.
  assert.deepEqual(toolIndexesOf(chunks), [0, 1, 1, 0]);
}

async function testJsonParallelCallsAllSurvive() {
  const tf = new OpenAIResponsesTransformer();
  tf.logger = logger;
  const payload = {
    id: "resp_json",
    object: "response",
    model: "gpt",
    created_at: 1,
    output: [
      {
        type: "message",
        content: [{ type: "output_text", text: "running tools" }],
      },
      {
        type: "function_call",
        call_id: "call_1",
        name: "Read",
        arguments: '{"path":"a.ts"}',
      },
      {
        type: "function_call",
        call_id: "call 2",
        name: "Grep",
        arguments: '{"query":"foo"}',
      },
      {
        type: "function_call",
        call_id: "call_3",
        name: "Edit",
        arguments: '{"path":"b.ts"}',
      },
    ],
    usage: { input_tokens: 1, output_tokens: 2, total_tokens: 3 },
  };
  const out = await tf.transformResponseOut(
    new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    })
  );
  const chat = await out.json();
  const message = chat.choices[0].message;
  assert.equal(message.content, "running tools");
  assert.equal(message.tool_calls.length, 3);
  assert.deepEqual(
    message.tool_calls.map((call: any) => call.function.name),
    ["Read", "Grep", "Edit"]
  );
  assert.equal(message.tool_calls[0].id, "call_1");
  // The non-conforming id is sanitized, not dropped.
  assert.ok(message.tool_calls[1].id);
  assert.equal(message.tool_calls[2].function.arguments, '{"path":"b.ts"}');
  assert.equal(chat.choices[0].finish_reason, "tool_calls");
}

async function testJsonNoToolCallsOmitsThem() {
  const tf = new OpenAIResponsesTransformer();
  tf.logger = logger;
  const payload = {
    id: "resp_plain",
    object: "response",
    model: "gpt",
    created_at: 1,
    output: [
      {
        type: "message",
        content: [{ type: "output_text", text: "plain answer" }],
      },
    ],
  };
  const out = await tf.transformResponseOut(
    new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    })
  );
  const chat = await out.json();
  assert.equal(chat.choices[0].message.tool_calls, null);
  assert.equal(chat.choices[0].finish_reason, "stop");
}

async function testFailedBeforeOutput() {
  const { chunks, doneCount } = await convertStream([
    '{"type":"response.failed","response":{"id":"resp_f1","status":"failed","error":{"message":"rate limit hit","type":"rate_limit_error","code":"rate_limit"}}}',
  ]);
  assert.equal(chunks.length, 1);
  assert.ok(chunks[0].error);
  assert.equal(chunks[0].error.message, "rate limit hit");
  assert.equal(chunks[0].error.type, "rate_limit_error");
  assert.equal(chunks[0].error.code, "rate_limit");
  assert.equal(doneCount, 1);
  // No successful terminal event.
  assert.ok(
    chunks.every((chunk) => !chunk.choices?.[0]?.finish_reason)
  );
}

async function testFailedAfterPartialText() {
  const { chunks, doneCount } = await convertStream([
    '{"type":"response.output_text.delta","item_id":"msg_1","delta":"partial"}',
    '{"type":"response.failed","response":{"id":"resp_f2","status":"failed","error":{"message":"server exploded","type":"server_error"}}}',
    '{"type":"response.completed","response":{"id":"resp_f2","model":"gpt","output":[]}}',
  ]);
  // Partial output preserved…
  assert.equal(chunks[0].choices[0].delta.content, "partial");
  // …then exactly one error terminator; the trailing response.completed is
  // ignored — no success chunk, no duplicate failure, single [DONE].
  const errors = chunks.filter((chunk) => chunk.error);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].error.message, "server exploded");
  assert.equal(
    chunks.filter((chunk) => chunk.choices?.[0]?.finish_reason).length,
    0
  );
  assert.equal(doneCount, 1);
}

async function testFailedAfterPartialToolCall() {
  const { chunks, doneCount } = await convertStream([
    '{"type":"response.output_item.added","item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"Read"}}',
    '{"type":"response.function_call_arguments.delta","item_id":"fc_1","delta":"{\\"pa"}',
    '{"type":"response.failed","response":{"id":"resp_f3","status":"failed","error":{"message":"connection reset","type":"server_error"}}}',
  ]);
  assert.equal(chunks[0].choices[0].delta.tool_calls[0].index, 0);
  assert.equal(chunks[1].choices[0].delta.tool_calls[0].index, 0);
  assert.equal(chunks[2].error.message, "connection reset");
  assert.equal(chunks.length, 3);
  assert.equal(doneCount, 1);
}

async function testFailedMessageIsRedacted() {
  const secret = "sk-zzzzzzzzzzzzzzzzzzzzzzzz";
  const { chunks, raw } = await convertStream([
    `{"type":"response.failed","response":{"id":"resp_f4","status":"failed","error":{"message":"bad key ${secret}","type":"authentication_error"}}}`,
  ]);
  assert.equal(chunks.length, 1);
  assert.ok(!raw.includes(secret));
  assert.ok(chunks[0].error.message.includes("[redacted-secret]"));
}

async function main() {
  await testInterleavedParallelCalls();
  await testNonContiguousOutputIndexFallback();
  await testSanitizeCollisionProneIdsKeepDistinctIndexes();
  await testJsonParallelCallsAllSurvive();
  await testJsonNoToolCallsOmitsThem();
  await testFailedBeforeOutput();
  await testFailedAfterPartialText();
  await testFailedAfterPartialToolCall();
  await testFailedMessageIsRedacted();
  console.log("responses.parallel-and-failure: all tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

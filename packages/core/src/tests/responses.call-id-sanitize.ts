import assert from "node:assert/strict";
import { CodexTransformer } from "../transformer/codex.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { sanitizeResponsesCallId } from "../utils/toolCallId";

const CURSOR_CONCATENATED_ID =
  "call-901b1ddc-d889-4a6e-8c58-564ad17bc095-3\nfc_b466705e-df33-9395-8d4a-21a95066affe_0";
const RESPONSES_CALL_ID_PATTERN = /^[a-zA-Z0-9_-]{1,64}$/;

function sanitizerContract() {
  const valid = "call_901b1ddc_d889_4a6e_8c58_564ad17bc095";
  assert.equal(sanitizeResponsesCallId(valid), valid);
  assert.equal(sanitizeResponsesCallId("x".repeat(64)), "x".repeat(64));

  const sanitized = sanitizeResponsesCallId(CURSOR_CONCATENATED_ID)!;
  assert.match(sanitized, RESPONSES_CALL_ID_PATTERN);
  assert.ok(sanitized.length <= 64);
  assert.equal(sanitizeResponsesCallId(CURSOR_CONCATENATED_ID), sanitized);
  assert.equal(sanitizeResponsesCallId(sanitized), sanitized);

  const sharedPrefix = "x".repeat(80);
  assert.notEqual(
    sanitizeResponsesCallId(`${sharedPrefix}-a`),
    sanitizeResponsesCallId(`${sharedPrefix}-b`)
  );
  assert.notEqual(
    sanitizeResponsesCallId("call_a\nb"),
    sanitizeResponsesCallId("call_a_b")
  );

  assert.equal(sanitizeResponsesCallId(""), undefined);
  assert.equal(sanitizeResponsesCallId(undefined), undefined);
}

function pairedRequest() {
  return {
    model: "gpt-5.6-sol",
    messages: [
      { role: "user", content: "run it" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: CURSOR_CONCATENATED_ID,
            type: "function",
            function: { name: "Bash", arguments: '{"command":"pwd"}' },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: CURSOR_CONCATENATED_ID,
        content: "/tmp",
      },
    ],
  };
}

function assertPairedInput(input: any[]) {
  const call = input.find((item) => item.type === "function_call");
  const output = input.find((item) => item.type === "function_call_output");
  assert.ok(call);
  assert.ok(output);
  assert.match(call.call_id, RESPONSES_CALL_ID_PATTERN);
  assert.equal(call.call_id, output.call_id);
}

async function codexRequestIsSanitized() {
  const transformer = new CodexTransformer();
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });

  const result = await transformer.transformRequestIn(
    pairedRequest() as any,
    { baseUrl: "https://example.test" },
    {}
  );
  assertPairedInput((result.body as any).input);
}

async function openAIResponsesRequestIsSanitized() {
  const transformer = new OpenAIResponsesTransformer();
  const result = await transformer.transformRequestIn(
    pairedRequest() as any,
    {},
    {}
  );
  assertPairedInput((result as any).input);
}

function functionCallEvent() {
  return {
    type: "response.output_item.added",
    output_index: 0,
    item: {
      type: "function_call",
      id: "fc_test",
      call_id: CURSOR_CONCATENATED_ID,
      name: "Bash",
    },
  };
}

function streamingResponse() {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(
          encoder.encode(`data: ${JSON.stringify(functionCallEvent())}\n\n`)
        );
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } }
  );
}

async function streamingResponsesAreSanitized() {
  for (const transformer of [
    new OpenAIResponsesTransformer(),
    new CodexTransformer(),
  ]) {
    const response = await transformer.transformResponseOut(streamingResponse());
    const text = await response.text();
    const id = JSON.parse(
      text
        .split("\n")
        .find((line) => line.startsWith("data: {"))!
        .slice(6)
    ).choices[0].delta.tool_calls[0].id;
    assert.match(id, RESPONSES_CALL_ID_PATTERN);
  }
}

async function nonStreamingResponsesAreSanitized() {
  const payload = {
    id: "resp_test",
    object: "response",
    model: "gpt-5.6-sol",
    created_at: 1,
    output: [
      {
        type: "function_call",
        id: "fc_test",
        call_id: CURSOR_CONCATENATED_ID,
        name: "Bash",
        arguments: '{"command":"pwd"}',
      },
    ],
  };

  for (const transformer of [
    new OpenAIResponsesTransformer(),
    new CodexTransformer(),
  ]) {
    (transformer as any).logger = { debug() {} };
    const response = await transformer.transformResponseOut(
      new Response(JSON.stringify(payload), {
        headers: { "Content-Type": "application/json" },
      })
    );
    const id = (await response.json() as any).choices[0].message.tool_calls[0].id;
    assert.match(id, RESPONSES_CALL_ID_PATTERN);
  }
}

async function main() {
  sanitizerContract();
  await codexRequestIsSanitized();
  await openAIResponsesRequestIsSanitized();
  await streamingResponsesAreSanitized();
  await nonStreamingResponsesAreSanitized();
  console.log("responses.call-id-sanitize: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

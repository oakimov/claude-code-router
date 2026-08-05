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

function redosAdversarialInputIsLinear() {
  // `/_+$/` is quadratic on a run of "_" followed by a non-underscore: the
  // engine re-scans the run for every start position. A client-supplied call
  // id of unbounded length must sanitize in linear time (CodeQL
  // js/polynomial-redos). The fixed implementation trims without a regex.
  const adversarial = `${"_".repeat(1_000_000)}x`;
  const started = performance.now();
  const result = sanitizeResponsesCallId(adversarial)!;
  const elapsed = performance.now() - started;

  assert.match(result, RESPONSES_CALL_ID_PATTERN);
  assert.equal(sanitizeResponsesCallId(adversarial), result); // idempotent
  assert.ok(
    elapsed < 2000,
    `ReDoS regression: sanitizeResponsesCallId took ${elapsed.toFixed(0)}ms`
  );
}

async function perTurnMapAvoidsSanitizedCollisions() {
  const { createCallIdMap, mapCallId } = await import(
    "../utils/openai.responses.util"
  );
  const invalid = "call_with_a_newline\n";
  const sanitized = sanitizeResponsesCallId(invalid)!;
  const map = createCallIdMap();
  assert.equal(mapCallId(map, invalid), sanitized);
  const collidingValidId = mapCallId(map, sanitized)!;
  assert.notEqual(collidingValidId, sanitized);
  assert.match(collidingValidId, RESPONSES_CALL_ID_PATTERN);
  assert.equal(mapCallId(map, sanitized), collidingValidId);
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

async function openAIResponsesProviderRequestPreservesAssistantText() {
  const transformer = new OpenAIResponsesTransformer();
  const result = await transformer.transformRequestIn(
    {
      model: "gpt-5.6-sol",
      messages: [
        {
          role: "assistant",
          content: "I will use a tool",
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "Read", arguments: "{}" },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "done" },
      ],
      tool_choice: {
        type: "function",
        function: { name: "Read" },
      },
      reasoning: { enabled: true, effort: "high" },
    } as any,
    {},
    {}
  );
  const input = (result as any).input;
  assert.ok(
    input.some(
      (item: any) =>
        item.role === "assistant" && item.content === "I will use a tool"
    )
  );
  assert.ok(input.some((item: any) => item.type === "function_call"));
  assert.deepEqual((result as any).tool_choice, {
    type: "function",
    name: "Read",
  });
  assert.equal((result as any).reasoning.summary, undefined);
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

async function clientBoundCallIdsAreMapped() {
  const {
    createCallIdMap,
    mapCallId,
    responsesRequestToUnified,
    unifiedResponseToResponses,
  } = await import("../utils/openai.responses.util");

  const map = createCallIdMap();
  const unified = responsesRequestToUnified(
    {
      model: "openai,gpt-4o",
      input: [
        {
          type: "function_call",
          call_id: CURSOR_CONCATENATED_ID,
          name: "Bash",
          arguments: '{}',
        },
        {
          type: "function_call_output",
          call_id: CURSOR_CONCATENATED_ID,
          output: "/tmp",
        },
      ],
    },
    map
  );

  const assistant = unified.messages.find(
    (m: any) => m.role === "assistant" && m.tool_calls
  );
  const tool = unified.messages.find((m: any) => m.role === "tool");
  assert.ok(assistant);
  assert.ok(tool);
  const sanitized = sanitizeResponsesCallId(CURSOR_CONCATENATED_ID)!;
  assert.equal(assistant!.tool_calls![0].id, sanitized);
  assert.equal(tool!.tool_call_id, sanitized);

  // Client-bound response remains conforming; never restore an invalid id.
  const restored = mapCallId(map, sanitized, "unified_to_client");
  assert.equal(restored, sanitized);

  const responses = unifiedResponseToResponses(
    {
      id: "chatcmpl-x",
      created: 1,
      model: "gpt-4o",
      choices: [
        {
          finish_reason: "tool_calls",
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: sanitized,
                type: "function",
                function: { name: "Bash", arguments: '{}' },
              },
            ],
          },
        },
      ],
    },
    { callIdMap: map }
  );
  assert.equal(responses.output[0].call_id, sanitized);
  assert.match(responses.output[0].call_id, RESPONSES_CALL_ID_PATTERN);
}

async function main() {
  sanitizerContract();
  redosAdversarialInputIsLinear();
  await perTurnMapAvoidsSanitizedCollisions();
  await codexRequestIsSanitized();
  await openAIResponsesRequestIsSanitized();
  await openAIResponsesProviderRequestPreservesAssistantText();
  await streamingResponsesAreSanitized();
  await nonStreamingResponsesAreSanitized();
  await clientBoundCallIdsAreMapped();
  console.log("responses.call-id-sanitize: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

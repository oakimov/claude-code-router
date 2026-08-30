import assert from "node:assert/strict";
import { CodexTransformer } from "../transformer/codex.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { sanitizeResponsesCallId } from "../utils/toolCallId";

const CURSOR_CONCATENATED_ID =
  "call-901b1ddc-d889-4a6e-8c58-564ad17bc095-3\nfc_b466705e-df33-9395-8d4a-21a95066affe_0";
const CURSOR_RESPONSES_REPLAY_ID =
  "call-66dbf0b1-aad7-482f-baa2-647748651824-0_fc_49ff1230-042d-97ce-b451-5e3f019a21d8_0";
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

async function exactWireCallIdsAreSanitizedSelectively() {
  const {
    createCallIdMap,
    responsesRequestToUnified,
    sanitizeResponsesWireCallIds,
  } = await import("../utils/openai.responses.util");
  const map = createCallIdMap();
  const imageOutput = [
    { type: "input_text", text: "image" },
    {
      type: "input_image",
      image_url: "data:image/png;base64,iVBOR",
      detail: "high",
    },
  ];
  const wire = {
    model: "muse-spark-1.2-contributor-free",
    input: [
      {
        type: "function_call",
        id: "fc_original",
        call_id: CURSOR_RESPONSES_REPLAY_ID,
        name: "Read",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: CURSOR_RESPONSES_REPLAY_ID,
        output: imageOutput,
      },
    ],
    prompt_cache_key: "session-stable",
    include: ["reasoning.encrypted_content"],
    store: false,
  };

  // Normalization populates the same map that exact-wire repair later reuses.
  responsesRequestToUnified(wire, map);
  const repaired = sanitizeResponsesWireCallIds(wire, map);
  const call = repaired.input[0];
  const output = repaired.input[1];
  assert.match(call.call_id, RESPONSES_CALL_ID_PATTERN);
  assert.equal(call.call_id, output.call_id);
  assert.equal(call.id, "fc_original");
  assert.equal(repaired.prompt_cache_key, "session-stable");
  assert.deepEqual(repaired.include, ["reasoning.encrypted_content"]);
  assert.equal(repaired.store, false);
  assert.strictEqual(repaired.input[1].output, imageOutput);
  assert.strictEqual(wire.input[0].call_id, CURSOR_RESPONSES_REPLAY_ID);

  const fallbackOnly = sanitizeResponsesWireCallIds({
    model: wire.model,
    input: [
      {
        type: "function_call",
        id: CURSOR_RESPONSES_REPLAY_ID,
        name: "Read",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: CURSOR_RESPONSES_REPLAY_ID,
        output: "done",
      },
    ],
  });
  assert.equal(fallbackOnly.input[0].id, CURSOR_RESPONSES_REPLAY_ID);
  assert.match(fallbackOnly.input[0].call_id, RESPONSES_CALL_ID_PATTERN);
  assert.equal(
    fallbackOnly.input[0].call_id,
    fallbackOnly.input[1].call_id
  );
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

async function openAIResponsesRequestStripsStreamOptions() {
  const transformer = new OpenAIResponsesTransformer();
  const result = await transformer.transformRequestIn(
    {
      model: "gpt-5.4-mini",
      stream: true,
      stream_options: { include_usage: true },
      messages: [{ role: "user", content: "ping" }],
    } as any,
    {},
    {}
  );
  assert.equal((result as any).stream_options, undefined);
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

async function streamingFlatJsonIsReEmittedAsSse() {
  const transformer = new CodexTransformer();
  (transformer as any).logger = { debug() {} };
  const payload = {
    id: "resp_flat",
    object: "response",
    model: "gpt-5.6-sol",
    created_at: 1,
    output: [
      {
        type: "message",
        content: [{ type: "output_text", text: "flat answer" }],
      },
    ],
  };
  const response = await transformer.transformResponseOut(
    new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    }),
    { req: { id: "req-flat" } } as any
  );
  assert.match(response.headers.get("Content-Type") || "", /text\/event-stream/);
  const text = await response.text();
  assert.ok(text.includes('"content":"flat answer"'));
  assert.ok(text.includes("data: [DONE]"));
}

async function nonStreamingResponsesAreSanitized() {
  // Explicitly record non-streaming intent for the flat JSON branch.

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

function customToolRequest() {
  return {
    model: "gpt-5.6-sol",
    messages: [
      { role: "user", content: "edit a file" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_apply_patch_1",
            type: "function",
            function: {
              name: "apply_patch",
              arguments: JSON.stringify({
                input:
                  "*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch",
              }),
            },
          },
        ],
      },
      {
        role: "tool",
        tool_call_id: "call_apply_patch_1",
        content: "Success",
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "apply_patch",
          description: "Apply a patch envelope",
          parameters: {
            type: "object",
            properties: {
              input: { type: "string", description: "freeform patch text" },
            },
            required: ["input"],
          },
        },
      },
    ],
  };
}

async function codexOutboundCustomToolIsRestored() {
  const transformer = new CodexTransformer();
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });

  const result = await transformer.transformRequestIn(
    customToolRequest() as any,
    { baseUrl: "https://example.test" },
    { responsesCustomToolNames: new Set(["apply_patch"]) }
  );
  const body: any = result.body;

  const tool = body.tools.find((t: any) => t.name === "apply_patch");
  assert.ok(tool);
  assert.equal(tool.type, "custom");
  assert.equal(tool.parameters, undefined);

  const call = body.input.find((item: any) => item.type === "custom_tool_call");
  const output = body.input.find(
    (item: any) => item.type === "custom_tool_call_output"
  );
  assert.ok(call);
  assert.ok(output);
  assert.equal(
    call.input,
    "*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch"
  );
  assert.equal(output.output, "Success");
  assert.equal(call.call_id, output.call_id);
  assert.ok(!body.input.some((item: any) => item.type === "function_call"));
  assert.ok(
    !body.input.some((item: any) => item.type === "function_call_output")
  );
}

async function codexOutboundRegressionWithoutCustomToolNames() {
  // Same request shape, but no responsesCustomToolNames on context — must
  // stay on the plain function/function_call path unchanged (this is the
  // overwhelming majority case: any caller that isn't relaying a client's
  // Responses `type: "custom"` tool).
  const transformer = new CodexTransformer();
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });

  const result = await transformer.transformRequestIn(
    customToolRequest() as any,
    { baseUrl: "https://example.test" },
    {}
  );
  const body: any = result.body;

  const tool = body.tools.find((t: any) => t.name === "apply_patch");
  assert.equal(tool.type, "function");
  assert.ok(tool.parameters);
  assert.ok(body.input.some((item: any) => item.type === "function_call"));
  assert.ok(
    body.input.some((item: any) => item.type === "function_call_output")
  );
  assert.ok(!body.input.some((item: any) => item.type === "custom_tool_call"));
}

function customToolStreamingResponse() {
  const encoder = new TextEncoder();
  const events = [
    {
      type: "response.output_item.added",
      output_index: 0,
      item: {
        type: "custom_tool_call",
        id: "ct_test",
        call_id: "ct_test",
        name: "apply_patch",
      },
    },
    {
      type: "response.custom_tool_call_input.delta",
      item_id: "ct_test",
      delta: "*** Begin",
    },
    {
      type: "response.custom_tool_call_input.delta",
      item_id: "ct_test",
      delta: " Patch",
    },
    {
      type: "response.custom_tool_call_input.done",
      item_id: "ct_test",
      input: "*** Begin Patch",
    },
  ];
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const event of events) {
          controller.enqueue(
            encoder.encode(`data: ${JSON.stringify(event)}\n\n`)
          );
        }
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } }
  );
}

async function codexInboundCustomToolStreamingIsConverted() {
  const transformer = new CodexTransformer();
  const response = await transformer.transformResponseOut(
    customToolStreamingResponse()
  );
  const text = await response.text();
  const chunks = text
    .split("\n")
    .filter((line) => line.startsWith("data: {"))
    .map((line) => JSON.parse(line.slice(6)));

  const toolCallChunks = chunks.filter(
    (c: any) => c.choices?.[0]?.delta?.tool_calls
  );
  // Only two tool_calls-bearing chunks reach the client: output_item.added
  // (empty arguments placeholder) and custom_tool_call_input.done (the full
  // JSON-wrapped freeform text). The two .delta events accumulate silently —
  // a partial freeform string can't be JSON-wrapped mid-stream.
  assert.equal(toolCallChunks.length, 2);
  assert.equal(
    toolCallChunks[0].choices[0].delta.tool_calls[0].function.arguments,
    ""
  );
  assert.equal(
    toolCallChunks[1].choices[0].delta.tool_calls[0].function.arguments,
    JSON.stringify({ input: "*** Begin Patch" })
  );
}

async function codexInboundCustomToolNonStreamingIsConverted() {
  const payload = {
    id: "resp_test2",
    object: "response",
    model: "gpt-5.6-sol",
    created_at: 1,
    output: [
      {
        type: "custom_tool_call",
        id: "ct_test2",
        call_id: "ct_test2",
        name: "apply_patch",
        input: "*** Begin Patch\n*** End Patch",
      },
    ],
  };

  const transformer = new CodexTransformer();
  (transformer as any).logger = { debug() {} };
  const response = await transformer.transformResponseOut(
    new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    })
  );
  const json: any = await response.json();
  const toolCall = json.choices[0].message.tool_calls[0];
  assert.equal(toolCall.function.name, "apply_patch");
  assert.equal(
    toolCall.function.arguments,
    JSON.stringify({ input: "*** Begin Patch\n*** End Patch" })
  );
}

function responseFormatRequest() {
  return {
    model: "gpt-5.6-sol",
    messages: [{ role: "user", content: "give me json" }],
    response_format: {
      type: "json_schema",
      json_schema: {
        name: "result",
        schema: { type: "object", properties: { ok: { type: "boolean" } } },
        strict: true,
      },
    },
  };
}

async function codexOutboundResponseFormatIsRestored() {
  const transformer = new CodexTransformer();
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });

  const result = await transformer.transformRequestIn(
    responseFormatRequest() as any,
    { baseUrl: "https://example.test" },
    {}
  );
  const body: any = result.body;

  assert.deepEqual(body.text.format, {
    type: "json_schema",
    name: "result",
    schema: { type: "object", properties: { ok: { type: "boolean" } } },
    strict: true,
  });
  assert.equal(body.response_format, undefined);
}

async function codexInboundParallelCustomToolsKeepDistinctIndexes() {
  const transformer = new CodexTransformer();
  const encoder = new TextEncoder();
  const events = [
    {
      type: "response.output_item.added",
      item: {
        type: "custom_tool_call",
        id: "ct_a",
        call_id: "ct_a",
        name: "apply_patch",
      },
    },
    {
      type: "response.output_item.added",
      item: {
        type: "custom_tool_call",
        id: "ct_b",
        call_id: "ct_b",
        name: "apply_patch",
      },
    },
    {
      type: "response.custom_tool_call_input.done",
      item_id: "ct_a",
      input: "patch A",
    },
    {
      type: "response.custom_tool_call_input.done",
      item_id: "ct_b",
      input: "patch B",
    },
  ];
  const response = await transformer.transformResponseOut(
    new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          for (const event of events) {
            controller.enqueue(
              encoder.encode(`data: ${JSON.stringify(event)}\n\n`)
            );
          }
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          controller.close();
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } }
    )
  );
  const chunks = (await response.text())
    .split("\n")
    .filter((line) => line.startsWith("data: {"))
    .map((line) => JSON.parse(line.slice(6)));
  const toolCalls = chunks.flatMap(
    (chunk: any) => chunk.choices?.[0]?.delta?.tool_calls ?? []
  );
  const starts = toolCalls.filter((call: any) => call.id);
  assert.equal(starts.length, 2);
  assert.equal(starts[0].index, 0);
  assert.equal(starts[1].index, 1);
  const args = toolCalls.filter((call: any) => call.function?.arguments);
  assert.equal(args[0].index, 0);
  assert.equal(args[1].index, 1);
  assert.equal(args[0].function.arguments, JSON.stringify({ input: "patch A" }));
  assert.equal(args[1].function.arguments, JSON.stringify({ input: "patch B" }));
}

async function openAIResponsesOutboundRestoresResponseFormat() {
  // Chat Completions response_format is the Unified stand-in for Responses
  // text.format. Generic Responses destinations (xAI, OpenAI) accept the
  // native field, so restore it and drop the Chat-only property.
  const transformer = new OpenAIResponsesTransformer();
  const result = await transformer.transformRequestIn(
    responseFormatRequest() as any,
    {},
    {}
  );
  assert.equal((result as any).response_format, undefined);
  assert.deepEqual((result as any).text.format, {
    type: "json_schema",
    name: "result",
    schema: { type: "object", properties: { ok: { type: "boolean" } } },
    strict: true,
  });
}

function jsonObjectFormatRequest() {
  return {
    model: "gpt-5.6-sol",
    messages: [{ role: "user", content: "give me json" }],
    response_format: { type: "json_object" },
  };
}

async function outboundJsonObjectFormatIsRestoredOnBothPaths() {
  const responses = new OpenAIResponsesTransformer();
  const responsesResult = await responses.transformRequestIn(
    jsonObjectFormatRequest() as any,
    {},
    {}
  );
  assert.deepEqual((responsesResult as any).text.format, { type: "json_object" });
  assert.equal((responsesResult as any).response_format, undefined);

  const codex = new CodexTransformer();
  (codex as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });
  const codexResult = await codex.transformRequestIn(
    jsonObjectFormatRequest() as any,
    { baseUrl: "https://example.test" },
    {}
  );
  assert.deepEqual((codexResult as any).body.text.format, { type: "json_object" });
  assert.equal((codexResult as any).body.response_format, undefined);
}

async function codexOutboundHistoryStripsHeredoc() {
  // History replay is intentionally normalized: a whole-value heredoc wrapper
  // is stripped so the backend is not re-fed a wrapper the client never kept.
  const transformer = new CodexTransformer();
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });
  const patch =
    "*** Begin Patch\n*** Add File: hello.txt\n+hi\n*** End Patch";
  const request = customToolRequest();
  const assistant = request.messages[1] as {
    tool_calls: Array<{ function: { arguments: string } }>;
  };
  assistant.tool_calls[0].function.arguments = JSON.stringify({
    input: `<<EOF\n${patch}\nEOF`,
  });
  const result = await transformer.transformRequestIn(
    request as any,
    { baseUrl: "https://example.test" },
    { responsesCustomToolNames: new Set(["apply_patch"]) }
  );
  const call = (result as any).body.input.find(
    (item: any) => item.type === "custom_tool_call"
  );
  assert.equal(call.input, patch);
}

async function openAIResponsesOutboundMapsChatAndAnthropicFields() {
  const transformer = new OpenAIResponsesTransformer();
  const result = await transformer.transformRequestIn(
    {
      model: "grok-4.6",
      messages: [{ role: "user", content: "hi" }],
      tool_choice: "required",
      parallel_tool_calls: false,
      stop: ["END"],
      tools: [
        {
          type: "function",
          function: {
            name: "Read",
            description: "read",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    } as any,
    {},
    {}
  );
  assert.equal((result as any).tool_choice, "required");
  assert.equal((result as any).parallel_tool_calls, false);
  assert.equal((result as any).stop, undefined);
}

async function main() {
  sanitizerContract();
  redosAdversarialInputIsLinear();
  await perTurnMapAvoidsSanitizedCollisions();
  await exactWireCallIdsAreSanitizedSelectively();
  await codexRequestIsSanitized();
  await openAIResponsesRequestStripsStreamOptions();
  await openAIResponsesRequestIsSanitized();
  await openAIResponsesProviderRequestPreservesAssistantText();
  await streamingResponsesAreSanitized();
  await streamingFlatJsonIsReEmittedAsSse();
  await nonStreamingResponsesAreSanitized();
  await clientBoundCallIdsAreMapped();
  await codexOutboundCustomToolIsRestored();
  await codexOutboundRegressionWithoutCustomToolNames();
  await codexInboundCustomToolStreamingIsConverted();
  await codexInboundCustomToolNonStreamingIsConverted();
  await codexInboundParallelCustomToolsKeepDistinctIndexes();
  await codexOutboundResponseFormatIsRestored();
  await openAIResponsesOutboundRestoresResponseFormat();
  await outboundJsonObjectFormatIsRestoredOnBothPaths();
  await codexOutboundHistoryStripsHeredoc();
  await openAIResponsesOutboundMapsChatAndAnthropicFields();
  console.log("responses.call-id-sanitize: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

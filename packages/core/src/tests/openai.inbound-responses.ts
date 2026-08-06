/**
 * Inbound OpenAI Responses: request normalization, unsupported state,
 * JSON output, exact text/tool SSE lifecycle (including content_part),
 * usage, errors, and call-id mapping.
 */
import assert from "node:assert/strict";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import {
  createCallIdMap,
  createResponsesStreamState,
  finalizeResponsesStream,
  responsesRequestToUnified,
  unifiedChunkToResponsesEvents,
  unifiedResponseToResponses,
} from "../utils/openai.responses.util";
import { sanitizeResponsesCallId } from "../utils/toolCallId";

const CURSOR_CONCATENATED_ID =
  "call-901b1ddc-d889-4a6e-8c58-564ad17bc095-3\nfc_b466705e-df33-9395-8d4a-21a95066affe_0";

async function expectReject(
  fn: () => Promise<unknown> | unknown,
  code: string
): Promise<void> {
  let caught: any;
  try {
    await fn();
  } catch (e) {
    caught = e;
  }
  assert.ok(caught, `expected reject with code ${code}`);
  assert.equal(caught.code, code);
  assert.equal(caught.statusCode, 400);
}

function parseSseEvents(text: string): any[] {
  return text
    .split("\n")
    .filter((line) => line.startsWith("data: ") && line !== "data: [DONE]")
    .map((line) => JSON.parse(line.slice(6)));
}

async function testStringAndMessageInput() {
  const unified = responsesRequestToUnified({
    model: "openai,gpt-4o",
    instructions: "be brief",
    input: "hello",
    stream: false,
  });
  assert.ok(unified.messages.some((m: any) => m.role === "system"));
  assert.ok(
    unified.messages.some(
      (m: any) => m.role === "user" && m.content === "hello"
    )
  );

  const withItems = responsesRequestToUnified({
    model: "openai,gpt-4o",
    input: [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "hi" }],
      },
      {
        type: "message",
        role: "user",
        content: [
          {
            type: "input_image",
            image_url: "data:image/png;base64,aa",
          },
        ],
      },
    ],
  });
  assert.ok(
    withItems.messages.some(
      (m: any) =>
        m.role === "user" &&
        (m.content === "hi" ||
          (Array.isArray(m.content) &&
            m.content.some((p: any) => p.type === "text")))
    )
  );
  assert.ok(
    withItems.messages.some(
      (m: any) =>
        Array.isArray(m.content) &&
        m.content.some((p: any) => p.type === "image_url")
    )
  );
}

async function testFunctionCallRoundTrip() {
  const map = createCallIdMap();
  const unified = responsesRequestToUnified(
    {
      model: "openai,gpt-4o",
      input: [
        {
          type: "function_call",
          call_id: CURSOR_CONCATENATED_ID,
          name: "Read",
          arguments: '{"path":"a.ts"}',
        },
        {
          type: "function_call_output",
          call_id: CURSOR_CONCATENATED_ID,
          output: "ok",
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
  assert.ok(assistant.tool_calls?.length);
  assert.equal(assistant.tool_calls![0].id, sanitized);
  assert.equal(tool.tool_call_id, sanitized);
  assert.equal(map.reverse.get(sanitized), CURSOR_CONCATENATED_ID);
}

/**
 * DeepSeek / Chat Completions require one assistant message carrying every
 * parallel tool_call. Responses emits consecutive function_call items; those
 * must coalesce, otherwise upstream rejects with "insufficient tool messages".
 */
async function testParallelFunctionCallsCoalesce() {
  const unified = responsesRequestToUnified({
    model: "opencode-openai,deepseek-v4-flash-free",
    input: [
      {
        type: "function_call",
        call_id: "call_A",
        name: "exec_command",
        arguments: '{"cmd":"ls"}',
      },
      {
        type: "function_call",
        call_id: "call_B",
        name: "exec_command",
        arguments: '{"cmd":"pwd"}',
      },
      {
        type: "function_call_output",
        call_id: "call_A",
        output: "a.ts",
      },
      {
        type: "function_call_output",
        call_id: "call_B",
        output: "/tmp",
      },
    ],
  });

  const assistants = unified.messages.filter(
    (m: any) => m.role === "assistant" && m.tool_calls
  );
  const tools = unified.messages.filter((m: any) => m.role === "tool");
  assert.equal(assistants.length, 1, "parallel calls must share one assistant");
  const parallelAssistant = assistants[0]!;
  assert.ok(parallelAssistant.tool_calls);
  assert.equal(parallelAssistant.tool_calls.length, 2);
  assert.equal(parallelAssistant.tool_calls[0]!.id, "call_A");
  assert.equal(parallelAssistant.tool_calls[1]!.id, "call_B");
  assert.equal(tools.length, 2);
  assert.equal(tools[0]!.tool_call_id, "call_A");
  assert.equal(tools[1]!.tool_call_id, "call_B");
}

async function testParallelCustomToolCallsCoalesce() {
  const unified = responsesRequestToUnified({
    model: "m",
    input: [
      {
        type: "custom_tool_call",
        call_id: "call_X",
        name: "exec",
        input: "echo x",
      },
      {
        type: "custom_tool_call",
        call_id: "call_Y",
        name: "exec",
        input: "echo y",
      },
      {
        type: "custom_tool_call_output",
        call_id: "call_X",
        output: "x",
      },
      {
        type: "custom_tool_call_output",
        call_id: "call_Y",
        output: "y",
      },
    ],
  });

  const assistants = unified.messages.filter(
    (m: any) => m.role === "assistant" && m.tool_calls
  );
  assert.equal(assistants.length, 1);
  const customAssistant = assistants[0]!;
  assert.ok(customAssistant.tool_calls);
  assert.equal(customAssistant.tool_calls.length, 2);
  assert.equal(customAssistant.tool_calls[0]!.function.name, "exec");
  assert.equal(
    customAssistant.tool_calls[0]!.function.arguments,
    JSON.stringify({ input: "echo x" })
  );
}

async function testSequentialToolTurnsDoNotMerge() {
  const unified = responsesRequestToUnified({
    model: "m",
    input: [
      {
        type: "function_call",
        call_id: "call_1",
        name: "a",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: "ok1",
      },
      {
        type: "function_call",
        call_id: "call_2",
        name: "b",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: "call_2",
        output: "ok2",
      },
    ],
  });

  const assistants = unified.messages.filter(
    (m: any) => m.role === "assistant" && m.tool_calls
  );
  assert.equal(
    assistants.length,
    2,
    "call→output→call must stay two assistant turns"
  );
  assert.ok(assistants[0]?.tool_calls);
  assert.ok(assistants[1]?.tool_calls);
  assert.equal(assistants[0]!.tool_calls.length, 1);
  assert.equal(assistants[1]!.tool_calls.length, 1);
}

async function testUnsupportedState() {
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: "x",
        store: true,
      }),
    "unsupported_store"
  );
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: "x",
        previous_response_id: "resp_1",
      }),
    "unsupported_previous_response_id"
  );
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: "x",
        conversation: "conv_1",
      }),
    "unsupported_conversation"
  );
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: "x",
        background: true,
      }),
    "unsupported_background"
  );
  // Hosted tools that are not function/custom also convert to function tools
  // (name defaults to the Responses type), not reject.
  const fileSearch = responsesRequestToUnified({
    model: "m",
    input: "x",
    tools: [{ type: "file_search" }],
  });
  assert.equal(fileSearch.tools?.[0]?.type, "function");
  assert.equal(fileSearch.tools?.[0]?.function?.name, "file_search");
  await expectReject(
    () => responsesRequestToUnified({ model: "m" }),
    "invalid_input"
  );
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: [{ type: "input_file", file_id: "file_1" }],
      }),
    "unsupported_input_item"
  );
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: [
          {
            type: "message",
            role: "user",
            content: [{ type: "input_image", file_id: "file_1" }],
          },
        ],
      }),
    "unsupported_file_id"
  );
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: "x",
        tool_choice: { type: "allowed_tools", tools: [] },
      }),
    "unsupported_tool_choice"
  );
}

async function testReasoningAndTools() {
  const unified = responsesRequestToUnified({
    model: "m",
    input: "think",
    reasoning: { effort: "high" },
    tools: [
      {
        type: "function",
        name: "Read",
        description: "read",
        parameters: { type: "object", properties: {} },
      },
      { type: "web_search" },
    ],
    tool_choice: { type: "function", name: "Read" },
    prompt_cache_key: "opaque-key",
  });
  assert.equal(unified.reasoning?.effort, "high");
  assert.ok(
    unified.tools?.some((t: any) => t.function?.name === "Read")
  );
  // Hosted Responses tools (web_search) become Unified function tools.
  const webSearch = unified.tools?.find(
    (t: any) => t.function?.name === "web_search"
  );
  assert.ok(webSearch);
  assert.equal(webSearch.type, "function");
  assert.equal((unified as any).prompt_cache_key, "opaque-key");
  assert.equal(
    typeof unified.tool_choice === "object" &&
      unified.tool_choice &&
      "function" in unified.tool_choice
      ? (unified.tool_choice as any).function.name
      : null,
    "Read"
  );
}

async function testCustomHostedToolConvertsToFunction() {
  // Codex-hosted tools (MCP / plugin) arrive as `type: "custom"`. They are not
  // dropped: they project onto a function tool so the model can call them and
  // the client executes them by name.
  const customToolNames = new Set<string>();
  const unified = responsesRequestToUnified(
    {
      model: "m",
      input: "use the tool",
      tools: [
        {
          type: "custom",
          name: "search_code",
          description: "Search the codebase",
        },
      ],
    },
    createCallIdMap(),
    customToolNames
  );

  assert.ok(Array.isArray(unified.tools));
  assert.equal(unified.tools.length, 1);
  const converted = unified.tools[0];
  assert.equal(converted.type, "function");
  assert.equal(converted.function.name, "search_code");
  assert.equal(converted.function.description, "Search the codebase");
  assert.deepEqual(converted.function.parameters, {
    type: "object",
    properties: {
      input: {
        type: "string",
        description: "The freeform text/code input for this tool.",
      },
    },
    required: ["input"],
  });
  assert.ok(customToolNames.has("search_code"));

  // Responses custom tools are freeform regardless of incidental schema-like
  // fields: the Chat projection always uses the synthetic string carrier.
  const viaSchema = responsesRequestToUnified({
    model: "m",
    input: "x",
    tools: [
      {
        type: "custom",
        name: "list_files",
        schema: { type: "object", properties: { dir: { type: "string" } } },
      },
    ],
  });
  const listed = viaSchema.tools?.[0] as any;
  assert.equal(listed.function.name, "list_files");
  assert.deepEqual(listed.function.parameters.required, ["input"]);

  // A custom tool missing a name is still rejected rather than silently lost.
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: "x",
        tools: [{ type: "custom" }],
      }),
    "invalid_tool"
  );
}

async function testJsonOutput() {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = { debug() {} };
  const map = createCallIdMap();
  const context = { responsesCallIdMap: map } as any;

  const chat = {
    id: "chatcmpl-1",
    object: "chat.completion",
    created: 1,
    model: "gpt-4o",
    choices: [
      {
        index: 0,
        finish_reason: "stop",
        message: { role: "assistant", content: "hello" },
      },
    ],
    usage: {
      prompt_tokens: 2,
      completion_tokens: 1,
      total_tokens: 3,
      prompt_tokens_details: { cached_tokens: 1, cache_write_tokens: 0 },
    },
  };

  const out = await tf.transformResponseIn(
    new Response(JSON.stringify(chat), {
      headers: { "Content-Type": "application/json" },
    }),
    context
  );
  const json = await out.json();
  assert.equal(json.object, "response");
  assert.equal(json.status, "completed");
  assert.ok(json.output.some((i: any) => i.type === "message"));
  assert.equal(json.output[0].content[0].type, "output_text");
  assert.equal(json.output[0].content[0].text, "hello");
  assert.equal(json.usage.input_tokens, 2);
  assert.equal(json.usage.input_tokens_details.cached_tokens, 1);

  // Direct helper path with tools
  const withTools = unifiedResponseToResponses(
    {
      id: "chatcmpl-2",
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
                id: "call_1",
                type: "function",
                function: { name: "Read", arguments: "{}" },
              },
            ],
          },
        },
      ],
    },
    { callIdMap: map }
  );
  assert.ok(withTools.output.some((i: any) => i.type === "function_call"));

  const custom = unifiedResponseToResponses(
    {
      id: "chatcmpl-custom",
      choices: [
        {
          finish_reason: "tool_calls",
          message: {
            tool_calls: [
              {
                id: "call_exec",
                type: "function",
                function: {
                  name: "exec",
                  arguments: JSON.stringify({ input: "await tools.apply_patch(...)" }),
                },
              },
            ],
          },
        },
      ],
    },
    { callIdMap: map, customToolNames: new Set(["exec"]) }
  );
  assert.equal(custom.output[0].type, "custom_tool_call");
  assert.equal(custom.output[0].name, "exec");
  assert.equal(custom.output[0].input, "await tools.apply_patch(...)");
  assert.equal(custom.output[0].arguments, undefined);
}

async function testCustomToolSseLifecycle() {
  const state = createResponsesStreamState({
    model: "gpt-4o",
    customToolNames: new Set(["exec"]),
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-custom-stream",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_exec",
                  function: {
                    name: "exec",
                    arguments: '{"input":"echo hello"}',
                  },
                },
              ],
            },
          },
        ],
      },
      state
    ),
    ...finalizeResponsesStream(state),
  ];

  const added = events.find(
    (event) => event.type === "response.output_item.added"
  );
  assert.equal(added.item.type, "custom_tool_call");
  const delta = events.find(
    (event) => event.type === "response.custom_tool_call_input.delta"
  );
  assert.equal(delta.delta, "echo hello");
  const done = events.find(
    (event) => event.type === "response.custom_tool_call_input.done"
  );
  assert.equal(done.input, "echo hello");
  const itemDone = events.find(
    (event) => event.type === "response.output_item.done"
  );
  assert.equal(itemDone.item.type, "custom_tool_call");
  assert.equal(itemDone.item.input, "echo hello");
  assert.ok(
    !events.some(
      (event) => event.type === "response.function_call_arguments.delta"
    )
  );
}

async function testCustomToolHistoryInput() {
  const customToolNames = new Set<string>();
  const unified = responsesRequestToUnified(
    {
      model: "m",
      input: [
        {
          type: "custom_tool_call",
          call_id: "call_exec",
          name: "exec",
          input: "echo hello",
        },
        {
          type: "custom_tool_call_output",
          call_id: "call_exec",
          output: "hello",
        },
      ],
      tools: [{ type: "custom", name: "exec" }],
    },
    createCallIdMap(),
    customToolNames
  );
  assert.equal((unified.messages[0] as any).tool_calls[0].function.name, "exec");
  assert.equal(
    (unified.messages[0] as any).tool_calls[0].function.arguments,
    '{"input":"echo hello"}'
  );
  assert.equal(unified.messages[1].role, "tool");
  assert.equal(unified.messages[1].content, "hello");

  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: [
          {
            type: "custom_tool_call",
            call_id: "call_bad",
            name: "exec",
            input: { command: "echo hello" },
          },
        ],
      }),
    "invalid_custom_tool_input"
  );
}

async function testExactTextSseLifecycle() {
  const state = createResponsesStreamState({ model: "gpt-4o" });
  const events: any[] = [];

  events.push(
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-s",
        object: "chat.completion.chunk",
        model: "gpt-4o",
        choices: [
          {
            index: 0,
            delta: { content: "Hel" },
            finish_reason: null,
          },
        ],
      },
      state
    )
  );
  events.push(
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-s",
        object: "chat.completion.chunk",
        choices: [
          {
            index: 0,
            delta: { content: "lo" },
            finish_reason: null,
          },
        ],
      },
      state
    )
  );
  events.push(
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-s",
        object: "chat.completion.chunk",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
      state
    )
  );
  events.push(...finalizeResponsesStream(state));

  const types = events.map((e) => e.type);
  assert.deepEqual(types, [
    "response.created",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
  ]);

  // Codex failure mode: missing content_part events cause text loss.
  assert.ok(types.includes("response.content_part.added"));
  assert.ok(types.includes("response.content_part.done"));
  assert.ok(!types.includes("response.text.done"));

  const completed = events.find((e) => e.type === "response.completed");
  assert.ok(completed.response.output[0].content[0].text === "Hello");
  assert.equal(completed.response.usage.total_tokens, 2);
  assert.deepEqual(
    events.map((event) => event.sequence_number),
    events.map((_, index) => index)
  );
  assert.deepEqual(
    events.find((e) => e.type === "response.output_text.done").logprobs,
    []
  );
}

async function testToolSseLifecycle() {
  const state = createResponsesStreamState({ model: "gpt-4o" });
  const events: any[] = [];

  events.push(
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-t",
        object: "chat.completion.chunk",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_abc",
                  type: "function",
                  function: { name: "Read", arguments: "" },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      state
    )
  );
  events.push(
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-t",
        object: "chat.completion.chunk",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: '{"p":"' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      state
    )
  );
  events.push(
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-t",
        object: "chat.completion.chunk",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: 0,
                  function: { arguments: 'a.ts"}' },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      },
      state
    )
  );
  events.push(
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-t",
        object: "chat.completion.chunk",
        choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
      },
      state
    )
  );
  events.push(...finalizeResponsesStream(state));

  const types = events.map((e) => e.type);
  assert.ok(types.includes("response.created"));
  assert.ok(types.includes("response.output_item.added"));
  assert.ok(types.includes("response.function_call_arguments.delta"));
  assert.ok(types.includes("response.function_call_arguments.done"));
  assert.ok(types.includes("response.output_item.done"));
  assert.ok(types.includes("response.completed"));

  const doneArgs = events.find(
    (e) => e.type === "response.function_call_arguments.done"
  );
  assert.equal(doneArgs.arguments, '{"p":"a.ts"}');
  assert.equal(doneArgs.name, "Read");

  const completed = events.find((e) => e.type === "response.completed");
  assert.ok(
    completed.response.output.some((i: any) => i.type === "function_call")
  );
}

async function testOutputIndicesStayStableWhenTextFollowsTool() {
  const state = createResponsesStreamState({ model: "gpt-4o" });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-order",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_first",
                  function: { name: "Read", arguments: "{}" },
                },
              ],
            },
          },
        ],
      },
      state
    ),
    ...unifiedChunkToResponsesEvents(
      { choices: [{ delta: { content: "after" }, finish_reason: null }] },
      state
    ),
    ...finalizeResponsesStream(state),
  ];

  const toolAdded = events.find(
    (event) =>
      event.type === "response.output_item.added" &&
      event.item.type === "function_call"
  );
  const toolDone = events.find(
    (event) =>
      event.type === "response.output_item.done" &&
      event.item.type === "function_call"
  );
  const textAdded = events.find(
    (event) =>
      event.type === "response.output_item.added" &&
      event.item.type === "message"
  );
  assert.equal(toolAdded.output_index, 0);
  assert.equal(toolDone.output_index, toolAdded.output_index);
  assert.equal(textAdded.output_index, 1);

  const completed = events.at(-1).response;
  assert.equal(completed.output[0].type, "function_call");
  assert.equal(completed.output[1].type, "message");
}

async function testTransformerStreamIntegration() {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = { debug() {} };

  const sse = [
    'data: {"id":"chatcmpl-i","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
    "",
    'data: {"id":"chatcmpl-i","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
    "",
    "data: [DONE]",
    "",
  ].join("\n");

  const out = await tf.transformResponseIn(
    new Response(sse, {
      headers: { "Content-Type": "text/event-stream" },
    }),
    { responsesCallIdMap: createCallIdMap() } as any
  );

  const text = await out.text();
  assert.ok(!text.includes("[DONE]"));
  const events = parseSseEvents(text);
  const types = events.map((e) => e.type);
  assert.ok(types.includes("response.content_part.added"));
  assert.ok(types.includes("response.content_part.done"));
  assert.ok(types.includes("response.completed"));
  assert.equal(types[types.length - 1], "response.completed");
}

async function testSeparateUsageChunkIsRetained() {
  const tf = new OpenAIResponsesTransformer();
  const sse = [
    'data: {"id":"chatcmpl-u","object":"chat.completion.chunk","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
    "",
    'data: {"id":"chatcmpl-u","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    "",
    'data: {"id":"chatcmpl-u","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}',
    "",
    "data: [DONE]",
    "",
  ].join("\n");

  const out = await tf.transformResponseIn(
    new Response(sse, {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const events = parseSseEvents(await out.text());
  const completed = events.find((event) => event.type === "response.completed");
  assert.equal(completed.response.usage.total_tokens, 9);
}

async function testMalformedStreamBecomesFailedEvent() {
  const tf = new OpenAIResponsesTransformer();
  const out = await tf.transformResponseIn(
    new Response("data: {not-json}\n\n", {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const events = parseSseEvents(await out.text());
  assert.equal(events.at(-1).type, "response.failed");
}

async function testUpstreamStreamErrorBecomesFailedEvent() {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = { error() {} };
  const encoder = new TextEncoder();
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encoder.encode(
          'data: {"id":"chatcmpl-e","choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        )
      );
      controller.error(new Error("upstream socket broke Bearer secret-token"));
    },
  });
  const out = await tf.transformResponseIn(
    new Response(upstream, {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const events = parseSseEvents(await out.text());
  assert.equal(events.at(-1).type, "response.failed");
  assert.ok(!events.at(-1).response.error.message.includes("secret-token"));
  assert.ok(!events.some((event) => event.type === "response.completed"));
}

async function testCallIdMapPersistsInProtocolContext() {
  const tf = new OpenAIResponsesTransformer();
  const protocolContext: any = {};
  await tf.transformRequestOut(
    { model: "m", input: "hello" },
    { protocolContext } as any
  );
  assert.ok(protocolContext.responsesCallIdMap);
}

async function testClientTransformRequestOut() {
  const tf = new OpenAIResponsesTransformer();
  const context: any = {};
  const unified = await tf.transformRequestOut(
    {
      model: "openai,gpt-4o",
      instructions: "sys",
      input: [{ type: "message", role: "user", content: "hi" }],
      reasoning: { effort: "medium" },
      tools: [{ type: "custom", name: "exec" }],
    },
    context
  );
  assert.equal(unified.model, "openai,gpt-4o");
  assert.ok(unified.messages.some((m: any) => m.role === "system"));
  assert.equal(unified.reasoning?.effort, "medium");
  assert.ok(context.responsesCustomToolNames.has("exec"));

  const out = await tf.transformResponseIn(
    new Response(
      JSON.stringify({
        id: "chatcmpl-context-custom",
        choices: [
          {
            message: {
              tool_calls: [
                {
                  id: "call_exec",
                  type: "function",
                  function: {
                    name: "exec",
                    arguments: '{"input":"echo hello"}',
                  },
                },
              ],
            },
          },
        ],
      }),
      { headers: { "Content-Type": "application/json" } }
    ),
    context
  );
  const json = await out.json();
  assert.equal(json.output[0].type, "custom_tool_call");
  assert.equal(json.output[0].input, "echo hello");
}

async function main() {
  await testStringAndMessageInput();
  await testFunctionCallRoundTrip();
  await testParallelFunctionCallsCoalesce();
  await testParallelCustomToolCallsCoalesce();
  await testSequentialToolTurnsDoNotMerge();
  await testUnsupportedState();
  await testReasoningAndTools();
  await testCustomHostedToolConvertsToFunction();
  await testJsonOutput();
  await testCustomToolSseLifecycle();
  await testCustomToolHistoryInput();
  await testExactTextSseLifecycle();
  await testToolSseLifecycle();
  await testOutputIndicesStayStableWhenTextFollowsTool();
  await testTransformerStreamIntegration();
  await testSeparateUsageChunkIsRetained();
  await testMalformedStreamBecomesFailedEvent();
  await testUpstreamStreamErrorBecomesFailedEvent();
  await testCallIdMapPersistsInProtocolContext();
  await testClientTransformRequestOut();
  console.log("openai.inbound-responses: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

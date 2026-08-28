/**
 * Inbound OpenAI Responses: request normalization, unsupported state,
 * JSON output, exact text/tool SSE lifecycle (including content_part),
 * usage, errors, and call-id mapping.
 */
import assert from "node:assert/strict";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import {
  CUSTOM_TOOL_INPUT_KEY,
  createCallIdMap,
  createResponsesStreamState,
  finalizeResponsesStream,
  responsesReasoningItemFromThinking,
  responsesRequestToUnified,
  responsesTextFormatFromResponseFormat,
  unifiedChunkToResponsesEvents,
  unifiedResponseToResponses,
  uniquifyReasoningItemIds,
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

function assertSingleToolTurn(
  unified: { messages: any[] },
  content: string,
  callIds: string[],
  outputs: string[]
) {
  const assistants = unified.messages.filter(
    (m: any) => m.role === "assistant"
  );
  assert.equal(
    assistants.length,
    1,
    "same-turn text + tools must be one assistant"
  );
  const assistant = assistants[0]!;
  assert.equal(assistant.content, content);
  assert.ok(assistant.tool_calls);
  assert.equal(assistant.tool_calls.length, callIds.length);
  assert.deepEqual(
    assistant.tool_calls.map((t: any) => t.id),
    callIds
  );
  const tools = unified.messages.filter((m: any) => m.role === "tool");
  assert.equal(tools.length, outputs.length);
  assert.deepEqual(
    tools.map((t: any) => t.tool_call_id),
    callIds
  );
  assert.deepEqual(
    tools.map((t: any) => t.content),
    outputs
  );
}

async function assertWireKeepsToolPair(
  unified: any,
  callIds: string[]
) {
  const tf = new OpenAIResponsesTransformer();
  const out = await tf.transformRequestIn(unified, {}, {});
  const input = (out as any).input as any[];
  const calls = input.filter((item) => item.type === "function_call");
  const outputs = input.filter((item) => item.type === "function_call_output");
  assert.equal(calls.length, callIds.length);
  assert.equal(outputs.length, callIds.length);
  assert.deepEqual(
    calls.map((item) => item.call_id),
    callIds
  );
  assert.deepEqual(
    outputs.map((item) => item.call_id),
    callIds
  );
}

/**
 * Some clients (and Chat→Responses encoders) emit function_call before the
 * same-turn assistant text. That is valid Responses; items pair by call_id.
 * Inbound must still produce one Chat assistant so validateOpenAIToolCalls
 * does not strip the tool history.
 */
async function testFunctionCallThenAssistantTextCoalesce() {
  const unified = responsesRequestToUnified({
    model: "m",
    input: [
      {
        type: "function_call",
        call_id: "call_1",
        name: "Read",
        arguments: '{"path":"TODO.md"}',
      },
      { role: "assistant", content: "I'll read TODO.md" },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: "# TODO",
      },
    ],
  });
  assertSingleToolTurn(unified, "I'll read TODO.md", ["call_1"], ["# TODO"]);
  await assertWireKeepsToolPair(unified, ["call_1"]);
}

async function testAssistantTextThenFunctionCallCoalesce() {
  const unified = responsesRequestToUnified({
    model: "m",
    input: [
      { role: "assistant", content: "I'll read TODO.md" },
      {
        type: "function_call",
        call_id: "call_1",
        name: "Read",
        arguments: '{"path":"TODO.md"}',
      },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: "# TODO",
      },
    ],
  });
  assertSingleToolTurn(unified, "I'll read TODO.md", ["call_1"], ["# TODO"]);
  await assertWireKeepsToolPair(unified, ["call_1"]);
}

async function testAssistantTextThenParallelFunctionCallsCoalesce() {
  const unified = responsesRequestToUnified({
    model: "m",
    input: [
      { role: "assistant", content: "checking" },
      {
        type: "function_call",
        call_id: "call_A",
        name: "Read",
        arguments: "{}",
      },
      {
        type: "function_call",
        call_id: "call_B",
        name: "Grep",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: "call_A",
        output: "a",
      },
      {
        type: "function_call_output",
        call_id: "call_B",
        output: "b",
      },
    ],
  });
  assertSingleToolTurn(unified, "checking", ["call_A", "call_B"], ["a", "b"]);
  await assertWireKeepsToolPair(unified, ["call_A", "call_B"]);
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

async function testResponseFormatConversion() {
  // text.format: "text" (or absent) is the default — no response_format set.
  const plain = responsesRequestToUnified({
    model: "m",
    input: "x",
    text: { format: { type: "text" } },
  });
  assert.equal((plain as any).response_format, undefined);

  const noText = responsesRequestToUnified({ model: "m", input: "x" });
  assert.equal((noText as any).response_format, undefined);

  // json_schema converts into the Chat Completions response_format shape —
  // Unified is Chat-Completions-shaped, so this is the verified mapping
  // Responses destinations later reconstruct back into a native text.format.
  const schema = { type: "object", properties: { ok: { type: "boolean" } } };
  const jsonSchema = responsesRequestToUnified({
    model: "m",
    input: "x",
    text: {
      format: { type: "json_schema", name: "result", schema, strict: true },
    },
  });
  assert.deepEqual((jsonSchema as any).response_format, {
    type: "json_schema",
    json_schema: { name: "result", schema, strict: true },
  });

  const jsonObject = responsesRequestToUnified({
    model: "m",
    input: "x",
    text: { format: { type: "json_object" } },
  });
  assert.deepEqual((jsonObject as any).response_format, {
    type: "json_object",
  });

  // Genuinely unverified format types still reject rather than risk a wrong
  // conversion (e.g. a tool-grammar-shaped "grammar" type at the top-level
  // text.format, which is not a real Responses field but must not be
  // silently accepted either).
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "m",
        input: "x",
        text: { format: { type: "grammar" } },
      }),
    "unsupported_response_format"
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

async function testClientIncludeAndStorePassThrough() {
  // Responses→Responses: client include / store:false survive Unified and
  // outbound rebuild. Chat/Anthropic never send these; do not invent them.
  const unified = responsesRequestToUnified({
    model: "muse-spark",
    input: "think",
    reasoning: { effort: "high", summary: "auto" },
    store: false,
    include: ["reasoning.encrypted_content", 1, "", "file_search_call.results"],
  });
  assert.deepEqual(unified.include, [
    "reasoning.encrypted_content",
    "file_search_call.results",
  ]);
  assert.equal(unified.store, false);

  const tf = new OpenAIResponsesTransformer();
  const out = await tf.transformRequestIn(unified, {}, {});
  assert.deepEqual((out as any).include, [
    "reasoning.encrypted_content",
    "file_search_call.results",
  ]);
  assert.equal((out as any).store, false);
  assert.equal((out as any).reasoning?.summary, "auto");

  const fromChat = await tf.transformRequestIn(
    {
      model: "muse-spark",
      messages: [{ role: "user", content: "hi" }],
      reasoning: { effort: "high", summary: "auto" },
    } as any,
    {},
    {}
  );
  assert.equal((fromChat as any).include, undefined);
  assert.equal((fromChat as any).store, undefined);
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
  // The original description is preserved verbatim, followed by explicit
  // JSON-wrapping guidance for models unfamiliar with the custom-tool proxy
  // convention (see normalizeResponsesTools's comment).
  assert.ok(converted.function.description.startsWith("Search the codebase"));
  assert.match(converted.function.description, /single "input" argument/);
  assert.match(converted.function.description, /shell heredoc/);
  assert.deepEqual(converted.function.parameters, {
    type: "object",
    properties: {
      input: {
        type: "string",
        description:
          "The complete freeform text/code input for this tool, exactly as described in the tool description — as a plain string, with no markdown code fences or shell heredoc wrapping.",
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

// Grok reaches for the trailing-asterisk patch markers (Claude Code dialect)
// when driving Codex's exec → apply_patch; Codex only accepts the exact
// `*** Begin Patch` / `*** End Patch` tokens. The client-facing emission must
// normalize the variant so the first write is not rejected.
async function testCodexPatchMarkersNormalizedOnStream() {
  const state = createResponsesStreamState({
    model: "grok-4.6",
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-patch",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_patch",
                  function: {
                    name: "exec",
                    arguments:
                      '{"input":"await tools.apply_patch(`*** Begin Patch ***\\n*** Update File: demo.txt\\n@@\\n-old\\n+new\\n*** End Patch ***\\n`);"}',
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

  const done = events.find(
    (event) => event.type === "response.function_call_arguments.done"
  );
  assert.ok(
    !done.arguments.includes("*** Begin Patch ***"),
    "no trailing-asterisk Begin"
  );
  assert.ok(done.arguments.includes("*** Begin Patch"), "Begin marker present");
  assert.ok(
    !done.arguments.includes("*** End Patch ***"),
    "no trailing-asterisk End"
  );
  assert.ok(done.arguments.includes("*** End Patch"), "End marker present");

  const itemDone = events.find(
    (event) => event.type === "response.output_item.done"
  );
  assert.equal(itemDone.item.name, "exec");
  assert.ok(
    !itemDone.item.arguments.includes("*** Begin Patch ***"),
    "output_item.done normalized"
  );

  const completed = events.find(
    (event) => event.type === "response.completed"
  );
  const [call] = completed.response.output;
  assert.ok(
    !call.arguments.includes("*** Begin Patch ***"),
    "response.completed normalized"
  );
}

async function testCodexPatchMarkersNormalizedOnJson() {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = { debug() {} };
  const out = await tf.transformResponseIn(
    new Response(
      JSON.stringify({
        id: "chatcmpl-json-patch",
        object: "chat.completion",
        model: "grok-4.6",
        created: 1,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "Applying patch.",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: {
                    name: "exec",
                    arguments:
                      '{"input":"await tools.apply_patch(`*** Begin Patch ***\\n*** Add File: a.txt\\n+line\\n*** End Patch ***\\n`);"}',
                  },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      }),
      { headers: { "Content-Type": "application/json" } }
    ),
    {
      protocolContext: { originalModel: "grok-4.6" },
    }
  );

  const json = await out.json();
  const [call] = json.output.filter(
    (item: any) => item.type === "function_call"
  );
  assert.ok(call.arguments.includes("*** Begin Patch"), "JSON Begin present");
  assert.ok(
    !call.arguments.includes("*** Begin Patch ***"),
    "JSON Begin no trailing asterisks"
  );
  assert.ok(
    !call.arguments.includes("*** End Patch ***"),
    "JSON End no trailing asterisks"
  );
}

// Grok sometimes fills Codex's freeform `exec` (raw JS) with a JSON
// shell-envelope (`{"cmd": "ls"}`) instead of JS; the client runs it as JS and
// fails with SyntaxError. The client-facing custom_tool_call input must become
// the `await tools.exec_command({cmd: …})` call the model retries with anyway.
async function testExecShellEnvelopeNormalizedOnStream() {
  const state = createResponsesStreamState({
    model: "grok-4.6",
    customToolNames: new Set(["exec"]),
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-exec",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_exec",
                  function: {
                    name: "exec",
                    arguments:
                      '{"input":"{\\"cmd\\": \\"ls -la /tmp\\"}"}',
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

  const delta = events.find(
    (event) => event.type === "response.custom_tool_call_input.delta"
  );
  assert.equal(
    delta.delta,
    'await tools.exec_command({"cmd":"ls -la /tmp"});'
  );
  const done = events.find(
    (event) => event.type === "response.custom_tool_call_input.done"
  );
  assert.equal(done.input, 'await tools.exec_command({"cmd":"ls -la /tmp"});');
  const itemDone = events.find(
    (event) => event.type === "response.output_item.done"
  );
  assert.equal(itemDone.item.name, "exec");
  assert.ok(
    itemDone.item.input.startsWith("await tools.exec_command("),
    "output_item.done input rewritten"
  );
  // Non-cmd freeform content is untouched.
  const plain = createResponsesStreamState({
    model: "grok-4.6",
    customToolNames: new Set(["exec"]),
  });
  const plainEvents = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-exec2",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_exec2",
                  function: {
                    name: "exec",
                    arguments:
                      '{"input":"await tools.exec_command({cmd: \\"echo hi\\"});"}',
                  },
                },
              ],
            },
          },
        ],
      },
      plain
    ),
    ...finalizeResponsesStream(plain),
  ];
  const plainDone = plainEvents.find(
    (event) => event.type === "response.custom_tool_call_input.done"
  );
  assert.equal(plainDone.input, 'await tools.exec_command({cmd: "echo hi"});');
}

// A generic Responses destination with a tool named `exec` that legitimately
// accepts `{"cmd": …}` must not be rewritten into Codex V8 isolate JS.
async function testExecShellEnvelopeUnmodifiedWithoutCodexConventions() {
  const envelope = '{"cmd": "ls -la /tmp"}';
  const json = unifiedResponseToResponses(
    {
      id: "chatcmpl-exec-generic",
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
                  arguments: JSON.stringify({ input: envelope }),
                },
              },
            ],
          },
        },
      ],
    },
    {
      customToolNames: new Set(["exec"]),
      codexIsolateConventions: false,
    }
  );
  assert.equal(json.output[0].type, "custom_tool_call");
  assert.equal(json.output[0].input, envelope);

  const state = createResponsesStreamState({
    model: "gpt-4o",
    customToolNames: new Set(["exec"]),
    codexIsolateConventions: false,
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-exec-generic-stream",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_exec",
                  function: {
                    name: "exec",
                    arguments: JSON.stringify({ input: envelope }),
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
  const done = events.find(
    (event) => event.type === "response.custom_tool_call_input.done"
  );
  assert.equal(done.input, envelope);
}

async function testResponsesTextFormatHelper() {
  assert.deepEqual(
    responsesTextFormatFromResponseFormat({
      type: "json_schema",
      json_schema: {
        name: "result",
        schema: { type: "object", properties: { ok: { type: "boolean" } } },
        strict: true,
      },
    }),
    {
      type: "json_schema",
      name: "result",
      schema: { type: "object", properties: { ok: { type: "boolean" } } },
      strict: true,
    }
  );
  assert.deepEqual(
    responsesTextFormatFromResponseFormat({ type: "json_object" }),
    { type: "json_object" }
  );
  assert.equal(responsesTextFormatFromResponseFormat(undefined), undefined);
}

// Grok alternates the shell-envelope key between `cmd` and `command`; both
// must land on exec_command's accepted `cmd` key.
async function testExecCommandKeyVariantNormalized() {
  const state = createResponsesStreamState({
    model: "grok-4.6",
    customToolNames: new Set(["exec"]),
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-exec-command",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_execc",
                  function: {
                    name: "exec",
                    arguments:
                      '{"input":"{\\"command\\": \\"ls -la /tmp\\"}"}',
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
  const done = events.find(
    (event) => event.type === "response.custom_tool_call_input.done"
  );
  assert.equal(
    done.input,
    'await tools.exec_command({"cmd":"ls -la /tmp"});'
  );
}

// Grok sometimes invokes the patch tool through a shell heredoc instead of JS:
//   apply_patch << 'PATCH' … PATCH
// That is not valid JS for Codex's exec, so rewrite it into the JS call form.
async function testExecApplyPatchHeredocNormalized() {
  const state = createResponsesStreamState({
    model: "grok-4.6",
    customToolNames: new Set(["exec"]),
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-exec-heredoc",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_exech",
                  function: {
                    name: "exec",
                    arguments: JSON.stringify({
                      input:
                        "apply_patch << 'PATCH'\n*** Begin Patch\n*** Add File: a.txt\n+line 1\n+line 2\n*** End Patch\nPATCH",
                    }),
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
  const done = events.find(
    (event) => event.type === "response.custom_tool_call_input.done"
  );
  assert.equal(
    done.input,
    'await tools.apply_patch("*** Begin Patch\\n*** Add File: a.txt\\n+line 1\\n+line 2\\n*** End Patch");'
  );
  // The marker normalization must not fire on the already-canonical result.
  assert.ok(!done.input.includes("*** Begin Patch ***"));
}

// When `exec` is declared as a plain function tool its arguments stay wrapped
// in {"input": "…"}. The envelope must be normalized inside the wrapper even
// without the custom-tool unwrap path.
async function testExecFunctionToolWrapperNormalized() {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = { debug() {} };
  const out = await tf.transformResponseIn(
    new Response(
      JSON.stringify({
        id: "chatcmpl-fn-exec",
        object: "chat.completion",
        model: "grok-4.6",
        created: 1,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "running",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: {
                    name: "exec",
                    arguments:
                      '{"input":"{\\"cmd\\": \\"ls /tmp\\"}"}',
                  },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      }),
      { headers: { "Content-Type": "application/json" } }
    ),
    {
      protocolContext: { originalModel: "grok-4.6" },
      req: { protocolContext: { originalModel: "grok-4.6" } },
    }
  );
  const json = await out.json();
  const [call] = json.output.filter(
    (item: any) => item.type === "function_call"
  );
  assert.equal(
    call.arguments,
    '{"input":"await tools.exec_command({\\"cmd\\":\\"ls /tmp\\"});"}'
  );
}

// Flatten is provider-neutral: `{input}` contract only. Codex V8 calling
// conventions stay off the description so Anthropic / Chat Completions are
// not prompted as if they were Codex's isolate.
async function testExecToolDescriptionIsProviderNeutral() {
  const customToolNames = new Set<string>();
  const unified = responsesRequestToUnified(
    {
      model: "openai,gpt-4o",
      input: [
        {
          role: "user",
          content: "list the dir",
        },
      ],
      tools: [
        {
          type: "custom",
          name: "exec",
          description: "Run JavaScript code to orchestrate tool calls.",
        },
      ],
    },
    undefined,
    customToolNames
  );
  assert.ok(customToolNames.has("exec"));
  const tool = unified.tools?.find((t: any) => t.function?.name === "exec");
  assert.ok(tool);
  const description = tool.function!.description;
  assert.ok(description.includes("freeform text"), "keeps the input contract");
  assert.ok(description.includes(`"${CUSTOM_TOOL_INPUT_KEY}"`), "names the wrapper key");
  assert.ok(!description.includes("exec_command"), "no Codex V8 exec_command coaching");
  assert.ok(!description.includes("await tools."), "no Codex V8 JS coaching");
  assert.ok(!description.includes("Calling conventions"), "no destination-specific conventions");
}

async function testExecShellEnvelopeNormalizedOnJson() {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = { debug() {} };
  const map = createCallIdMap();
  const customToolNames = new Set(["exec"]);
  const out = await tf.transformResponseIn(
    new Response(
      JSON.stringify({
        id: "chatcmpl-json-exec",
        object: "chat.completion",
        model: "grok-4.6",
        created: 1,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: "running",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: {
                    name: "exec",
                    arguments:
                      '{"input":"{\\"cmd\\": \\"ls -la /tmp\\"}"}',
                  },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      }),
      { headers: { "Content-Type": "application/json" } }
    ),
    {
      protocolContext: { originalModel: "grok-4.6" },
      responsesCallIdMap: map,
      responsesCustomToolNames: customToolNames,
      req: { protocolContext: { originalModel: "grok-4.6" } },
    }
  );
  const json = await out.json();
  const [call] = json.output.filter(
    (item: any) => item.type === "custom_tool_call"
  );
  assert.equal(
    call.input,
    'await tools.exec_command({"cmd":"ls -la /tmp"});'
  );
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

async function testCustomToolHeredocUnwrap() {
  const map = createCallIdMap();
  const heredocPatch =
    '*** Begin Patch\n*** Add File: hello.txt\n+hello world\n*** End Patch';

  // Non-streaming: a model that invokes the freeform tool shell-style
  // (`apply_patch <<EOF ... EOF`) instead of passing the freeform text
  // directly gets unwrapped down to the clean text, not left heredoc-wrapped.
  const wrapped = unifiedResponseToResponses(
    {
      id: "chatcmpl-heredoc",
      choices: [
        {
          finish_reason: "tool_calls",
          message: {
            tool_calls: [
              {
                id: "call_patch",
                type: "function",
                function: {
                  name: "apply_patch",
                  arguments: JSON.stringify({
                    input: `<<EOF\n${heredocPatch}\nEOF`,
                  }),
                },
              },
            ],
          },
        },
      ],
    },
    { callIdMap: map, customToolNames: new Set(["apply_patch"]) }
  );
  assert.equal(wrapped.output[0].type, "custom_tool_call");
  assert.equal(wrapped.output[0].input, heredocPatch);

  // Quoted heredoc delimiter and a leading `cat` both unwrap the same way.
  const quotedCat = unifiedResponseToResponses(
    {
      id: "chatcmpl-heredoc-2",
      choices: [
        {
          finish_reason: "tool_calls",
          message: {
            tool_calls: [
              {
                id: "call_patch_2",
                type: "function",
                function: {
                  name: "apply_patch",
                  arguments: JSON.stringify({
                    input: `cat <<'PATCH'\n${heredocPatch}\nPATCH`,
                  }),
                },
              },
            ],
          },
        },
      ],
    },
    { callIdMap: createCallIdMap(), customToolNames: new Set(["apply_patch"]) }
  );
  assert.equal(quotedCat.output[0].input, heredocPatch);

  // A genuine patch that merely contains "<<" or "EOF"-like text mid-body
  // (not wrapping the *entire* input) must pass through untouched — this is
  // a syntactic unwrap, not a content rewrite.
  const untouched = unifiedResponseToResponses(
    {
      id: "chatcmpl-heredoc-3",
      choices: [
        {
          finish_reason: "tool_calls",
          message: {
            tool_calls: [
              {
                id: "call_patch_3",
                type: "function",
                function: {
                  name: "apply_patch",
                  arguments: JSON.stringify({ input: heredocPatch }),
                },
              },
            ],
          },
        },
      ],
    },
    { callIdMap: createCallIdMap(), customToolNames: new Set(["apply_patch"]) }
  );
  assert.equal(untouched.output[0].input, heredocPatch);
}

async function testCustomToolHeredocUnwrapSse() {
  const state = createResponsesStreamState({
    model: "grok-4.6",
    customToolNames: new Set(["apply_patch"]),
  });
  const heredocPatch =
    '*** Begin Patch\n*** Add File: hello.txt\n+hello world\n*** End Patch';
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-heredoc-sse",
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_patch",
                  function: {
                    name: "apply_patch",
                    arguments: JSON.stringify({
                      input: `<<EOF\n${heredocPatch}\nEOF`,
                    }),
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

  const done = events.find(
    (event) => event.type === "response.custom_tool_call_input.done"
  );
  assert.equal(done.input, heredocPatch);
  const itemDone = events.find(
    (event) => event.type === "response.output_item.done"
  );
  assert.equal(itemDone.item.input, heredocPatch);
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

async function testTrailingTextAfterToolOpensNewMessageItem() {
  // Regression: closing the preamble message before a tool call must not
  // leave later text as output_text.delta against that already-done item.
  const state = createResponsesStreamState({
    model: "grok-4.6",
    customToolNames: new Set(["apply_patch"]),
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-trail",
        choices: [{ delta: { content: "I'll create the file." } }],
      },
      state
    ),
    ...unifiedChunkToResponsesEvents(
      {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_patch",
                  function: {
                    name: "apply_patch",
                    arguments: '{"input":"*** Begin Patch\\n*** End Patch"}',
                  },
                },
              ],
            },
          },
        ],
      },
      state
    ),
    ...unifiedChunkToResponsesEvents(
      { choices: [{ delta: { content: "Done" } }] },
      state
    ),
    ...finalizeResponsesStream(state),
  ];

  const messageAdds = events.filter(
    (event) =>
      event.type === "response.output_item.added" &&
      event.item?.type === "message"
  );
  assert.equal(messageAdds.length, 2);
  assert.equal(messageAdds[0].output_index, 0);
  assert.equal(messageAdds[1].output_index, 2);

  const trailingDeltas = events.filter(
    (event) =>
      event.type === "response.output_text.delta" && event.delta === "Done"
  );
  assert.equal(trailingDeltas.length, 1);
  assert.equal(trailingDeltas[0].item_id, messageAdds[1].item.id);
  assert.equal(trailingDeltas[0].output_index, 2);

  const completed = events.at(-1).response;
  assert.deepEqual(
    completed.output.map((item: any) => item.type),
    ["message", "custom_tool_call", "message"]
  );
  assert.equal(completed.output[0].content[0].text, "I'll create the file.");
  assert.equal(completed.output[2].content[0].text, "Done");
  assert.notEqual(completed.output[0].id, completed.output[2].id);
}

async function testTextItemClosesBeforeToolCallOpens() {
  // Regression: a text preamble followed by a tool call in the same
  // response must close the text item (in item-index order) before the
  // tool call's own added/delta/done sequence begins. Previously the text
  // item only closed at finalizeResponsesStream (end of stream), so the
  // tool call's `output_item.added` was emitted live while the text item
  // was still open — a client rendering items as they complete (e.g. the
  // Codex/ChatGPT UI) could fail to render the tool call as a distinct
  // block when items close out of their own start order.
  const state = createResponsesStreamState({
    model: "grok-4.6",
    customToolNames: new Set(["apply_patch"]),
  });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-preamble",
        choices: [{ delta: { content: "I'll create the file." } }],
      },
      state
    ),
    ...unifiedChunkToResponsesEvents(
      {
        choices: [
          {
            delta: {
              tool_calls: [
                {
                  index: 0,
                  id: "call_patch",
                  function: {
                    name: "apply_patch",
                    arguments: '{"input":"*** Begin Patch\\n*** End Patch"}',
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

  const types = events.map((e) => e.type);
  const textDoneIndex = types.indexOf("response.output_item.done");
  const toolAddedIndex = types.findIndex(
    (t, i) =>
      t === "response.output_item.added" &&
      (events[i] as any).item?.type === "custom_tool_call"
  );
  assert.ok(textDoneIndex !== -1, "text item must close");
  assert.ok(toolAddedIndex !== -1, "tool call must open");
  assert.ok(
    textDoneIndex < toolAddedIndex,
    `text item must close (index ${textDoneIndex}) before the tool call opens (index ${toolAddedIndex})`
  );

  const textItemDone = events[textDoneIndex] as any;
  assert.equal(textItemDone.item.type, "message");
  assert.equal(textItemDone.item.content[0].text, "I'll create the file.");

  const toolCallDone = events.find(
    (e) =>
      e.type === "response.output_item.done" &&
      (e as any).item?.type === "custom_tool_call"
  ) as any;
  assert.equal(toolCallDone.item.input, "*** Begin Patch\n*** End Patch");
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

async function testReasoningInputItemBecomesThinking() {
  const unified = responsesRequestToUnified({
    model: "m",
    input: [
      {
        type: "reasoning",
        id: "rs_1",
        summary: [{ type: "summary_text", text: "I should read the file." }],
        encrypted_content: "enc-sig",
      },
      {
        type: "function_call",
        call_id: "call_r",
        name: "Read",
        arguments: "{}",
      },
      { role: "user", content: "now continue" },
    ],
  });
  const assistant = unified.messages.find((m: any) => m.role === "assistant");
  assert.ok(assistant);
  assert.equal(assistant.thinking?.content, "I should read the file.");
  assert.equal(assistant.thinking?.encrypted_content, "enc-sig");
  assert.equal(assistant.thinking?.id, "rs_1");
  assert.equal(assistant.thinking?.signature, undefined);
  assert.equal(assistant.tool_calls?.[0]?.id, "call_r");
}

async function testThinkingRoundTripsToResponsesReasoningItem() {
  const responses = unifiedResponseToResponses({
    id: "chatcmpl-think",
    model: "gpt-4o",
    created: 1,
    choices: [
      {
        message: {
          role: "assistant",
          content: "done",
          thinking: {
            content: "plan first",
            encrypted_content: "enc-blob",
            id: "rs_plan",
          },
        },
      },
    ],
  });
  assert.equal(responses.output[0].type, "reasoning");
  assert.equal(responses.output[0].summary[0].text, "plan first");
  assert.equal(responses.output[0].encrypted_content, "enc-blob");
  assert.equal(responses.output[0].id, "rs_plan");
  assert.equal(responses.output[1].type, "message");
}

async function testThinkingHistoryIsEmittedOnResponsesRequest() {
  const tf = new OpenAIResponsesTransformer();
  const result = await tf.transformRequestIn(
    {
      model: "grok-4.6",
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: "ok",
          thinking: { content: "think aloud", signature: "sig-hist" },
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "Read", arguments: "{}" },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "file" },
      ],
    } as any,
    {},
    {}
  );
  const input = (result as any).input;
  const reasoning = input.find((item: any) => item.type === "reasoning");
  assert.ok(reasoning, "thinking history must become a reasoning item");
  assert.equal(reasoning.summary[0].text, "think aloud");
  // Anthropic/Chat signatures are not Codex ciphertext — omit encrypted_content.
  assert.equal(reasoning.encrypted_content, undefined);
  assert.ok(input.every((item: any) => !item.thinking));
  assert.ok(input.some((item: any) => item.type === "function_call"));
}

async function testPoisonedReasoningItemIdIsNotReplayedAsEncryptedContent() {
  const tf = new OpenAIResponsesTransformer();
  // Live Codex 400 shape: thinking.signature held a prior reasoning item id
  // (`rs_0fbe4a00…`), which was then copied into encrypted_content.
  const poisonedId = "rs_0fbe4a009e888583016a8258d857e081918ffc9d6f6f3e0d6c";
  const result = await tf.transformRequestIn(
    {
      model: "gpt-5.6-sol",
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: "ok",
          thinking: {
            content: "I should check the logs",
            signature: poisonedId,
          },
        },
      ],
    } as any,
    {},
    {}
  );
  const reasoning = (result as any).input.find((item: any) => item.type === "reasoning");
  assert.ok(reasoning);
  assert.equal(reasoning.summary[0].text, "I should check the logs");
  assert.equal(reasoning.id, poisonedId);
  assert.equal(reasoning.encrypted_content, undefined);
}

async function testChatReasoningContentHistoryBecomesResponsesReasoning() {
  const tf = new OpenAIResponsesTransformer();
  const result = await tf.transformRequestIn(
    {
      model: "grok-4.6",
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: "ok",
          reasoning_content: "plan first",
        },
      ],
    } as any,
    {},
    {}
  );
  const reasoning = (result as any).input.find((item: any) => item.type === "reasoning");
  assert.ok(reasoning);
  assert.equal(reasoning.summary[0].text, "plan first");
  assert.ok((result as any).input.every((item: any) => !item.reasoning_content));
}

async function testStreamedThinkingLandsOnCompletedReasoningItem() {
  const state = createResponsesStreamState({ model: "gpt-4o" });
  const events = [
    ...unifiedChunkToResponsesEvents(
      {
        id: "chatcmpl-th",
        choices: [{ delta: { thinking: { content: "hmm", id: "rs_stream" } } }],
      },
      state
    ),
    ...unifiedChunkToResponsesEvents(
      {
        choices: [
          {
            delta: {
              thinking: {
                encrypted_content: "enc-stream",
                id: "rs_stream",
              },
            },
          },
        ],
      },
      state
    ),
    ...unifiedChunkToResponsesEvents(
      { choices: [{ delta: { content: "answer" }, finish_reason: "stop" }] },
      state
    ),
    ...finalizeResponsesStream(state),
  ];
  const types = events.map((event) => event.type);
  assert.ok(types.includes("response.output_item.added"));
  assert.ok(types.includes("response.reasoning_summary_part.added"));
  assert.ok(types.includes("response.reasoning_summary_text.delta"));
  assert.ok(types.includes("response.reasoning_summary_text.done"));
  assert.ok(types.includes("response.reasoning_summary_part.done"));
  const reasoningAdded = events.find(
    (event) =>
      event.type === "response.output_item.added" &&
      event.item?.type === "reasoning"
  );
  assert.equal(reasoningAdded?.item?.id, "rs_stream");
  const deltas = events
    .filter((event) => event.type === "response.reasoning_summary_text.delta")
    .map((event) => event.delta)
    .join("");
  assert.equal(deltas, "hmm");
  // Reasoning must close before the message item opens (Zen / OpenAI order).
  const reasoningDoneAt = types.indexOf("response.output_item.done");
  const messageAddedAt = types.findIndex(
    (type, index) =>
      type === "response.output_item.added" &&
      events[index].item?.type === "message"
  );
  assert.ok(reasoningDoneAt >= 0);
  assert.ok(messageAddedAt > reasoningDoneAt);
  const completed = events.at(-1).response;
  assert.equal(completed.output[0].type, "reasoning");
  assert.equal(completed.output[0].summary[0].text, "hmm");
  assert.equal(completed.output[0].encrypted_content, "enc-stream");
  assert.equal(completed.output[0].id, "rs_stream");
  assert.equal(completed.output[1].type, "message");
}

async function testDuplicateRsAnonReasoningIdsAreRewritten() {
  const a = responsesReasoningItemFromThinking({
    encrypted_content: "cipher-turn-a",
    id: "rs_anon",
  });
  const b = responsesReasoningItemFromThinking({
    encrypted_content: "cipher-turn-b",
    id: "rs_anon",
  });
  assert.ok(a);
  assert.ok(b);
  assert.notEqual(a.id, "rs_anon");
  assert.notEqual(b.id, "rs_anon");
  assert.notEqual(a.id, b.id);

  const items = [
    { type: "reasoning", id: "rs_anon", encrypted_content: "enc-1", summary: [] },
    { type: "reasoning", id: "rs_anon", encrypted_content: "enc-2", summary: [] },
    { type: "reasoning", id: "rs_same", encrypted_content: "enc-3", summary: [] },
    { type: "reasoning", id: "rs_same", encrypted_content: "enc-4", summary: [] },
  ];
  uniquifyReasoningItemIds(items);
  const ids = items.map((item) => item.id);
  assert.equal(new Set(ids).size, 4);
  assert.ok(ids.every((id) => id !== "rs_anon"));
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

async function testMaxOutputTokensClampedToApiFloor() {
  const tf = new OpenAIResponsesTransformer();
  // Anthropic clients may send max_tokens < 16, which Responses rejects.
  const small = await tf.transformRequestIn(
    {
      model: "muse-spark-1.2",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 4,
    } as any,
    {},
    {}
  );
  assert.equal((small as any).max_output_tokens, 16);
  assert.equal((small as any).max_tokens, undefined);

  const normal = await tf.transformRequestIn(
    {
      model: "muse-spark-1.2",
      messages: [{ role: "user", content: "hi" }],
      max_tokens: 1024,
    } as any,
    {},
    {}
  );
  assert.equal((normal as any).max_output_tokens, 1024);
}

async function main() {
  await testStringAndMessageInput();
  await testFunctionCallRoundTrip();
  await testParallelFunctionCallsCoalesce();
  await testParallelCustomToolCallsCoalesce();
  await testSequentialToolTurnsDoNotMerge();
  await testFunctionCallThenAssistantTextCoalesce();
  await testAssistantTextThenFunctionCallCoalesce();
  await testAssistantTextThenParallelFunctionCallsCoalesce();
  await testUnsupportedState();
  await testResponseFormatConversion();
  await testReasoningAndTools();
  await testClientIncludeAndStorePassThrough();
  await testCustomHostedToolConvertsToFunction();
  await testJsonOutput();
  await testCustomToolSseLifecycle();
  await testExecShellEnvelopeNormalizedOnStream();
  await testExecShellEnvelopeUnmodifiedWithoutCodexConventions();
  await testResponsesTextFormatHelper();
  await testExecCommandKeyVariantNormalized();
  await testExecApplyPatchHeredocNormalized();
  await testExecShellEnvelopeNormalizedOnJson();
  await testExecFunctionToolWrapperNormalized();
  await testExecToolDescriptionIsProviderNeutral();
  await testCodexPatchMarkersNormalizedOnStream();
  await testCodexPatchMarkersNormalizedOnJson();
  await testCustomToolHistoryInput();
  await testCustomToolHeredocUnwrap();
  await testCustomToolHeredocUnwrapSse();
  await testExactTextSseLifecycle();
  await testToolSseLifecycle();
  await testOutputIndicesStayStableWhenTextFollowsTool();
  await testTrailingTextAfterToolOpensNewMessageItem();
  await testTextItemClosesBeforeToolCallOpens();
  await testTransformerStreamIntegration();
  await testSeparateUsageChunkIsRetained();
  await testMalformedStreamBecomesFailedEvent();
  await testUpstreamStreamErrorBecomesFailedEvent();
  await testCallIdMapPersistsInProtocolContext();
  await testReasoningInputItemBecomesThinking();
  await testThinkingRoundTripsToResponsesReasoningItem();
  await testThinkingHistoryIsEmittedOnResponsesRequest();
  await testPoisonedReasoningItemIdIsNotReplayedAsEncryptedContent();
  await testChatReasoningContentHistoryBecomesResponsesReasoning();
  await testStreamedThinkingLandsOnCompletedReasoningItem();
  await testDuplicateRsAnonReasoningIdsAreRewritten();
  await testClientTransformRequestOut();
  await testMaxOutputTokensClampedToApiFloor();
  console.log("openai.inbound-responses: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

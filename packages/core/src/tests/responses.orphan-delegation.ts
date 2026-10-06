/**
 * Responses inbound multi-agent compatibility: orphan delegation outputs
 * (missing call_id) become user messages when enabled + collab_spawn header,
 * and `agent_message` items become user messages when multi-agent mode is on.
 * Missing call ids and role-less agent_message items stay strict by default;
 * role-bearing agent_message items keep their existing message projection.
 */
import assert from "node:assert/strict";
import {
  hasCollabSpawnSubagentHeader,
  resolveResponsesInboundCompat,
  responsesRequestToUnified,
  repairResponsesWireMultiAgentCompat,
} from "../utils/openai.responses.util";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";

process.exitCode = 1;

const COMPAT = {
  orphanDelegationCompatibility: true,
  subagentCollabSpawn: true,
};

async function expectReject(
  fn: () => Promise<unknown> | unknown,
  code: string
): Promise<void> {
  let caught: any;
  try {
    await fn();
  } catch (error) {
    caught = error;
  }
  assert.ok(caught, `expected reject with code ${code}`);
  assert.equal(caught.code, code);
}

async function testOrphanOutputBecomesUserMessage() {
  const unified = responsesRequestToUnified(
    {
      model: "codex,gpt-5",
      input: [
        { role: "user", content: "continue the thread" },
        {
          type: "function_call_output",
          output: "delegation result text",
        },
      ],
    },
    undefined,
    undefined,
    COMPAT
  );
  const last = unified.messages.at(-1) as any;
  assert.equal(last.role, "user");
  assert.equal(last.content, "delegation result text");
}

async function testOrphanCustomOutputBecomesUserMessage() {
  const unified = responsesRequestToUnified(
    {
      model: "codex,gpt-5",
      input: [
        {
          type: "custom_tool_call_output",
          call_id: "",
          output: "custom delegation result",
        },
      ],
    },
    undefined,
    undefined,
    COMPAT
  );
  assert.equal(unified.messages[0].role, "user");
  assert.equal(unified.messages[0].content, "custom delegation result");
}

async function testOrphanStillRejectsByDefault() {
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "codex,gpt-5",
        input: [{ type: "function_call_output", output: "x" }],
      }),
    "invalid_call_id"
  );
  // Enabled flag but no collab_spawn header signal: still strict.
  await expectReject(
    () =>
      responsesRequestToUnified(
        {
          model: "codex,gpt-5",
          input: [{ type: "function_call_output", output: "x" }],
        },
        undefined,
        undefined,
        { orphanDelegationCompatibility: true, subagentCollabSpawn: false }
      ),
    "invalid_call_id"
  );
  // Header signal but flag off: still strict.
  await expectReject(
    () =>
      responsesRequestToUnified(
        {
          model: "codex,gpt-5",
          input: [{ type: "function_call_output", output: "x" }],
        },
        undefined,
        undefined,
        { orphanDelegationCompatibility: false, subagentCollabSpawn: true }
      ),
    "invalid_call_id"
  );
}

async function testValidOutputsUnaffected() {
  const unified = responsesRequestToUnified(
    {
      model: "codex,gpt-5",
      input: [
        { type: "function_call", call_id: "call_1", name: "Read", arguments: "{}" },
        { type: "function_call_output", call_id: "call_1", output: "file" },
      ],
    },
    undefined,
    undefined,
    COMPAT
  );
  assert.equal(unified.messages[1].role, "tool");
  assert.equal((unified.messages[1] as any).tool_call_id, "call_1");
}

async function testAgentMessageBecomesUserMessage() {
  const unified = responsesRequestToUnified(
    {
      model: "codex,gpt-5",
      input: [
        { type: "agent_message", content: "spawned agent says hi" },
      ],
    },
    undefined,
    undefined,
    { optimizeMultiAgentV2: true }
  );
  assert.equal(unified.messages[0].role, "user");
  assert.equal(unified.messages[0].content, "spawned agent says hi");
}

async function testAgentMessageRejectsByDefault() {
  await expectReject(
    () =>
      responsesRequestToUnified({
        model: "codex,gpt-5",
        input: [{ type: "agent_message", content: "hi" }],
      }),
    "unsupported_input_item"
  );
}

async function testHeaderDetection() {
  assert.equal(
    hasCollabSpawnSubagentHeader({ "X-Openai-Subagent": "collab_spawn" }),
    true
  );
  assert.equal(
    hasCollabSpawnSubagentHeader({ "x-openai-subagent": "other" }),
    false
  );
  assert.equal(hasCollabSpawnSubagentHeader({}), false);
  assert.equal(hasCollabSpawnSubagentHeader(undefined), false);
}

async function testCompatResolution() {
  const on = resolveResponsesInboundCompat(
    { "x-openai-subagent": "collab_spawn" },
    { get: (key: string) =>
        key === "orphanDelegationCompatibility" || key === "optimizeMultiAgentV2"
          ? true
          : undefined }
  );
  assert.equal(on.orphanDelegationCompatibility, true);
  assert.equal(on.optimizeMultiAgentV2, true);
  assert.equal(on.subagentCollabSpawn, true);
  const off = resolveResponsesInboundCompat({}, { get: () => undefined });
  assert.equal(off.orphanDelegationCompatibility, false);
  assert.equal(off.optimizeMultiAgentV2, false);
  assert.equal(off.subagentCollabSpawn, false);
  // snake_case spellings are accepted.
  const snake = resolveResponsesInboundCompat(
    {},
    { get: (key: string) =>
        key === "orphan_delegation_compatibility" ? true : undefined }
  );
  assert.equal(snake.orphanDelegationCompatibility, true);
}

async function testOwnerTransformerThreadsCompat() {
  const tf = new OpenAIResponsesTransformer();
  const unified = await tf.transformRequestOut(
    {
      model: "codex,gpt-5",
      input: [{ type: "function_call_output", output: "orphan" }],
    },
    {
      req: {
        headers: { "x-openai-subagent": "collab_spawn" },
        server: {
          configService: {
            get: (key: string) =>
              key === "orphanDelegationCompatibility" ? true : undefined,
          },
        },
      },
    } as any
  );
  assert.equal(unified.messages[0].role, "user");
  assert.equal(unified.messages[0].content, "orphan");
}

/**
 * Compat resolved once by the inbound pipeline wins over the transformer's
 * own request: the Unified projection and the kept wire gate on one value.
 */
async function testPipelineCompatWinsOverRequestConfig() {
  const tf = new OpenAIResponsesTransformer();
  const request = {
    model: "codex,gpt-5",
    input: [{ type: "agent_message", content: "from a worker" }],
  };
  const req = {
    headers: {},
    server: { configService: { get: () => undefined } },
  };
  const unified = await tf.transformRequestOut(request, {
    req,
    responsesCompat: { optimizeMultiAgentV2: true },
  } as any);
  assert.equal(unified.messages[0].role, "user");
  assert.equal(unified.messages[0].content, "from a worker");
  await expectReject(
    () => tf.transformRequestOut(request, { req, responsesCompat: {} } as any),
    "unsupported_input_item"
  );
}

async function testRoleBearingAgentMessagesKeepDefaultBehavior() {
  for (const compat of [{}, { optimizeMultiAgentV2: true }]) {
    for (const role of ["user", "assistant"]) {
      const unified = responsesRequestToUnified(
        { model: "p,m", input: [{ type: "agent_message", role, content: "hi" }] },
        undefined,
        undefined,
        compat
      );
      assert.deepEqual(unified.messages, [{ role, content: "hi" }]);
    }
  }
}

async function testAgentMessageContentAndValidation() {
  const content = [
    { type: "input_text", text: "part one" },
    { type: "input_text", text: "part two" },
    { type: "input_image", image_url: "data:image/png;base64,aGVsbG8=", detail: "high" },
    { type: "input_file", filename: "note.txt", file_data: "aGVsbG8=", mime_type: "text/plain" },
  ];
  const body = { model: "p,m", input: [{ type: "agent_message", content }] };
  const compat = { optimizeMultiAgentV2: true };
  const unified = responsesRequestToUnified(body, undefined, undefined, compat);
  assert.deepEqual(unified.messages, [{
    role: "user",
    content: [
      { type: "text", text: "part one" },
      { type: "text", text: "part two" },
      { type: "image_url", image_url: { url: content[2].image_url, detail: "high" } },
      { type: "file", filename: "note.txt", file_data: "aGVsbG8=", media_type: "text/plain" },
    ],
  }]);
  const wire = repairResponsesWireMultiAgentCompat(body, compat);
  assert.deepEqual(wire.input, [{ type: "message", role: "user", content }]);
  assert.equal(body.input[0].type, "agent_message", "original input stays immutable");
  for (const type of ["input_image", "input_file"]) {
    await expectReject(
      () => responsesRequestToUnified(
        { model: "p,m", input: [{ type: "agent_message", content: [{ type, file_id: "file_1" }] }] },
        undefined,
        undefined,
        compat
      ),
      "unsupported_file_id"
    );
  }
}

async function testOrphansWaitForParallelToolResults() {
  const input = [
    { type: "function_call", call_id: "call_1", name: "Read", arguments: "{}" },
    { type: "function_call", call_id: "call_2", name: "Read", arguments: "{}" },
    { type: "function_call_output", output: "orphan one" },
    { type: "function_call_output", call_id: "call_2", output: "second" },
    { type: "custom_tool_call_output", call_id: "", output: "orphan two" },
    { type: "function_call_output", call_id: "call_1", output: "first" },
  ];
  const body = { model: "p,m", input };
  const unified = responsesRequestToUnified(body, undefined, undefined, COMPAT);
  assert.deepEqual(unified.messages.map((message) => message.role), [
    "assistant", "tool", "tool", "user", "user",
  ]);
  assert.deepEqual(unified.messages.slice(1).map((message) => message.content), [
    "second", "first", "orphan one", "orphan two",
  ]);
  const wire = repairResponsesWireMultiAgentCompat(body, COMPAT);
  assert.deepEqual(wire.input.slice(0, 4), [input[0], input[1], input[3], input[5]]);
  assert.deepEqual(wire.input.slice(4), [
    { type: "message", role: "user", content: "orphan one" },
    { type: "message", role: "user", content: "orphan two" },
  ]);
}

async function testCompatMessagesWaitAcrossAssistantItems() {
  const input = [
    { type: "function_call", call_id: "call_1", name: "Read", arguments: "{}" },
    { type: "agent_message", content: "agent update" },
    { type: "custom_tool_call_output", output: "orphan" },
    { type: "reasoning", id: "rs_1", summary: [{ type: "summary_text", text: "thinking" }] },
    { type: "message", role: "assistant", content: "working" },
    { type: "web_search_call", id: "search_1", status: "completed" },
    { type: "function_call_output", call_id: "call_1", output: "file" },
  ];
  const body = { model: "p,m", input };
  const compat = { ...COMPAT, optimizeMultiAgentV2: true };
  const unified = responsesRequestToUnified(body, undefined, undefined, compat);
  assert.deepEqual(unified.messages.map((message) => message.role), [
    "assistant", "tool", "user", "user",
  ]);
  assert.deepEqual(unified.messages.slice(1).map((message) => message.content), [
    "file", "agent update", "orphan",
  ]);
  const wire = repairResponsesWireMultiAgentCompat(body, compat);
  assert.deepEqual(wire.input.slice(0, 5), [input[0], input[3], input[4], input[5], input[6]]);
  assert.deepEqual(wire.input.slice(5), [
    { type: "message", role: "user", content: "agent update" },
    { type: "message", role: "user", content: "orphan" },
  ]);
}

async function testWireGatesAndOpaqueItemsStayIntact() {
  const reasoning = { type: "reasoning", id: "rs_1", encrypted_content: "opaque", summary: [] };
  const output = { type: "custom_tool_call_output", output: "delegated" };
  const body = { model: "p,m", store: false, input: [reasoning, output] };
  for (const compat of [{}, { orphanDelegationCompatibility: true }, { subagentCollabSpawn: true }]) {
    assert.equal(repairResponsesWireMultiAgentCompat(body, compat), body);
    await expectReject(
      () => responsesRequestToUnified(body, undefined, undefined, compat),
      "invalid_call_id"
    );
  }
  const wire = repairResponsesWireMultiAgentCompat(body, COMPAT);
  assert.equal(wire.input[0], reasoning);
  assert.equal(wire.store, false);
  assert.deepEqual(wire.input[1], { type: "message", role: "user", content: "delegated" });
  const withId = { model: "p,m", input: [{ ...output, call_id: "remote_call" }] };
  assert.equal(repairResponsesWireMultiAgentCompat(withId, COMPAT), withId);
  assert.equal(responsesRequestToUnified(withId, undefined, undefined, COMPAT).messages[0].role, "tool");
}

async function main() {
  await testRoleBearingAgentMessagesKeepDefaultBehavior();
  await testAgentMessageContentAndValidation();
  await testOrphansWaitForParallelToolResults();
  await testCompatMessagesWaitAcrossAssistantItems();
  await testWireGatesAndOpaqueItemsStayIntact();
  await testOrphanOutputBecomesUserMessage();
  await testOrphanCustomOutputBecomesUserMessage();
  await testOrphanStillRejectsByDefault();
  await testValidOutputsUnaffected();
  await testAgentMessageBecomesUserMessage();
  await testAgentMessageRejectsByDefault();
  await testHeaderDetection();
  await testCompatResolution();
  await testOwnerTransformerThreadsCompat();
  await testPipelineCompatWinsOverRequestConfig();
  console.log("responses.orphan-delegation: PASS");
}

main().then(() => { process.exitCode = 0; }).catch((err) => {
  console.error(err);
  process.exit(1);
});

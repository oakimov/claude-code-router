import assert from "node:assert/strict";
import {
  bodyToLogString,
  isTruthyConfigFlag,
  logKeepWire,
  logMessageBody,
  resolveLogBodyMaxBytes,
  shouldLogRequestBodies,
  shouldLogSSEEvents,
  summarizeKeepWire,
} from "../utils/message-debug";

function truthyFlags() {
  assert.equal(isTruthyConfigFlag(true), true);
  assert.equal(isTruthyConfigFlag("true"), true);
  assert.equal(isTruthyConfigFlag("1"), true);
  assert.equal(isTruthyConfigFlag(false), false);
  assert.equal(isTruthyConfigFlag("false"), false);
  assert.equal(isTruthyConfigFlag(undefined), false);
}

function configHelpers() {
  const on = {
    get(key: string) {
      if (key === "LOG_REQUEST_BODY") return true;
      if (key === "LOG_SSE_EVENTS") return "true";
      if (key === "LOG_REQUEST_BODY_MAX_BYTES") return 4096;
      return undefined;
    },
  };
  assert.equal(shouldLogRequestBodies(on), true);
  assert.equal(shouldLogSSEEvents(on), true);
  assert.equal(resolveLogBodyMaxBytes(on), 4096);

  const off = { get() { return undefined; } };
  assert.equal(shouldLogRequestBodies(off), false);
  assert.equal(resolveLogBodyMaxBytes(off), 32768);
}

function logsWithDirection() {
  const records: Record<string, unknown>[] = [];
  const logger = {
    debug(payload: Record<string, unknown>) {
      records.push(payload);
    },
  };

  logMessageBody(
    { model: "muse", reasoning: { effort: "high", summary: "auto" }, include: ["reasoning.encrypted_content"] },
    {
      logger,
      direction: "client→ccr",
      protocol: "openai_responses",
      reqId: "req-1",
    }
  );

  assert.equal(records.length, 1);
  assert.equal(records[0].type, "message body");
  assert.equal(records[0].direction, "client→ccr");
  assert.equal(records[0].protocol, "openai_responses");
  assert.ok(String(records[0].data).includes('"summary":"auto"'));
  assert.ok(String(records[0].data).includes("reasoning.encrypted_content"));
}

function keepWireDigestCountsEncryptedItems() {
  const summary = summarizeKeepWire({
    model: "gpt-5.6-luna",
    store: false,
    stream: true,
    include: ["reasoning.encrypted_content"],
    prompt_cache_key: "01a05730-c3d2-74e0-a162-ef80bfd53b73",
    input: [
      {
        type: "reasoning",
        id: "rs_keep",
        summary: [],
        encrypted_content: "gAAAAABlciphertext",
      },
      {
        type: "function_call",
        call_id: "call_1",
        name: "Read",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: "ok",
      },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "continue" }],
      },
    ],
    tools: [{ type: "function", name: "Read" }],
  });
  assert.equal(summary.store, false);
  assert.equal(summary.stream, true);
  assert.deepEqual(summary.include, ["reasoning.encrypted_content"]);
  assert.equal(summary.prompt_cache_key, "01a05730-c3d2-74e0-a162-ef80bfd53b73");
  assert.equal(summary.input_n, 4);
  assert.deepEqual(summary.input_types, {
    reasoning: 1,
    function_call: 1,
    function_call_output: 1,
    message: 1,
  });
  assert.equal(summary.encrypted_content_items, 1);
  assert.deepEqual(summary.reasoning, [
    {
      type: "reasoning",
      encrypted_len: "gAAAAABlciphertext".length,
      id: "rs_keep",
      summary_n: 0,
    },
  ]);
  assert.equal(summary.tools_n, 1);
  assert.equal(JSON.stringify(summary).includes("gAAAAABl"), false);

  const records: Record<string, unknown>[] = [];
  logKeepWire(
    { store: false, input: [{ type: "reasoning", encrypted_content: "abc" }] },
    {
      logger: {
        debug(payload: Record<string, unknown>) {
          records.push(payload);
        },
      },
      reqId: "req-keep",
      provider: "codex",
      model: "gpt-5.6-luna",
    }
  );
  assert.equal(records[0].type, "keep wire");
  assert.equal(records[0].direction, "ccr→provider");
  assert.equal(records[0].encrypted_content_items, 1);
  assert.equal(records[0].provider, "codex");
}

function serializesBodies() {
  assert.equal(bodyToLogString("raw"), "raw");
  assert.equal(bodyToLogString({ a: 1 }), '{"a":1}');
  assert.equal(bodyToLogString(null), "");
}

function main() {
  truthyFlags();
  configHelpers();
  logsWithDirection();
  keepWireDigestCountsEncryptedItems();
  serializesBodies();
  console.log("message-debug: ok");
}

main();

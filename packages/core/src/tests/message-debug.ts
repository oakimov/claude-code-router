import assert from "node:assert/strict";
import {
  bodyToLogString,
  isTruthyConfigFlag,
  logMessageBody,
  resolveLogBodyMaxBytes,
  shouldLogRequestBodies,
  shouldLogSSEEvents,
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

function serializesBodies() {
  assert.equal(bodyToLogString("raw"), "raw");
  assert.equal(bodyToLogString({ a: 1 }), '{"a":1}');
  assert.equal(bodyToLogString(null), "");
}

function main() {
  truthyFlags();
  configHelpers();
  logsWithDirection();
  serializesBodies();
  console.log("message-debug: ok");
}

main();

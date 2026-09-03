import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import {
  bodyToLogString,
  isTruthyConfigFlag,
  logKeepWire,
  logMessageBody,
  resolveLogBodyMaxBytes,
  resolveLogBodySelection,
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
      if (key === "LOG_REQUEST_BODY_PARTS") return "system,tools";
      if (key === "LOG_SSE_EVENTS") return "true";
      if (key === "LOG_REQUEST_BODY_MAX_BYTES") return 4096;
      return undefined;
    },
  };
  assert.deepEqual(resolveLogBodySelection(on), ["system", "tools"]);
  assert.equal(shouldLogSSEEvents(on), true);
  assert.equal(resolveLogBodyMaxBytes(on), 4096);

  const off = { get() { return undefined; } };
  assert.equal(resolveLogBodySelection(off), undefined);
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

function resolvesBodyParts() {
  const pick = (value: unknown) =>
    resolveLogBodySelection({
      get: (key: string) => (key === "LOG_REQUEST_BODY_PARTS" ? value : undefined),
    });
  assert.equal(pick(undefined), undefined);
  assert.equal(pick(""), undefined);
  assert.equal(pick("full"), "full");
  assert.equal(pick("*"), "full");
  assert.equal(pick("FULL"), "full");
  assert.deepEqual(pick("system, tools,system , MESSAGES"), [
    "system",
    "tools",
    "messages",
  ]);
  assert.deepEqual(pick("tools,unknown-part"), ["tools", "unknown-part"]);
}

function storesBodyPartsInFiles() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-body-parts-"));
  try {
    const records: Record<string, unknown>[] = [];
    const logger = {
      debug(payload: Record<string, unknown>) {
        records.push(payload);
      },
    };
    const body = {
      model: "muse",
      system: "be helpful",
      tools: [{ type: "function", name: "Read" }],
      messages: [{ role: "user", content: "hi" }],
    };

    logMessageBody(body, {
      logger,
      direction: "ccr→provider",
      reqId: "req-9",
      provider: "cursor",
      model: "muse",
      selection: ["system", "tools", "missing-part"],
      bodiesDir: dir,
    });

    // Two manifests for found parts + one not-found line; no inline dump.
    assert.equal(records.length, 3);
    assert.ok(records.every((record) => record.type === "message body part"));
    assert.ok(records.every((record) => !("data" in record)));
    const system = records.find((record) => record.part === "system")!;
    const tools = records.find((record) => record.part === "tools")!;
    const missing = records.find((record) => record.part === "missing-part")!;
    assert.equal(system.found, true);
    assert.equal(tools.found, true);
    assert.equal(missing.found, false);

    for (const manifest of [system, tools]) {
      assert.match(String(manifest.sha256), /^[0-9a-f]{16}$/);
      assert.ok(typeof manifest.path === "string");
      assert.ok(existsSync(String(manifest.path)));
      const stored = readFileSync(String(manifest.path), "utf-8");
      // Hash in the manifest matches the stored bytes.
      assert.equal(
        createHash("sha256").update(stored, "utf8").digest("hex").slice(0, 16),
        manifest.sha256
      );
      assert.equal(manifest.bytes, stored.length);
      assert.equal(manifest.truncated, false);
    }
    assert.match(
      String(system.path),
      /req-9\.ccr-provider\.system\.[0-9a-f]{16}\.json$/
    );
    assert.ok(readFileSync(String(system.path), "utf-8").includes("be helpful"));
    assert.ok(
      readFileSync(String(tools.path), "utf-8").includes('"name":"Read"')
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function redactsAndIsolatesFailures() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-body-parts-"));
  try {
    const records: Record<string, unknown>[] = [];
    const logger = {
      debug(payload: Record<string, unknown>) {
        records.push(payload);
      },
    };
    logMessageBody(
      {
        // Non-string secret key shapes still redact inside stored files.
        tools: [{ name: "x", api_key: "sk-abcdefghijklmnop" }],
        messages: [
          { type: "reasoning", encrypted_content: "gAAAAABlciphertext" },
        ],
      },
      {
        logger,
        direction: "client→ccr",
        reqId: "req-secret/../../evil",
        selection: ["tools", "messages"],
        bodiesDir: dir,
      }
    );

    assert.equal(records.length, 2);
    for (const manifest of records) {
      const stored = readFileSync(String(manifest.path), "utf-8");
      assert.ok(!stored.includes("sk-abcdefghijklmnop"));
      assert.ok(!stored.includes("gAAAAABlciphertext"));
      // reqId is neutralized for the filename (no traversal, no separators).
      const name = basename(String(manifest.path));
      assert.ok(!name.includes(".."));
      assert.ok(!name.includes("/"));
    }
    assert.match(
      String(records[0].path),
      /req-secret_evil\.client-ccr\.tools\.[0-9a-f]{16}\.json$/
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }

  // Unwritable dir degrades to an error manifest, never throws.
  const records: Record<string, unknown>[] = [];
  logMessageBody(
    { tools: [] },
    {
      logger: { debug: (payload: Record<string, unknown>) => records.push(payload) },
      direction: "ccr→provider",
      reqId: "req-1",
      selection: ["tools"],
      bodiesDir: join(dir, "no-such-parent", "\0"),
    }
  );
  assert.equal(records.length, 1);
  assert.ok(typeof records[0].error === "string");
}

function storesFullBodyInOneFile() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-body-full-"));
  try {
    const records: Record<string, unknown>[] = [];
    const body = { model: "muse", system: "be helpful" };
    logMessageBody(body, {
      logger: { debug: (payload: Record<string, unknown>) => records.push(payload) },
      direction: "client→ccr",
      reqId: "req-full",
      selection: "full",
      bodiesDir: dir,
    });
    assert.equal(records.length, 1);
    assert.equal(records[0].type, "message body part");
    assert.equal(records[0].part, "full");
    assert.equal(records[0].found, true);
    const stored = readFileSync(String(records[0].path), "utf-8");
    assert.ok(stored.includes('"model":"muse"'));
    assert.ok(stored.includes("be helpful"));
    assert.match(
      String(records[0].path),
      /req-full\.client-ccr\.full\.[0-9a-f]{16}\.json$/
    );
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function main() {
  truthyFlags();
  configHelpers();
  logsWithDirection();
  keepWireDigestCountsEncryptedItems();
  serializesBodies();
  resolvesBodyParts();
  storesBodyPartsInFiles();
  storesFullBodyInOneFile();
  redactsAndIsolatesFailures();
  console.log("message-debug: ok");
}

main();

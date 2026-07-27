import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Cursor } from "@cursor/sdk";
import { extractHostEnvironment } from "../cursor-sdk/host-env";
import { runCursor } from "../cursor-sdk/runner";
import { globalSessionManager } from "../cursor-sdk/session";

/**
 * Reporting regressions found only in live traffic:
 *
 * 1. `@cursor/sdk` can invoke `customTools.execute` as soon as `agent.send`
 *    resolves — before the response stream is constructed — so a start/end
 *    delta over the cumulative session counter reported zero.
 * 2. A bridge turn normally ends by emitting host tool calls and returning
 *    early, bypassing any report placed on the run-completed path.
 */

const WORKSPACE = mkdtempSync(join(tmpdir(), "ccr-report-"));
const HOST_ROOT = "/Users/dev/Projects/app";

const hostEnv = extractHostEnvironment({
  messages: [
    {
      role: "system",
      content: `<env>\nWorking directory: ${HOST_ROOT}\nPlatform: darwin\n</env>`,
    },
  ],
} as any);

function fakeRun() {
  return {
    status: "running",
    usage: undefined,
    stream() {
      return {
        [Symbol.asyncIterator]() {
          return {
            // Never completes: the turn must end via the host-tool path, not
            // by the run finishing.
            next() {
              return new Promise<any>(() => undefined);
            },
          };
        },
      };
    },
    cancel: async () => undefined,
  };
}

function fakeSession(send: (prompt: any, options: any) => Promise<any>): any {
  const session: any = {
    key: "scratch-report",
    agentId: "agent-1",
    agent: { send, close: () => undefined },
    mode: "bridge",
    workspaceDir: WORKSPACE,
    hostEnv,
    parked: [],
    pendingEmit: [],
    emitWaiters: [],
    pendingSdkMessages: [],
    sdkMessageWaiters: [],
    sendChain: Promise.resolve(),
    hasSentPrompt: false,
    lastActiveAt: Date.now(),
    metrics: {
      customToolCalls: 0,
      builtinToolCallsSeen: 0,
      scratchPathViolations: 0,
      scratchPathCorrections: 0,
    },
    parkHostTool({ id, name, args }: any) {
      const entry: any = { id, name, args, runToken: session.activeRunToken };
      entry.promise = new Promise((resolve, reject) => {
        entry.resolve = resolve;
        entry.reject = reject;
      });
      session.parked.push(entry);
      session.pendingEmit.push(entry);
      session.notifyEmit();
      return entry.promise;
    },
    notifyEmit() {
      const waiters = session.emitWaiters.splice(0, session.emitWaiters.length);
      for (const waiter of waiters) waiter();
    },
    waitForEmit() {
      return new Promise<void>((resolve) => {
        if (session.pendingEmit.length) {
          resolve();
          return;
        }
        session.emitWaiters.push(resolve);
      });
    },
    enqueueSdkMessage(message: any, runToken = session.activeRunToken) {
      session.pendingSdkMessages.push({ message, runToken, source: "delta" });
      session.notifySdkMessage();
    },
    notifySdkMessage() {
      const waiters = session.sdkMessageWaiters.splice(
        0,
        session.sdkMessageWaiters.length
      );
      for (const waiter of waiters) waiter();
    },
    waitForSdkMessage() {
      return new Promise<void>((resolve) => {
        if (session.pendingSdkMessages.length) {
          resolve();
          return;
        }
        session.sdkMessageWaiters.push(resolve);
      });
    },
  };
  return session;
}

async function main() {
  const originalGetOrCreate =
    globalSessionManager.getOrCreate.bind(globalSessionManager);
  const originalModelList = Cursor.models.list;

  const warns: Array<{ payload: any; msg: string }> = [];
  const logger = {
    warn: (payload: any, msg: string) => warns.push({ payload, msg }),
    info: () => undefined,
    error: () => undefined,
    debug: () => undefined,
  };

  const session = fakeSession(async (_prompt, options) => {
    const tools = options.local.customTools;
    // Fires before runCursor builds the stream — the ordering that hid the bug.
    void tools.Read.execute(
      { file_path: `${WORKSPACE}/AGENTS.md` },
      { toolCallId: "bad" }
    );
    // A clean call parks and ends the turn through the tool_use early return.
    // Must stay referenced — it is the only thing driving the turn forward.
    setTimeout(() => {
      void tools.Read.execute(
        { file_path: `${HOST_ROOT}/package.json` },
        { toolCallId: "good" }
      );
    }, 5);
    return fakeRun();
  });

  (Cursor.models as any).list = async () => [];
  (globalSessionManager as any).getOrCreate = async () => session;

  try {
    const response = await runCursor(
      {
        model: "glm-5.2",
        stream: true,
        messages: [
          {
            role: "system",
            content: `<env>\nWorking directory: ${HOST_ROOT}\nPlatform: darwin\n</env>`,
          },
          { role: "user", content: "read the config" },
        ],
        tools: [
          {
            function: {
              name: "Read",
              description: "Read a file",
              parameters: { type: "object", properties: {} },
            },
          },
        ],
      } as any,
      { apiKey: "crsr_test" },
      { req: { headers: { "x-ccr-cursor-session": "scratch-report" } } },
      { cursorMode: "bridge", logger }
    );
    const body = await response.text();

    // The turn ended by emitting the clean host tool call, not by run completion.
    assert.match(body, /"finish_reason":"tool_calls"/);
    assert.match(body, /package\.json/);
    // The corrected call was never forwarded to Claude Code.
    assert.doesNotMatch(body, /AGENTS\.md/);

    const perCall = warns.filter((w) =>
      w.msg.includes("referenced the scratch workspace")
    );
    assert.equal(perCall.length, 1, "per-call warn fires");
    assert.equal(perCall[0].payload.tool, "Read");
    assert.equal(perCall[0].payload.corrected, true);

    const summary = warns.filter((w) =>
      w.msg.includes("turn produced scratch-workspace tool paths")
    );
    assert.equal(summary.length, 1, "per-turn summary fires on the tool_use path");
    assert.equal(summary[0].payload.scratchViolations, 1);
    assert.equal(summary[0].payload.scratchCorrections, 1);
    assert.equal(summary[0].payload.model, "glm-5.2");
    assert.equal(summary[0].payload.hostProjectRoot, HOST_ROOT);
  } finally {
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    (Cursor.models as any).list = originalModelList;
    rmSync(WORKSPACE, { recursive: true, force: true });
  }

  console.log("cursor-sdk.scratch-report: ok");
}

await main();

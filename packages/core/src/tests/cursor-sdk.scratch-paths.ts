import assert from "node:assert/strict";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { extractHostEnvironment } from "../cursor-sdk/host-env";
import { CURSOR_SDK_WORKSPACES_ROOT, isManagedWorkspacePath } from "../cursor-sdk/shared";
import {
  buildScratchPathCorrection,
  findScratchPaths,
  scratchDetectionApplies,
} from "../cursor-sdk/tool-paths";
import { createTurnToolMetrics, toCustomTools } from "../cursor-sdk/tools";

const WORKSPACE = join(CURSOR_SDK_WORKSPACES_ROOT, "a".repeat(32));

const hostEnv = extractHostEnvironment({
  messages: [
    {
      role: "system",
      content: "<env>\nWorking directory: /Users/dev/Projects/app\nPlatform: darwin\n</env>",
    },
  ],
} as any);

// --- detection --------------------------------------------------------------

assert.deepEqual(
  findScratchPaths({ file_path: "/Users/dev/Projects/app/src/index.ts" }, WORKSPACE),
  []
);

const direct = findScratchPaths({ file_path: `${WORKSPACE}/src/index.ts` }, WORKSPACE);
assert.equal(direct.length, 1);
assert.equal(direct[0].argPath, "file_path");

// Shell commands arrive as one string — a prefix-only test would miss these.
const shell = findScratchPaths({ command: `cd ${WORKSPACE} && ls -la` }, WORKSPACE);
assert.equal(shell.length, 1);
assert.equal(shell[0].argPath, "command");

// Nested and array arguments are walked.
const nested = findScratchPaths(
  { edits: [{ file_path: "/Users/dev/ok.ts" }, { file_path: `${WORKSPACE}/bad.ts` }] },
  WORKSPACE
);
assert.equal(nested.length, 1);
assert.equal(nested[0].argPath, "edits.1.file_path");

// Another session's workspace, or the root itself, counts too.
const sibling = findScratchPaths(
  { path: join(CURSOR_SDK_WORKSPACES_ROOT, "b".repeat(32), "x.ts") },
  WORKSPACE
);
assert.equal(sibling.length, 1);

// Long values are truncated in the report.
const long = findScratchPaths({ command: `${WORKSPACE}/${"x".repeat(500)}` }, WORKSPACE);
assert.ok(long[0].value.endsWith("…"));
assert.ok(long[0].value.length <= 201);

// Cycles and depth must not hang the walk.
const cyclic: any = { a: {} };
cyclic.a.self = cyclic;
assert.doesNotThrow(() => findScratchPaths(cyclic, WORKSPACE));

// Non-string leaves are ignored.
assert.deepEqual(findScratchPaths({ n: 1, b: true, z: null }, WORKSPACE), []);

// --- applicability guard ----------------------------------------------------

assert.equal(scratchDetectionApplies(hostEnv), true);
const selfHosted = extractHostEnvironment({
  messages: [
    { role: "system", content: `<env>\nWorking directory: ${WORKSPACE}/real\n</env>` },
  ],
} as any);
assert.equal(scratchDetectionApplies(selfHosted), false);

// --- correction text --------------------------------------------------------

const correction = buildScratchPathCorrection(direct, "Read", WORKSPACE, hostEnv);
assert.match(correction, /^Error: this Read call was not executed/);
assert.match(correction, /file_path: .*a{32}/);
assert.match(correction, /normally under \/Users\/dev\/Projects\/app/);
assert.match(correction, /Retry the call with a host path\./);

// --- execute() gate ---------------------------------------------------------

function fakeSession() {
  const parked: Array<{ name: string; args: Record<string, unknown> }> = [];
  return {
    key: "test",
    workspaceDir: WORKSPACE,
    hostEnv,
    metrics: {
      customToolCalls: 0,
      builtinToolCallsSeen: 0,
      scratchPathViolations: 0,
      scratchPathCorrections: 0,
    },
    parked,
    parkHostTool: (tool: any) => {
      parked.push({ name: tool.name, args: tool.args });
      return Promise.resolve("host result");
    },
  } as any;
}

const session = fakeSession();
const tools = toCustomTools(
  { messages: [], tools: [{ function: { name: "Read", description: "d" } }] } as any,
  session
);

const good = await tools.Read.execute(
  { file_path: "/Users/dev/Projects/app/a.ts" } as any,
  { toolCallId: "1" } as any
);
assert.equal(good, "host result");
assert.equal(session.metrics.customToolCalls, 1);
assert.equal(session.metrics.scratchPathViolations, 0);
assert.equal(session.parked.length, 1);

const bad = await tools.Read.execute(
  { file_path: `${WORKSPACE}/a.ts` } as any,
  { toolCallId: "2" } as any
);
assert.match(String(bad), /was not executed/);
// Corrected calls must not reach the host or count as host tool calls.
assert.equal(session.metrics.customToolCalls, 1);
assert.equal(session.parked.length, 1);
assert.equal(session.metrics.scratchPathViolations, 1);
assert.equal(session.metrics.scratchPathCorrections, 1);

// After the cap, forward instead of looping corrections forever.
for (let i = 0; i < 4; i++) {
  await tools.Read.execute(
    { file_path: `${WORKSPACE}/a.ts` } as any,
    { toolCallId: `loop-${i}` } as any
  );
}
assert.equal(session.metrics.scratchPathCorrections, 3);
assert.equal(session.metrics.scratchPathViolations, 5);
assert.equal(session.parked.length, 3); // 1 good + 2 forwarded past the cap

// Per-turn metrics are independent of the cumulative session counters.
// Regression: the SDK can invoke execute() as soon as agent.send resolves —
// before the response stream is built — so a start/end delta over the session
// counter reported zero for a turn that actually had a violation.
const turnSession = fakeSession();
turnSession.metrics.scratchPathViolations = 7; // earlier turns on this session
turnSession.metrics.scratchPathCorrections = 1;
const turn = createTurnToolMetrics();
const turnTools = toCustomTools(
  { messages: [], tools: [{ function: { name: "Read", description: "d" } }] } as any,
  turnSession,
  undefined,
  turn
);
await turnTools.Read.execute(
  { file_path: `${WORKSPACE}/a.ts` } as any,
  { toolCallId: "t1" } as any
);
assert.equal(turn.scratchViolations, 1);
assert.equal(turn.scratchCorrections, 1);
assert.equal(turnSession.metrics.scratchPathViolations, 8);
// A clean call must not inflate the turn tally.
await turnTools.Read.execute(
  { file_path: "/Users/dev/Projects/app/a.ts" } as any,
  { toolCallId: "t2" } as any
);
assert.equal(turn.scratchViolations, 1);

// Detection is skipped entirely when the project legitimately lives there.
const selfSession = fakeSession();
selfSession.hostEnv = selfHosted;
const selfTools = toCustomTools(
  { messages: [], tools: [{ function: { name: "Read", description: "d" } }] } as any,
  selfSession
);
const passthrough = await selfTools.Read.execute(
  { file_path: `${WORKSPACE}/real/a.ts` } as any,
  { toolCallId: "3" } as any
);
assert.equal(passthrough, "host result");
assert.equal(selfSession.metrics.scratchPathViolations, 0);

// --- managed workspace guard ------------------------------------------------

assert.equal(isManagedWorkspacePath(WORKSPACE), true);
assert.equal(isManagedWorkspacePath(`${WORKSPACE}/`), true);
assert.equal(isManagedWorkspacePath(CURSOR_SDK_WORKSPACES_ROOT), false);
assert.equal(isManagedWorkspacePath("/Users/dev/Projects/app"), false);
assert.equal(isManagedWorkspacePath(join(WORKSPACE, "nested")), false);
assert.equal(
  isManagedWorkspacePath(join(CURSOR_SDK_WORKSPACES_ROOT, "not-a-session-key")),
  false
);
assert.equal(isManagedWorkspacePath("/"), false);

// A user-supplied `cursorCwd` (agent mode) is never managed.
const cwdLike = join(CURSOR_SDK_WORKSPACES_ROOT, "..", "my-project");
assert.equal(isManagedWorkspacePath(cwdLike), false);

// --- orphan sweep -----------------------------------------------------------
// Runs against a temp root: never touch the real ~/.claude-code-router.

const { SessionManager } = await import("../cursor-sdk/session");
const manager = new SessionManager();

const sweepRoot = mkdtempSync(join(tmpdir(), "ccr-sweep-"));
const orphan = join(sweepRoot, "c".repeat(32));
const fresh = join(sweepRoot, "d".repeat(32));
const foreign = join(sweepRoot, "keep-me");
for (const dir of [orphan, fresh, foreign]) {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "AGENTS.md"), "", "utf-8");
}
// Age the orphan past the TTL.
const old = new Date(Date.now() - 48 * 60 * 60 * 1000);
utimesSync(orphan, old, old);

const removed = manager.sweepOrphanWorkspaces(Date.now(), sweepRoot);
assert.equal(removed, 1);
assert.equal(existsSync(orphan), false, "aged orphan removed");
assert.equal(existsSync(fresh), true, "recent workspace kept");
assert.equal(existsSync(foreign), true, "non-session directory untouched");

// Rate limited: an immediate second sweep does nothing.
assert.equal(manager.sweepOrphanWorkspaces(Date.now(), sweepRoot), 0);

rmSync(sweepRoot, { recursive: true, force: true });

console.log("cursor-sdk.scratch-paths: ok");

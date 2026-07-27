import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  EMPTY_HOST_ENVIRONMENT,
  describeHostEnvironment,
  extractHostEnvironment,
} from "../cursor-sdk/host-env";
import {
  buildBridgeSystemGuidance,
  buildBridgeTailReminder,
  buildWorkspaceRulesDocument,
  toSdkPrompt,
} from "../cursor-sdk/prompt";
import { ensureDenyHooksWorkspace } from "../cursor-sdk/hooks-template";

const WORKSPACE = "/root/.claude-code-router/cursor-sdk-workspaces/abc123";

// Shape observed in real Claude Code traffic (packages/server/ccr-config/logs).
const claudeCodeSystemPrompt = `You are Claude Code.

Here is useful information about the environment you are running in:
<env>
Working directory: /Users/dev/Projects/app
Is directory a git repo: Yes
Platform: darwin
OS Version: Darwin 24.6.0
</env>
`;

const request: any = {
  messages: [
    { role: "system", content: claudeCodeSystemPrompt },
    { role: "user", content: "what changed in the router?" },
  ],
  tools: [
    { function: { name: "Read", description: "Read a file" } },
  ],
};

// --- extraction -------------------------------------------------------------

const env = extractHostEnvironment(request);
assert.equal(env.known, true);
assert.equal(env.projectRoot, "/Users/dev/Projects/app");
assert.equal(env.platform, "darwin");
assert.equal(env.osVersion, "Darwin 24.6.0");

// Non-canonical facts survive verbatim so host prompt changes need no code change.
const described = describeHostEnvironment(env);
assert.ok(described.includes("- Project root: /Users/dev/Projects/app"));
assert.ok(described.includes("- Is directory a git repo: Yes"));

// "Primary working directory" wins over "Working directory".
const dualRoot = extractHostEnvironment({
  messages: [
    {
      role: "system",
      content: [
        "<env>",
        "Working directory: /Users/dev/Projects/app/sub",
        "Primary working directory: /Users/dev/Projects/app",
        "Additional working directories: /Users/dev/Projects/other, relative/ignored",
        "</env>",
      ].join("\n"),
    },
  ],
} as any);
assert.equal(dualRoot.projectRoot, "/Users/dev/Projects/app");
assert.deepEqual(dualRoot.additionalRoots, ["/Users/dev/Projects/other"]);

// No env block: fall back to loose key scanning, and never guess a root.
const loose = extractHostEnvironment({
  messages: [
    { role: "system", content: "Primary working directory: C:\\Users\\dev\\app" },
  ],
} as any);
assert.equal(loose.projectRoot, "C:\\Users\\dev\\app");

const unknown = extractHostEnvironment({ messages: [] } as any);
assert.equal(unknown.known, false);
assert.equal(unknown.projectRoot, undefined);
assert.equal(unknown.fingerprint, EMPTY_HOST_ENVIRONMENT.fingerprint);

// Relative paths are not accepted as a root.
const relative = extractHostEnvironment({
  messages: [{ role: "system", content: "<env>\nWorking directory: ./app\n</env>" }],
} as any);
assert.equal(relative.projectRoot, undefined);

// Fingerprint tracks the facts, so guidance is only re-stamped on real change.
assert.equal(extractHostEnvironment(request).fingerprint, env.fingerprint);
assert.notEqual(dualRoot.fingerprint, env.fingerprint);

// --- guidance ---------------------------------------------------------------

const guidance = buildBridgeSystemGuidance(request, WORKSPACE, env);
assert.match(guidance, /Tools do not run where you are running/);
assert.match(guidance, /disposable scratch space: \/root\/\.claude-code-router/);
assert.match(guidance, /Project root: \/Users\/dev\/Projects\/app/);
assert.match(guidance, /Platform: darwin/);
assert.match(guidance, /normally under \/Users\/dev\/Projects\/app/);
assert.match(guidance, /not a restriction on the user's project/);
// Progress-narration rules must survive the rewrite.
assert.match(guidance, /Never end a turn with progress narration alone\./);

// Unknown host root: conservative wording, no invented path.
const blindGuidance = buildBridgeSystemGuidance(
  { messages: [], tools: [] } as any,
  WORKSPACE,
  EMPTY_HOST_ENVIRONMENT
);
assert.match(blindGuidance, /never invent one/);
assert.doesNotMatch(blindGuidance, /normally under/);

// --- prompt placement -------------------------------------------------------

const prompt = toSdkPrompt(request, {
  mode: "bridge",
  workspaceDir: WORKSPACE,
  hostEnv: env,
});
assert.ok(prompt.text.startsWith("You are a remote reasoning agent"));
// Restated last, after the flattened transcript.
assert.ok(prompt.text.trimEnd().endsWith(buildBridgeTailReminder(WORKSPACE, env)));
assert.ok(prompt.text.indexOf("[bridge reminder]") > prompt.text.indexOf("[user]"));

// Plan mode stays tool-free and carries no bridge topology.
const planPrompt = toSdkPrompt(request, {
  mode: "plan",
  workspaceDir: WORKSPACE,
  hostEnv: env,
});
assert.doesNotMatch(planPrompt.text, /\[bridge reminder\]/);

// --- workspace files --------------------------------------------------------

const rules = buildWorkspaceRulesDocument(WORKSPACE, env);
assert.match(rules, /^# Agent rules/);
assert.match(rules, /- Tools do not run where you are running/);
assert.match(rules, /- Project root: \/Users\/dev\/Projects\/app/);

const dir = mkdtempSync(join(tmpdir(), "ccr-cursor-ws-"));
ensureDenyHooksWorkspace(dir, env);

const agentsMd = readFileSync(join(dir, "AGENTS.md"), "utf-8");
assert.equal(agentsMd, buildWorkspaceRulesDocument(dir, env));
assert.match(agentsMd, /- Project root: \/Users\/dev\/Projects\/app/);

const denyScript = readFileSync(join(dir, ".cursor/hooks/deny-builtin.cjs"), "utf-8");
assert.match(denyScript, /do not use Cursor built-in tools/);
assert.match(denyScript, /empty scratch space on a different machine/);
assert.match(denyScript, /\/Users\/dev\/Projects\/app/);
assert.match(denyScript, /custom-user-tools/);

const hooksJson = JSON.parse(readFileSync(join(dir, ".cursor/hooks.json"), "utf-8"));
assert.equal(hooksJson.version, 1);
assert.match(hooksJson.hooks.preToolUse[0].matcher, /Shell\|Read/);

// Rewrite with a different host root updates the workspace in place.
ensureDenyHooksWorkspace(dir, dualRoot);
assert.match(readFileSync(join(dir, "AGENTS.md"), "utf-8"), /\/Users\/dev\/Projects\/other/);

console.log("cursor-sdk.host-env: ok");

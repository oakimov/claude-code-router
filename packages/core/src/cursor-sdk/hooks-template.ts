import { chmodSync, mkdirSync, readFileSync, writeFileSync } from "fs";
import { join } from "path";
import { EMPTY_HOST_ENVIRONMENT, type HostEnvironment } from "./host-env";
import { buildDenyGuidance, buildWorkspaceRulesDocument } from "./prompt";
import { CURSOR_BUILTIN_DENY_LIST, CUSTOM_USER_TOOLS_SERVER } from "./shared";

function buildDenyScript(denyMessage: string): string {
  return `#!/usr/bin/env node
const chunks = [];
process.stdin.on("data", (c) => chunks.push(c));
process.stdin.on("end", () => {
  let input = {};
  try {
    const raw = Buffer.concat(chunks).toString("utf8").trim();
    input = raw ? JSON.parse(raw) : {};
  } catch {
    input = {};
  }

  const server = String(
    input.server ||
      input.providerIdentifier ||
      input.mcpServer ||
      input.mcp_server ||
      ""
  );
  const toolName = String(
    input.tool_name || input.toolName || input.name || input.tool || ""
  );

  // Always allow the SDK synthetic MCP host tools Claude Code bridges through.
  if (server.includes(${JSON.stringify(CUSTOM_USER_TOOLS_SERVER)})) {
    process.stdout.write(JSON.stringify({ permission: "allow" }));
    return;
  }

  const denyMessage = ${JSON.stringify(denyMessage)};

  process.stdout.write(
    JSON.stringify({
      permission: "deny",
      agentMessage: denyMessage,
      userMessage: denyMessage + (toolName ? \` (blocked: \${toolName})\` : ""),
    })
  );
});
`;
}

/**
 * A denied built-in is where the model forms its "I am confined" belief, so the
 * denial has to carry the topology with it — not just the routing rule.
 */
function denyMessageFor(hostEnv: HostEnvironment): string {
  return [
    "CCR Cursor SDK bridge: do not use Cursor built-in tools.",
    `Call host tools via MCP server '${CUSTOM_USER_TOOLS_SERVER}' using the bare Claude Code tool names (Read, Bash, Edit, …).`,
    buildDenyGuidance(hostEnv),
  ].join(" ");
}

/** Write only on change — this runs per turn once host facts are known. */
function writeIfChanged(
  path: string,
  content: string,
  mode: number
): void {
  try {
    if (readFileSync(path, "utf-8") === content) return;
  } catch {
    // missing or unreadable — fall through to write
  }
  writeFileSync(path, content, { encoding: "utf-8", mode });
}

export function ensureDenyHooksWorkspace(
  workspaceDir: string,
  hostEnv: HostEnvironment = EMPTY_HOST_ENVIRONMENT
): void {
  const cursorDir = join(workspaceDir, ".cursor");
  const hooksDir = join(cursorDir, "hooks");
  mkdirSync(hooksDir, { recursive: true });

  const denyScriptPath = join(hooksDir, "deny-builtin.cjs");
  writeIfChanged(denyScriptPath, buildDenyScript(denyMessageFor(hostEnv)), 0o700);
  try {
    chmodSync(denyScriptPath, 0o700);
  } catch {
    // best-effort on platforms without chmod semantics
  }

  const matcher = CURSOR_BUILTIN_DENY_LIST.join("|");
  const denyCommand = `node "${denyScriptPath}"`;

  const hooksJson = {
    version: 1,
    hooks: {
      preToolUse: [
        {
          matcher,
          command: denyCommand,
        },
      ],
      beforeShellExecution: [
        {
          command: denyCommand,
        },
      ],
      beforeReadFile: [
        {
          command: denyCommand,
        },
      ],
      beforeMCPExecution: [
        {
          // Deny non-bridge MCP; allow custom-user-tools via the script logic.
          command: denyCommand,
        },
      ],
    },
  };

  writeIfChanged(
    join(cursorDir, "hooks.json"),
    JSON.stringify(hooksJson, null, 2),
    0o600
  );

  // AGENTS.md is Cursor's project-rules channel: it is injected at a higher
  // priority than the flattened user turn, so it is the only place the bridge
  // can contradict the server-built claim that this workspace is the project.
  writeIfChanged(
    join(workspaceDir, "AGENTS.md"),
    buildWorkspaceRulesDocument(workspaceDir, hostEnv),
    0o600
  );
}

import { chmodSync, mkdirSync, writeFileSync } from "fs";
import { join } from "path";
import { CURSOR_BUILTIN_DENY_LIST, CUSTOM_USER_TOOLS_SERVER } from "./shared";

const DENY_SCRIPT = `#!/usr/bin/env node
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

  const denyMessage =
    "CCR Cursor SDK bridge: do not use Cursor built-in tools. " +
    "Call host tools via MCP server '${CUSTOM_USER_TOOLS_SERVER}' " +
    "using the bare Claude Code tool names (Read, Bash, Edit, …).";

  process.stdout.write(
    JSON.stringify({
      permission: "deny",
      agentMessage: denyMessage,
      userMessage: denyMessage + (toolName ? \` (blocked: \${toolName})\` : ""),
    })
  );
});
`;

export function ensureDenyHooksWorkspace(workspaceDir: string): void {
  const cursorDir = join(workspaceDir, ".cursor");
  const hooksDir = join(cursorDir, "hooks");
  mkdirSync(hooksDir, { recursive: true });

  const denyScriptPath = join(hooksDir, "deny-builtin.cjs");
  writeFileSync(denyScriptPath, DENY_SCRIPT, { encoding: "utf-8", mode: 0o700 });
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

  writeFileSync(
    join(cursorDir, "hooks.json"),
    JSON.stringify(hooksJson, null, 2),
    { encoding: "utf-8", mode: 0o600 }
  );

  // Empty AGENTS.md so ambient project instructions aren't invented.
  writeFileSync(join(workspaceDir, "AGENTS.md"), "", {
    encoding: "utf-8",
    mode: 0o600,
  });
}

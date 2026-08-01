import assert from "node:assert/strict";
import { homedir } from "node:os";
import { join, resolve } from "node:path";
import { resolveLogFilePath } from "../server";

const LOG_DIR = resolve(join(homedir(), ".claude-code-router", "logs"));

function expectReject(filePath: string | undefined, label: string): void {
  assert.throws(
    () => resolveLogFilePath(filePath),
    (err: unknown) =>
      err instanceof Error && err.message.includes("logs directory"),
    label
  );
}

function main(): void {
  // Default when no file is provided
  assert.equal(resolveLogFilePath(), join(LOG_DIR, "app.log"));
  assert.equal(resolveLogFilePath(undefined), join(LOG_DIR, "app.log"));

  // Basename-only and absolute paths under the log dir both resolve safely
  assert.equal(resolveLogFilePath("ccr-1.log"), join(LOG_DIR, "ccr-1.log"));
  assert.equal(
    resolveLogFilePath(join(LOG_DIR, "ccr-1.log")),
    join(LOG_DIR, "ccr-1.log")
  );

  // Absolute / traversal inputs are reduced to basename — still confined to LOG_DIR
  assert.equal(
    resolveLogFilePath("/etc/passwd.log"),
    join(LOG_DIR, "passwd.log")
  );
  assert.equal(
    resolveLogFilePath("../secrets.log"),
    join(LOG_DIR, "secrets.log")
  );

  // Invalid basenames must be rejected
  expectReject("..", "dot-dot alone");
  expectReject(".", "dot alone");
  expectReject("not-a-log.txt", "non-.log extension");
  expectReject("evil..log", "embedded ..");
  // Empty string is falsy and defaults to app.log (same as undefined).
  assert.equal(resolveLogFilePath(""), join(LOG_DIR, "app.log"));
  // basename of this join is config.json — not a .log file
  expectReject(join(LOG_DIR, "..", "config.json"), "escape via join (non-.log)");

  console.log("logs-path-allowlist tests passed.");
}

main();

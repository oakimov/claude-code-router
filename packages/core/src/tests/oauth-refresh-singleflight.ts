import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getValidAccessToken as getClaudeToken } from "../utils/claude-auth";
import { getValidAccessToken as getQwenToken } from "../utils/qwen-auth";

const tempDir = mkdtempSync(join(tmpdir(), "ccr-oauth-singleflight-"));
const originalFetch = globalThis.fetch;

async function testClaudeRefreshSingleFlight(): Promise<void> {
  const authFile = join(tempDir, "claude.json");
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "old",
      refresh_token: "refresh",
      token_type: "Bearer",
      expires_at: 0,
    })
  );

  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    await new Promise((resolve) => setTimeout(resolve, 10));
    return Response.json({
      access_token: "new",
      refresh_token: "rotated",
      token_type: "Bearer",
      expires_in: 3600,
    });
  };

  const [first, second] = await Promise.all([
    getClaudeToken({ force: true }),
    getClaudeToken({ force: true }),
  ]);
  assert.equal(calls, 1);
  assert.equal(first.access_token, "new");
  assert.equal(second.refresh_token, "rotated");
  assert.equal(JSON.parse(readFileSync(authFile, "utf8")).refresh_token, "rotated");
}

async function testQwenRefreshSingleFlight(): Promise<void> {
  const authFile = join(tempDir, "qwen.json");
  process.env.CCR_QWEN_AUTH_FILE = authFile;
  writeFileSync(
    authFile,
    JSON.stringify({ token: "old", expiresAt: null, updatedAt: Date.now() })
  );

  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    await new Promise((resolve) => setTimeout(resolve, 10));
    return Response.json({ access_token: "new" });
  };

  const [first, second] = await Promise.all([
    getQwenToken({ force: true }),
    getQwenToken({ force: true }),
  ]);
  assert.equal(calls, 1);
  assert.equal(first.token, "new");
  assert.equal(second.token, "new");
  assert.equal(JSON.parse(readFileSync(authFile, "utf8")).token, "new");
}

try {
  await testClaudeRefreshSingleFlight();
  await testQwenRefreshSingleFlight();
  console.log("oauth refresh single-flight: PASS");
} finally {
  globalThis.fetch = originalFetch;
  delete process.env.CCR_CLAUDE_AUTH_FILE;
  delete process.env.CCR_QWEN_AUTH_FILE;
  rmSync(tempDir, { recursive: true, force: true });
}

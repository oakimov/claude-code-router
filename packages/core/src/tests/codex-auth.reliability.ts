import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { CodexTransformer } from "../transformer/codex.transformer";
import {
  getValidAccessToken,
  loadTokens,
  toCodexOAuthAuth,
} from "../utils/codex-auth";
import { sendWithUnauthorizedAuthRecovery } from "../utils/auth-recovery";

function jwt(payload: Record<string, unknown>): string {
  const encode = (value: unknown) =>
    Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(payload)}.signature`;
}

function idToken(accountId: string, fedramp = false): string {
  return jwt({
    "https://api.openai.com/auth": {
      chatgpt_account_id: accountId,
      chatgpt_account_is_fedramp: fedramp,
    },
  });
}

function accessToken(expiresAt: number, marker: string): string {
  return jwt({ exp: expiresAt, marker });
}

async function main() {
  const originalFetch = globalThis.fetch;
  const originalAuthFile = process.env.CCR_CODEX_AUTH_FILE;
  const tempDir = mkdtempSync(join(tmpdir(), "ccr-codex-auth-"));
  const authFile = join(tempDir, "codex_auth.json");
  process.env.CCR_CODEX_AUTH_FILE = authFile;

  try {
    const now = Math.floor(Date.now() / 1000);
    const originalIdToken = idToken("workspace-1", true);
    const originalRefreshToken = "refresh-original";
    writeFileSync(
      authFile,
      JSON.stringify({
        access_token: accessToken(now + 30, "old"),
        refresh_token: originalRefreshToken,
        id_token: originalIdToken,
        token_type: "Bearer",
        expires_at: now + 30,
      }),
      { mode: 0o600 }
    );

    let refreshCalls = 0;
    globalThis.fetch = async (_url, init) => {
      refreshCalls += 1;
      assert.equal(
        new Headers(init?.headers).get("content-type"),
        "application/json"
      );
      const body = JSON.parse(String(init?.body));
      assert.equal(body.refresh_token, originalRefreshToken);
      await new Promise((resolve) => setTimeout(resolve, 20));
      return new Response(
        JSON.stringify({
          access_token: accessToken(now + 3600, "new"),
          expires_in: 3600,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    };

    const refreshed = await Promise.all(
      Array.from({ length: 5 }, () => getValidAccessToken())
    );
    assert.equal(refreshCalls, 1, "concurrent refreshes must be single-flight");
    assert.ok(refreshed.every((tokens) => tokens.access_token.includes(".")));

    const stored = loadTokens();
    assert.ok(stored);
    assert.equal(stored.refresh_token, originalRefreshToken);
    assert.equal(stored.id_token, originalIdToken);
    assert.equal(stored.account_id, "workspace-1");
    assert.equal(statSync(authFile).mode & 0o777, 0o600);

    const resolved = toCodexOAuthAuth(stored);
    assert.equal(resolved.accountId, "workspace-1");
    assert.equal(resolved.isFedramp, true);

    const transformer = new CodexTransformer();
    const transformed = await transformer.transformRequestIn(
      {
        model: "gpt-test",
        messages: [{ role: "user", content: "hello" }],
      },
      {
        name: "codex",
        apiKey: "oauth",
        baseUrl: "https://chatgpt.com/backend-api/codex",
      },
      { req: { id: "oauth-header-test", sessionId: "session-1" } }
    );
    assert.equal(
      transformed.config.headers["ChatGPT-Account-ID"],
      "workspace-1"
    );
    assert.equal(transformed.config.headers["X-OpenAI-Fedramp"], "true");
    assert.equal(typeof transformed.config.__authRecovery, "function");

    let patCalls = 0;
    globalThis.fetch = async (_url, init) => {
      patCalls += 1;
      assert.equal(
        new Headers(init?.headers).get("authorization"),
        "Bearer at-reliability-test"
      );
      return new Response(
        JSON.stringify({
          chatgpt_account_id: "pat-workspace",
          chatgpt_account_is_fedramp: false,
          chatgpt_user_id: "user-1",
          chatgpt_plan_type: "pro",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    };
    const patTransformer = new CodexTransformer();
    const [patA, patB] = await Promise.all([
      patTransformer.transformRequestIn(
        {
          model: "gpt-test",
          messages: [{ role: "user", content: "one" }],
        },
        { apiKey: "at-reliability-test" },
        { req: { id: "pat-a" } }
      ),
      patTransformer.transformRequestIn(
        {
          model: "gpt-test",
          messages: [{ role: "user", content: "two" }],
        },
        { apiKey: "at-reliability-test" },
        { req: { id: "pat-b" } }
      ),
    ]);
    assert.equal(patCalls, 1, "PAT metadata requests must be single-flight");
    assert.equal(patA.config.headers["ChatGPT-Account-ID"], "pat-workspace");
    assert.equal(patB.config.headers["ChatGPT-Account-ID"], "pat-workspace");
    assert.equal(await patA.config.__authRecovery(), null);

    const attempts: string[] = [];
    let recoveryCalls = 0;
    const retried = await sendWithUnauthorizedAuthRecovery(
      async (headers) => {
        attempts.push(headers.Authorization);
        return attempts.length === 1
          ? new Response("unauthorized", { status: 401 })
          : new Response("ok", { status: 200 });
      },
      { Authorization: "Bearer old" },
      async () => {
        recoveryCalls += 1;
        return { Authorization: "Bearer new" };
      }
    );
    assert.equal(retried.status, 200);
    assert.deepEqual(attempts, ["Bearer old", "Bearer new"]);
    assert.equal(recoveryCalls, 1);

    let boundedAttempts = 0;
    const stillUnauthorized = await sendWithUnauthorizedAuthRecovery(
      async () => {
        boundedAttempts += 1;
        return new Response("unauthorized", { status: 401 });
      },
      { Authorization: "Bearer old" },
      async () => ({ Authorization: "Bearer new" })
    );
    assert.equal(stillUnauthorized.status, 401);
    assert.equal(boundedAttempts, 2, "401 recovery must retry only once");

    const rawStored = JSON.parse(readFileSync(authFile, "utf8"));
    assert.equal(rawStored.refresh_token, originalRefreshToken);
    console.log("Codex server/core auth reliability tests passed.");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAuthFile === undefined) delete process.env.CCR_CODEX_AUTH_FILE;
    else process.env.CCR_CODEX_AUTH_FILE = originalAuthFile;
    rmSync(tempDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

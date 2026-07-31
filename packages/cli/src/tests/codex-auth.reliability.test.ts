import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  buildCliCodexHeaders,
  resolveCliCodexAuth,
} from "../utils/codex-auth";
import { fetchWithCodexAuth } from "../utils/modelGet";
import { buildAuthorizeUrl } from "../utils/codex-cli-auth";

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

async function main() {
  const originalFetch = globalThis.fetch;
  const originalAuthFile = process.env.CCR_CODEX_AUTH_FILE;
  const tempDir = mkdtempSync(join(tmpdir(), "ccr-cli-codex-auth-"));
  const authFile = join(tempDir, "codex_auth.json");
  process.env.CCR_CODEX_AUTH_FILE = authFile;

  try {
    const authorizeUrl = new URL(
      buildAuthorizeUrl("challenge", "oauth-state")
    );
    assert.equal(
      authorizeUrl.searchParams.get("id_token_add_organizations"),
      "true"
    );
    assert.equal(
      authorizeUrl.searchParams.get("codex_cli_simplified_flow"),
      "true"
    );
    assert.equal(authorizeUrl.searchParams.get("originator"), "codex_cli_rs");
    assert.match(
      authorizeUrl.searchParams.get("scope") || "",
      /offline_access/
    );

    const now = Math.floor(Date.now() / 1000);
    const originalIdToken = idToken("cli-workspace", true);
    writeFileSync(
      authFile,
      JSON.stringify({
        access_token: jwt({ exp: now + 30, marker: "old" }),
        refresh_token: "cli-refresh",
        id_token: originalIdToken,
        token_type: "Bearer",
        expires_at: now + 30,
      }),
      { mode: 0o600 }
    );

    let refreshCalls = 0;
    globalThis.fetch = async (_url, init) => {
      refreshCalls += 1;
      const body = JSON.parse(String(init?.body));
      assert.equal(body.refresh_token, "cli-refresh");
      await new Promise((resolve) => setTimeout(resolve, 20));
      return new Response(
        JSON.stringify({
          access_token: jwt({ exp: now + 3600, marker: "new" }),
          expires_in: 3600,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    };

    const auths = await Promise.all(
      Array.from({ length: 5 }, () => resolveCliCodexAuth(undefined))
    );
    assert.equal(refreshCalls, 1, "CLI refresh must be single-flight");
    assert.ok(auths.every((auth) => auth.accountId === "cli-workspace"));
    assert.ok(auths.every((auth) => auth.isFedramp));

    const stored = JSON.parse(readFileSync(authFile, "utf8"));
    assert.equal(stored.refresh_token, "cli-refresh");
    assert.equal(stored.id_token, originalIdToken);
    assert.equal(stored.account_id, "cli-workspace");

    const headers = buildCliCodexHeaders(auths[0], "0.test");
    assert.equal(headers["ChatGPT-Account-ID"], "cli-workspace");
    assert.equal(headers["X-OpenAI-Fedramp"], "true");
    assert.equal(headers.originator, "codex_cli_rs");

    let patCalls = 0;
    globalThis.fetch = async (_url, init) => {
      patCalls += 1;
      assert.equal(
        new Headers(init?.headers).get("authorization"),
        "Bearer at-cli-test"
      );
      return new Response(
        JSON.stringify({
          chatgpt_account_id: "pat-cli-workspace",
          chatgpt_account_is_fedramp: false,
          chatgpt_user_id: "pat-user",
          chatgpt_plan_type: "pro",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    };
    const pat = await resolveCliCodexAuth("at-cli-test");
    assert.equal(patCalls, 1);
    assert.equal(pat.accountId, "pat-cli-workspace");

    process.env.CODEX_CLI_PAT_TEST = "at-cli-test";
    const envPat = await resolveCliCodexAuth("${CODEX_CLI_PAT_TEST}");
    const bareEnvPat = await resolveCliCodexAuth("CODEX_CLI_PAT_TEST");
    assert.equal(envPat.accountId, "pat-cli-workspace");
    assert.equal(bareEnvPat.accountId, "pat-cli-workspace");
    delete process.env.CODEX_CLI_PAT_TEST;

    writeFileSync(
      authFile,
      JSON.stringify({
        access_token: jwt({ exp: now + 3600, marker: "request-old" }),
        refresh_token: "request-refresh",
        id_token: idToken("request-workspace"),
        token_type: "Bearer",
        expires_at: now + 3600,
      }),
      { mode: 0o600 }
    );

    const calls: Array<{ url: string; headers: Headers }> = [];
    globalThis.fetch = async (url, init) => {
      calls.push({ url: String(url), headers: new Headers(init?.headers) });
      if (calls.length === 1) {
        return new Response("unauthorized", { status: 401 });
      }
      if (calls.length === 2) {
        assert.match(String(url), /oauth\/token/);
        return new Response(
          JSON.stringify({
            access_token: jwt({ exp: now + 3600, marker: "request-new" }),
            expires_in: 3600,
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      return new Response(JSON.stringify({ models: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };

    const response = await fetchWithCodexAuth(
      "https://chatgpt.com/backend-api/codex/models",
      {
        name: "codex",
        api_key: "oauth",
        api_base_url: "https://chatgpt.com/backend-api/codex",
        models: [],
      } as any
    );
    assert.equal(response.status, 200);
    assert.equal(calls.length, 3);
    assert.equal(
      calls[0].headers.get("chatgpt-account-id"),
      "request-workspace"
    );
    assert.equal(
      calls[2].headers.get("chatgpt-account-id"),
      "request-workspace"
    );
    assert.match(calls[0].url, /client_version=/);
    assert.match(calls[2].url, /client_version=/);

    console.log("Codex CLI auth reliability tests passed.");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAuthFile === undefined) delete process.env.CCR_CODEX_AUTH_FILE;
    else process.env.CCR_CODEX_AUTH_FILE = originalAuthFile;
    delete process.env.CODEX_CLI_PAT_TEST;
    rmSync(tempDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

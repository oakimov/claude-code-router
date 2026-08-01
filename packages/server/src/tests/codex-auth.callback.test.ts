import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify from "fastify";
import { registerCodexAuthRoutes } from "../routes/codex-auth";

function jwt(payload: Record<string, unknown>): string {
  const encode = (value: unknown) =>
    Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(payload)}.signature`;
}

async function main() {
  const originalFetch = globalThis.fetch;
  const originalAuthFile = process.env.CCR_CODEX_AUTH_FILE;
  const originalVerifierFile = process.env.CCR_CODEX_VERIFIER_FILE;
  const tempDir = mkdtempSync(join(tmpdir(), "ccr-codex-callback-"));
  const authFile = join(tempDir, "codex_auth.json");
  const verifierFile = join(tempDir, "codex_verifier.tmp");
  process.env.CCR_CODEX_AUTH_FILE = authFile;
  process.env.CCR_CODEX_VERIFIER_FILE = verifierFile;

  const app = Fastify({ logger: false });
  await registerCodexAuthRoutes(app);

  try {
    writeFileSync(
      verifierFile,
      JSON.stringify({
        code_verifier: "verifier",
        state: "expected-state",
        created_at: Date.now(),
      }),
      { mode: 0o600 }
    );

    let exchanges = 0;
    globalThis.fetch = async (_url, init) => {
      exchanges += 1;
      assert.equal(
        new Headers(init?.headers).get("content-type"),
        "application/x-www-form-urlencoded"
      );
      const body = new URLSearchParams(String(init?.body));
      assert.equal(body.get("code"), "authorization-code");
      assert.equal(body.get("code_verifier"), "verifier");
      const now = Math.floor(Date.now() / 1000);
      return new Response(
        JSON.stringify({
          access_token: jwt({ exp: now + 3600 }),
          refresh_token: "refresh-secret",
          id_token: jwt({
            "https://api.openai.com/auth": {
              chatgpt_account_id: "callback-workspace",
              chatgpt_account_is_fedramp: true,
            },
          }),
          expires_in: 3600,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    };

    const response = await app.inject({
      method: "GET",
      url: "/auth/callback?code=authorization-code&state=expected-state",
    });
    assert.equal(response.statusCode, 200);
    assert.match(response.body, /Authentication Successful/);
    assert.equal(exchanges, 1);
    assert.equal(existsSync(verifierFile), false);

    const stored = JSON.parse(readFileSync(authFile, "utf8"));
    assert.equal(stored.account_id, "callback-workspace");
    assert.equal(stored.account_is_fedramp, true);
    assert.equal(stored.refresh_token, "refresh-secret");

    writeFileSync(
      verifierFile,
      JSON.stringify({
        code_verifier: "expired",
        state: "expired-state",
        created_at: Date.now() - 6 * 60 * 1000,
      }),
      { mode: 0o600 }
    );
    const expired = await app.inject({
      method: "GET",
      url: "/auth/callback?code=unused&state=expired-state",
    });
    assert.match(expired.body, /authorization request expired/);
    assert.equal(exchanges, 1, "expired verifier must not exchange a code");
    assert.equal(existsSync(verifierFile), false);

    const escaped = await app.inject({
      method: "GET",
      url: "/auth/callback?error=%3Cscript%3E&error_description=%3Cb%3Ebad%3C%2Fb%3E",
    });
    assert.doesNotMatch(escaped.body, /<script>/i);
    assert.doesNotMatch(escaped.body, /<b>bad<\/b>/i);

    console.log("Codex OAuth callback tests passed.");
  } finally {
    await app.close();
    globalThis.fetch = originalFetch;
    if (originalAuthFile === undefined) delete process.env.CCR_CODEX_AUTH_FILE;
    else process.env.CCR_CODEX_AUTH_FILE = originalAuthFile;
    if (originalVerifierFile === undefined) {
      delete process.env.CCR_CODEX_VERIFIER_FILE;
    } else {
      process.env.CCR_CODEX_VERIFIER_FILE = originalVerifierFile;
    }
    rmSync(tempDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

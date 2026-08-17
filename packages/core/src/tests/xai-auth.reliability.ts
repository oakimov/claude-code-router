import assert from "node:assert/strict";
import { mkdtempSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { XaiAuthTransformer } from "../transformer/xai-auth.transformer";
import {
  getValidAccessToken,
  isTokenExpiring,
  loadTokens,
  pollDeviceCodeToken,
  refreshTokens,
  type XaiTokens,
} from "../utils/xai-auth";

function jwt(payload: Record<string, unknown>): string {
  const encode = (value: unknown) => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(payload)}.signature`;
}

async function testIsTokenExpiring() {
  const now = Date.now() / 1000;

  // Stored expires_at path.
  const opaqueToken: XaiTokens = {
    access_token: "opaque-token",
    token_type: "Bearer",
    expires_at: now + 30,
  };
  assert.equal(isTokenExpiring(opaqueToken, 120), true, "within skew window must be expiring");
  assert.equal(isTokenExpiring({ ...opaqueToken, expires_at: now + 3600 }, 120), false);

  // JWT exp claim takes priority over a stale/missing stored expires_at.
  const jwtToken: XaiTokens = {
    access_token: jwt({ exp: Math.floor(now + 30) }),
    token_type: "Bearer",
    expires_at: now + 3600, // stale/incorrect stored value
  };
  assert.equal(isTokenExpiring(jwtToken, 120), true, "JWT exp must win over stored expires_at");

  // No expiry information at all (opaque token, no stored expires_at) never
  // proactively expires — the 401-on-call path drives refresh instead.
  const noExpiry: XaiTokens = { access_token: "opaque", token_type: "Bearer" };
  assert.equal(isTokenExpiring(noExpiry, 120), false);
}

async function testPollDeviceCodeToken() {
  const device = {
    device_code: "device-abc",
    user_code: "ABCD-EFGH",
    verification_uri: "https://x.ai/device",
    expires_in: 60,
    interval: 1,
  };

  // authorization_pending then success.
  {
    let calls = 0;
    const sleeps: number[] = [];
    globalThis.fetch = (async () => {
      calls += 1;
      if (calls < 3) {
        return new Response(JSON.stringify({ error: "authorization_pending" }), { status: 400 });
      }
      return new Response(
        JSON.stringify({ access_token: "at-1", refresh_token: "rt-1", expires_in: 3600 }),
        { status: 200 }
      );
    }) as typeof fetch;

    const tokens = await pollDeviceCodeToken(device, {
      sleep: async (ms) => {
        sleeps.push(ms);
      },
      now: (() => {
        let t = 0;
        return () => (t += 1);
      })(),
    });
    assert.equal(tokens.access_token, "at-1");
    assert.equal(calls, 3, "authorization_pending must keep polling until success");
    assert.equal(sleeps.length, 2, "must sleep once per pending response");
  }

  // slow_down backs off, then success.
  {
    let calls = 0;
    const sleeps: number[] = [];
    globalThis.fetch = (async () => {
      calls += 1;
      if (calls === 1) return new Response(JSON.stringify({ error: "slow_down" }), { status: 400 });
      return new Response(JSON.stringify({ access_token: "at-2", expires_in: 3600 }), { status: 200 });
    }) as typeof fetch;

    await pollDeviceCodeToken(device, {
      sleep: async (ms) => {
        sleeps.push(ms);
      },
      now: (() => {
        let t = 0;
        return () => (t += 1);
      })(),
    });
    assert.equal(sleeps.length, 1);
    assert.ok(sleeps[0] >= 5_000 + 3_000, "slow_down must extend the poll interval");
  }

  // expired_token is terminal.
  {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ error: "expired_token" }), { status: 400 })) as typeof fetch;
    await assert.rejects(
      () => pollDeviceCodeToken(device, { sleep: async () => {}, now: () => 0 }),
      /expired/
    );
  }

  // access_denied is terminal.
  {
    globalThis.fetch = (async () =>
      new Response(JSON.stringify({ error: "access_denied" }), { status: 400 })) as typeof fetch;
    await assert.rejects(
      () => pollDeviceCodeToken(device, { sleep: async () => {}, now: () => 0 }),
      /denied/
    );
  }
}

async function testRefreshAndAuthRecovery() {
  const originalFetch = globalThis.fetch;
  const originalAuthFile = process.env.CCR_XAI_AUTH_FILE;
  const tempDir = mkdtempSync(join(tmpdir(), "ccr-xai-auth-"));
  const authFile = join(tempDir, "xai_auth.json");
  process.env.CCR_XAI_AUTH_FILE = authFile;

  try {
    const now = Date.now() / 1000;
    const originalRefreshToken = "refresh-original";
    writeFileSync(
      authFile,
      JSON.stringify({
        access_token: "opaque-old",
        refresh_token: originalRefreshToken,
        token_type: "Bearer",
        expires_at: now + 30,
      }),
      { mode: 0o600 }
    );

    let refreshCalls = 0;
    globalThis.fetch = (async (_url, init) => {
      refreshCalls += 1;
      assert.equal(
        new Headers(init?.headers).get("content-type"),
        "application/x-www-form-urlencoded"
      );
      const params = new URLSearchParams(String(init?.body));
      assert.ok(params.get("refresh_token"), "refresh request must send a refresh_token");
      assert.equal(params.get("grant_type"), "refresh_token");
      await new Promise((resolve) => setTimeout(resolve, 10));
      return new Response(
        JSON.stringify({
          access_token: "opaque-new",
          refresh_token: "refresh-rotated",
          expires_in: 3600,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      );
    }) as typeof fetch;

    // Concurrent refreshes collapse onto a single HTTP call (single-flight).
    const refreshed = await Promise.all(Array.from({ length: 5 }, () => getValidAccessToken()));
    assert.equal(refreshCalls, 1, "concurrent refreshes must be single-flight");
    assert.ok(refreshed.every((t) => t.access_token === "opaque-new"));

    // The exported refreshTokens() used by 401 recovery must share that
    // single-flight; xAI rotates refresh tokens on every use.
    refreshCalls = 0;
    const forced = await Promise.all(
      Array.from({ length: 3 }, () => refreshTokens("refresh-rotated"))
    );
    assert.equal(
      refreshCalls,
      1,
      "refreshTokens must share getValidAccessToken single-flight"
    );
    assert.ok(forced.every((t) => t.access_token === "opaque-new"));

    const stored = loadTokens();
    assert.ok(stored);
    assert.equal(stored!.refresh_token, "refresh-rotated", "rotated refresh token must be persisted");
    assert.equal(statSync(authFile).mode & 0o777, 0o600);

    // __authRecovery: PAT mode short-circuits to null, no network call.
    // Also verifies the outbound URL: openai-responses only converts the
    // body (it doesn't own the URL), so xai-auth must build `${baseUrl}/responses`
    // itself, matching CodexTransformer's pattern (routes.ts falls back to a
    // bare provider.baseUrl with no path when no transformer sets config.url).
    const patTransformer = new XaiAuthTransformer();
    const patResult = await patTransformer.transformRequestIn(
      { model: "grok-test", messages: [] },
      { apiKey: "xai-literal-key", baseUrl: "https://api.x.ai/v1" }
    );
    assert.equal(patResult.config.headers.Authorization, "Bearer xai-literal-key");
    assert.equal(patResult.config.url, "https://api.x.ai/v1/responses");
    assert.equal(await patResult.config.__authRecovery(), null);

    // __authRecovery: OAuth mode reload-then-refresh.
    const oauthTransformer = new XaiAuthTransformer();
    const oauthResult = await oauthTransformer.transformRequestIn(
      { model: "grok-test", messages: [] },
      { apiKey: "no-key", baseUrl: "https://api.x.ai/v1" }
    );
    assert.equal(oauthResult.config.headers.Authorization, "Bearer opaque-new");
    assert.equal(oauthResult.config.url, "https://api.x.ai/v1/responses");

    // Simulate another process having already refreshed on disk.
    writeFileSync(
      authFile,
      JSON.stringify({
        access_token: "opaque-newer",
        refresh_token: "refresh-rotated",
        token_type: "Bearer",
        expires_at: now + 3600,
      }),
      { mode: 0o600 }
    );
    refreshCalls = 0;
    const recovered = await oauthResult.config.__authRecovery();
    assert.deepEqual(recovered, { Authorization: "Bearer opaque-newer" });
    assert.equal(refreshCalls, 0, "reload-newer-token must short-circuit before any refresh call");

    console.log("xAI core auth reliability tests passed.");
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAuthFile === undefined) delete process.env.CCR_XAI_AUTH_FILE;
    else process.env.CCR_XAI_AUTH_FILE = originalAuthFile;
    rmSync(tempDir, { recursive: true, force: true });
  }
}

async function main() {
  await testIsTokenExpiring();
  await testPollDeviceCodeToken();
  await testRefreshAndAuthRecovery();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

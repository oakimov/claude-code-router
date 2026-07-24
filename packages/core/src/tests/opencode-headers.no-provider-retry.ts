import assert from "node:assert/strict";
import { OpencodeHeadersTransformer } from "../transformer/opencode-headers.transformer";

// Verifies the opencode session self-heal: a Zen `No provider available` 401
// triggers a session re-roll + retry (contained in the transformer), while any
// other non-ok response is surfaced as a normal provider_response_error.

type Captured = { headers: Record<string, string>; url: string };

function makeContext() {
  return {
    req: {
      sessionId: "conv-1",
      log: { warn() {}, info() {}, debug() {} },
      server: { configService: { getHttpsProxy: () => undefined } },
    },
  };
}

const NO_PROVIDER_BODY = JSON.stringify({
  error: { type: "ModelError", message: "No provider available" },
});

const UPSTREAM_FAILED_BODY = JSON.stringify({
  error: {
    message: "Error from provider (Console): Upstream request failed",
    type: "invalid_request_error",
    code: "invalid_request_error",
  },
});

function installFetch(sequence: Array<() => Response>): Captured[] {
  const calls: Captured[] = [];
  (globalThis as any).fetch = async (url: any, init: any) => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((v, k) => (headers[k] = v));
    calls.push({ headers, url: String(url) });
    return sequence[Math.min(calls.length - 1, sequence.length - 1)]();
  };
  return calls;
}

const provider = {
  name: "opencode-openai",
  apiKey: "test-key",
  baseUrl: "https://opencode.ai/zen/v1/chat/completions",
};

const body = {
  model: "deepseek-v4-flash-free",
  messages: [{ role: "user", content: "hi" }],
};

async function main() {
  const originalFetch = (globalThis as any).fetch;

  try {
    // --- Case 1: recover after two "No provider available" 401s ---
    {
      const calls = installFetch([
        () => new Response(NO_PROVIDER_BODY, { status: 401 }),
        () => new Response(NO_PROVIDER_BODY, { status: 401 }),
        () =>
          new Response(JSON.stringify({ id: "ok", choices: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
      ]);

      const t = new OpencodeHeadersTransformer();
      const result = await t.transformRequestIn(body, provider, makeContext());
      const response: Response = result.config.__providerResponse;

      assert.equal(response.status, 200, "should recover to a 200 after retries");
      assert.equal(calls.length, 3, "should have attempted 3 upstream calls");

      const sessions = calls.map((c) => c.headers["x-opencode-session"]);
      assert.ok(
        sessions[0] && sessions[1] && sessions[2],
        "every attempt sends a session header"
      );
      assert.notEqual(
        sessions[0],
        sessions[1],
        "session must be re-rolled between attempt 1 and 2"
      );
      assert.notEqual(
        sessions[1],
        sessions[2],
        "session must be re-rolled between attempt 2 and 3"
      );
      // Auth + client identity preserved on every attempt.
      for (const c of calls) {
        assert.equal(c.headers["x-api-key"], "test-key");
        assert.equal(c.headers["user-agent"], "opencode/1.18.4");
        assert.equal(c.headers["x-opencode-client"], "cli");
        // sendUnifiedRequest drops falsy header values, so authorization is
        // never sent upstream (the "undefined" string is filtered out).
        assert.notEqual(c.headers.authorization, "Bearer test-key");
      }
      console.log("✓ case 1: recovers after No provider available with new sessions");
    }

    // --- Case 2: a genuine 401 (bad key) is NOT retried, surfaces as error ---
    {
      const calls = installFetch([
        () =>
          new Response(
            JSON.stringify({ error: { message: "Invalid API key" } }),
            { status: 401 }
          ),
      ]);

      const t = new OpencodeHeadersTransformer();
      await assert.rejects(
        () => t.transformRequestIn(body, provider, makeContext()),
        (err: any) => {
          assert.equal(err.statusCode, 401);
          assert.equal(err.code, "provider_response_error");
          return true;
        },
        "genuine 401 must throw provider_response_error"
      );
      assert.equal(calls.length, 1, "genuine 401 must NOT be retried");
      console.log("✓ case 2: genuine 401 is not retried and surfaces normally");
    }

    // --- Case 2b: 400 "Upstream request failed" is treated as routing failure ---
    {
      const calls = installFetch([
        () => new Response(UPSTREAM_FAILED_BODY, { status: 400 }),
        () =>
          new Response(JSON.stringify({ id: "ok", choices: [] }), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
      ]);

      const t = new OpencodeHeadersTransformer();
      const result = await t.transformRequestIn(body, provider, makeContext());
      const response: Response = result.config.__providerResponse;

      assert.equal(response.status, 200, "400 upstream-failed should recover");
      assert.equal(calls.length, 2, "should retry once after the 400 routing failure");
      assert.notEqual(
        calls[0].headers["x-opencode-session"],
        calls[1].headers["x-opencode-session"],
        "session must be re-rolled after a 400 routing failure"
      );
      console.log("✓ case 2b: 400 'Upstream request failed' re-rolls and recovers");
    }

    // --- Case 2c: a genuine 400 (validation) is NOT retried ---
    {
      const calls = installFetch([
        () =>
          new Response(
            JSON.stringify({
              error: { message: "max_tokens is required", type: "invalid_request_error" },
            }),
            { status: 400 }
          ),
      ]);

      const t = new OpencodeHeadersTransformer();
      await assert.rejects(
        () => t.transformRequestIn(body, provider, makeContext()),
        (err: any) => {
          assert.equal(err.statusCode, 400);
          assert.equal(err.code, "provider_response_error");
          return true;
        },
        "genuine 400 must throw provider_response_error"
      );
      assert.equal(calls.length, 1, "genuine 400 must NOT be retried");
      console.log("✓ case 2c: genuine 400 validation error is not retried");
    }

    // --- Case 3: persistent No provider available exhausts retries then throws ---
    {
      const calls = installFetch([
        () => new Response(NO_PROVIDER_BODY, { status: 401 }),
      ]);

      const t = new OpencodeHeadersTransformer();
      await assert.rejects(
        () => t.transformRequestIn(body, provider, makeContext()),
        (err: any) => {
          assert.equal(err.statusCode, 401);
          assert.equal(err.code, "provider_response_error");
          return true;
        },
        "exhausted retries must throw"
      );
      assert.equal(calls.length, 3, "should try exactly MAX_NO_PROVIDER_RETRIES times");
      console.log("✓ case 3: persistent failure exhausts retries then errors");
    }

    console.log("\nAll opencode no-provider retry tests passed.");
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

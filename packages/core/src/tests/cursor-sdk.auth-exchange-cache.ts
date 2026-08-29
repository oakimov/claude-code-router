import assert from "node:assert/strict";
import {
  __resetCursorAuthExchangeCacheForTests,
  installCursorAuthExchangeCache,
} from "../cursor-sdk/auth-exchange-cache";

const EXCHANGE = "https://api2.cursor.sh/auth/exchange_user_api_key";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function isAbortError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const candidate = error as { name?: string; code?: string };
  return (
    candidate.name === "AbortError" ||
    candidate.code === "ABORT_ERR" ||
    (typeof DOMException !== "undefined" &&
      error instanceof DOMException &&
      error.name === "AbortError")
  );
}

async function main() {
  const original = globalThis.fetch;
  try {
    __resetCursorAuthExchangeCacheForTests();
    let exchanges = 0;
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(
        typeof input === "string" ? input : (input as any).url || input
      );
      if (url.includes("exchange_user_api_key")) {
        exchanges += 1;
        const auth = new Headers(init?.headers).get("authorization");
        assert.equal(auth, "Bearer crsr_test");
        return jsonResponse({ accessToken: "tok_live" });
      }
      if (url.includes("/secret")) {
        const auth = new Headers(init?.headers).get("authorization");
        if (auth === "Bearer tok_live") {
          return jsonResponse({ ok: false }, 401);
        }
        return jsonResponse({ ok: true });
      }
      return jsonResponse({ ok: true });
    }) as typeof fetch;

    installCursorAuthExchangeCache();

    const first = await fetch(EXCHANGE, {
      method: "POST",
      headers: { Authorization: "Bearer crsr_test" },
      body: "{}",
    });
    const second = await fetch(EXCHANGE, {
      method: "POST",
      headers: { Authorization: "Bearer crsr_test" },
      body: "{}",
    });
    assert.equal((await first.json()).accessToken, "tok_live");
    assert.equal((await second.json()).accessToken, "tok_live");
    assert.equal(exchanges, 1, "second exchange must use the cached access token");

    exchanges = 0;
    const parallel = await Promise.all([
      fetch(EXCHANGE, {
        method: "POST",
        headers: { Authorization: "Bearer crsr_test" },
        body: "{}",
      }),
      fetch(EXCHANGE, {
        method: "POST",
        headers: { Authorization: "Bearer crsr_test" },
        body: "{}",
      }),
    ]);
    assert.equal((await parallel[0].json()).accessToken, "tok_live");
    assert.equal((await parallel[1].json()).accessToken, "tok_live");
    assert.equal(exchanges, 0, "cached token must satisfy parallel callers");

    const unauthorized = await fetch("https://api2.cursor.sh/secret", {
      headers: { Authorization: "Bearer tok_live" },
    });
    assert.equal(unauthorized.status, 401);

    exchanges = 0;
    const afterInvalidate = await fetch(EXCHANGE, {
      method: "POST",
      headers: { Authorization: "Bearer crsr_test" },
      body: "{}",
    });
    assert.equal((await afterInvalidate.json()).accessToken, "tok_live");
    assert.equal(
      exchanges,
      1,
      "401 must drop the cached token so the next exchange hits the network"
    );

    // Cancel during coalesce must not abort the shared network exchange.
    __resetCursorAuthExchangeCacheForTests();
    exchanges = 0;
    let lastExchangeHadSignal = false;
    let releaseSlow: (() => void) | undefined;
    const slowGate = new Promise<void>((resolve) => {
      releaseSlow = resolve;
    });
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(
        typeof input === "string" ? input : (input as any).url || input
      );
      if (url.includes("exchange_user_api_key")) {
        exchanges += 1;
        lastExchangeHadSignal = Boolean(init?.signal);
        await slowGate;
        return jsonResponse({ accessToken: "tok_after_abort" });
      }
      return jsonResponse({ ok: true });
    }) as typeof fetch;
    installCursorAuthExchangeCache();

    const controllerA = new AbortController();
    const controllerB = new AbortController();

    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown) => {
      unhandled.push(reason);
    };
    process.on("unhandledRejection", onUnhandled);

    try {
      const aborted = fetch(EXCHANGE, {
        method: "POST",
        headers: { Authorization: "Bearer crsr_test" },
        body: "{}",
        signal: controllerA.signal,
      });
      const survivor = fetch(EXCHANGE, {
        method: "POST",
        headers: { Authorization: "Bearer crsr_test" },
        body: "{}",
        signal: controllerB.signal,
      });

      await new Promise((r) => setTimeout(r, 20));
      assert.equal(exchanges, 1, "coalesced callers must share one network exchange");
      assert.equal(
        lastExchangeHadSignal,
        false,
        "shared exchange must not carry a caller AbortSignal"
      );

      controllerA.abort();
      await assert.rejects(aborted, (err: unknown) => isAbortError(err));

      releaseSlow!();
      const survivorBody = await survivor;
      assert.equal((await survivorBody.json()).accessToken, "tok_after_abort");
      assert.equal(exchanges, 1, "aborted waiter must not force a second exchange");

      await new Promise((r) => setImmediate(r));
      assert.equal(
        unhandled.length,
        0,
        `canceled waiter must not leave unhandledRejection: ${String(unhandled[0])}`
      );
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  } finally {
    __resetCursorAuthExchangeCacheForTests();
    globalThis.fetch = original;
  }
  console.log("cursor-sdk.auth-exchange-cache: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

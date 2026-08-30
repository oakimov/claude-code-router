import assert from "node:assert/strict";
import { setTimeout as sleep } from "node:timers/promises";
import { OpencodeHeadersTransformer } from "../transformer/opencode-headers.transformer";

type Captured = {
  headers: Record<string, string>;
  url: string;
};

function makeContext(signal?: AbortSignal) {
  return {
    signal,
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

function installFetch(sequence: Array<() => Response | Promise<Response>>): Captured[] {
  const calls: Captured[] = [];
  (globalThis as any).fetch = async (url: any, init: any) => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((value, key) => (headers[key] = value));
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

function okResponse() {
  return new Response(JSON.stringify({ id: "ok", choices: [] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function assertIdentityHeaders(calls: Captured[]) {
  for (const call of calls) {
    assert.equal(call.headers["x-api-key"], "test-key");
    assert.equal(call.headers["x-opencode-project"], "global");
    assert.equal(call.headers["x-opencode-client"], "cli");
    assert.equal(call.headers["user-agent"], "opencode/1.18.25");
    assert.match(call.headers["x-opencode-session"], /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/);
    assert.match(call.headers["x-opencode-request"], /^msg_[0-9a-f]{12}[0-9A-Za-z]{14}$/);
    assert.equal(call.headers["x-session-affinity"], undefined);
    assert.equal(call.headers["x-session-id"], undefined);
    assert.equal(call.headers["x-parent-session-id"], undefined);
    assert.equal(call.headers.authorization, undefined);
  }
}

async function recoverRoutingFailures() {
  const calls = installFetch([
    () => new Response(NO_PROVIDER_BODY, { status: 401 }),
    () => new Response(NO_PROVIDER_BODY, { status: 401 }),
    okResponse,
  ]);

  const result = await new OpencodeHeadersTransformer().transformRequestIn(
    body,
    provider,
    makeContext()
  );

  assert.equal(result.config.__providerResponse.status, 200);
  assert.equal(calls.length, 3);
  assertIdentityHeaders(calls);
  assert.notEqual(calls[0].headers["x-opencode-session"], calls[1].headers["x-opencode-session"]);
  assert.notEqual(calls[1].headers["x-opencode-session"], calls[2].headers["x-opencode-session"]);
  assert.equal(calls[0].headers["x-opencode-request"], calls[1].headers["x-opencode-request"]);
}

async function recoverWrappedRoutingFailure() {
  const calls = installFetch([
    () => new Response(UPSTREAM_FAILED_BODY, { status: 400 }),
    okResponse,
  ]);

  const result = await new OpencodeHeadersTransformer().transformRequestIn(
    body,
    provider,
    makeContext()
  );

  assert.equal(result.config.__providerResponse.status, 200);
  assert.equal(calls.length, 2);
  assert.notEqual(calls[0].headers["x-opencode-session"], calls[1].headers["x-opencode-session"]);
}

async function retryTransientStatusesWithAffinity() {
  for (const status of [429, 529]) {
    const calls = installFetch([
      () =>
        new Response(JSON.stringify({ error: { message: "temporary" } }), {
          status,
          headers: { "retry-after-ms": "0" },
        }),
      okResponse,
    ]);

    const result = await new OpencodeHeadersTransformer().transformRequestIn(
      body,
      provider,
      makeContext()
    );

    assert.equal(result.config.__providerResponse.status, 200);
    assert.equal(calls.length, 2);
    assert.equal(calls[0].headers["x-opencode-session"], calls[1].headers["x-opencode-session"]);
    assert.equal(calls[0].headers["x-opencode-request"], calls[1].headers["x-opencode-request"]);
  }
}

async function preserveTerminalErrors() {
  for (const fixture of [
    {
      status: 401,
      body: JSON.stringify({ error: { message: "Invalid API key" } }),
    },
    {
      status: 400,
      body: JSON.stringify({
        error: { message: "max_tokens is required", type: "invalid_request_error" },
      }),
    },
    {
      status: 403,
      body: JSON.stringify({ error: { message: "Workspace is blocked" } }),
    },
    {
      status: 400,
      body: JSON.stringify({
        error: {
          message:
            "Error from provider (Console): Upstream request failed: [400] 6 validation errors: Field required",
          type: "invalid_request_error",
        },
      }),
    },
  ]) {
    const calls = installFetch([
      () => new Response(fixture.body, { status: fixture.status }),
    ]);
    await assert.rejects(
      () =>
        new OpencodeHeadersTransformer().transformRequestIn(
          body,
          provider,
          makeContext()
        ),
      (error: any) => {
        assert.equal(error.statusCode, fixture.status);
        assert.equal(error.code, "provider_response_error");
        return true;
      }
    );
    assert.equal(calls.length, 1);
  }
}

async function preserveFinalRetryHeaders() {
  const calls = installFetch([
    () =>
      new Response(JSON.stringify({ error: { message: "busy" } }), {
        status: 529,
        headers: { "retry-after-ms": "0" },
      }),
  ]);

  await assert.rejects(
    () =>
      new OpencodeHeadersTransformer().transformRequestIn(
        body,
        provider,
        makeContext()
      ),
    (error: any) => {
      assert.equal(error.statusCode, 529);
      assert.equal(error.code, "provider_response_error");
      assert.equal(error.headers["Retry-After-Ms"], "0");
      return true;
    }
  );
  assert.equal(calls.length, 5);
  assert.equal(new Set(calls.map((call) => call.headers["x-opencode-session"])).size, 1);
}

async function abortsPendingBackoff() {
  const controller = new AbortController();
  const calls = installFetch([
    () =>
      new Response(JSON.stringify({ error: { message: "busy" } }), {
        status: 429,
        headers: { "retry-after": "30" },
      }),
  ]);

  const pending = new OpencodeHeadersTransformer().transformRequestIn(
    body,
    provider,
    makeContext(controller.signal)
  );
  await sleep(10);
  controller.abort("client disconnected");
  await assert.rejects(pending, (error: any) => error?.name === "AbortError");
  assert.equal(calls.length, 1);
}

async function retriesTransportFailureWithAffinity() {
  const calls = installFetch([
    async () => {
      throw Object.assign(new Error("fetch failed"), { code: "ECONNRESET" });
    },
    okResponse,
  ]);

  const result = await new OpencodeHeadersTransformer().transformRequestIn(
    body,
    provider,
    makeContext()
  );
  assert.equal(result.config.__providerResponse.status, 200);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].headers["x-opencode-session"], calls[1].headers["x-opencode-session"]);
  assert.equal(calls[0].headers["x-opencode-request"], calls[1].headers["x-opencode-request"]);
}

async function streamFailuresAreErrors() {
  for (const finishReason of ["network error", "network-error", "network_error"]) {
    const transformer = new OpencodeHeadersTransformer();
    const response = await transformer.transformResponseOut(
      new Response(
        `data: ${JSON.stringify({
          id: "c",
          choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
        })}\n\n`,
        { headers: { "content-type": "text/event-stream" } }
      )
    );
    await assert.rejects(
      () => response.text(),
      (error: any) => {
        assert.equal(error.code, "provider_network_error");
        assert.match(error.message, /finish_reason/);
        return true;
      }
    );
  }

  const structured = await new OpencodeHeadersTransformer().transformResponseOut(
    new Response(
      'data: {"error":{"message":"The model is temporarily at capacity"}}\n\n',
      { headers: { "content-type": "text/event-stream" } }
    )
  );
  await assert.rejects(
    () => structured.text(),
    (error: any) => {
      assert.equal(error.code, "provider_network_error");
      assert.match(error.message, /temporarily at capacity/);
      return true;
    }
  );
}

async function streamCancellationPropagates() {
  let cancelled = false;
  const upstream = new ReadableStream<Uint8Array>({
    pull() {},
    cancel() {
      cancelled = true;
    },
  });
  const response = await new OpencodeHeadersTransformer().transformResponseOut(
    new Response(upstream, { headers: { "content-type": "text/event-stream" } })
  );
  await response.body!.cancel("stop");
  assert.equal(cancelled, true);
}

async function zenModelIsNeverSent() {
  // The real opencode client never emits x-zen-model; Zen's edge worker derives
  // it from the request body after backend selection and overwrites/deletes any
  // inbound value. Wire fidelity means we must not send it either.
  const calls = installFetch([okResponse]);
  const result = await new OpencodeHeadersTransformer().transformRequestIn(
    body,
    provider,
    makeContext()
  );
  assert.equal(result.config.__providerResponse.status, 200);
  for (const c of calls) assert.equal(c.headers["x-zen-model"], undefined);
}

async function forwardsParentSessionIdWhenPresent() {
  const parentId = "ses_parent_abc123";
  const withParent = {
    signal: undefined,
    req: {
      sessionId: "conv-1",
      headers: { "x-parent-session-id": parentId },
      log: { warn() {}, info() {}, debug() {} },
      server: { configService: { getHttpsProxy: () => undefined } },
    },
  };
  const calls = installFetch([okResponse]);
  const result = await new OpencodeHeadersTransformer().transformRequestIn(body, provider, withParent);
  assert.equal(result.config.__providerResponse.status, 200);
  assert.equal(calls[0].headers["x-parent-session-id"], parentId);
}

async function parentIsPreservedAcrossZenRetries() {
  const parentId = "ses_parent_xyz";
  const withParent = {
    signal: undefined,
    req: {
      sessionId: "conv-parent-retry",
      headers: { "x-parent-session-id": parentId },
      log: { warn() {}, info() {}, debug() {} },
      server: { configService: { getHttpsProxy: () => undefined } },
    },
  };
  const calls = installFetch([
    () => new Response(NO_PROVIDER_BODY, { status: 401 }),
    okResponse,
  ]);
  const result = await new OpencodeHeadersTransformer().transformRequestIn(body, provider, withParent);
  assert.equal(result.config.__providerResponse.status, 200);
  assert.equal(calls.length, 2);
  for (const c of calls) assert.equal(c.headers["x-parent-session-id"], parentId);
  assert.notEqual(calls[0].headers["x-opencode-session"], calls[1].headers["x-opencode-session"]);
  assert.equal(calls[0].headers["x-opencode-request"], calls[1].headers["x-opencode-request"]);
}

async function exhaustedRoutingFailureInvalidatesSessionForNextTurn() {
  // Regression for /tmp/ccr-logs ccr-20260830000000_1.log L6772-L6817:
  // req-1fa exhausted 5 routing 400s and cached poisoned g0sb; req-1fc
  // reused it and burned its first attempt on the same bad bucket.
  const transformer = new OpencodeHeadersTransformer();
  const ctx = makeContext(); // conv-1

  const calls = installFetch(
    Array.from({ length: 5 }, () => () => new Response(UPSTREAM_FAILED_BODY, { status: 400 }))
  );
  await assert.rejects(
    () => transformer.transformRequestIn(body, provider, ctx),
    (error: any) => {
      assert.equal(error.statusCode, 400);
      return true;
    }
  );
  assert.equal(calls.length, 5);
  // Each attempt re-rolled — 5 distinct sessions exhausted.
  assert.equal(new Set(calls.map((c) => c.headers["x-opencode-session"])).size, 5);
  const poisoned = calls[4].headers["x-opencode-session"];

  // Next turn in the same conversation must not reuse the poisoned session.
  const nextCalls = installFetch([okResponse]);
  const result = await transformer.transformRequestIn(body, provider, ctx);
  assert.equal(result.config.__providerResponse.status, 200);
  assert.equal(nextCalls.length, 1);
  assert.notEqual(nextCalls[0].headers["x-opencode-session"], poisoned);
}

async function exhaustedTransientKeepsSessionForNextTurn() {
  const transformer = new OpencodeHeadersTransformer();
  const ctx = { signal: undefined, req: { sessionId: "conv-transient-exhaust", log: { warn() {}, info() {}, debug() {} }, server: { configService: { getHttpsProxy: () => undefined } } } };

  const calls = installFetch(
    Array.from({ length: 5 }, () => () =>
      new Response(JSON.stringify({ error: { message: "busy" } }), { status: 529, headers: { "retry-after-ms": "0" } })
    )
  );
  await assert.rejects(
    () => transformer.transformRequestIn(body, provider, ctx),
    (error: any) => {
      assert.equal(error.statusCode, 529);
      return true;
    }
  );
  assert.equal(calls.length, 5);
  assert.equal(new Set(calls.map((c) => c.headers["x-opencode-session"])).size, 1);
  const sticky = calls[4].headers["x-opencode-session"];

  const nextCalls = installFetch([okResponse]);
  const result = await transformer.transformRequestIn(body, provider, ctx);
  assert.equal(result.config.__providerResponse.status, 200);
  assert.equal(nextCalls[0].headers["x-opencode-session"], sticky);
}

async function main() {
  const originalFetch = (globalThis as any).fetch;
  try {
    await recoverRoutingFailures();
    await recoverWrappedRoutingFailure();
    await retryTransientStatusesWithAffinity();
    await preserveTerminalErrors();
    await preserveFinalRetryHeaders();
    await abortsPendingBackoff();
    await retriesTransportFailureWithAffinity();
    await streamFailuresAreErrors();
    await streamCancellationPropagates();
    await zenModelIsNeverSent();
    await forwardsParentSessionIdWhenPresent();
    await parentIsPreservedAcrossZenRetries();
    await exhaustedRoutingFailureInvalidatesSessionForNextTurn();
    await exhaustedTransientKeepsSessionForNextTurn();
    console.log("opencode-headers reliability: PASS");
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

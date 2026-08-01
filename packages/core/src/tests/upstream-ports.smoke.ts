import assert from "node:assert/strict";
import {
  sanitizeUpstreamErrorText,
  sanitizeErrorForLog,
  sanitizeHeadersForLog,
} from "../utils/redact";
import {
  parseRetryAfterHeaderMs,
  exponentialRetryBackoffMs,
  retryDelayAfterFailure,
  isProviderNetworkError,
  isFallbackEligibleError,
  isClientAbortError,
  createClientDisconnectSignal,
  CLIENT_DISCONNECT_REASON,
  delay,
  selectFallbackModels,
  toClientAbortError,
} from "../utils/retry";
import { EventEmitter } from "node:events";
import type { FastifyReply, FastifyRequest } from "fastify";
import { shouldBypassProxy } from "../utils/request";
import {
  normalizeModelSelector,
  removeClaudeCodeBillingSystemHeader,
  extractAndRemoveClaudeCodeSubagentModelTag,
} from "../utils/router";

function testRedact() {
  const text = sanitizeUpstreamErrorText(
    'fetch failed Connecting to api.example.com:443 Bearer sk-abcdefghijklmnop Authorization: "Bearer tok" {"api_key":"secret-value"} from 10.0.0.5:8443'
  );
  assert.ok(!/\bapi\.example\.com\b/.test(text), text);
  assert.ok(!text.includes("sk-abcdefghijklmnop"), text);
  assert.ok(!text.includes("secret-value"), text);
  assert.ok(!text.includes("10.0.0.5"), text);
  assert.match(text, /\[redacted/);

  const diagnostic = sanitizeUpstreamErrorText(
    "connect ECONNREFUSED 10.0.0.5:443 from provider to backend"
  );
  assert.ok(diagnostic.includes("ECONNREFUSED"), diagnostic);
  assert.ok(diagnostic.includes("from provider to backend"), diagnostic);
  assert.ok(!diagnostic.includes("10.0.0.5"), diagnostic);

  const loggedError = Object.assign(new Error("token=abc123xyz999"), {
    code: "ECONNRESET",
    cause: new Error("connect to api.example.com with token=othersecret"),
  });
  const logged = sanitizeErrorForLog(loggedError);
  assert.equal(logged.code, "ECONNRESET");
  assert.ok(!logged.message.includes("abc123xyz999"));
  assert.ok(
    logged.stack?.includes("Error: token=[redacted]"),
    logged.stack ?? "missing sanitized stack"
  );
  assert.ok(
    !logged.stack?.includes("abc123xyz999"),
    logged.stack ?? "missing sanitized stack"
  );
  assert.ok(
    logged.cause?.includes("connect to [redacted-address]"),
    logged.cause ?? "missing sanitized cause"
  );
  assert.ok(
    !logged.cause?.includes("othersecret"),
    logged.cause ?? "missing sanitized cause"
  );

  const headers = sanitizeHeadersForLog({
    Authorization: "Bearer secret",
    "x-api-key": "key",
    "content-type": "application/json",
  });
  assert.equal(headers.Authorization, "[redacted]");
  assert.equal(headers["x-api-key"], "[redacted]");
  assert.equal(headers["content-type"], "application/json");
}

function testRetry() {
  assert.equal(parseRetryAfterHeaderMs("2"), 2000);
  assert.equal(parseRetryAfterHeaderMs("not-a-date"), undefined);
  assert.equal(exponentialRetryBackoffMs(0), 1000);
  assert.equal(exponentialRetryBackoffMs(1), 2000);
  assert.equal(retryDelayAfterFailure(0, "5"), 5000);
  assert.equal(retryDelayAfterFailure(2, null), 4000);

  const networkErr = Object.assign(new Error("fetch failed"), {
    code: "ECONNRESET",
  });
  assert.equal(isProviderNetworkError(networkErr), true);
  assert.equal(isFallbackEligibleError(networkErr), true);

  const abortErr = Object.assign(new Error("aborted"), { name: "AbortError" });
  assert.equal(isClientAbortError(abortErr), true);
  assert.equal(isFallbackEligibleError(abortErr), false);

  const timeoutErr = Object.assign(
    new Error("The operation was aborted due to timeout"),
    { name: "TimeoutError", code: 23 }
  );
  assert.equal(isClientAbortError(timeoutErr), false);
  assert.equal(isProviderNetworkError(timeoutErr), true);
  assert.equal(isFallbackEligibleError(timeoutErr), true);

  const providerErr = Object.assign(new Error("upstream 500"), {
    code: "provider_response_error",
  });
  assert.equal(isFallbackEligibleError(providerErr), true);
}

async function testDelayAbort() {
  const controller = new AbortController();
  const pending = delay(5_000, controller.signal);
  controller.abort();
  await assert.rejects(pending, (err: any) => err?.name === "AbortError");
}

function testNoProxy(env: NodeJS.ProcessEnv = process.env) {
  const prevNoProxy = env.NO_PROXY;
  const prevNoProxyLower = env.no_proxy;
  try {
    delete env.NO_PROXY;
    delete env.no_proxy;

    assert.equal(shouldBypassProxy("https://127.0.0.1:3456"), true);
    assert.equal(shouldBypassProxy("http://localhost:3456"), true);
    assert.equal(shouldBypassProxy("https://[::1]/"), true);
    assert.equal(shouldBypassProxy("https://api.anthropic.com"), false);

    env.NO_PROXY = "example.com:443,::1,.internal,10.0.0.0/8,*.corp.local";
    assert.equal(shouldBypassProxy("https://example.com/v1"), true);
    assert.equal(shouldBypassProxy("https://[::1]/"), true);
    assert.equal(shouldBypassProxy("https://svc.internal/path"), true);
    assert.equal(shouldBypassProxy("https://10.1.2.3/"), true);
    assert.equal(shouldBypassProxy("https://app.corp.local/"), true);
    assert.equal(shouldBypassProxy("https://api.anthropic.com"), false);
  } finally {
    if (prevNoProxy === undefined) delete env.NO_PROXY;
    else env.NO_PROXY = prevNoProxy;
    if (prevNoProxyLower === undefined) delete env.no_proxy;
    else env.no_proxy = prevNoProxyLower;
  }
}

function testSubagentHelpers() {
  assert.equal(normalizeModelSelector("openrouter/gpt-4o"), "openrouter,gpt-4o");
  assert.equal(normalizeModelSelector("openrouter,gpt-4o"), "openrouter,gpt-4o");
  assert.deepEqual(
    selectFallbackModels(
      { default: ["provider,default"], subagent: ["provider,subagent"] },
      "subagent"
    ),
    ["provider,subagent"]
  );
  assert.deepEqual(
    selectFallbackModels({ default: ["provider,default"] }, "subagent"),
    ["provider,default"]
  );
  assert.equal(selectFallbackModels({ default: [] }, "think"), undefined);

  const bodyWithBilling = {
    system: [
      {
        type: "text",
        text: 'x-anthropic-billing-header: {"cc_is_subagent":true}',
      },
      { type: "text", text: "real system prompt" },
    ],
  };
  assert.equal(removeClaudeCodeBillingSystemHeader(bodyWithBilling), true);
  assert.equal(bodyWithBilling.system.length, 1);
  assert.equal(bodyWithBilling.system[0].text, "real system prompt");

  const bodyWithTag = {
    system: [
      {
        type: "text",
        text: "prefix <CCR-SUBAGENT-MODEL>provider,model</CCR-SUBAGENT-MODEL> suffix",
      },
    ],
    messages: [],
  };
  assert.equal(
    extractAndRemoveClaudeCodeSubagentModelTag(bodyWithTag),
    "provider,model"
  );
  assert.equal(bodyWithTag.system[0].text, "prefix  suffix");

  const bodyWithMessageTag = {
    messages: [
      {
        role: "user",
        content: "<CCR-SUBAGENT-MODEL>acme/fast</CCR-SUBAGENT-MODEL> do work",
      },
    ],
  };
  assert.equal(
    extractAndRemoveClaudeCodeSubagentModelTag(bodyWithMessageTag),
    "acme,fast"
  );
  assert.equal(bodyWithMessageTag.messages[0].content, " do work");
}

async function testFromWebCleanupDoesNotCancelLockedBody() {
  // Regression: after Readable.fromWeb(body), body.cancel() rejects with
  // ERR_INVALID_STATE ("ReadableStream is locked") as an unhandledRejection.
  // Cleanup must destroy the Node stream only.
  const { Readable } = await import("node:stream");
  const body = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode("data: hi\n\n"));
      controller.close();
    },
  });
  const nodeStream = Readable.fromWeb(body as any);
  assert.equal(body.locked, true);

  await assert.rejects(
    () => body.cancel(),
    (err: any) =>
      err?.code === "ERR_INVALID_STATE" ||
      /ReadableStream is locked/.test(String(err?.message || err))
  );

  await new Promise<void>((resolve, reject) => {
    nodeStream.once("error", reject);
    nodeStream.once("close", () => resolve());
    nodeStream.destroy();
  });
}

function testClientDisconnectSignal() {
  // Regression: listening on request socket.close falsely aborted Cursor SDK
  // requests during bootstrap while the client was still connected.
  // createClientDisconnectSignal returns { signal, arm } — listeners attach only on arm().
  const socket = new EventEmitter() as any;
  const rawReq = Object.assign(new EventEmitter(), {
    destroyed: false,
    aborted: false,
    complete: true,
    socket,
  });
  const responseSocket = Object.assign(new EventEmitter(), {
    destroyed: false,
    readable: true,
    writable: true,
  });
  const rawRes = Object.assign(new EventEmitter(), {
    destroyed: false,
    writableEnded: false,
    socket: responseSocket,
  });
  const req = { raw: rawReq } as FastifyRequest;
  const reply = { raw: rawRes } as FastifyReply;

  const handle = createClientDisconnectSignal(req, reply);
  assert.equal(handle.signal.aborted, false);

  // Request close/destroyed state is not a reliable disconnect signal after
  // Fastify has parsed the body, so request events never abort this handle.
  socket.emit("close");
  rawReq.emit("close");
  assert.equal(handle.signal.aborted, false);

  // Arm immediately so upstream fetches and fallback waits can be cancelled.
  // A response close with a live socket is not enough evidence of disconnect.
  handle.arm();
  rawRes.emit("close");
  assert.equal(handle.signal.aborted, false);

  // A response close with a destroyed socket is a real disconnect.
  const handle2 = createClientDisconnectSignal(req, reply);
  handle2.arm();
  responseSocket.destroyed = true;
  rawRes.emit("close");
  assert.equal(handle2.signal.aborted, true);
  assert.equal(handle2.signal.reason, `${CLIENT_DISCONNECT_REASON} (response close)`);
  responseSocket.destroyed = false;

  // Finished responses closing cleanly must not abort.
  const handleFinished = createClientDisconnectSignal(req, reply);
  handleFinished.arm();
  rawRes.writableEnded = true;
  rawRes.emit("close");
  assert.equal(handleFinished.signal.aborted, false);
  rawRes.writableEnded = false;

  // Without reply, create does not attach request listeners (arm is a no-op).
  const rawReq2 = Object.assign(new EventEmitter(), {
    destroyed: false,
    aborted: false,
    complete: false,
    socket: new EventEmitter(),
  });
  const handle3 = createClientDisconnectSignal({
    raw: rawReq2,
  } as FastifyRequest);
  handle3.arm();
  (rawReq2.socket as EventEmitter).emit("close");
  rawReq2.emit("close");
  assert.equal(handle3.signal.aborted, false);

  // Fastify sets req.raw.destroyed after JSON body parsing while the client is
  // still connected. That must NOT pre-abort the signal (OpenCode/Zen 500s).
  const rawReqDestroyed = Object.assign(new EventEmitter(), {
    destroyed: true,
    aborted: false,
    complete: true,
    socket: new EventEmitter(),
  });
  const rawResLive = Object.assign(new EventEmitter(), {
    destroyed: false,
    writableEnded: false,
    socket: {
      destroyed: false,
      readable: true,
      writable: true,
    },
  });
  const handle4 = createClientDisconnectSignal(
    { raw: rawReqDestroyed } as FastifyRequest,
    { raw: rawResLive } as FastifyReply
  );
  assert.equal(handle4.signal.aborted, false);
  handle4.arm();
  assert.equal(handle4.signal.aborted, false);

  // Response already destroyed when arm() runs ⇒ abort.
  const rawResDead = Object.assign(new EventEmitter(), {
    destroyed: true,
    writableEnded: false,
  });
  const handle5 = createClientDisconnectSignal(req, {
    raw: rawResDead,
  } as FastifyReply);
  assert.equal(handle5.signal.aborted, false);
  handle5.arm();
  assert.equal(handle5.signal.aborted, true);
  assert.equal(
    handle5.signal.reason,
    `${CLIENT_DISCONNECT_REASON} (already destroyed)`
  );

  // fetch() rejects with the abort string itself — must classify as client abort
  // so errorHandler returns 499 instead of 500.
  assert.equal(
    isClientAbortError(`${CLIENT_DISCONNECT_REASON} (already destroyed)`),
    true
  );
  assert.equal(isClientAbortError("ENOTFOUND upstream"), false);

  const normalized = toClientAbortError(
    `${CLIENT_DISCONNECT_REASON} (already destroyed)`
  );
  assert.equal(normalized.name, "AbortError");
  assert.equal((normalized as any).code, "ABORT_ERR");
  assert.equal(isClientAbortError(normalized), true);
}

async function main() {
  testRedact();
  testRetry();
  await testDelayAbort();
  testNoProxy();
  testSubagentHelpers();
  await testFromWebCleanupDoesNotCancelLockedBody();
  testClientDisconnectSignal();
}

main().catch((error) => {
  process.stderr.write(
    `${error instanceof Error ? error.stack || error.message : String(error)}\n`
  );
  process.exit(1);
});

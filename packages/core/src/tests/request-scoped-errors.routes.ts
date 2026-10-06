/**
 * Request-scoped error rules wired through the real routes: top-level global
 * rules, late markers in raw upstream bodies (never leaked to clients or
 * logs), provider rule aliases on registration and update, `[1m]`-normalized
 * cooldown ids, and no fallback waits spent on cooled-down candidates.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { errorHandler } from "../api/middleware";
import { registerApiRoutes } from "../api/routes";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TokenizerService } from "../services/tokenizer";
import { TransformerService } from "../services/transformer";

process.exitCode = 1;

const silent = { debug() {}, info() {}, warn() {}, error() {} };

type Upstream = (body: any) => Response;

function chatOk(model: string): Response {
  return new Response(
    JSON.stringify({
      id: "chatcmpl-ok",
      object: "chat.completion",
      created: 1,
      model,
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: { role: "assistant", content: `ok from ${model}` },
        },
      ],
    }),
    { status: 200, headers: { "content-type": "application/json" } }
  );
}

function failure(
  status: number,
  body: string,
  headers: Record<string, string> = {}
): Response {
  return new Response(body, { status, headers });
}

function provider(name: string, extra: Record<string, unknown> = {}) {
  return {
    name,
    api_base_url: `https://${name}.invalid/v1/chat/completions`,
    api_key: `${name}-key`,
    models: [name[0]],
    ...extra,
  };
}

async function buildApp(config: Record<string, unknown>) {
  const logLines: string[] = [];
  const providerWarnings: string[] = [];
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: config,
  });
  const transformerService = new TransformerService(configService, silent);
  await transformerService.initialize();
  const providerService = new ProviderService(configService, transformerService, {
    ...silent,
    warn: (_obj: unknown, message: string) => providerWarnings.push(message),
  });
  const tokenizerService = new TokenizerService(configService, silent);
  await tokenizerService.initialize();

  const app = Fastify({
    logger: {
      level: "trace",
      stream: { write: (line: string) => logLines.push(line) },
    },
  });
  app.decorate("configService", configService);
  app.decorate("transformerService", transformerService);
  app.decorate("providerService", providerService);
  app.decorate("tokenizerService", tokenizerService);
  app.setErrorHandler(errorHandler);
  await registerApiRoutes(app);
  return { app, logLines, providerWarnings, providerService };
}

/** Route mocked fetches by upstream host; records every call in order. */
async function withUpstreams(
  upstreams: Record<string, Upstream>,
  run: (calls: Array<{ host: string; body: any }>) => Promise<void>
) {
  const originalFetch = globalThis.fetch;
  const calls: Array<{ host: string; body: any }> = [];
  globalThis.fetch = (async (input: any, init?: any) => {
    const url = new URL(typeof input === "string" ? input : input?.url ?? String(input));
    const host = url.hostname.replace(/\.invalid$/, "");
    const rawBody = init?.body ?? (typeof input?.text === "function" ? await input.text() : undefined);
    const body = typeof rawBody === "string" ? JSON.parse(rawBody) : rawBody;
    calls.push({ host, body });
    const handler = upstreams[host];
    if (!handler) throw new Error(`unexpected upstream ${host}`);
    return handler(body);
  }) as typeof fetch;
  try {
    await run(calls);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

function chat(app: any, model: string) {
  return app.inject({
    method: "POST",
    url: "/v1/chat/completions",
    payload: { model, messages: [{ role: "user", content: "hi" }] },
  });
}

/** Filler longer than the 240-char sanitized message/body cut. */
const FILLER = "upstream diagnostic detail ".repeat(30);

async function testGlobalRuleMatchesLateRawMarker() {
  const marker = "server_is_overloaded";
  const { app } = await buildApp({
    Router: { default: "alpha,a" },
    fallback: { default: ["gamma,g"] },
    providers: [provider("alpha"), provider("gamma")],
    // Top level, as users write it in config.json.
    request_scoped_errors: [
      { status: 400, match: [marker], action: "continue" },
    ],
  });
  try {
    // Non-JSON body: the marker only survives in the raw classification text.
    await withUpstreams(
      {
        alpha: () =>
          failure(400, `${FILLER}${marker}`, { "retry-after": "0.05" }),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a");
        assert.equal(res.statusCode, 200, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha", "gamma"]);
      }
    );
    // JSON body: the marker sits late inside a single long string value.
    await withUpstreams(
      {
        alpha: () =>
          failure(
            400,
            JSON.stringify({
              error: { message: `${FILLER}${marker}`, type: "invalid_request_error" },
            }),
            { "content-type": "application/json", "retry-after": "0.05" }
          ),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a");
        assert.equal(res.statusCode, 200, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha", "gamma"]);
      }
    );
  } finally {
    await app.close();
  }
}

async function testGlobalStopRuleKeepsRawTextPrivate() {
  const marker = "context_length_exceeded_marker";
  const { app, logLines } = await buildApp({
    Router: { default: "alpha,a" },
    fallback: { default: ["gamma,g"] },
    providers: [provider("alpha"), provider("gamma")],
    requestScopedErrors: [{ status: 429, match: [marker], action: "stop" }],
  });
  try {
    await withUpstreams(
      {
        alpha: () => failure(429, `${FILLER}${marker}`),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a");
        // 429 is fallback-eligible by status; the global stop rule wins.
        assert.equal(res.statusCode, 429, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha"]);
        assert.ok(!res.body.includes(marker), "raw text never reaches the client");
        assert.ok(
          logLines.some((line) => line.includes("terminal request-scoped-error")),
          "stop decision is logged"
        );
        assert.ok(
          !logLines.some((line) => line.includes(marker)),
          "raw text never reaches the logs"
        );
      }
    );
  } finally {
    await app.close();
  }
}

async function testInvalidRulesWarn() {
  const { app, logLines, providerWarnings } = await buildApp({
    Router: { default: "alpha,a" },
    providers: [
      provider("alpha", {
        request_scoped_errors: [{ matches: ["typo"], action: "stop" }],
      }),
    ],
    "request-scoped-errors": [{ status: "429", action: "continue" }],
  });
  try {
    assert.ok(
      providerWarnings.some((line) => line.includes('unknown key(s): "matches"')),
      providerWarnings.join("\n")
    );
    assert.ok(
      logLines.some(
        (line) =>
          line.includes("request_scoped_errors global rule ignored") &&
          line.includes('\\"status\\"')
      ),
      logLines.join("\n")
    );
    // The typo rule was dropped, not widened: a 400 stays a plain 400.
    await withUpstreams({ alpha: () => failure(400, "bad request") }, async (calls) => {
      const res = await chat(app, "alpha,a");
      assert.equal(res.statusCode, 400);
      assert.equal(calls.length, 1);
    });
  } finally {
    await app.close();
  }
}

async function testProviderAliasesRegisterAndUpdate() {
  const quota = "usage_limit_reached";
  const { app } = await buildApp({
    Router: { default: "alpha,a" },
    fallback: { default: ["gamma,g"] },
    providers: [
      provider("alpha", {
        requestScopedErrors: [{ status: 400, match: [quota], action: "continue" }],
      }),
      provider("gamma"),
    ],
  });
  try {
    const registered = await app.inject({ method: "GET", url: "/providers/alpha" });
    const stored = registered.json();
    assert.deepEqual(Object.keys(stored).filter((k) => /scoped/i.test(k)), [
      "request_scoped_errors",
    ]);
    assert.equal(stored.request_scoped_errors[0].action, "continue");

    // The camelCase provider rule turns a terminal 400 into a fallback.
    await withUpstreams(
      {
        alpha: () => failure(400, quota, { "retry-after": "0.05" }),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a");
        assert.equal(res.statusCode, 200, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha", "gamma"]);
      }
    );

    // Update through the kebab alias: it replaces the stored rules instead of
    // being shadowed by the canonical key.
    const updated = await app.inject({
      method: "PUT",
      url: "/providers/alpha",
      payload: {
        "request-scoped-errors": [{ status: 400, match: [quota], action: "stop" }],
      },
    });
    assert.equal(updated.statusCode, 200, updated.body);
    const after = updated.json();
    assert.deepEqual(Object.keys(after).filter((k) => /scoped/i.test(k)), [
      "request_scoped_errors",
    ]);
    assert.equal(after.request_scoped_errors[0].action, "stop");

    await withUpstreams(
      {
        alpha: () => failure(400, quota),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a");
        assert.equal(res.statusCode, 400, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha"]);
      }
    );

    // An update without any rule key keeps the stored rules.
    await app.inject({
      method: "PUT",
      url: "/providers/alpha",
      payload: { models: ["a"] },
    });
    const kept = await app.inject({ method: "GET", url: "/providers/alpha" });
    assert.equal(kept.json().request_scoped_errors[0].action, "stop");
  } finally {
    await app.close();
  }
}

async function testCooldownIdsAndWaits() {
  const quota = "usage_limit_reached";
  const cooling = [{ status: 429, match: [quota], action: "continue-and-cooldown" }];
  const { app } = await buildApp({
    Router: { default: "alpha,a" },
    fallback: { default: ["beta,b[1m]", "gamma,g"] },
    providers: [
      provider("alpha", { request_scoped_errors: cooling }),
      provider("beta", { request_scoped_errors: cooling }),
      provider("gamma"),
    ],
  });
  try {
    // 1) beta (configured as `b[1m]`) hits quota and cools down; gamma serves.
    //    The picker marker never reaches the upstream model id.
    await withUpstreams(
      {
        alpha: () => failure(503, "busy", { "retry-after": "0.05" }),
        beta: () => failure(429, quota, { "retry-after": "0.05" }),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a");
        assert.equal(res.statusCode, 200, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha", "beta", "gamma"]);
        assert.equal(calls[1].body.model, "b");
      }
    );

    // 2) The cooled `beta,b[1m]` candidate is skipped on the next request.
    await withUpstreams(
      {
        alpha: () => failure(503, "busy", { "retry-after": "0.05" }),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a");
        assert.equal(res.statusCode, 200, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha", "gamma"]);
      }
    );

    // 3) The primary `alpha,a[1m]` cools the same id as a fallback `alpha,a`.
    //    The primary itself is still attempted (fallback-only semantics).
    await withUpstreams(
      {
        alpha: () => failure(429, quota, { "retry-after": "0.05" }),
        beta: () => chatOk("b"),
        gamma: () => chatOk("g"),
      },
      async (calls) => {
        const res = await chat(app, "alpha,a[1m]");
        assert.equal(res.statusCode, 200, res.body);
        // beta still cooling, so gamma serves.
        assert.deepEqual(calls.map((c) => c.host), ["alpha", "gamma"]);
        assert.equal(calls[0].body.model, "a");
      }
    );
  } finally {
    await app.close();
  }
}

async function testNoWaitForCooledCandidates() {
  const quota = "usage_limit_reached";
  const cooling = [{ status: 429, match: [quota], action: "continue-and-cooldown" }];
  const { app } = await buildApp({
    Router: { default: "gamma,g" },
    fallback: { default: ["beta,b"] },
    providers: [
      provider("beta", { request_scoped_errors: cooling }),
      provider("gamma"),
    ],
  });
  try {
    // Cool beta down through a direct request (primary attempts still run).
    await withUpstreams({ beta: () => failure(429, quota) }, async () => {
      const res = await chat(app, "beta,b");
      assert.equal(res.statusCode, 429);
    });

    // Every fallback candidate is cooling: fail fast, no backoff wait. The
    // upstream asks for a 5 s Retry-After that must not be honored.
    await withUpstreams(
      { gamma: () => failure(503, "busy", { "retry-after": "5" }) },
      async (calls) => {
        const started = Date.now();
        const res = await chat(app, "gamma,g");
        const elapsed = Date.now() - started;
        assert.equal(res.statusCode, 503, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["gamma"]);
        assert.ok(elapsed < 2_000, `waited ${elapsed}ms for cooled candidates`);
      }
    );
  } finally {
    await app.close();
  }

  // A remaining cooled candidate must not trigger the between-attempt wait.
  const second = await buildApp({
    Router: { default: "alpha,a" },
    fallback: { default: ["gamma,g", "beta,b"] },
    providers: [
      provider("alpha"),
      provider("beta", { request_scoped_errors: cooling }),
      provider("gamma"),
    ],
  });
  try {
    await withUpstreams({ beta: () => failure(429, quota) }, async () => {
      await chat(second.app, "beta,b");
    });
    await withUpstreams(
      {
        alpha: () => failure(503, "busy", { "retry-after": "0.05" }),
        gamma: () => failure(503, "busy", { "retry-after": "5" }),
      },
      async (calls) => {
        const started = Date.now();
        const res = await chat(second.app, "alpha,a");
        const elapsed = Date.now() - started;
        assert.equal(res.statusCode, 503, res.body);
        assert.deepEqual(calls.map((c) => c.host), ["alpha", "gamma"]);
        assert.ok(elapsed < 2_000, `waited ${elapsed}ms before a cooled candidate`);
      }
    );
  } finally {
    await second.app.close();
  }
}

async function main() {
  await testGlobalRuleMatchesLateRawMarker();
  await testGlobalStopRuleKeepsRawTextPrivate();
  await testInvalidRulesWarn();
  await testProviderAliasesRegisterAndUpdate();
  await testCooldownIdsAndWaits();
  await testNoWaitForCooledCandidates();
  console.log("request-scoped-errors.routes: PASS");
}

main().then(() => { process.exitCode = 0; }).catch((error) => {
  console.error(error);
  process.exit(1);
});

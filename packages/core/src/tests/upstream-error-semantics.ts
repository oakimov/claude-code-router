/**
 * Native upstream error semantics and status-aware fallback.
 * Fix 1: Anthropic/OpenAI wire errors keep status, envelope, and safe headers.
 * Fix 2: terminal 4xx never trigger fallback; transient failures do.
 * Fix 8: safe upstream headers reach the client; sensitive ones do not.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { errorHandler } from "../api/middleware";
import { registerApiRoutes } from "../api/routes";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TokenizerService } from "../services/tokenizer";
import { TransformerService } from "../services/transformer";
import {
  isFallbackEligibleError,
  isFallbackEligibleStatus,
} from "../utils/retry";
import { selectSafeDownstreamHeaders } from "../utils/headers";

const logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
};

async function buildApp() {
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      Router: { default: "primary,claude" },
      fallback: { default: ["generic,gpt"] },
      providers: [
        {
          name: "primary",
          api_base_url: "https://primary.invalid",
          api_key: "primary-key",
          models: ["claude", "gpt"],
          transformer: { use: ["Anthropic"] },
        },
        {
          name: "generic",
          api_base_url: "https://generic.invalid/v1/chat/completions",
          api_key: "generic-key",
          models: ["gpt"],
        },
      ],
    },
  });
  const transformerService = new TransformerService(configService, logger);
  await transformerService.initialize();
  const providerService = new ProviderService(
    configService,
    transformerService,
    logger
  );
  const tokenizerService = new TokenizerService(configService, logger);
  await tokenizerService.initialize();

  const app = Fastify({ logger: false });
  app.decorate("configService", configService);
  app.decorate("transformerService", transformerService);
  app.decorate("providerService", providerService);
  app.decorate("tokenizerService", tokenizerService);
  app.setErrorHandler(errorHandler);
  await registerApiRoutes(app);
  return app;
}

function testStatusClassifier() {
  const retryable = [408, 409, 425, 429, 500, 502, 503, 504, 529];
  for (const status of retryable) {
    assert.equal(isFallbackEligibleStatus(status), true, `${status} retryable`);
  }
  const terminal = [400, 401, 403, 404, 422, 451, 200, 301];
  for (const status of terminal) {
    assert.equal(isFallbackEligibleStatus(status), false, `${status} terminal`);
  }
  assert.equal(isFallbackEligibleStatus(undefined), false);

  // Error-level classification
  assert.equal(
    isFallbackEligibleError({ code: "provider_response_error", statusCode: 400 }),
    false
  );
  assert.equal(
    isFallbackEligibleError({ code: "provider_response_error", statusCode: 401 }),
    false
  );
  assert.equal(
    isFallbackEligibleError({ code: "provider_response_error", statusCode: 404 }),
    false
  );
  assert.equal(
    isFallbackEligibleError({ code: "provider_response_error", statusCode: 422 }),
    false
  );
  assert.equal(
    isFallbackEligibleError({ code: "provider_response_error", statusCode: 429 }),
    true
  );
  assert.equal(
    isFallbackEligibleError({ code: "provider_response_error", statusCode: 500 }),
    true
  );
  assert.equal(
    isFallbackEligibleError({ upstream: { status: 403 } }),
    false
  );
  assert.equal(
    isFallbackEligibleError({ code: "provider_network_error" }),
    true
  );
  assert.equal(
    isFallbackEligibleError(new Error("fetch failed")),
    true
  );
  assert.equal(
    isFallbackEligibleError(
      Object.assign(new Error("aborted"), { name: "TimeoutError" })
    ),
    true
  );
  assert.equal(
    isFallbackEligibleError(
      Object.assign(new Error("client disconnected"), { name: "AbortError" })
    ),
    false
  );
}

function testSafeDownstreamHeaders() {
  const headers = new Headers({
    "retry-after": "7",
    "request-id": "req_123",
    "x-request-id": "x-req-123",
    "anthropic-ratelimit-remaining": "42",
    "anthropic-ratelimit-unified-overage": "disabled",
    "x-ratelimit-limit-requests": "100",
    "set-cookie": "session=secret",
    authorization: "Bearer sk-secret",
    server: "cloudfront",
    via: "1.1 proxy",
    "content-length": "512",
    "content-encoding": "gzip",
    "content-type": "application/json",
    "transfer-encoding": "chunked",
    connection: "keep-alive",
    "x-private-internal": "nope",
  });
  const safe = selectSafeDownstreamHeaders(headers);
  assert.deepEqual(safe, {
    "retry-after": "7",
    "request-id": "req_123",
    "x-request-id": "x-req-123",
    "anthropic-ratelimit-remaining": "42",
    "anthropic-ratelimit-unified-overage": "disabled",
    "x-ratelimit-limit-requests": "100",
  });
}

async function withMockedFetch(
  handler: (callCount: { count: number }) => Response | Promise<Response>,
  run: () => Promise<void>
) {
  const originalFetch = globalThis.fetch;
  const callCount = { count: 0 };
  globalThis.fetch = (async () => {
    callCount.count += 1;
    return handler(callCount);
  }) as any;
  try {
    await run();
  } finally {
    globalThis.fetch = originalFetch;
  }
}

async function testAnthropicWireErrorPreserved() {
  const app = await buildApp();
  try {
    await withMockedFetch(
      () =>
        new Response(
          JSON.stringify({
            type: "error",
            error: {
              type: "invalid_request_error",
              message: "prompt too long: 250000 > 200000 tokens",
            },
          }),
          {
            status: 400,
            headers: {
              "content-type": "application/json",
              "retry-after": "3",
              "request-id": "req_abc",
              "anthropic-ratelimit-remaining": "0",
              "set-cookie": "session=secret",
            },
          }
        ),
      async () => {
        const res = await app.inject({
          method: "POST",
          url: "/v1/messages",
          payload: {
            model: "claude",
            max_tokens: 16,
            messages: [{ role: "user", content: "hi" }],
          },
        });
        assert.equal(res.statusCode, 400);
        const body = res.json();
        assert.equal(body.type, "error");
        assert.equal(body.error.type, "invalid_request_error");
        assert.equal(
          body.error.message,
          "prompt too long: 250000 > 200000 tokens"
        );
        // Safe headers forwarded; sensitive ones dropped.
        assert.equal(res.headers["retry-after"], "3");
        assert.equal(res.headers["request-id"], "req_abc");
        assert.equal(res.headers["anthropic-ratelimit-remaining"], "0");
        assert.equal(res.headers["set-cookie"], undefined);
      }
    );
  } finally {
    await app.close();
  }
}

async function testTerminalErrorSkipsFallback() {
  const app = await buildApp();
  try {
    let calls = 0;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          type: "error",
          error: { type: "authentication_error", message: "invalid x-api-key" },
        }),
        { status: 401, headers: { "content-type": "application/json" } }
      );
    }) as any;
    try {
      const res = await app.inject({
        method: "POST",
        url: "/v1/messages",
        payload: {
          model: "claude",
          max_tokens: 16,
          messages: [{ role: "user", content: "hi" }],
        },
      });
      assert.equal(res.statusCode, 401);
      assert.equal(res.json().error.type, "authentication_error");
      // Terminal 4xx: the configured fallback model must not be retried.
      assert.equal(calls, 1);
    } finally {
      globalThis.fetch = originalFetch;
    }
  } finally {
    await app.close();
  }
}

async function testLongContextLiteralMessagePreserved() {
  const app = await buildApp();
  for (const literal of [
    "Extra usage is required for long context",
    "Usage credits are required for long context",
  ]) {
    await withMockedFetch(
      () =>
        new Response(
          JSON.stringify({
            type: "error",
            error: { type: "invalid_request_error", message: literal },
          }),
          { status: 400, headers: { "content-type": "application/json" } }
        ),
      async () => {
        const res = await app.inject({
          method: "POST",
          url: "/v1/messages",
          payload: {
            model: "claude",
            max_tokens: 16,
            messages: [{ role: "user", content: "hi" }],
          },
        });
        assert.equal(res.statusCode, 400);
        assert.equal(res.json().error.message, literal);
      }
    );
  }
  await app.close();
}

async function testChatCallerGetsOpenAIEnvelope() {
  const app = await buildApp();
  try {
    let calls = 0;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      calls += 1;
      return new Response(
        JSON.stringify({
          error: {
            message:
              "This model's maximum context length is 8192 tokens",
            type: "invalid_request_error",
            param: "messages",
            code: "context_length_exceeded",
          },
        }),
        { status: 400, headers: { "content-type": "application/json" } }
      );
    }) as any;
    try {
      const res = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        payload: {
          model: "gpt",
          messages: [{ role: "user", content: "hi" }],
        },
      });
      assert.equal(res.statusCode, 400);
      const body = res.json();
      assert.equal(
        body.error.message,
        "This model's maximum context length is 8192 tokens"
      );
      assert.equal(body.error.type, "invalid_request_error");
      assert.equal(body.error.code, "context_length_exceeded");
      assert.equal(calls, 1, "terminal 400 must not fall back");
    } finally {
      globalThis.fetch = originalFetch;
    }
  } finally {
    await app.close();
  }
}

async function testRetryableFailureFallsBack() {
  const app = await buildApp();
  try {
    let calls = 0;
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () => {
      calls += 1;
      if (calls === 1) {
        return new Response(
          JSON.stringify({
            error: { message: "rate limited", type: "rate_limit_error" },
          }),
          { status: 429, headers: { "content-type": "application/json" } }
        );
      }
      return new Response(
        JSON.stringify({
          id: "chatcmpl-fallback",
          object: "chat.completion",
          created: 1,
          model: "gpt",
          choices: [
            {
              index: 0,
              finish_reason: "stop",
              message: { role: "assistant", content: "fallback ok" },
            },
          ],
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      );
    }) as any;
    try {
      const res = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        payload: {
          model: "gpt",
          messages: [{ role: "user", content: "hi" }],
        },
      });
      assert.equal(res.statusCode, 200);
      assert.equal(res.json().choices[0].message.content, "fallback ok");
      assert.equal(calls, 2, "429 must try the fallback model");
    } finally {
      globalThis.fetch = originalFetch;
    }
  } finally {
    await app.close();
  }
}

async function testUpstreamSecretsRedacted() {
  const app = await buildApp();
  const secret = "sk-abcdefghijklmnopqrstuvwxyz123456";
  try {
    await withMockedFetch(
      () =>
        new Response(
          JSON.stringify({
            type: "error",
            error: {
              type: "invalid_request_error",
              message: `request rejected for credential ${secret}`,
            },
          }),
          { status: 400, headers: { "content-type": "application/json" } }
        ),
      async () => {
        const res = await app.inject({
          method: "POST",
          url: "/v1/messages",
          payload: {
            model: "claude",
            max_tokens: 16,
            messages: [{ role: "user", content: "hi" }],
          },
        });
        assert.equal(res.statusCode, 400);
        assert.ok(!res.body.includes(secret), "secret must be redacted");
        assert.ok(res.body.includes("[redacted-secret]"));
      }
    );
  } finally {
    await app.close();
  }
}

async function main() {
  testStatusClassifier();
  testSafeDownstreamHeaders();
  await testAnthropicWireErrorPreserved();
  await testTerminalErrorSkipsFallback();
  await testLongContextLiteralMessagePreserved();
  await testChatCallerGetsOpenAIEnvelope();
  await testRetryableFailureFallsBack();
  await testUpstreamSecretsRedacted();
  console.log("upstream-error-semantics: all tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

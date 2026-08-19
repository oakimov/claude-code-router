import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseOpenAiTools, stubToolExecute } from "../debug/tools";
import {
  guessInboundProtocol,
  oauthKindForProvider,
  parseDebugChatBody,
  parseReasoningEffort,
  applyReasoningEffortToBody,
  authHeadersForProvider,
  interpolateConfigString,
  resolveDebugModel,
} from "../debug/model";
import {
  redactCapturedHeaders,
  parseTokenUsage,
  parseTokenUsageFromPayload,
  rewriteSseReasoningLine,
  splitChatCompletionsDoneLine,
  createReasoningNormalizeTransform,
  readCapturedBody,
} from "../debug/llm-capture";
import { errorExchangeFromMessage } from "../debug/types";
import {
  setStreamDebugChatForTests,
  streamDebugChat,
} from "../debug/ai-sdk-agent";
import { createServer } from "../server";

const PROVIDERS = [
  {
    name: "openai",
    api_key: "sk-provider-secret",
    api_base_url: "https://api.openai.com/v1/chat/completions",
    models: ["gpt-4o"],
  },
  {
    name: "claude",
    api_key: "oauth",
    api_base_url: "https://api.anthropic.com/v1/messages",
    models: ["claude-sonnet-4"],
    transformer: { use: ["claude-auth"] },
  },
  {
    name: "enved",
    api_key: "$OPENAI_API_KEY",
    api_base_url: "https://api.openai.com/v1/chat/completions",
    models: ["gpt-4o"],
  },
  {
    name: "codex-pat",
    api_key: "at-example-pat",
    api_base_url: "https://chatgpt.com/backend-api/codex/responses",
    models: ["gpt-5"],
    transformer: { use: ["codex"] },
  },
];

function testParseToolsIgnoresUserCode(): void {
  const specs = parseOpenAiTools([
    {
      type: "function",
      function: {
        name: "get_weather",
        description: "weather",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
        },
        execute: "throw new Error('should never run')",
      },
    },
  ]);
  assert.equal(specs.length, 1);
  assert.equal(specs[0].id, "get_weather");
  assert.equal((specs[0] as any).execute, undefined);
  const result = stubToolExecute({ city: "Berlin" });
  assert.deepEqual(result, { ok: true, stub: true, args: { city: "Berlin" } });
}

function testMalformedToolIdRejected(): void {
  assert.throws(
    () =>
      parseOpenAiTools([
        { type: "function", function: { name: "evil;process.exit(1)" } },
      ]),
    /Invalid tool name/
  );
}

function testRedactCapturedHeaders(): void {
  const redacted = redactCapturedHeaders({
    Authorization: "Bearer sk-secret",
    "x-api-key": "abc-secret",
    "Content-Type": "application/json",
  });
  assert.equal(redacted.Authorization, "[redacted]");
  assert.equal(redacted["x-api-key"], "[redacted]");
  assert.equal(redacted["Content-Type"], "application/json");
  assert.doesNotMatch(JSON.stringify(redacted), /sk-secret|abc-secret/);
}

function testErrorExchangeFromMessage(): void {
  const fallback = errorExchangeFromMessage("provider failed");
  assert.equal(fallback.status, 0);
  assert.match(fallback.responseBody, /provider failed/);

  const captured = errorExchangeFromMessage("ignored", {
    url: "http://example/v1/chat/completions",
    method: "POST",
    requestHeaders: {},
    requestBody: {},
    status: 429,
    responseHeaders: { "retry-after": "2" },
    responseBody: '{"error":{"message":"rate limited"}}',
    streaming: false,
  });
  assert.equal(captured.status, 429);
  assert.equal(captured.responseHeaders["retry-after"], "2");
  assert.match(captured.responseBody, /rate limited/);
}

function testRewriteSseReasoningLine(): void {
  const thinkingOnly = rewriteSseReasoningLine(
    'data: {"choices":[{"delta":{"thinking":{"content":"The user"},"finish_reason":null}}]}'
  );
  const thinkingParsed = JSON.parse(thinkingOnly.slice(5).trim());
  assert.equal(thinkingParsed.choices[0].delta.reasoning_content, "The user");

  const alreadySet = rewriteSseReasoningLine(
    'data: {"choices":[{"delta":{"thinking":{"content":"The"},"reasoning_content":"The"}}]}'
  );
  const alreadyParsed = JSON.parse(alreadySet.slice(5).trim());
  assert.equal(alreadyParsed.choices[0].delta.reasoning_content, "The");

  assert.equal(rewriteSseReasoningLine("data: [DONE]"), "data: [DONE]");
  assert.deepEqual(splitChatCompletionsDoneLine("data: [DONE]"), [
    "data: [DONE]",
    "",
  ]);
  assert.deepEqual(
    splitChatCompletionsDoneLine('data: [DONE] {"choices":[],"cost":"0"}'),
    ["data: [DONE]", ""]
  );
}

async function testStreamSplitsDoneFromCostTrailer(): Promise<void> {
  const sse = [
    'data: {"choices":[{"delta":{"content":"hi"}}]}',
    "",
    "data: [DONE]",
    'data: {"choices":[],"cost":"0"}',
    "",
  ].join("\n");
  const transformed = new Response(sse).body!.pipeThrough(
    createReasoningNormalizeTransform()
  );
  const text = await new Response(transformed).text();
  assert.ok(text.includes("data: [DONE]\n\n"));
  assert.equal(text.includes('"cost":"0"'), false, "cost trailer must not follow [DONE]");
}

async function testCapturedBodyCancellation(): Promise<void> {
  let cancelled = false;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(new TextEncoder().encode("data: partial\n\n"));
    },
    cancel() {
      cancelled = true;
    },
  });
  const controller = new AbortController();
  const captured = readCapturedBody(stream, controller.signal);
  controller.abort(new Error("client disconnected"));
  await assert.rejects(captured, /client disconnected/);
  assert.equal(cancelled, true);
}

function testParseTokenUsage(): void {
  const usage = parseTokenUsage({
    usage: {
      prompt_tokens: 100,
      completion_tokens: 20,
      total_tokens: 120,
      prompt_tokens_details: { cached_tokens: 40, cache_write_tokens: 8 },
    },
  });
  assert.equal(usage?.input, 100);
  assert.equal(usage?.output, 20);
  assert.equal(usage?.total, 120);
  assert.equal(usage?.cacheRead, 40);
  assert.equal(usage?.cacheWrite, 8);

  const anthropic = parseTokenUsage({
    type: "message_start",
    message: {
      usage: {
        input_tokens: 80,
        output_tokens: 0,
        cache_read_input_tokens: 12,
        cache_creation_input_tokens: 4,
      },
    },
  });
  assert.equal(anthropic?.input, 80);
  assert.equal(anthropic?.cacheRead, 12);
  assert.equal(anthropic?.cacheWrite, 4);

  const sse = parseTokenUsageFromPayload(
    [
      "data: {\"message\":{\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}",
      "data: {\"usage\":{\"output_tokens\":7}}",
      "data: [DONE]",
    ].join("\n")
  );
  assert.equal(sse?.input, 10);
  assert.equal(sse?.output, 7);
}

async function testResolveCcrAndDirect(): Promise<void> {
  const config = { PORT: 3456, APIKEY: "router-secret", Providers: PROVIDERS };
  const ccr = await resolveDebugModel(
    {
      target: "ccr",
      protocol: "chat_completions",
      provider: "openai",
      model: "gpt-4o",
    },
    config
  );
  assert.equal(ccr.url, "http://127.0.0.1:3456/v1");
  assert.equal(ccr.id, "openai/openai,gpt-4o");
  assert.equal(ccr.apiKey, "router-secret");

  const direct = await resolveDebugModel(
    {
      target: "direct",
      protocol: "chat_completions",
      provider: "openai",
      model: "gpt-4o",
    },
    config
  );
  assert.equal(direct.url, "https://api.openai.com/v1");
  assert.equal(direct.id, "openai/gpt-4o");
  assert.equal(direct.apiKey, "sk-provider-secret");

  const messages = await resolveDebugModel(
    {
      target: "ccr",
      protocol: "messages",
      provider: "claude",
      model: "claude-sonnet-4",
    },
    config
  );
  assert.equal(messages.url, "http://127.0.0.1:3456/v1");
  assert.equal(messages.id, "anthropic/claude,claude-sonnet-4");

  await assert.rejects(
    () =>
      resolveDebugModel(
        {
          target: "direct",
          protocol: "responses",
          provider: "codex-pat",
          model: "gpt-5",
        },
        config
      ),
    /Codex PAT direct mode is not supported/
  );

  const tempDir = mkdtempSync(join(tmpdir(), "ccr-codex-direct-"));
  const authFile = join(tempDir, "codex.json");
  const previousAuthFile = process.env.CCR_CODEX_AUTH_FILE;
  process.env.CCR_CODEX_AUTH_FILE = authFile;
  const claims = Buffer.from(
    JSON.stringify({
      "https://api.openai.com/auth": {
        chatgpt_account_id: "account-123",
        chatgpt_account_is_fedramp: true,
      },
    })
  ).toString("base64url");
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "oauth-token",
      id_token: `e30.${claims}.sig`,
      token_type: "Bearer",
      expires_at: Date.now() / 1000 + 3600,
    })
  );
  try {
    const oauth = await resolveDebugModel(
      {
        target: "direct",
        protocol: "responses",
        provider: "codex-oauth",
        model: "gpt-5",
      },
      {
        Providers: [
          {
            name: "codex-oauth",
            api_key: "oauth",
            api_base_url: "https://chatgpt.com/backend-api/codex/responses",
            models: ["gpt-5"],
            transformer: { use: ["codex"] },
          },
        ],
      }
    );
    assert.equal(oauth.headers.Authorization, "Bearer oauth-token");
    assert.equal(oauth.headers["ChatGPT-Account-ID"], "account-123");
    assert.equal(oauth.headers.originator, "codex_cli_rs");
    assert.equal(oauth.headers["X-OpenAI-Fedramp"], "true");
  } finally {
    if (previousAuthFile === undefined) delete process.env.CCR_CODEX_AUTH_FILE;
    else process.env.CCR_CODEX_AUTH_FILE = previousAuthFile;
    rmSync(tempDir, { recursive: true, force: true });
  }

  const previous = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "sk-from-env";
  try {
    const enved = await resolveDebugModel(
      {
        target: "direct",
        protocol: "chat_completions",
        provider: "enved",
        model: "gpt-4o",
      },
      config
    );
    assert.equal(enved.apiKey, "sk-from-env");
    assert.equal(enved.url, "https://api.openai.com/v1");
  } finally {
    if (previous === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = previous;
  }

  const previousFoo = process.env.CCR_DEBUG_TEST_KEY;
  process.env.CCR_DEBUG_TEST_KEY = "from-config";
  try {
    assert.equal(interpolateConfigString("$CCR_DEBUG_TEST_KEY"), "from-config");
    assert.equal(interpolateConfigString("${CCR_DEBUG_TEST_KEY}"), "from-config");
  } finally {
    if (previousFoo === undefined) delete process.env.CCR_DEBUG_TEST_KEY;
    else process.env.CCR_DEBUG_TEST_KEY = previousFoo;
  }
}

function testProtocolAndOauthKind(): void {
  assert.equal(guessInboundProtocol(PROVIDERS[0]), "chat_completions");
  assert.equal(guessInboundProtocol(PROVIDERS[1]), "messages");
  assert.equal(oauthKindForProvider(PROVIDERS[1]), "claude-auth");
  assert.equal(oauthKindForProvider(PROVIDERS[0]), null);
  assert.equal(oauthKindForProvider(PROVIDERS[3]), null);
  const previousPat = process.env.CCR_DEBUG_CODEX_PAT;
  process.env.CCR_DEBUG_CODEX_PAT = "at-env-pat";
  try {
    assert.equal(
      oauthKindForProvider({
        api_key: "$CCR_DEBUG_CODEX_PAT",
        transformer: { use: ["codex"] },
      }),
      null
    );
  } finally {
    if (previousPat === undefined) delete process.env.CCR_DEBUG_CODEX_PAT;
    else process.env.CCR_DEBUG_CODEX_PAT = previousPat;
  }

  const oauthHeaders = authHeadersForProvider(
    "messages",
    "oauth-token",
    "claude-auth"
  );
  assert.equal(oauthHeaders.Authorization, "Bearer oauth-token");
  assert.equal(oauthHeaders["anthropic-beta"], "oauth-2025-04-20");
  assert.equal(oauthHeaders["anthropic-version"], "2023-06-01");

  const apiKeyHeaders = authHeadersForProvider(
    "messages",
    "sk-ant",
    null
  );
  assert.equal(apiKeyHeaders["x-api-key"], "sk-ant");
  assert.equal(apiKeyHeaders["anthropic-version"], "2023-06-01");
}

function testParseBodyDefaults(): void {
  const parsed = parseDebugChatBody({
    messages: [{ role: "user", content: "hi" }],
    provider: "openai",
    model: "gpt-4o",
  });
  assert.equal(parsed.target, "ccr");
  assert.equal(parsed.protocol, "chat_completions");
  assert.equal(parsed.stream, true);
  assert.equal(parsed.reasoningEffort, undefined);
  assert.equal(
    parseDebugChatBody({
      provider: "openai",
      model: "gpt-4o",
      stream: false,
    }).stream,
    true
  );

  const withEffort = parseDebugChatBody({
    messages: [{ role: "user", content: "hi" }],
    provider: "openai",
    model: "gpt-4o",
    reasoningEffort: "ultra",
  });
  assert.equal(withEffort.reasoningEffort, "ultra");
  assert.equal(parseReasoningEffort("XHigh"), "xhigh");
  assert.equal(parseReasoningEffort("nope"), undefined);

  const chatBody = applyReasoningEffortToBody(
    { model: "p,m", messages: [] },
    "chat_completions",
    "high"
  );
  assert.equal(chatBody.reasoning_effort, "high");
  const messagesBody = applyReasoningEffortToBody(
    { model: "p,m", messages: [] },
    "messages",
    "minimal"
  );
  assert.deepEqual(messagesBody.thinking, { type: "adaptive" });
  assert.equal((messagesBody.output_config as any).effort, "minimal");
  const noneBody = applyReasoningEffortToBody(
    { model: "p,m", messages: [], output_config: { effort: "high" } },
    "messages",
    "none"
  );
  assert.deepEqual(noneBody.thinking, { type: "disabled" });
  assert.equal(noneBody.output_config, undefined);
  const responsesBody = applyReasoningEffortToBody(
    { model: "p,m", input: [] },
    "responses",
    "max"
  );
  assert.equal((responsesBody.reasoning as any).effort, "max");
}

async function testDirectAiSdkStream(): Promise<void> {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () =>
    new Response(
      [
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}',
        "",
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "",
        "data: [DONE]",
        "",
      ].join("\n"),
      { headers: { "content-type": "text/event-stream" } }
    );
  try {
    const response = await streamDebugChat(
      {
        messages: [
          {
            id: "user-1",
            role: "user",
            parts: [{ type: "text", text: "ping" }],
          },
        ],
        target: "direct",
        protocol: "chat_completions",
        provider: "openai",
        model: "gpt-4o",
        system: "",
        tools: [],
        stream: true,
      },
      { Providers: PROVIDERS }
    );
    assert.equal(response.status, 200);
    assert.match(await response.text(), /pong/);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

async function testDebugChatRouteStubbed(): Promise<void> {
  setStreamDebugChatForTests(async (input) => {
    assert.equal(input.provider, "openai");
    assert.equal(input.model, "gpt-4o");
    assert.equal(input.target, "ccr");
    return new Response("data: ok\n\n", {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    });
  });
  const server = await createServer({
    initialConfig: {
      APIKEY: "test-router-secret",
      HOST: "127.0.0.1",
      PORT: 3456,
      Providers: PROVIDERS,
    },
    logger: false,
    useJsonFile: false,
  });
  try {
    const missing = await server.app.inject({
      method: "POST",
      url: "/api/debug/chat",
      headers: { "x-api-key": "test-router-secret" },
      payload: { messages: [] },
    });
    assert.equal(missing.statusCode, 400);

    const ok = await server.app.inject({
      method: "POST",
      url: "/api/debug/chat",
      headers: { "x-api-key": "test-router-secret" },
      payload: {
        messages: [{ role: "user", parts: [{ type: "text", text: "hi" }] }],
        target: "ccr",
        protocol: "chat_completions",
        provider: "openai",
        model: "gpt-4o",
        system: "",
        tools: [],
        stream: true,
      },
    });
    assert.equal(ok.statusCode, 200);
    assert.match(String(ok.headers["content-type"]), /text\/event-stream/);
    assert.equal(ok.body.includes("ok"), true);
  } finally {
    setStreamDebugChatForTests(null);
    await server.app.close();
  }
}

async function testOauthRefreshValidation(): Promise<void> {
  const server = await createServer({
    initialConfig: {
      APIKEY: "test-router-secret",
      HOST: "127.0.0.1",
      PORT: 3456,
      Providers: PROVIDERS,
    },
    logger: false,
    useJsonFile: false,
  });
  try {
    const unknown = await server.app.inject({
      method: "POST",
      url: "/api/oauth/refresh",
      headers: { "x-api-key": "test-router-secret" },
      payload: { provider: "missing" },
    });
    assert.equal(unknown.statusCode, 400);

    const pat = await server.app.inject({
      method: "POST",
      url: "/api/oauth/refresh",
      headers: { "x-api-key": "test-router-secret" },
      payload: { provider: "codex-pat" },
    });
    assert.equal(pat.statusCode, 400);
    assert.match(String(pat.json().error), /PAT|OAuth/i);

    const noOauth = await server.app.inject({
      method: "POST",
      url: "/api/oauth/refresh",
      headers: { "x-api-key": "test-router-secret" },
      payload: { provider: "openai" },
    });
    assert.equal(noOauth.statusCode, 400);
  } finally {
    await server.app.close();
  }
}

async function testOauthRefreshSuccess(): Promise<void> {
  const tempDir = mkdtempSync(join(tmpdir(), "ccr-qwen-refresh-"));
  const authFile = join(tempDir, "qwen.json");
  const originalFetch = globalThis.fetch;
  process.env.CCR_QWEN_AUTH_FILE = authFile;
  writeFileSync(
    authFile,
    JSON.stringify({ token: "old", expiresAt: null, updatedAt: Date.now() })
  );
  globalThis.fetch = async () => Response.json({ access_token: "rotated" });

  const server = await createServer({
    initialConfig: {
      APIKEY: "test-router-secret",
      HOST: "127.0.0.1",
      PORT: 3456,
      Providers: [
        ...PROVIDERS,
        {
          name: "qwen",
          api_key: "oauth",
          api_base_url: "https://qwen.aikit.club/v1",
          models: ["qwen3-coder-plus"],
          transformer: { use: ["qwen-auth"] },
        },
      ],
    },
    logger: false,
    useJsonFile: false,
  });
  try {
    const refreshed = await server.app.inject({
      method: "POST",
      url: "/api/oauth/refresh",
      headers: { "x-api-key": "test-router-secret" },
      payload: { provider: "qwen" },
    });
    assert.equal(refreshed.statusCode, 200);
    assert.equal(refreshed.json().success, true);
  } finally {
    await server.app.close();
    globalThis.fetch = originalFetch;
    delete process.env.CCR_QWEN_AUTH_FILE;
    rmSync(tempDir, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  testParseToolsIgnoresUserCode();
  testMalformedToolIdRejected();
  testRedactCapturedHeaders();
  testParseTokenUsage();
  testRewriteSseReasoningLine();
  await testStreamSplitsDoneFromCostTrailer();
  await testCapturedBodyCancellation();
  testErrorExchangeFromMessage();
  await testResolveCcrAndDirect();
  testProtocolAndOauthKind();
  testParseBodyDefaults();
  await testDirectAiSdkStream();
  await testDebugChatRouteStubbed();
  await testOauthRefreshValidation();
  await testOauthRefreshSuccess();
  console.log("debug-chat tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

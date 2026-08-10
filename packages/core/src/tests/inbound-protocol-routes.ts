/**
 * Full Fastify route lifecycle for inbound OpenAI protocols. Covers exact-wire
 * credential isolation, Responses↔Chat conversion, and mutation-safe fallback.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify from "fastify";
import { errorHandler } from "../api/middleware";
import { registerApiRoutes } from "../api/routes";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TokenizerService } from "../services/tokenizer";
import { TransformerService } from "../services/transformer";
import { encodeClaudeModelAlias } from "@caeliq/ccr-shared";

const logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
};

interface CapturedRequest {
  url: string;
  headers: Headers;
  body: any;
}

async function buildApp() {
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      Router: { default: "generic,gpt" },
      fallback: { default: ["generic,gpt"] },
      providers: [
        {
          name: "same",
          api_base_url: "https://same.invalid/v1/chat/completions",
          api_key: "same-provider-key",
          models: ["gpt"],
          transformer: { use: ["OpenAI"] },
        },
        {
          name: "responses",
          api_base_url: "https://responses.invalid/v1/responses",
          api_key: "responses-provider-key",
          models: ["gpt"],
          transformer: { use: ["openai-responses"] },
        },
        {
          name: "generic",
          api_base_url: "https://generic.invalid/v1/chat/completions",
          api_key: "generic-provider-key",
          models: ["gpt"],
        },
        {
          // No transformer.use, but a recognized native-caching host — the
          // Responses→Chat cache-key deletion must be followed by
          // applyProviderNativeChatCaching deriving a fresh ccr_ key here.
          name: "openai",
          api_base_url: "https://api.openai.com/v1/chat/completions",
          api_key: "openai-provider-key",
          models: ["gpt"],
        },
        {
          name: "anthropic",
          // Bare origin: the transformer must derive /v1/messages?beta=true.
          api_base_url: "https://anthropic.invalid",
          api_key: "anthropic-provider-key",
          models: ["claude"],
          transformer: { use: ["Anthropic"] },
        },
        {
          name: "manual",
          api_base_url: "https://manual.invalid/v1/messages",
          api_key: "manual-provider-key",
          models: ["claude"],
          transformer: {
            use: ["Anthropic", "test-noop"],
            passthrough: true,
          },
        },
        {
          // Claude subscription (OAuth) chain: claude-auth supplies the
          // bearer/identity and Anthropic owns wire-body conversion. The
          // api_key is a placeholder only — registerProvider requires a
          // truthy value, but claude-auth overrides it with the OAuth token.
          name: "subscription",
          api_base_url: "https://subscription.invalid",
          api_key: "placeholder-unused-key",
          models: ["claude"],
          transformer: { use: ["claude-auth", "Anthropic"] },
        },
      ],
    },
  });
  const transformerService = new TransformerService(configService, logger);
  await transformerService.initialize();
  transformerService.registerTransformer("test-noop", {
    name: "test-noop",
    async transformRequestIn(request: any) {
      return request;
    },
    async transformResponseOut(response: Response) {
      return response;
    },
  });
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

function chatResponse(content = "ok"): Response {
  return new Response(
    JSON.stringify({
      id: "chatcmpl-test",
      object: "chat.completion",
      created: 1,
      model: "gpt",
      choices: [
        {
          index: 0,
          finish_reason: "stop",
          message: { role: "assistant", content },
        },
      ],
      usage: { prompt_tokens: 2, completion_tokens: 1, total_tokens: 3 },
    }),
    { headers: { "content-type": "application/json" } }
  );
}

// REWRITE_SYSTEM_PROMPT is permitted only on the third-party Anthropic path.
// The rewritten foreign system content is relocated into the first user turn
// by the Claude Code emulation policy.
async function testRewriteSystemPromptAffectsCanonicalBody() {
  const promptDir = mkdtempSync(join(tmpdir(), "ccr-rewrite-prompt-"));
  const promptFile = join(promptDir, "prompt.txt");
  writeFileSync(promptFile, "REWRITTEN PROMPT");

  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      REWRITE_SYSTEM_PROMPT: promptFile,
      Router: { default: "anthropic,claude" },
      providers: [
        {
          name: "anthropic",
          api_base_url: "https://anthropic.invalid",
          api_key: "anthropic-provider-key",
          models: ["claude"],
          transformer: { use: ["Anthropic"] },
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

  const captured: CapturedRequest[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    captured.push({
      url: String(input),
      headers: new Headers(init?.headers),
      body: JSON.parse(String(init?.body || "{}")),
    });
    return new Response(
      JSON.stringify({
        id: "msg_rewrite",
        type: "message",
        role: "assistant",
        model: "claude",
        content: [{ type: "text", text: "ok" }],
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 1 },
      }),
      { headers: { "content-type": "application/json" } }
    );
  };

  try {
    const result = await app.inject({
      method: "POST",
      url: "/v1/messages",
      payload: {
        model: "anthropic,claude",
        max_tokens: 32,
        system: [
          {
            type: "text",
            text: "x-anthropic-billing-header: cc_version=2.1.226.abc;",
          },
          { type: "text", text: "original prefix<env>original env body</env>" },
        ],
        messages: [{ role: "user", content: "hi" }],
      },
    });
    assert.equal(result.statusCode, 200, result.body);
    const upstream = captured.at(-1)!;
    assert.equal(
      upstream.body.system[1].text,
      "You are Claude Code, Anthropic's official CLI for Claude."
    );
    assert.equal(
      upstream.body.messages[0].content[0].text,
      "REWRITTEN PROMPT<env>original env body</env>\n\nhi"
    );
  } finally {
    globalThis.fetch = originalFetch;
    await app.close();
    rmSync(promptDir, { recursive: true, force: true });
  }
}

async function main() {
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDeviceFile = process.env.CCR_CLAUDE_DEVICE_FILE;
  const authTempDir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-"));
  process.env.CCR_CLAUDE_AUTH_FILE = join(authTempDir, "claude_auth.json");
  process.env.CCR_CLAUDE_DEVICE_FILE = join(authTempDir, "claude_device.json");
  writeFileSync(
    process.env.CCR_CLAUDE_AUTH_FILE,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );

  const app = await buildApp();
  const captured: CapturedRequest[] = [];
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (input, init) => {
    const request = {
      url: String(input),
      headers: new Headers(init?.headers),
      body: JSON.parse(String(init?.body || "{}")),
    };
    captured.push(request);
    if (
      request.url.includes("manual.invalid") ||
      request.url.includes("anthropic.invalid") ||
      request.url.includes("subscription.invalid")
    ) {
      return new Response(
        JSON.stringify({
          id: "msg_manual",
          type: "message",
          role: "assistant",
          model: "claude",
          content: [{ type: "text", text: "manual ok" }],
          stop_reason: "end_turn",
          usage: { input_tokens: 1, output_tokens: 1 },
        }),
        { headers: { "content-type": "application/json" } }
      );
    }
    if (request.url.includes("responses.invalid")) {
      if (typeof request.body.input === "string") {
        return new Response(
          JSON.stringify({
            id: "resp_exact",
            object: "response",
            created_at: 1,
            status: "completed",
            model: "gpt",
            output: [],
          }),
          { headers: { "content-type": "application/json" } }
        );
      }
      return new Response('{"error":{"message":"temporary"}}', {
        status: 500,
        headers: { "content-type": "application/json" },
      });
    }
    return chatResponse();
  };

  try {
    // No latent Gemini/legacy route is exposed by endpoint transformer metadata.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1beta/models/gemini-test:generateContent",
        payload: { contents: [] },
      });
      assert.equal(result.statusCode, 404);
      const legacy = await app.inject({
        method: "POST",
        url: "/v1/completions",
        payload: { model: "generic,gpt", prompt: "hello" },
      });
      assert.equal(legacy.statusCode, 404);
    }

    // Protocol detection runs before JSON parsing, so parse failures still use
    // the OpenAI error envelope and never reach an upstream.
    {
      const before = captured.length;
      const result = await app.inject({
        method: "POST",
        url: "/v1/responses",
        headers: { "content-type": "application/json" },
        payload: "{",
      });
      assert.equal(result.statusCode, 400, result.body);
      assert.equal(result.json().error.type, "invalid_request_error");
      assert.equal(captured.length, before);
    }

    // Exact Chat passthrough keeps OpenAI metadata but replaces every inbound
    // credential with the configured provider credential.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        headers: {
          authorization: "Bearer ccr-client-secret",
          "x-api-key": "ccr-client-secret",
          "x-auth-token": "another-client-secret",
          "openai-beta": "responses=v1",
        },
        payload: {
          model: "same,gpt",
          messages: [{ role: "user", content: "hello" }],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(upstream.body.model, "gpt");
      assert.equal(
        upstream.headers.get("authorization"),
        "Bearer same-provider-key",
        JSON.stringify(
          captured.map((request) => ({ url: request.url, body: request.body }))
        )
      );
      assert.equal(upstream.headers.get("x-api-key"), null);
      assert.equal(upstream.headers.get("x-auth-token"), null);
      assert.equal(upstream.headers.get("openai-beta"), "responses=v1");
    }

    // Models advertised by gateway discovery as 1M-capable round-trip through
    // non-Anthropic providers: the picker suffix is CCR metadata, never part
    // of the upstream model id.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        payload: {
          model: "generic,gpt[1m]",
          messages: [{ role: "user", content: "hello" }],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(upstream.body.model, "gpt");
    }

    // Claude-filter-safe aliases are decoded before routing and accept the
    // Desktop-generated [1m] suffix. The canonical route reaches the provider.
    {
      const alias = encodeClaudeModelAlias("generic,gpt");
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        payload: {
          model: `${alias}[1m]`,
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(upstream.url, "https://generic.invalid/v1/chat/completions");
      assert.equal(upstream.body.model, "gpt");

      const before = captured.length;
      const missing = await app.inject({
        method: "POST",
        url: "/v1/messages",
        payload: {
          model: encodeClaudeModelAlias("missing,model"),
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
        },
      });
      assert.equal(missing.statusCode, 404, missing.body);
      assert.equal(missing.json().error.code, "model_not_found");
      assert.equal(captured.length, before);
    }

    // Existing manual passthrough chains keep exact Anthropic-only fields while
    // still allowing adjacent provider middleware to run.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        payload: {
          model: "manual,claude",
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
          output_config: { effort: "high" },
          custom_provider_field: { keep: true },
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.deepEqual(upstream.body.output_config, { effort: "high" });
      assert.deepEqual(upstream.body.custom_provider_field, { keep: true });
      assert.equal(result.json().type, "message");
      assert.equal(result.json().content[0].text, "manual ok");
    }

    // Native Claude Code → Anthropic exact path. The upstream call must be
    // indistinguishable from a direct Claude Code request: derived
    // /v1/messages?beta=true URL, provider-generated auth only, complete safe
    // identity headers, and a byte-faithful cache-rich body.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        headers: {
          // Client credentials must never reach the upstream.
          authorization: "Bearer ccr-client-secret",
          "x-api-key": "ccr-client-secret",
          cookie: "session=client-cookie",
          "x-forwarded-for": "10.0.0.1",
          // Claude Code identity headers must survive verbatim.
          "user-agent": "claude-cli/2.0.14 (external, cli)",
          "x-app": "cli",
          "anthropic-version": "2023-06-01",
          "anthropic-beta": "claude-code-20250219,fine-grained-tool-streaming-2025-05-14",
          "anthropic-dangerous-direct-browser-access": "true",
          "x-stainless-lang": "js",
          "x-stainless-package-version": "0.94.0",
          "x-stainless-retry-count": "0",
          "x-claude-code-session-id": "session-abc",
          "x-client-request-id": "req-abc",
        },
        payload: {
          model: "anthropic,claude",
          max_tokens: 512,
          metadata: { user_id: "user_abc_session_xyz" },
          thinking: { type: "adaptive" },
          output_config: { effort: "high" },
          stop_sequences: ["</done>"],
          system: [
            {
              type: "text",
              text: "x-anthropic-billing-header: {\"is_subagent\":false}",
            },
            {
              type: "text",
              text: "You are Claude Code, Anthropic's official CLI for Claude.",
              cache_control: { type: "ephemeral" },
            },
          ],
          messages: [
            {
              role: "user",
              content: [
                {
                  type: "text",
                  text: "hello",
                  cache_control: { type: "ephemeral" },
                },
              ],
            },
          ],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;

      assert.equal(
        upstream.url,
        "https://anthropic.invalid/v1/messages?beta=true"
      );

      // Provider auth replaces every inbound credential.
      assert.equal(
        upstream.headers.get("x-api-key"),
        "anthropic-provider-key"
      );
      assert.equal(upstream.headers.get("authorization"), null);
      assert.equal(upstream.headers.get("cookie"), null);
      assert.equal(upstream.headers.get("x-forwarded-for"), null);

      // Identity headers are forwarded unchanged.
      assert.equal(
        upstream.headers.get("user-agent"),
        "claude-cli/2.0.14 (external, cli)"
      );
      assert.equal(upstream.headers.get("x-app"), "cli");
      assert.equal(upstream.headers.get("anthropic-version"), "2023-06-01");
      assert.equal(
        upstream.headers.get("anthropic-beta"),
        "claude-code-20250219,fine-grained-tool-streaming-2025-05-14"
      );
      assert.equal(
        upstream.headers.get("anthropic-dangerous-direct-browser-access"),
        "true"
      );
      assert.equal(upstream.headers.get("x-stainless-lang"), "js");
      assert.equal(upstream.headers.get("x-stainless-retry-count"), "0");
      assert.equal(
        upstream.headers.get("x-claude-code-session-id"),
        "session-abc"
      );
      assert.equal(upstream.headers.get("x-client-request-id"), "req-abc");

      // Body is the client wire body with only the model rewritten.
      assert.equal(upstream.body.model, "claude");
      assert.equal(upstream.body.max_tokens, 512);
      assert.deepEqual(upstream.body.metadata, {
        user_id: "user_abc_session_xyz",
      });
      assert.deepEqual(upstream.body.thinking, { type: "adaptive" });
      assert.deepEqual(upstream.body.output_config, { effort: "high" });
      assert.deepEqual(upstream.body.stop_sequences, ["</done>"]);

      // The billing marker is part of Claude Code's own wire format; removing
      // it would make the request identifiable as proxied.
      assert.ok(
        String(upstream.body.system[0].text).startsWith(
          "x-anthropic-billing-header"
        )
      );

      // Exact cache preservation: client breakpoints intact and no automatic
      // top-level breakpoint injected (applyRawAnthropicPromptCaching would
      // add one for a body carrying fewer than four explicit markers).
      assert.deepEqual(upstream.body.system[1].cache_control, {
        type: "ephemeral",
      });
      assert.deepEqual(upstream.body.messages[0].content[0].cache_control, {
        type: "ephemeral",
      });
      assert.equal(upstream.body.cache_control, undefined);
      assert.equal(upstream.body.system[0].cache_control, undefined);

      assert.equal(result.json().type, "message");
    }

    // Native Claude Desktop → Anthropic exact path. Desktop's SDK fingerprint
    // is distinct from Claude Code, but it receives the same raw-wire promise:
    // opaque fields, system shape, cache placement and application headers are
    // preserved while only the routed model and provider credential change.
    {
      const desktopBody = {
        model: "anthropic,claude",
        max_tokens: 256,
        system: [{ type: "text", text: "Desktop system" }],
        messages: [
          {
            role: "user",
            content: [{ type: "text", text: "hello" }],
          },
        ],
        custom_desktop_field: { keep: true },
      };
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        headers: {
          "user-agent": "Anthropic/JS 0.94.0",
          "anthropic-desktop-topbar": "1",
          "x-desktop-custom-header": "keep-me",
          authorization: "Bearer desktop-client-secret",
          "x-api-key": "desktop-client-secret",
        },
        payload: desktopBody,
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.deepEqual(upstream.body, {
        ...desktopBody,
        model: "claude",
      });
      assert.equal(
        upstream.headers.get("x-desktop-custom-header"),
        "keep-me"
      );
      assert.equal(upstream.headers.get("x-api-key"), "anthropic-provider-key");
      assert.equal(upstream.headers.get("authorization"), null);
      assert.equal(result.json().type, "message");
    }

    // Current Desktop 3P uses its bundled Agent SDK/CLI transport. The
    // Desktop entrypoint in that otherwise CLI-shaped fingerprint must still
    // select Desktop raw passthrough; all application headers and native cache
    // markers survive while CCR replaces only auth and adds the OAuth beta.
    {
      const desktopBody = {
        model: "subscription,claude",
        max_tokens: 128,
        system: [
          {
            type: "text",
            text: "x-anthropic-billing-header: cc_version=2.1.222.abc;",
          },
          {
            type: "text",
            text: "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.",
            cache_control: { type: "ephemeral" },
          },
        ],
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "hello",
                cache_control: { type: "ephemeral" },
              },
            ],
          },
        ],
        desktop_opaque_field: { keep: true },
      };
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        headers: {
          "user-agent":
            "claude-cli/2.1.222 (external, claude-desktop-3p, agent-sdk/0.3.222)",
          "x-app": "cli",
          "x-claude-code-session-id": "desktop-agent-session",
          "x-stainless-package-version": "0.94.0",
          "x-stainless-runtime": "node",
          "anthropic-client-platform": "desktop_app",
          "anthropic-client-version": "1.26832.0",
          "x-desktop-custom-header": "keep-agent-header",
          "anthropic-beta": "desktop-agent-beta",
          authorization: "Bearer desktop-client-secret",
        },
        payload: desktopBody,
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.deepEqual(upstream.body, { ...desktopBody, model: "claude" });
      assert.equal(
        upstream.headers.get("authorization"),
        "Bearer hermetic-subscription-token"
      );
      assert.equal(upstream.headers.get("x-api-key"), null);
      assert.equal(
        upstream.headers.get("anthropic-beta"),
        "desktop-agent-beta,oauth-2025-04-20"
      );
      assert.equal(
        upstream.headers.get("user-agent"),
        "claude-cli/2.1.222 (external, claude-desktop-3p, agent-sdk/0.3.222)"
      );
      assert.equal(upstream.headers.get("x-app"), "cli");
      assert.equal(
        upstream.headers.get("x-claude-code-session-id"),
        "desktop-agent-session"
      );
      assert.equal(upstream.headers.get("x-stainless-runtime"), "node");
      assert.equal(
        upstream.headers.get("anthropic-client-platform"),
        "desktop_app"
      );
      assert.equal(
        upstream.headers.get("anthropic-client-version"),
        "1.26832.0"
      );
      assert.equal(
        upstream.headers.get("x-desktop-custom-header"),
        "keep-agent-header"
      );
    }

    // CCR-only markers are stripped from the exact-wire body. The subagent tag
    // both selects the destination and must never reach the upstream.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        payload: {
          // Bare Claude Code model name: the tag decides the destination.
          model: "claude-sonnet-4-5-20250929",
          max_tokens: 32,
          messages: [
            {
              role: "user",
              content:
                "<CCR-SUBAGENT-MODEL>anthropic,claude</CCR-SUBAGENT-MODEL>analyze this",
            },
          ],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.ok(upstream.url.startsWith("https://anthropic.invalid/v1/messages"));
      const wire = JSON.stringify(upstream.body);
      assert.ok(
        !wire.includes("CCR-SUBAGENT-MODEL"),
        `subagent tag leaked upstream: ${wire}`
      );
      assert.equal(upstream.body.messages[0].content[0].text, "analyze this");
      assert.deepEqual(upstream.body.messages[0].content[0].cache_control, {
        type: "ephemeral",
      });
    }

    // Normalized Anthropic (Claude Code client) → subscription (claude-auth)
    // chain. Source provenance (metadata/thinking/output_config/stop
    // sequences), the client's own billing block, and explicit cache
    // directives must survive pre-routing normalization even though the
    // destination provider builds its wire body from the Unified form
    // rather than exact passthrough. Auth must be the OAuth bearer only —
    // never a synthesized "Bearer no-key" fallback.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        headers: {
          authorization: "Bearer ccr-client-secret",
          "user-agent": "claude-cli/2.0.14 (external, cli)",
          "x-app": "cli",
          "x-claude-code-session-id": "session-subscription",
          "x-stainless-package-version": "0.94.0",
        },
        payload: {
          model: "subscription,claude",
          max_tokens: 256,
          metadata: { user_id: "user_sub_session_xyz" },
          thinking: { type: "adaptive" },
          output_config: { effort: "high" },
          stop_sequences: ["</done>"],
          system: [
            {
              type: "text",
              text: "x-anthropic-billing-header: {\"is_subagent\":false}",
            },
            {
              type: "text",
              text: "You are Claude Code, Anthropic's official CLI for Claude.",
              cache_control: { type: "ephemeral" },
            },
          ],
          messages: [{ role: "user", content: "hello" }],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(
        upstream.headers.get("authorization"),
        "Bearer hermetic-subscription-token"
      );
      assert.notEqual(upstream.headers.get("authorization"), "Bearer no-key");
      assert.equal(upstream.headers.get("x-api-key"), null);
      assert.deepEqual(upstream.body.metadata, {
        user_id: "user_sub_session_xyz",
      });
      assert.deepEqual(upstream.body.thinking, { type: "adaptive" });
      assert.deepEqual(upstream.body.output_config, { effort: "high" });
      assert.deepEqual(upstream.body.stop_sequences, ["</done>"]);
      assert.ok(
        String(upstream.body.system[0].text).startsWith(
          "x-anthropic-billing-header"
        )
      );
      assert.deepEqual(upstream.body.system[1].cache_control, {
        type: "ephemeral",
      });
    }

    // Non-Claude-Code Chat and Responses clients → subscription
    // (claude-auth) chain. The full synthesized marker/header set must be
    // present (billing block, identity text, synthesized claude-cli
    // user-agent and stainless headers), while client credentials and
    // opaque cross-protocol cache keys must be absent.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        headers: {
          authorization: "Bearer ccr-client-secret",
          "user-agent": "curl/8.0",
        },
        payload: {
          model: "subscription,claude",
          messages: [{ role: "user", content: "hello" }],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(
        upstream.headers.get("authorization"),
        "Bearer hermetic-subscription-token"
      );
      assert.ok(
        String(upstream.headers.get("user-agent")).startsWith("claude-cli/")
      );
      assert.equal(upstream.headers.get("x-app"), "cli");
      assert.equal(
        upstream.headers.get("anthropic-dangerous-direct-browser-access"),
        "true"
      );
      assert.ok(upstream.headers.get("x-claude-code-session-id"));
      assert.ok(upstream.headers.get("x-stainless-lang"));
      assert.ok(
        Array.isArray(upstream.body.system) &&
          String(upstream.body.system[0]?.text).startsWith(
            "x-anthropic-billing-header"
          )
      );
      assert.ok(
        upstream.body.system.some(
          (block: any) =>
            block.text === "You are Claude Code, Anthropic's official CLI for Claude."
        )
      );
    }
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/responses",
        headers: { "user-agent": "opencode/0.5" },
        payload: {
          model: "subscription,claude",
          input: "hello",
          prompt_cache_key: "client-private-cache-key",
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(
        upstream.headers.get("authorization"),
        "Bearer hermetic-subscription-token"
      );
      // Anthropic wire has no cache-key field; the Responses-scoped opaque
      // key must not leak across the protocol boundary.
      assert.equal(upstream.body.prompt_cache_key, undefined);
      assert.ok(
        Array.isArray(upstream.body.system) &&
          String(upstream.body.system[0]?.text).startsWith(
            "x-anthropic-billing-header"
          )
      );
    }

    // Responses client → generic Chat provider → Responses client.
    {
      const result = await app.inject({
        method: "POST",
        url: "/responses/",
        payload: {
          model: "generic,gpt",
          instructions: "Be brief",
          input: "hello",
          prompt_cache_key: "client-private-cache-key",
          reasoning: { effort: "ultra" },
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(upstream.body.model, "gpt");
      assert.ok(Array.isArray(upstream.body.messages));
      assert.equal(upstream.body.prompt_cache_key, undefined);
      assert.equal(upstream.body.reasoning_effort, "ultra");
      assert.equal(upstream.body.reasoning, undefined);
      // Unrecognized destination host: applyProviderNativeChatCaching's
      // default branch only strips cache_control, it does not synthesize a
      // replacement key — the client's opaque key must not be forwarded, but
      // none should be invented either.
      assert.equal("prompt_cache_key" in upstream.body, false);
      const body = result.json();
      assert.equal(body.object, "response");
      assert.equal(body.model, "generic,gpt");
      assert.equal(body.output[0].content[0].text, "ok");
    }

    // Responses client → recognized native-caching Chat provider (no
    // transformer.use configured). The client's Responses-scoped opaque key
    // must still be dropped crossing the protocol boundary, but
    // applyProviderNativeChatCaching's OpenAI branch must then derive its own
    // ccr_ session key rather than leaving prompt_cache_key absent.
    {
      const result = await app.inject({
        method: "POST",
        url: "/responses/",
        payload: {
          model: "openai,gpt",
          instructions: "Be brief",
          input: "hello",
          prompt_cache_key: "client-private-cache-key",
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(upstream.body.model, "gpt");
      assert.notEqual(upstream.body.prompt_cache_key, "client-private-cache-key");
      assert.ok(
        String(upstream.body.prompt_cache_key).startsWith("ccr_"),
        `expected a re-derived ccr_ cache key, got ${upstream.body.prompt_cache_key}`
      );
    }

    // An opaque cache key survives only an exact Responses destination.
    {
      const result = await app.inject({
        method: "POST",
        url: "/v1/responses",
        payload: {
          model: "responses,gpt",
          input: "exact responses",
          prompt_cache_key: "same-protocol-cache-key",
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const upstream = captured.at(-1)!;
      assert.equal(
        upstream.body.prompt_cache_key,
        "same-protocol-cache-key"
      );
      assert.equal(result.json().object, "response");
    }

    // A Responses provider mutates its attempt body into input[]. Its failed
    // attempt must not corrupt the canonical Unified body reused by fallback.
    {
      const before = captured.length;
      const result = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        payload: {
          model: "responses,gpt",
          messages: [{ role: "user", content: "fallback me" }],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const attempts = captured.slice(before);
      assert.equal(attempts.length, 2);
      assert.ok(Array.isArray(attempts[0].body.input));
      assert.equal(attempts[0].body.messages, undefined);
      assert.ok(Array.isArray(attempts[1].body.messages));
      assert.equal(attempts[1].body.input, undefined);
      assert.equal(attempts[1].body.messages[0].content, "fallback me");
    }
  } finally {
    globalThis.fetch = originalFetch;
    await app.close();
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    if (originalDeviceFile === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDeviceFile;
    rmSync(authTempDir, { recursive: true, force: true });
  }

  await testRewriteSystemPromptAffectsCanonicalBody();

  console.log("inbound-protocol-routes: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

/**
 * Third-party Anthropic emulation renames client tools to Claude Code's
 * mcp_PascalCase spelling on the way upstream. The route must restore the
 * caller's names on every response path, including the exact-protocol
 * Anthropic → Anthropic response that reaches the client without a Unified
 * round trip (e.g. Claude Code behind a gateway that drops its fingerprint).
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

const logger = { debug() {}, info() {}, warn() {}, error() {} };

// Headers a gateway forwards from Claude Code: x-* survive, but the claude-cli
// user agent and x-stainless-* do not, so CCR classifies the client as "other".
const GATEWAY_HEADERS = {
  "user-agent": "litellm/1.80.0",
  "x-app": "cli",
  "x-claude-code-session-id": "gateway-session",
};

const MCP_TOOL = "mcp__srv__lookup";

const USAGE = {
  input_tokens: 3,
  output_tokens: 7,
  cache_creation_input_tokens: 11,
  cache_read_input_tokens: 13,
  cache_creation: { ephemeral_5m_input_tokens: 0, ephemeral_1h_input_tokens: 11 },
  service_tier: "standard",
};

function anthropicToolMessage(): Record<string, unknown> {
  return {
    id: "msg_tools",
    type: "message",
    role: "assistant",
    model: "claude",
    content: [
      { type: "text", text: "running" },
      { type: "tool_use", id: "toolu_bash", name: "mcp_Bash", input: { command: "ls" } },
      { type: "tool_use", id: "toolu_mcp", name: "mcp_Mcp__srv__lookup", input: {} },
    ],
    stop_reason: "tool_use",
    stop_sequence: null,
    usage: USAGE,
  };
}

function sseEvent(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
}

function anthropicToolStream(bashName: string, mcpName: string): string {
  return [
    sseEvent("message_start", {
      message: {
        id: "msg_stream",
        type: "message",
        role: "assistant",
        model: "claude",
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { ...USAGE, output_tokens: 1 },
      },
    }),
    sseEvent("content_block_start", {
      index: 0,
      content_block: { type: "tool_use", id: "toolu_bash", name: bashName, input: {} },
    }),
    sseEvent("content_block_delta", {
      index: 0,
      delta: { type: "input_json_delta", partial_json: '{"command":"mcp_Bash"}' },
    }),
    sseEvent("content_block_stop", { index: 0 }),
    sseEvent("content_block_start", {
      index: 1,
      content_block: { type: "tool_use", id: "toolu_mcp", name: mcpName, input: {} },
    }),
    sseEvent("content_block_stop", { index: 1 }),
    sseEvent("message_delta", {
      delta: { stop_reason: "tool_use", stop_sequence: null },
      usage: { output_tokens: 7 },
    }),
    sseEvent("message_stop", {}),
  ].join("");
}

async function buildApp() {
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      Router: { default: "anthropic,claude" },
      providers: [
        {
          name: "anthropic",
          api_base_url: "https://anthropic.invalid",
          api_key: "anthropic-provider-key",
          models: ["claude"],
          transformer: { use: ["Anthropic"] },
        },
        {
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
  const providerService = new ProviderService(configService, transformerService, logger);
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

function anthropicPayload(model: string, stream: boolean) {
  return {
    model,
    max_tokens: 64,
    stream,
    system: [{ type: "text", text: "gateway harness prompt" }],
    tools: [
      { name: "Bash", input_schema: { type: "object", properties: {} } },
      { name: MCP_TOOL, input_schema: { type: "object", properties: {} } },
    ],
    messages: [
      { role: "user", content: [{ type: "text", text: "list files" }] },
      {
        role: "assistant",
        content: [{ type: "tool_use", id: "toolu_prev", name: "Bash", input: {} }],
      },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "toolu_prev", content: "a.ts" }],
      },
    ],
  };
}

async function main() {
  const authTempDir = mkdtempSync(join(tmpdir(), "ccr-tool-names-"));
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDeviceFile = process.env.CCR_CLAUDE_DEVICE_FILE;
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
  const upstreamBodies: any[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (_input, init) => {
    const body = JSON.parse(String(init?.body || "{}"));
    upstreamBodies.push(body);
    if (body.stream) {
      return new Response(anthropicToolStream("mcp_Bash", "mcp_Mcp__srv__lookup"), {
        headers: { "content-type": "text/event-stream" },
      });
    }
    return new Response(JSON.stringify(anthropicToolMessage()), {
      headers: { "content-type": "application/json" },
    });
  };

  try {
    for (const provider of ["anthropic", "subscription"]) {
      // Non-streaming: names restored, everything else (usage including the
      // cache_creation TTL breakdown) is the provider's response verbatim.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/messages",
          headers: GATEWAY_HEADERS,
          payload: anthropicPayload(`${provider},claude`, false),
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);

        const wire = upstreamBodies.at(-1);
        assert.deepEqual(
          wire.tools.map((tool: any) => tool.name),
          ["mcp_Bash", "mcp_Mcp__srv__lookup"],
          `${provider}: upstream tool definitions`
        );
        assert.equal(wire.messages[1].content[0].name, "mcp_Bash");

        const expected = anthropicToolMessage();
        (expected.content as any[])[1].name = "Bash";
        (expected.content as any[])[2].name = MCP_TOOL;
        assert.deepEqual(result.json(), expected, `${provider}: JSON response`);
      }

      // Streaming: only the tool_use content_block_start events change; every
      // other event reaches the client byte-identical.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/messages",
          headers: GATEWAY_HEADERS,
          payload: anthropicPayload(`${provider},claude`, true),
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        assert.equal(
          result.body,
          anthropicToolStream("Bash", MCP_TOOL),
          `${provider}: SSE response`
        );
      }

      // Chat Completions clients take the Unified conversion path. An MCP-style
      // name must be restored exactly once, never stripped a second time.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/chat/completions",
          payload: {
            model: `${provider},claude`,
            messages: [{ role: "user", content: "look it up" }],
            tools: ["Bash", MCP_TOOL].map((name) => ({
              type: "function",
              function: { name, parameters: { type: "object", properties: {} } },
            })),
          },
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        const names = result
          .json()
          .choices[0].message.tool_calls.map((call: any) => call.function.name);
        assert.deepEqual(names, ["Bash", MCP_TOOL], `${provider}: chat tool calls`);
      }
    }
    console.log("anthropic.third-party-tool-names: ok");
  } finally {
    globalThis.fetch = originalFetch;
    await app.close();
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    if (originalDeviceFile === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDeviceFile;
    rmSync(authTempDir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

import assert from "node:assert/strict";
import {
  applyThirdPartyAnthropicPolicy,
  classifyAnthropicClient,
  getAnthropicProviderMode,
  inspectAnthropicClientFingerprint,
} from "../utils/anthropic-client-policy";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

function desktopHeaders() {
  return {
    "user-agent": "Anthropic/JS 0.94.0",
    "anthropic-desktop-topbar": "1",
    "x-stainless-package-version": "0.94.0",
  };
}

function cliHeaders() {
  return {
    "user-agent": "claude-cli/2.1.226 (subscriber, cli)",
    "x-app": "cli",
    "x-claude-code-session-id": "session-1",
    "x-stainless-package-version": "0.94.0",
  };
}

function desktopAgentSdkHeaders() {
  return {
    "user-agent":
      "claude-cli/2.1.222 (external, claude-desktop-3p, agent-sdk/0.3.222)",
    "x-app": "cli",
    "x-claude-code-session-id": "desktop-session-1",
    "x-stainless-package-version": "0.94.0",
    "anthropic-client-platform": "desktop_app",
    "anthropic-client-version": "1.26832.0",
  };
}

function cliBody() {
  return {
    system: [
      {
        type: "text",
        text: "x-anthropic-billing-header: cc_version=2.1.226.abc;",
      },
      {
        type: "text",
        text: "You are Claude Code, Anthropic's official CLI for Claude.",
      },
    ],
    messages: [{ role: "user", content: "hello" }],
  };
}

function desktopAgentSdkBody() {
  return {
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
  };
}

function testClassification() {
  assert.equal(classifyAnthropicClient(desktopHeaders(), cliBody()), "claude_desktop");
  assert.equal(
    classifyAnthropicClient(desktopAgentSdkHeaders(), desktopAgentSdkBody()),
    "claude_desktop"
  );
  assert.equal(
    inspectAnthropicClientFingerprint(
      desktopAgentSdkHeaders(),
      desktopAgentSdkBody()
    ).desktopAgentSdkUserAgent,
    true
  );
  assert.equal(classifyAnthropicClient(cliHeaders(), cliBody()), "claude_code");
  assert.equal(
    classifyAnthropicClient(
      { "user-agent": "Anthropic/JS 0.94.0" },
      { messages: [] }
    ),
    "other"
  );
  assert.equal(
    classifyAnthropicClient(
      { "user-agent": "claude-cli/2.1.226", "x-app": "cli" },
      cliBody()
    ),
    "other"
  );
  assert.equal(
    classifyAnthropicClient(
      {
        ...desktopHeaders(),
        "x-app": "cli",
        "x-claude-code-session-id": "session-1",
        "x-stainless-package-version": "0.94.0",
      },
      cliBody()
    ),
    "other"
  );
  assert.equal(
    classifyAnthropicClient(
      {
        ...cliHeaders(),
        "x-app": "cli-bg",
      },
      {
        ...cliBody(),
        system: [
          cliBody().system[0],
          {
            type: "text",
            text: "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
          },
        ],
      }
    ),
    "claude_code"
  );
  assert.equal(
    classifyAnthropicClient(
      {
        "user-agent":
          "claude-cli/2.1.222 (external, claude-desktop-3p, agent-sdk/0.3.222)",
        "x-app": "cli",
      },
      desktopAgentSdkBody()
    ),
    "other",
    "an incomplete Desktop Agent SDK fingerprint must fail closed"
  );
}

function testProviderScope() {
  assert.equal(
    getAnthropicProviderMode({ transformer: { use: [{ name: "Anthropic" }] } }),
    "api_key"
  );
  assert.equal(
    getAnthropicProviderMode({
      transformer: { use: [{ name: "claude-auth" }, { name: "Anthropic" }] },
    }),
    "claude_oauth"
  );
  assert.equal(
    getAnthropicProviderMode({ transformer: { use: [{ name: "OpenAI" }] } }),
    "out_of_scope"
  );
  assert.equal(
    getAnthropicProviderMode({
      transformer: { use: [{ name: "Anthropic" }, { name: "test-noop" }] },
    }),
    "out_of_scope"
  );
}

async function testThirdPartyPolicyAndApiKeyWire() {
  const request: any = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    system: [{ type: "text", text: "foreign harness" }],
    messages: [{ role: "user", content: "hello" }],
    tools: [
      {
        type: "function",
        function: {
          name: "read",
          description: "read",
          parameters: { type: "object", properties: {} },
        },
      },
    ],
  };
  const context: any = {
    anthropicClientKind: "other",
    anthropicProviderMode: "api_key",
    anthropicDestinationInScope: true,
  };

  await applyThirdPartyAnthropicPolicy(request, context, { get: () => undefined });

  assert.equal(context.anthropicPolicyApplied, true);
  assert.equal(request.system.length, 2);
  assert.match(request.system[0].text, /^x-anthropic-billing-header:/);
  assert.equal(request.system[1].text, "You are Claude Code, Anthropic's official CLI for Claude.");
  assert.deepEqual(request.system[1].cache_control, { type: "ephemeral" });
  assert.equal(request.messages[0].content[0].text, "foreign harness\n\nhello");
  assert.deepEqual(request.messages[0].content[0].cache_control, {
    type: "ephemeral",
  });
  assert.equal(request.tools[0].function.name, "mcp_Read");
  assert.equal(request.tools[0].cache_control, undefined);

  const wire = await new AnthropicTransformer().transformRequestIn(
    request,
    {
      apiKey: "sk-test",
      transformer: { use: [{ name: "Anthropic" }] },
    } as any,
    { protocolContext: context } as any
  );
  assert.equal(wire.body.system[0].text.startsWith("x-anthropic-billing-header:"), true);
  assert.deepEqual(wire.body.system[1].cache_control, { type: "ephemeral" });
  assert.deepEqual(wire.body.messages[0].content[0].cache_control, {
    type: "ephemeral",
  });
  assert.equal(wire.body.tools[0].name, "mcp_Read");
  assert.equal(
    wire.body.betas,
    undefined,
    "SDK-only betas must not leak into the Anthropic JSON body"
  );
  assert.equal(wire.config.headers["x-api-key"], "sk-test");
  assert.equal(wire.config.headers.Authorization, undefined);
  assert.ok(wire.config.headers["anthropic-beta"].includes("claude-code-20250219"));
  assert.equal(
    wire.config.headers["anthropic-beta"].includes("oauth-2025-04-20"),
    false
  );

  const oauthRequest: any = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    messages: [
      { role: "user", content: "stable prefix" },
      {
        role: "assistant",
        content: [{ type: "thinking", thinking: "internal" }],
      },
    ],
  };
  const oauthContext: any = {
    anthropicClientKind: "other",
    anthropicProviderMode: "claude_oauth",
    anthropicDestinationInScope: true,
  };
  await applyThirdPartyAnthropicPolicy(oauthRequest, oauthContext, {
    get: () => undefined,
  });
  assert.deepEqual(oauthRequest.system[1].cache_control, {
    type: "ephemeral",
    ttl: "1h",
  });
  assert.deepEqual(oauthRequest.messages[0].content[0].cache_control, {
    type: "ephemeral",
    ttl: "1h",
  });
  assert.equal(oauthRequest.messages[1].content[0].cache_control, undefined);
}

async function main() {
  testClassification();
  testProviderScope();
  await testThirdPartyPolicyAndApiKeyWire();
  console.log("anthropic.client-policy: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

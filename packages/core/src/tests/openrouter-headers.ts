/**
 * OpenRouter outbound headers: Claude Code identity (UA / stainless) plus
 * OpenRouter attribution (HTTP-Referer, X-Title, categories).
 */
import assert from "node:assert/strict";
import {
  OpenrouterTransformer,
  buildOpenRouterAttributionHeaders,
  buildOpenRouterOutboundHeaders,
} from "../transformer/openrouter.transformer";

function testAttributionDefaults() {
  const headers = buildOpenRouterAttributionHeaders();
  assert.equal(
    headers["HTTP-Referer"],
    "https://github.com/caeliq/claude-code-router"
  );
  assert.equal(headers["X-Title"], "Claude Code Router");
  assert.equal(headers["X-OpenRouter-Title"], "Claude Code Router");
  assert.equal(headers["X-OpenRouter-Categories"], "cli-agent");
}

function testAttributionOptionOverrides() {
  const headers = buildOpenRouterAttributionHeaders({
    "HTTP-Referer": "https://example.test/app",
    "X-Title": "Custom App",
    "X-OpenRouter-Categories": "cli-agent,ide-extension",
  });
  assert.equal(headers["HTTP-Referer"], "https://example.test/app");
  assert.equal(headers["X-Title"], "Custom App");
  assert.equal(headers["X-OpenRouter-Title"], "Custom App");
  assert.equal(headers["X-OpenRouter-Categories"], "cli-agent,ide-extension");
}

function testSynthesizedClaudeIdentityWhenClientIsNotCli() {
  const headers = buildOpenRouterOutboundHeaders({
    "user-agent": "curl/8.0",
  });
  assert.match(String(headers["User-Agent"]), /^claude-cli\//);
  assert.equal(headers["x-app"], "cli");
  assert.equal(headers["x-stainless-lang"], "js");
  assert.ok(headers["x-stainless-package-version"]);
  assert.equal(headers["HTTP-Referer"], "https://github.com/caeliq/claude-code-router");
}

function testForwardsGenuineClaudeCodeIdentity() {
  const headers = buildOpenRouterOutboundHeaders({
    "user-agent": "claude-cli/2.1.226 (external, cli)",
    "x-app": "cli",
    "x-claude-code-session-id": "sess-from-client",
    "x-stainless-lang": "js",
    "x-stainless-package-version": "0.94.0",
  });
  assert.equal(headers["user-agent"], "claude-cli/2.1.226 (external, cli)");
  assert.equal(headers["x-app"], "cli");
  assert.equal(headers["x-claude-code-session-id"], "sess-from-client");
  assert.equal(headers["x-stainless-package-version"], "0.94.0");
  assert.equal(headers["X-Title"], "Claude Code Router");
}

async function testTransformRequestInReturnsHeadersAndKeepsProviderBodyOption() {
  const transformer = new OpenrouterTransformer({
    provider: { only: ["anthropic"] },
    "HTTP-Referer": "https://example.test/ccr",
    "X-Title": "CCR Test",
  });
  const result = await transformer.transformRequestIn(
    {
      model: "anthropic/claude-sonnet-4",
      messages: [{ role: "user", content: "hi" }],
    } as any,
    { name: "openrouter", baseUrl: "https://openrouter.ai/api/v1/chat/completions" },
    { req: { headers: { "user-agent": "litellm/1.0" } } }
  );

  assert.ok(result && typeof result === "object" && "body" in result);
  assert.deepEqual(result.body.provider, { only: ["anthropic"] });
  assert.equal(result.body["HTTP-Referer"], undefined);
  assert.equal(result.body["X-Title"], undefined);
  assert.match(String(result.config.headers["User-Agent"]), /^claude-cli\//);
  assert.equal(result.config.headers["HTTP-Referer"], "https://example.test/ccr");
  assert.equal(result.config.headers["X-Title"], "CCR Test");
  assert.equal(result.config.headers["X-OpenRouter-Title"], "CCR Test");
}

async function main() {
  testAttributionDefaults();
  testAttributionOptionOverrides();
  testSynthesizedClaudeIdentityWhenClientIsNotCli();
  testForwardsGenuineClaudeCodeIdentity();
  await testTransformRequestInReturnsHeadersAndKeepsProviderBodyOption();
  console.log("openrouter-headers: all tests passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

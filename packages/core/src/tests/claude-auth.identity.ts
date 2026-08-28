import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

import {
  ClaudeAuthTransformer,
  buildSynthesizedIdentityHeaders,
  isClaudeCodeClient,
  resolveClaudeAuthBetas,
  modelIdForRequestedOneMillionBeta,
  applyClaudeModelCapabilityAdjustments,
  __resetClaudeAuthTransformerStateForTests,
} from "../transformer/claude-auth.transformer";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import {
  buildClaudeBillingHeaderValue,
  computeVersionSuffix,
  sessionCch,
  SYSTEM_IDENTITY,
  CC_VERSION,
  __resetClaudeBillingStateForTests,
  prefixClaudeToolName,
  unprefixClaudeToolName,
} from "../utils/claude-billing";
import {
  CLAUDE_MODEL_CATALOG,
  catalogEntryHasCapability,
} from "../utils/claude-model-catalog";
import { mergeHeadersCaseInsensitive, canonicalizeOutboundHeaders } from "../utils/headers";
import { applyThirdPartyAnthropicPolicy } from "../utils/anthropic-client-policy";
import { listClientRouteRegistrations } from "../routing/protocol-endpoints";
import { TransformerContext } from "../types/transformer";
import { UnifiedChatRequest } from "../types/llm";

const routesSource = readFileSync(
  join(__dirname, "..", "api", "routes.ts"),
  "utf-8"
);

function resetState() {
  __resetClaudeAuthTransformerStateForTests();
  __resetClaudeBillingStateForTests();
}

/**
 * Keep OAuth-backed tests independent of a developer's credentials. Always
 * provision a fresh synthetic token instead of consulting the default auth
 * file, which may be absent, expired, or concurrently refreshed.
 */
function ensureHermeticAuthFileForTest(): string {
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-hermetic-"));
  const authFile = join(dir, "claude_auth.json");
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  process.env.CCR_CLAUDE_DEVICE_FILE = join(dir, "claude_device.json");
  return dir;
}

// --- Classification -------------------------------------------------------

function testIsClaudeCodeClient() {
  assert.equal(isClaudeCodeClient("claude-cli/2.1.226 (external, cli)"), true);
  assert.equal(isClaudeCodeClient("opencode/0.x"), false);
  assert.equal(isClaudeCodeClient("cursor"), false);
  assert.equal(isClaudeCodeClient(undefined), false);
}

// --- Billing suffix (pinned to live captures) ------------------------------

function testSuffixVectorsPinned() {
  assert.equal(computeVersionSuffix("say ok", "2.1.226"), "503");
  assert.equal(
    computeVersionSuffix("completely different prompt here xyz", "2.1.226"),
    "906"
  );
}

function testCchShape() {
  resetState();
  const first = sessionCch();
  const second = sessionCch();
  assert.equal(first, "00000");
  assert.equal(second, "00000");
  resetState();
}

function testClaudeToolNameMapping() {
  assert.equal(prefixClaudeToolName("bash"), "mcp_Bash");
  assert.equal(prefixClaudeToolName("read_file"), "mcp_Read_file");
  assert.equal(prefixClaudeToolName("mcp_Bash"), "mcp_Bash");
  assert.equal(unprefixClaudeToolName("mcp_Bash"), "bash");
  assert.equal(unprefixClaudeToolName("mcp_Read_file"), "read_file");
  assert.equal(unprefixClaudeToolName("bash"), "bash");
}

function testFullSystemZeroStringEquality() {
  resetState();
  const messages: UnifiedChatRequest["messages"] = [
    { role: "user", content: "say ok" },
  ];
  const suffix = computeVersionSuffix("say ok", CC_VERSION);
  const cch = sessionCch();
  const expected = `x-anthropic-billing-header: cc_version=${CC_VERSION}.${suffix}; cc_entrypoint=unknown; cch=${cch};`;
  assert.equal(buildClaudeBillingHeaderValue(messages), expected);
  resetState();
}

// --- Inbound protocol matrix: classification is UA-driven, suffix input is
// the canonical first user text, independent of which client protocol the
// request originated from (Chat/Responses/Anthropic all normalize to the
// same Unified shape before claude-auth runs). ------------------------------

async function testInboundProtocolMatrixSynthesis() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-identity-"));
  const authFile = join(dir, "claude_auth.json");
  const deviceFile = join(dir, "claude_device.json");
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDeviceFile = process.env.CCR_CLAUDE_DEVICE_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  process.env.CCR_CLAUDE_DEVICE_FILE = deviceFile;
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );

  try {
    // Three "origins" collapse to the same Unified shape by the time
    // claude-auth runs — the only variable that should matter is the actual
    // inbound user-agent, not the protocol the client used.
    const origins = ["anthropic_messages", "openai_chat_completions", "openai_responses"];
    const results: string[] = [];

    // The current first-party marker is stateless, so all three origins share
    // the same literal cch value.
    resetState();
    for (const protocol of origins) {
      const transformer = new ClaudeAuthTransformer();
      const request: UnifiedChatRequest = {
        model: "claude-sonnet-4-6",
        max_tokens: 100,
        messages: [{ role: "user", content: "say ok" }],
      } as UnifiedChatRequest;
      const context: TransformerContext = {
        req: { headers: { "user-agent": "some-third-party/1.0" } },
        protocolContext: {
          protocol,
          anthropicDestinationInScope: true,
        } as any,
      };
      await transformer.transformRequestIn(request, {}, context);
      const billing = (request.system as any[])[0].text as string;
      results.push(billing);
    }

    assert.equal(results[0], results[1]);
    assert.equal(results[1], results[2]);
    assert.match(results[0], /^x-anthropic-billing-header: cc_version=.*cc_entrypoint=unknown; cch=00000;$/);
  } finally {
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    if (originalDeviceFile === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDeviceFile;
    rmSync(dir, { recursive: true, force: true });
    resetState();
  }
}

// --- System handling regression guards -------------------------------------

async function testSystemHandlingRegressionGuards() {
  // Anthropic's OAuth billing validator rejects requests whose system[]
  // carries foreign content past the identity block ("out of extra usage"
  // 400 — see opencode-claude-auth's transforms.ts for the same finding).
  // The caller's own system content is relocated into the first user
  // message instead, so system[] holds only billing + identity.
  resetState();
  const transformer = new ClaudeAuthTransformer();
  const request: UnifiedChatRequest = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    system: [{ type: "text", text: "caller system", cache_control: { type: "ephemeral" } }],
    messages: [{ role: "user", content: "hello world" }],
  } as UnifiedChatRequest;
  const context: TransformerContext = {
    req: { headers: { "user-agent": "some-third-party/1.0" } },
  };

  await transformer.transformRequestIn(request, {}, context);

  const system = request.system as any[];
  assert.equal(system.length, 2);
  assert.ok(system[0].text.startsWith("x-anthropic-billing-header:"));
  assert.equal(system[1].text, SYSTEM_IDENTITY);
  assert.equal(
    request.messages[0].content as any,
    "caller system\n\nhello world"
  );
  assert.equal(
    request.messages.some((m) => m.role === "system"),
    false
  );
  resetState();
}

async function testForeignSystemContentRelocatedMultiBlock() {
  // Mirrors the pi/opencode scenario: a harness system prompt spanning
  // multiple system[] blocks, all relocated together in order.
  resetState();
  const transformer = new ClaudeAuthTransformer();
  const request: UnifiedChatRequest = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    system: [
      { type: "text", text: "You are pi, a coding agent harness." },
      { type: "text", text: "Available tools: read, bash, edit, write." },
    ],
    messages: [{ role: "user", content: "hi" }],
  } as UnifiedChatRequest;
  const context: TransformerContext = {
    req: { headers: { "user-agent": "pi/1.0" } },
  };

  await transformer.transformRequestIn(request, {}, context);

  const system = request.system as any[];
  assert.equal(system.length, 2);
  assert.ok(system[0].text.startsWith("x-anthropic-billing-header:"));
  assert.equal(system[1].text, SYSTEM_IDENTITY);
  assert.equal(
    request.messages[0].content as any,
    "You are pi, a coding agent harness.\n\nAvailable tools: read, bash, edit, write.\n\nhi"
  );
}

async function testForeignSystemContentKeptWithoutUserMessage() {
  // No user message to attach relocated text to — nothing is silently
  // dropped, so the caller's system content stays in system[].
  resetState();
  const transformer = new ClaudeAuthTransformer();
  const request: UnifiedChatRequest = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    system: [{ type: "text", text: "caller system" }],
    messages: [{ role: "assistant", content: "prior reply" }],
  } as UnifiedChatRequest;
  const context: TransformerContext = {
    req: { headers: { "user-agent": "some-third-party/1.0" } },
  };

  await transformer.transformRequestIn(request, {}, context);

  const system = request.system as any[];
  assert.equal(system.length, 3);
  assert.equal(system[2].text, "caller system");
}

async function testForeignSystemContentNotRelocatedForClaudeCode() {
  // Real Claude Code clients: claude-auth doesn't touch system[] at all, so
  // this only applies to the synthesis branch.
  resetState();
  const transformer = new ClaudeAuthTransformer();
  const request: UnifiedChatRequest = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    system: [{ type: "text", text: "own claude code system prompt" }],
    messages: [{ role: "user", content: "hi" }],
  } as UnifiedChatRequest;
  const context: TransformerContext = {
    req: { headers: { "user-agent": "claude-cli/2.1.226" } },
  };

  await transformer.transformRequestIn(request, {}, context);

  assert.deepEqual(request.system, [
    { type: "text", text: "own claude code system prompt" },
  ]);
  assert.equal(request.messages[0].content as any, "hi");
}

// --- Betas: exact expected lists --------------------------------------------

function testExactBetaLists() {
  const originalEnvBeta = process.env.ANTHROPIC_BETAS;
  delete process.env.ANTHROPIC_BETAS;
  try {
    assert.equal(
      resolveClaudeAuthBetas("claude-opus-5"),
      "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07"
    );
    assert.equal(
      resolveClaudeAuthBetas("claude-opus-5[1m]"),
      "claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07"
    );
    assert.equal(
      resolveClaudeAuthBetas("claude-sonnet-4-5"),
      "claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05"
    );
    assert.equal(
      resolveClaudeAuthBetas("claude-haiku-4-5"),
      "oauth-2025-04-20,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05"
    );

    // The current CLI appends ANTHROPIC_BETAS to its model-derived list.
    process.env.ANTHROPIC_BETAS = "custom-beta-2099-01-01, prompt-caching-scope-2026-01-05";
    const withCustomBeta = resolveClaudeAuthBetas("claude-opus-5");
    assert.ok(withCustomBeta.endsWith(",custom-beta-2099-01-01"));
    assert.equal(
      withCustomBeta.split(",").filter((beta) => beta === "prompt-caching-scope-2026-01-05").length,
      1
    );
    delete process.env.ANTHROPIC_BETAS;

    // Effort is request-driven in the current CLI; when explicitly requested,
    // only models advertising the capability receive its beta.
    assert.ok(!resolveClaudeAuthBetas("claude-sonnet-4-5", { includeEffort: true }).includes("effort-2025-11-24"));
    assert.ok(resolveClaudeAuthBetas("claude-opus-5", { includeEffort: true }).includes("effort-2025-11-24"));
  } finally {
    if (originalEnvBeta === undefined) delete process.env.ANTHROPIC_BETAS;
    else process.env.ANTHROPIC_BETAS = originalEnvBeta;
  }
}

// --- Usage-parity guards -----------------------------------------------------

function testUsageParityOneMillionSuffix() {
  const betas = resolveClaudeAuthBetas("claude-opus-5[1m]");
  assert.ok(betas.includes("context-1m-2025-08-07"));
  // Config validation accepts the explicit [1m] suffix — no rejection, and
  // the wire model id has the marker stripped elsewhere (AnthropicTransformer
  // strips it post-build); resolveClaudeAuthBetas itself never throws.
  assert.doesNotThrow(() => resolveClaudeAuthBetas("claude-opus-5[1m]"));
  assert.equal(
    modelIdForRequestedOneMillionBeta("claude-sonnet-4-6", true),
    "claude-sonnet-4-6[1m]"
  );
  assert.equal(
    modelIdForRequestedOneMillionBeta("claude-sonnet-5", true),
    "claude-sonnet-5"
  );
}

async function testNoPreflightCountTokensCall() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-preflight-"));
  const authFile = join(dir, "claude_auth.json");
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );

  const calledUrls: string[] = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: any) => {
    calledUrls.push(String(input));
    return new Response("{}", { status: 200 });
  }) as any;

  try {
    resetState();
    const transformer = new ClaudeAuthTransformer();
    const request: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "hello" }],
    } as UnifiedChatRequest;
    await transformer.transformRequestIn(request, {}, {
      req: { headers: { "user-agent": "claude-cli/2.1.226" } },
    });
    assert.equal(
      calledUrls.some((url) => url.includes("count_tokens")),
      false,
      "claude-auth must never preflight with /count_tokens"
    );
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    rmSync(dir, { recursive: true, force: true });
    resetState();
  }
}

async function testNoLocalContextRejection() {
  // A request whose content is far larger than the local 200K token estimate
  // must not be rejected or rerouted by claude-auth — configured routing
  // remains the operator's choice, and this layer has no size-based branch
  // at all.
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-oversize-"));
  const authFile = join(dir, "claude_auth.json");
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );

  try {
    resetState();
    const transformer = new ClaudeAuthTransformer();
    const oversizedContent = "x".repeat(2_000_000); // far above 200K tokens
    const request: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: oversizedContent }],
    } as UnifiedChatRequest;

    let thrown: unknown;
    let result: any;
    try {
      result = await transformer.transformRequestIn(request, {}, {
        req: { headers: { "user-agent": "claude-cli/2.1.226" } },
      });
    } catch (err) {
      thrown = err;
    }

    assert.equal(thrown, undefined, "claude-auth must not reject oversized requests");
    assert.equal(result.body, request);
    assert.ok(result.config.headers.Authorization);
  } finally {
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    rmSync(dir, { recursive: true, force: true });
    resetState();
  }
}

function testNoCompactionTriggerInvented() {
  // compaction_trigger is Codex/Responses-protocol data. claude-auth and the
  // shared error envelope must never synthesize it.
  assert.equal(routesSource.includes("compaction_trigger"), false);
  const claudeAuthSource = readFileSync(
    join(__dirname, "..", "transformer", "claude-auth.transformer.ts"),
    "utf-8"
  );
  assert.equal(claudeAuthSource.includes("compaction_trigger"), false);
  const protocolErrorsSource = readFileSync(
    join(__dirname, "..", "routing", "protocol-errors.ts"),
    "utf-8"
  );
  assert.equal(protocolErrorsSource.includes("compaction_trigger"), false);
}

async function testOverageHeaderPreservedThroughConversion() {
  resetState();
  const transformer = new ClaudeAuthTransformer();
  const response = new Response("{}", {
    status: 200,
    headers: {
      "anthropic-ratelimit-unified-overage-in-use": "true",
      "anthropic-ratelimit-unified-overage-status": "allowed_stop",
    },
  });
  const debugCalls: any[] = [];
  transformer.logger = { debug: (...args: any[]) => debugCalls.push(args) };

  const out = await transformer.transformResponseOut(response);

  // Must not warn/fail — only observability logging — and must preserve the
  // header on the returned response unmodified.
  assert.equal(out, response);
  assert.equal(
    out.headers.get("anthropic-ratelimit-unified-overage-in-use"),
    "true"
  );
  assert.equal(debugCalls.length, 1);
  resetState();
}

// --- Headers: synthesized set present for non-CC, absent for CC -----------

async function testSynthesizedHeadersPresenceByClient() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-headers-"));
  const authFile = join(dir, "claude_auth.json");
  const deviceFile = join(dir, "claude_device.json");
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDeviceFile = process.env.CCR_CLAUDE_DEVICE_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  process.env.CCR_CLAUDE_DEVICE_FILE = deviceFile;
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );

  try {
    resetState();
    const nonCcTransformer = new ClaudeAuthTransformer();
    const nonCcRequest: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    } as UnifiedChatRequest;
    const nonCcResult = await nonCcTransformer.transformRequestIn(nonCcRequest, {}, {
      req: { headers: { "user-agent": "non-claude-code-client/1.0" } },
    });
    for (const name of [
      "x-app",
      "anthropic-dangerous-direct-browser-access",
      "X-Claude-Code-Session-Id",
      "x-client-request-id",
      "x-stainless-arch",
      "x-stainless-lang",
      "x-stainless-os",
      "x-stainless-package-version",
      "x-stainless-retry-count",
      "x-stainless-runtime",
      "x-stainless-runtime-version",
      "x-stainless-timeout",
    ]) {
      assert.ok(
        nonCcResult.config.headers[name] !== undefined,
        `expected synthesized header ${name} for non-CC client`
      );
    }

    resetState();
    const ccTransformer = new ClaudeAuthTransformer();
    const ccRequest: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    } as UnifiedChatRequest;
    const ccResult = await ccTransformer.transformRequestIn(ccRequest, {}, {
      req: { headers: { "user-agent": "claude-cli/2.1.226" } },
    });
    for (const name of [
      "x-app",
      "anthropic-dangerous-direct-browser-access",
      "X-Claude-Code-Session-Id",
      "x-stainless-arch",
      "x-stainless-package-version",
    ]) {
      assert.equal(
        ccResult.config.headers[name],
        undefined,
        `did not expect synthesized header ${name} to be fabricated for a genuine CC client (only forwarded if present on the inbound request)`
      );
    }
  } finally {
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    if (originalDeviceFile === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDeviceFile;
    rmSync(dir, { recursive: true, force: true });
    resetState();
  }
}

function testSynthesizedCustomHeaderSafety() {
  const original = process.env.ANTHROPIC_CUSTOM_HEADERS;
  process.env.ANTHROPIC_CUSTOM_HEADERS =
    "x-client-app: custom-client\n" +
    "x-future-header: preserved\n" +
    "Authorization: attacker-token\n" +
    "x-api-key: attacker-key\n" +
    "anthropic-beta: attacker-beta";
  try {
    const headers = buildSynthesizedIdentityHeaders();
    assert.equal(headers["x-client-app"], "custom-client");
    assert.equal(headers["x-future-header"], "preserved");
    assert.equal(headers.Authorization, undefined);
    assert.equal(headers["x-api-key"], undefined);
    assert.equal(headers["anthropic-beta"], undefined);
  } finally {
    if (original === undefined) delete process.env.ANTHROPIC_CUSTOM_HEADERS;
    else process.env.ANTHROPIC_CUSTOM_HEADERS = original;
  }
}

// --- Tools: names pass through byte-identical -------------------------------

async function testToolNamesSurviveUnprefixed() {
  const transformer = new AnthropicTransformer();
  const unified = await transformer.transformRequestOut({
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    messages: [{ role: "user", content: "hi" }],
    tools: [
      { name: "bash", description: "d", input_schema: { type: "object", properties: {} } },
      { name: "Bash", description: "d", input_schema: { type: "object", properties: {} } },
      { name: "mcp__srv__tool", description: "d", input_schema: { type: "object", properties: {} } },
    ],
  });
  const names = unified.tools?.map((t: any) => t.function?.name ?? t.name);
  assert.deepEqual(names, ["bash", "Bash", "mcp__srv__tool"]);

  const rebuilt = AnthropicTransformer.buildAnthropicBody(unified);
  assert.deepEqual(
    rebuilt.tools?.map((t: any) => t.name),
    ["bash", "Bash", "mcp__srv__tool"]
  );
}

// --- __authRecovery contract -------------------------------------------------

async function testAuthRecoveryContract() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-recovery-"));
  const authFile = join(dir, "claude_auth.json");
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;

  const writeAuth = (tokens: Record<string, any>) =>
    writeFileSync(authFile, JSON.stringify(tokens), { mode: 0o600 });

  try {
    // Case 1: token changed externally (e.g. concurrent `ccr claude-auth`).
    writeAuth({
      access_token: "token-a",
      refresh_token: "refresh-a",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    });
    resetState();
    const transformer = new ClaudeAuthTransformer();
    const request: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    } as UnifiedChatRequest;
    const result = await transformer.transformRequestIn(request, {}, {
      req: { headers: { "user-agent": "claude-cli/2.1.226" } },
    });
    writeAuth({
      access_token: "token-b",
      refresh_token: "refresh-a",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    });
    const recovered1 = await result.config.__authRecovery();
    assert.deepEqual(recovered1, { Authorization: "Bearer token-b" });

    // Case 2: token unchanged, refresh_token present -> refresh path.
    writeAuth({
      access_token: "token-a",
      refresh_token: "refresh-a",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    });
    const originalFetch = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(
        JSON.stringify({
          access_token: "token-refreshed",
          refresh_token: "refresh-b",
          token_type: "Bearer",
          expires_in: 3600,
        }),
        { status: 200 }
      )) as any;
    try {
      const recovered2 = await result.config.__authRecovery();
      assert.deepEqual(recovered2, { Authorization: "Bearer token-refreshed" });
      const persisted = JSON.parse(readFileSync(authFile, "utf-8"));
      assert.equal(persisted.access_token, "token-refreshed");
    } finally {
      globalThis.fetch = originalFetch;
    }

    // Case 3: token unchanged, no refresh_token -> both fail -> null.
    writeAuth({
      access_token: "token-a",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    });
    const recovered3 = await result.config.__authRecovery();
    assert.equal(recovered3, null);
  } finally {
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    rmSync(dir, { recursive: true, force: true });
    resetState();
  }

  assert.equal(
    routesSource.includes("__responseRetry"),
    false,
    "no __responseRetry-style hook should be introduced in api/routes.ts"
  );
}

// --- Catalog-derived gating (Finding 14), table-driven ----------------------

function testCatalogDrivenGating() {
  for (const modelId of Object.keys(CLAUDE_MODEL_CATALOG)) {
    const entry = CLAUDE_MODEL_CATALOG[modelId];
    const betas = resolveClaudeAuthBetas(modelId);

    assert.equal(
      resolveClaudeAuthBetas(modelId, { includeEffort: true }).includes("effort-2025-11-24"),
      catalogEntryHasCapability(entry, "effort"),
      `effort-2025-11-24 mismatch for ${modelId}`
    );
    assert.equal(
      betas.includes("mid-conversation-system-2026-04-07"),
      catalogEntryHasCapability(entry, "mid_conv_system"),
      `mid-conversation-system-2026-04-07 mismatch for ${modelId}`
    );

    const body: Record<string, any> = { thinking: { type: "enabled", budget_tokens: 1024 } };
    applyClaudeModelCapabilityAdjustments(body, entry);
    if (entry.capabilities.length > 0) {
      assert.equal(
        body.thinking.type,
        catalogEntryHasCapability(entry, "adaptive_thinking") ? "adaptive" : "enabled",
        `thinking shape mismatch for ${modelId}`
      );
      assert.equal(
        body.thinking.display,
        "summarized",
        `thinking display must be summarized for ${modelId} (not omitted)`
      );
    }
  }

  // Explicit spot checks called out by the plan.
  assert.ok(resolveClaudeAuthBetas("claude-sonnet-5").includes("mid-conversation-system-2026-04-07"));
  assert.ok(resolveClaudeAuthBetas("claude-opus-4-8").includes("mid-conversation-system-2026-04-07"));
  assert.ok(!resolveClaudeAuthBetas("claude-opus-4-7").includes("mid-conversation-system-2026-04-07"));
}

// --- Chain composition (step 0) ---------------------------------------------

async function runProviderChain(
  requestBody: any,
  provider: any,
  context: TransformerContext
) {
  let body = requestBody;
  let config: any = {};
  for (const providerTransformer of provider.transformer.use as any[]) {
    const transformIn = await providerTransformer.transformRequestIn(body, provider, context);
    if (transformIn.body) body = transformIn.body;
    const nextConfig = transformIn.config || {};
    config = {
      ...config,
      ...nextConfig,
      headers: mergeHeadersCaseInsensitive(config.headers, nextConfig.headers),
    };
  }
  return { body, config };
}

async function testChainAnthropicAloneHasNoMarkers() {
  const provider = {
    apiKey: "sk-ant-live-key",
    baseUrl: "https://api.anthropic.com",
    transformer: { use: [new AnthropicTransformer()] },
  };
  const request: UnifiedChatRequest = {
    model: "claude-sonnet-4-6",
    max_tokens: 100,
    messages: [{ role: "user", content: "hi" }],
  } as UnifiedChatRequest;
  const context: TransformerContext = { req: { headers: {} } };

  const { body, config } = await runProviderChain(request, provider, context);

  const outbound = canonicalizeOutboundHeaders(config.headers, provider.apiKey);
  assert.equal(outbound["x-api-key"], "sk-ant-live-key");
  assert.equal(outbound.Authorization, undefined);
  assert.equal(
    Boolean(body.system?.some((s: any) => s.text?.startsWith("x-anthropic-billing-header:"))),
    false
  );
  assert.equal(
    Boolean(body.system?.some((s: any) => s.text === SYSTEM_IDENTITY)),
    false
  );
}

async function testChainClaudeAuthPlusAnthropicMergesNotReplaces() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-chain-"));
  const authFile = join(dir, "claude_auth.json");
  const deviceFile = join(dir, "claude_device.json");
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDeviceFile = process.env.CCR_CLAUDE_DEVICE_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  process.env.CCR_CLAUDE_DEVICE_FILE = deviceFile;
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );

  try {
    const provider = {
      apiKey: "sk-ant-should-not-be-used",
      baseUrl: "https://api.anthropic.com",
      transformer: { use: [new ClaudeAuthTransformer(), new AnthropicTransformer()] },
    };

    // Non-Claude-Code client: bearer present, no x-api-key, full marker set.
    resetState();
    const nonClaudeCodeRequest: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    } as UnifiedChatRequest;
    const nonClaudeCodeContext: TransformerContext = {
      req: { headers: { "user-agent": "non-claude-code-client/1.0" } },
      protocolContext: {
        anthropicClientKind: "other",
        anthropicProviderMode: "claude_oauth",
        anthropicDestinationInScope: true,
      } as any,
    };
    await applyThirdPartyAnthropicPolicy(
      nonClaudeCodeRequest,
      nonClaudeCodeContext.protocolContext as any,
      { get: () => undefined }
    );
    const nonClaudeCode = await runProviderChain(nonClaudeCodeRequest, provider, nonClaudeCodeContext);
    const nonClaudeCodeOutbound = canonicalizeOutboundHeaders(nonClaudeCode.config.headers, provider.apiKey);
    assert.equal(nonClaudeCodeOutbound.Authorization, "Bearer hermetic-subscription-token");
    assert.equal(nonClaudeCodeOutbound["x-api-key"], undefined);
    assert.equal(
      nonClaudeCode.body.betas,
      undefined,
      "SDK-only betas must not leak into the Anthropic JSON body"
    );
    assert.ok(nonClaudeCodeOutbound["anthropic-beta"].includes("claude-code-20250219"));
    assert.ok(nonClaudeCodeOutbound["anthropic-beta"].includes("oauth-2025-04-20"));
    assert.equal(
      nonClaudeCode.body.system?.some((s: any) => s.text?.startsWith("x-anthropic-billing-header:")),
      true
    );
    assert.equal(nonClaudeCode.body.system?.some((s: any) => s.text === SYSTEM_IDENTITY), true);

    const toolRequest: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "use a tool" }],
      tools: [
        {
          type: "function",
          function: {
            name: "bash",
            description: "Run a command",
            parameters: { type: "object", properties: {} },
          },
        },
        {
          type: "function",
          function: {
            name: "mcp__server__lookup",
            description: "Look something up",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
      tool_choice: { type: "function", function: { name: "bash" } },
    } as UnifiedChatRequest;
    const toolContext: TransformerContext = {
      req: { headers: { "user-agent": "non-claude-code-client/1.0" } },
    };
    const toolResult = await runProviderChain(toolRequest, provider, toolContext);
    assert.deepEqual(
      toolResult.body.tools.map((tool: any) => tool.name),
      ["mcp_Bash", "mcp_Mcp__server__lookup"]
    );
    assert.deepEqual(toolResult.body.tool_choice, {
      type: "tool",
      name: "mcp_Bash",
    });

    const restored = await new ClaudeAuthTransformer().transformResponseOut(
      new Response(
        JSON.stringify({
          choices: [{ message: { tool_calls: [{ function: { name: "mcp_Bash" } }] } }],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      ),
      toolContext
    );
    assert.equal((await restored.json()).choices[0].message.tool_calls[0].function.name, "bash");

    // Genuine Claude Code client: bearer present, no markers.
    resetState();
    const ccRequest: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "hi" }],
    } as UnifiedChatRequest;
    const ccContext: TransformerContext = {
      req: { headers: { "user-agent": "claude-cli/2.1.226" } },
    };
    const cc = await runProviderChain(ccRequest, provider, ccContext);
    const ccOutbound = canonicalizeOutboundHeaders(cc.config.headers, provider.apiKey);
    assert.equal(ccOutbound.Authorization, "Bearer hermetic-subscription-token");
    assert.equal(ccOutbound["x-api-key"], undefined);
    assert.equal(
      Boolean(cc.body.system?.some((s: any) => s?.text?.startsWith?.("x-anthropic-billing-header:"))),
      false
    );

    // request.system set by claude-auth survives buildAnthropicBody into
    // body.system in order, no duplicated role:"system" message.
    assert.equal(nonClaudeCode.body.system[0].text.startsWith("x-anthropic-billing-header:"), true);
    assert.equal(nonClaudeCode.body.system[1].text, SYSTEM_IDENTITY);
    assert.equal(
      nonClaudeCode.body.messages.some((m: any) => m.role === "system"),
      false
    );
  } finally {
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    if (originalDeviceFile === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDeviceFile;
    rmSync(dir, { recursive: true, force: true });
    resetState();
  }
}

async function testSingletonSafetyConcurrentRequests() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-singleton-"));
  const authFile = join(dir, "claude_auth.json");
  const deviceFile = join(dir, "claude_device.json");
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDeviceFile = process.env.CCR_CLAUDE_DEVICE_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = authFile;
  process.env.CCR_CLAUDE_DEVICE_FILE = deviceFile;
  writeFileSync(
    authFile,
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );

  try {
    resetState();
    // A single resolved transformer instance shared across two "concurrent"
    // requests, exactly as TransformerService.getTransformer() would hand
    // out the same singleton to every request referencing this provider.
    const sharedTransformer = new ClaudeAuthTransformer();

    const nativeRequest: UnifiedChatRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 100,
      messages: [{ role: "user", content: "native request" }],
    } as UnifiedChatRequest;
    const nativeContext: TransformerContext = {
      req: { headers: { "user-agent": "claude-cli/2.1.226" } },
    };

    const nonClaudeCodeRequest: UnifiedChatRequest = {
      model: "claude-opus-5",
      max_tokens: 100,
      messages: [{ role: "user", content: "non-claude-code request" }],
    } as UnifiedChatRequest;
    const nonClaudeCodeContext: TransformerContext = {
      req: { headers: { "user-agent": "non-claude-code-client/1.0" } },
    };

    const [nativeResult, nonClaudeCodeResult] = await Promise.all([
      sharedTransformer.transformRequestIn(nativeRequest, {}, nativeContext),
      sharedTransformer.transformRequestIn(nonClaudeCodeRequest, {}, nonClaudeCodeContext),
    ]);

    // Exactly one of the two bodies carries the marker system blocks — the
    // native-client body must never pick up the non-Claude-Code request's markers
    // (or vice versa) due to shared instance state.
    const nativeHasMarkers =
      Array.isArray(nativeRequest.system) &&
      nativeRequest.system.some((s: any) => s.text?.startsWith("x-anthropic-billing-header:"));
    const nonClaudeCodeHasMarkers =
      Array.isArray(nonClaudeCodeRequest.system) &&
      nonClaudeCodeRequest.system.some((s: any) => s.text?.startsWith("x-anthropic-billing-header:"));
    assert.equal(nativeHasMarkers, false);
    assert.equal(nonClaudeCodeHasMarkers, true);

    // Model-derived betas must not cross-contaminate: native uses the
    // client's own (absent) anthropic-beta header, the non-Claude-Code client derives from its
    // own model (claude-opus-5).
    assert.equal(nativeResult.config.headers["anthropic-beta"], "oauth-2025-04-20");
    assert.ok(nonClaudeCodeResult.config.headers["anthropic-beta"].includes("oauth-2025-04-20"));

    // Synthesized identity headers must appear only on the non-Claude-Code result.
    assert.equal(nativeResult.config.headers["x-app"], undefined);
    assert.equal(nonClaudeCodeResult.config.headers["x-app"], "cli");
  } finally {
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    if (originalDeviceFile === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDeviceFile;
    rmSync(dir, { recursive: true, force: true });
    resetState();
  }
}

function testRouteOwnershipUnchanged() {
  const registrations = listClientRouteRegistrations();
  const messagesRoute = registrations.find(
    (r) => r.path === "/v1/messages" && r.isCanonical
  );
  assert.ok(messagesRoute);
  assert.equal(messagesRoute?.ownerTransformerName, "Anthropic");
}

async function main() {
  const originalAuthFile = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDeviceFile = process.env.CCR_CLAUDE_DEVICE_FILE;
  const hermeticDir = ensureHermeticAuthFileForTest();
  try {
    testIsClaudeCodeClient();
    testSuffixVectorsPinned();
    testCchShape();
    testClaudeToolNameMapping();
    testFullSystemZeroStringEquality();
    await testInboundProtocolMatrixSynthesis();
    await testSystemHandlingRegressionGuards();
    await testForeignSystemContentRelocatedMultiBlock();
    await testForeignSystemContentKeptWithoutUserMessage();
    await testForeignSystemContentNotRelocatedForClaudeCode();
    testExactBetaLists();
    testUsageParityOneMillionSuffix();
    await testNoPreflightCountTokensCall();
    await testNoLocalContextRejection();
    testNoCompactionTriggerInvented();
    await testOverageHeaderPreservedThroughConversion();
    await testSynthesizedHeadersPresenceByClient();
    testSynthesizedCustomHeaderSafety();
    await testToolNamesSurviveUnprefixed();
    await testAuthRecoveryContract();
    testCatalogDrivenGating();
    await testChainAnthropicAloneHasNoMarkers();
    await testChainClaudeAuthPlusAnthropicMergesNotReplaces();
    await testSingletonSafetyConcurrentRequests();
    testRouteOwnershipUnchanged();
    console.log("claude-auth.identity: ok");
  } finally {
    if (originalAuthFile === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuthFile;
    if (originalDeviceFile === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDeviceFile;
    rmSync(hermeticDir, { recursive: true, force: true });
    resetState();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

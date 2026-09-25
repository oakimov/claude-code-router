import assert from "node:assert/strict";
import {
  applyNativeClaudeOAuthCacheTtl,
  applyThirdPartyAnthropicPolicy,
  classifyAnthropicClient,
  getAnthropicProviderMode,
  inspectAnthropicClientFingerprint,
  resolveNativeClaudeOAuthCacheTtlMode,
} from "../utils/anthropic-client-policy";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import {
  __resetCachePrefixSnapshotsForTests,
  rememberAndDiffOutboundCachePrefix,
} from "../utils/cache-prefix-debug";
import { CC_VERSION, computeVersionSuffix } from "../utils/claude-billing";

function desktopHeaders() {
  return {
    "user-agent": "Anthropic/JS 0.94.0",
    "anthropic-desktop-topbar": "1",
    "x-stainless-package-version": "0.94.0",
  };
}

function cliHeaders() {
  return {
    "user-agent": "claude-cli/2.1.280 (subscriber, cli)",
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
    "anthropic-client-version": "2.7032.0",
  };
}

function cliBody() {
  return {
    system: [
      {
        type: "text",
        text: "x-anthropic-billing-header: cc_version=2.1.280.abc;",
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
        text: "x-anthropic-billing-header: cc_version=2.1.280.abc;",
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
      { "user-agent": "claude-cli/2.1.280", "x-app": "cli" },
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
  // Relocated prompt is its own block with the Claude Code system-prompt
  // breakpoint; the user text after it carries the tail breakpoint.
  assert.equal(request.messages[0].content.length, 2);
  assert.equal(request.messages[0].content[0].text, "foreign harness");
  assert.deepEqual(request.messages[0].content[0].cache_control, {
    type: "ephemeral",
  });
  assert.equal(request.messages[0].content[1].text, "hello");
  assert.deepEqual(request.messages[0].content[1].cache_control, {
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
  assert.deepEqual(wire.body.messages[0].content[1].cache_control, {
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

/**
 * Chat Completions agent loops (e.g. Mastra) end most requests on role:"tool".
 * The tail breakpoint must reach the Anthropic tool_result block, otherwise
 * the whole history is re-billed as uncached input on every tool step.
 */
async function testToolTailKeepsMessageBreakpoint() {
  for (const toolContent of [
    '{"temp":20}',
    [{ type: "text", text: '{"temp":20}' }],
  ]) {
    for (const mode of ["api_key", "claude_oauth"] as const) {
      const request: any = {
        model: "claude-sonnet-4-6",
        max_tokens: 100,
        messages: [
          { role: "system", content: "agent instructions" },
          { role: "user", content: "weather?" },
          {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: { name: "weather", arguments: "{}" },
              },
              {
                id: "call_2",
                type: "function",
                function: { name: "weather", arguments: "{}" },
              },
            ],
          },
          { role: "tool", tool_call_id: "call_1", content: "sunny" },
          { role: "tool", tool_call_id: "call_2", content: toolContent },
        ],
      };
      const context: any = {
        anthropicClientKind: "other",
        anthropicProviderMode: mode,
        anthropicDestinationInScope: true,
      };
      await applyThirdPartyAnthropicPolicy(request, context, {
        get: () => undefined,
      });
      const expected =
        mode === "claude_oauth"
          ? { type: "ephemeral", ttl: "1h" }
          : { type: "ephemeral" };

      const body = AnthropicTransformer.buildAnthropicBody(request);
      assert.deepEqual(body.messages[0].content, [
        { type: "text", text: "agent instructions", cache_control: expected },
        { type: "text", text: "weather?" },
      ]);
      const tail = body.messages.at(-1);
      assert.equal(tail.role, "user");
      assert.equal(tail.content.length, 2);
      assert.equal(tail.content[0].cache_control, undefined);
      assert.equal(tail.content[1].type, "tool_result");
      assert.deepEqual(tail.content[1].cache_control, expected);
      const markers = JSON.stringify(body).match(/"cache_control"/g) || [];
      assert.equal(
        markers.length,
        3,
        "identity block + relocated system prompt + tool_result tail"
      );
    }
  }
}

/**
 * The Unified body after the policy is the client-stage cache snapshot. A user
 * message must keep one shape whether or not it is the tail, otherwise every
 * turn after a user tail logs a rewritten history.
 */
async function testTailPositionDoesNotChangeClientSnapshot() {
  const turn = async (messages: any[]) => {
    const request: any = {
      model: "claude-haiku-4-5",
      max_tokens: 100,
      messages: [
        { role: "system", content: "agent instructions" },
        ...JSON.parse(JSON.stringify(messages)),
      ],
    };
    await applyThirdPartyAnthropicPolicy(
      request,
      {
        anthropicClientKind: "other",
        anthropicProviderMode: "api_key",
        anthropicDestinationInScope: true,
      } as any,
      { get: () => undefined }
    );
    return request;
  };
  const first = [
    { role: "user", content: "plan a trip" },
    { role: "assistant", content: "ok" },
    { role: "user", content: "find hotels" },
  ];
  const next = [
    ...first,
    {
      role: "assistant",
      content: "",
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "search", arguments: "{}" },
        },
      ],
    },
    { role: "tool", tool_call_id: "call_1", content: "[]" },
  ];

  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("policy-sess", await turn(first), undefined, {
    stage: "client",
  });
  const diff = rememberAndDiffOutboundCachePrefix(
    "policy-sess",
    await turn(next),
    undefined,
    { stage: "client" }
  );
  assert.equal(diff?.change, "appended");
  assert.equal(diff?.prefixIntact, true);
  assert.equal(diff?.approxPrefixTokensLost, 0);
}

/**
 * An empty first user message has nothing to prefix; the relocated prompt must
 * still become its own block and keep the system-prompt breakpoint.
 */
async function testEmptyFirstUserKeepsRelocatedBreakpoint() {
  const request: any = {
    model: "claude-haiku-4-5",
    max_tokens: 100,
    messages: [
      { role: "system", content: "agent instructions" },
      { role: "user", content: "" },
      { role: "assistant", content: "ok" },
      { role: "user", content: "next" },
    ],
  };
  await applyThirdPartyAnthropicPolicy(
    request,
    {
      anthropicClientKind: "other",
      anthropicProviderMode: "api_key",
      anthropicDestinationInScope: true,
    } as any,
    { get: () => undefined }
  );
  const body = AnthropicTransformer.buildAnthropicBody(request);
  assert.deepEqual(body.messages[0].content, [
    {
      type: "text",
      text: "agent instructions",
      cache_control: { type: "ephemeral" },
    },
  ]);
  const markers = JSON.stringify(body).match(/"cache_control"/g) || [];
  assert.equal(markers.length, 3);
}

/** Runs the third-party policy and returns the built Anthropic wire body. */
async function policyWire(request: any): Promise<any> {
  await applyThirdPartyAnthropicPolicy(
    request,
    {
      anthropicClientKind: "other",
      anthropicProviderMode: "api_key",
      anthropicDestinationInScope: true,
    } as any,
    { get: () => undefined }
  );
  return AnthropicTransformer.buildAnthropicBody(request);
}

function cacheMarkerCount(body: any): number {
  return (JSON.stringify(body).match(/"cache_control"/g) || []).length;
}

/**
 * The billing suffix samples the first user text of the body actually sent,
 * i.e. the relocated prompt. Dropping or summarizing early turns must not
 * change system[0], or every breakpoint after it misses.
 */
async function testWindowedHistoryKeepsBillingBlock() {
  const instructions = "agent instructions, long and stable";
  const history = [
    { role: "user", content: "please plan a trip to Lisbon" },
    { role: "assistant", content: "ok" },
    { role: "user", content: "summary: user wants hotels in Lisbon" },
    { role: "assistant", content: "sure" },
    { role: "user", content: "next" },
  ];
  const wire = (messages: any[]) =>
    policyWire({
      model: "claude-haiku-4-5",
      max_tokens: 100,
      messages: [
        { role: "system", content: instructions },
        ...JSON.parse(JSON.stringify(messages)),
      ],
    });
  const full = await wire(history);
  const windowed = await wire(history.slice(2));
  assert.equal(windowed.system[0].text, full.system[0].text);
  assert.ok(
    full.system[0].text.includes(
      `cc_version=${CC_VERSION}.${computeVersionSuffix(instructions, CC_VERSION)};`
    )
  );
}

/**
 * Without a user message nothing is relocated and the caller's blocks stay in
 * system[]; the profile must still stay within Anthropic's 4 breakpoints.
 */
async function testNoUserMessageStaysWithinBreakpointLimit() {
  const body = await policyWire({
    model: "claude-haiku-4-5",
    max_tokens: 100,
    system: [
      { type: "text", text: "a" },
      { type: "text", text: "b" },
      { type: "text", text: "c" },
    ],
    messages: [{ role: "assistant", content: "prefill" }],
  });
  assert.deepEqual(
    body.system.map((block: any) => !!block.cache_control),
    [false, true, false, false, true]
  );
  assert.equal(cacheMarkerCount(body), 3);
}

/** A trailing part the body builder does not send cannot hold the tail marker. */
async function testUnsentTailPartKeepsBreakpoint() {
  const unsent = [
    { type: "text", text: "" },
    { type: "image_url", image_url: {} },
    { type: "file", filename: "a.pdf" },
    { type: "input_audio", input_audio: { data: "x", format: "wav" } },
  ];
  for (const part of unsent) {
    const body = await policyWire({
      model: "claude-haiku-4-5",
      max_tokens: 100,
      messages: [
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
        { role: "assistant", content: "ok" },
        { role: "user", content: [{ type: "text", text: "question" }, part] },
      ],
    });
    assert.deepEqual(
      body.messages.at(-1).content,
      [{ type: "text", text: "question", cache_control: { type: "ephemeral" } }],
      part.type
    );
    assert.equal(cacheMarkerCount(body), 3, part.type);
  }
}

// The billing slot is reserved ahead of identity and filled after relocation;
// with attribution disabled the reserved slot must vanish, leaving identity at
// system[0] and the caller's prompt still relocated.
async function testAttributionDisabledDropsBillingSlot() {
  const previous = process.env.CLAUDE_CODE_ATTRIBUTION_HEADER;
  process.env.CLAUDE_CODE_ATTRIBUTION_HEADER = "0";
  try {
    const body = await policyWire({
      model: "claude-haiku-4-5",
      max_tokens: 100,
      messages: [
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
      ],
    });
    assert.deepEqual(
      body.system.map((block: any) => block.text),
      ["You are Claude Code, Anthropic's official CLI for Claude."]
    );
    assert.equal(body.messages[0].content[0].text, "sys");
  } finally {
    if (previous === undefined) delete process.env.CLAUDE_CODE_ATTRIBUTION_HEADER;
    else process.env.CLAUDE_CODE_ATTRIBUTION_HEADER = previous;
  }
}

// A final user turn with nothing to send is omitted by the body builder; the
// tail breakpoint must move to the previous eligible message instead of
// vanishing with it.
async function testUnsentTailTurnMovesBreakpoint() {
  for (const tail of ["", [{ type: "text", text: "" }, { type: "image_url", image_url: {} }]]) {
    const body = await policyWire({
      model: "claude-haiku-4-5",
      max_tokens: 100,
      messages: [
        { role: "system", content: "sys" },
        { role: "user", content: "hi" },
        { role: "assistant", content: "ok" },
        { role: "user", content: tail },
      ],
    });
    const label = JSON.stringify(tail);
    assert.equal(body.messages.at(-1).role, "assistant", label);
    assert.deepEqual(
      body.messages.at(-1).content,
      [{ type: "text", text: "ok", cache_control: { type: "ephemeral" } }],
      label
    );
    assert.equal(cacheMarkerCount(body), 3, label);
  }
}

function testNativeOAuthCacheTtl() {
  const nativeBody = () => ({
    model: "claude-opus-5-5",
    cache_control: { type: "ephemeral" },
    system: [
      { type: "text", text: "x-anthropic-billing-header: cc_version=2.1.282.772;" },
      { type: "text", text: "You are Claude Code", cache_control: { type: "ephemeral" } },
    ],
    messages: [
      { role: "user", content: [{ type: "text", text: "hi" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "t1", name: "Bash", input: {} }] },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "t1", content: "ok", cache_control: { type: "ephemeral" } },
        ],
      },
      { role: "system", content: [{ type: "text", text: "notice", cache_control: { type: "ephemeral", scope: "turn" } }] },
    ],
  });
  const markers = (body: any) => {
    const found: any[] = [];
    const walk = (value: any) => {
      if (!value || typeof value !== "object") return;
      if (value.cache_control) found.push(value.cache_control);
      for (const child of Object.values(value)) walk(child);
    };
    walk(body);
    return found;
  };
  const oauth = { anthropicNativeWire: true, anthropicProviderMode: "claude_oauth" as const };

  const upgraded = nativeBody();
  assert.equal(applyNativeClaudeOAuthCacheTtl(upgraded, oauth), true);
  assert.deepEqual(markers(upgraded), [
    { type: "ephemeral", ttl: "1h" },
    { type: "ephemeral", ttl: "1h" },
    { type: "ephemeral", ttl: "1h" },
    { type: "ephemeral", scope: "turn", ttl: "1h" },
  ]);
  const placement = (body: any) => JSON.stringify(body, (key, value) => (key === "ttl" ? undefined : value));
  assert.equal(placement(upgraded), placement(nativeBody()));

  for (const context of [
    { anthropicNativeWire: true, anthropicProviderMode: "api_key" as const },
    { anthropicNativeWire: false, anthropicProviderMode: "claude_oauth" as const },
    undefined,
  ]) {
    const body = nativeBody();
    assert.equal(applyNativeClaudeOAuthCacheTtl(body, context), false);
    assert.deepEqual(body, nativeBody());
  }

  const explicit = nativeBody();
  (explicit.system[1] as any).cache_control.ttl = "5m";
  const explicitBefore = JSON.parse(JSON.stringify(explicit));
  assert.equal(applyNativeClaudeOAuthCacheTtl(explicit, oauth), false);
  assert.deepEqual(explicit, explicitBefore);

  const unmarked = { model: "m", messages: [{ role: "user", content: "hi" }] };
  assert.equal(applyNativeClaudeOAuthCacheTtl(unmarked, oauth), false);
  assert.deepEqual(unmarked, { model: "m", messages: [{ role: "user", content: "hi" }] });

  // Only real marker positions are scanned: a marker-shaped value inside tool
  // input is user data, neither rewritten nor treated as an explicit TTL;
  // markers inside tool_result content are real and are extended.
  const payload = (): any => ({
    model: "m",
    messages: [
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "t1",
            name: "Write",
            input: { cache_control: { type: "ephemeral", ttl: "5m" }, nested: { cache_control: { type: "ephemeral" } } },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "t1",
            content: [{ type: "text", text: "ok", cache_control: { type: "ephemeral" } }],
          },
        ],
      },
    ],
  });
  const scoped: any = payload();
  assert.equal(applyNativeClaudeOAuthCacheTtl(scoped, oauth), true);
  assert.deepEqual(scoped.messages[0].content[0].input, payload().messages[0].content[0].input);
  assert.deepEqual(scoped.messages[1].content[0].content[0].cache_control, { type: "ephemeral", ttl: "1h" });

  // `CLAUDE_AUTH_NATIVE_CACHE_TTL: "client"` keeps the client's markers.
  assert.equal(resolveNativeClaudeOAuthCacheTtlMode("client"), "client");
  assert.equal(resolveNativeClaudeOAuthCacheTtlMode(undefined), "1h");
  const clientMode = nativeBody();
  assert.equal(applyNativeClaudeOAuthCacheTtl(clientMode, oauth, "client"), false);
  assert.deepEqual(clientMode, nativeBody());
}

async function main() {
  testClassification();
  testNativeOAuthCacheTtl();
  testProviderScope();
  await testThirdPartyPolicyAndApiKeyWire();
  await testToolTailKeepsMessageBreakpoint();
  await testTailPositionDoesNotChangeClientSnapshot();
  await testEmptyFirstUserKeepsRelocatedBreakpoint();
  await testWindowedHistoryKeepsBillingBlock();
  await testNoUserMessageStaysWithinBreakpointLimit();
  await testUnsentTailPartKeepsBreakpoint();
  await testUnsentTailTurnMovesBreakpoint();
  await testAttributionDisabledDropsBillingSlot();
  console.log("anthropic.client-policy: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

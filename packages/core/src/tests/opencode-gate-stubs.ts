import assert from "node:assert/strict";
import { OpencodeHeadersTransformer } from "../transformer/opencode-headers.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { isFallbackEligibleError } from "../utils/retry";
import { tapUpstreamSSEDebug } from "../utils/sse-debug-tap";
import { tapResponseFirstByte } from "../utils/request-latency";

type Sent = {
  url: string;
  headers: Record<string, string>;
  body: any;
};

function makeContext() {
  return {
    req: {
      sessionId: `gate-stubs-${Math.random().toString(36).slice(2)}`,
      log: { warn() {}, info() {}, debug() {} },
      server: { configService: { getHttpsProxy: () => undefined } },
    },
  } as any;
}

function installFetch(): Sent[] {
  const sent: Sent[] = [];
  (globalThis as any).fetch = async (url: any, init: any) => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((value, key) => (headers[key] = value));
    sent.push({
      url: String(url),
      headers,
      body: JSON.parse(String(init?.body ?? "{}")),
    });
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };
  return sent;
}

const zenResponses = {
  name: "opencode-responses",
  apiKey: "test-key",
  baseUrl: "https://opencode.ai/zen/v1/responses",
};

const zenChat = {
  name: "opencode",
  apiKey: "test-key",
  baseUrl: "https://opencode.ai/zen/v1/chat/completions",
};

const FREE = "muse-spark-1.3-contributor-free";

function responsesClientTool(name: string) {
  return {
    type: "function",
    name,
    description: `${name} tool`,
    parameters: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  };
}

function chatClientTool(name: string) {
  return {
    type: "function",
    function: {
      name,
      description: `${name} tool`,
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
      },
    },
  };
}

async function injectsResponsesStubsWithClonedSchemas() {
  const sent = installFetch();
  const ctx = makeContext();
  const body = {
    model: FREE,
    input: [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "hi" }],
      },
    ],
    stream: false,
    tools: [responsesClientTool("Read"), responsesClientTool("Bash")],
  };
  await new OpencodeHeadersTransformer().transformRequestIn(
    structuredClone(body),
    zenResponses,
    ctx
  );
  assert.equal(sent.length, 1);
  const out = sent[0].body;
  // Stream forced for the free-tier gate.
  assert.equal(out.stream, true);
  // Client tools untouched and first.
  assert.equal(out.tools.length, 4);
  assert.equal(out.tools[0].name, "Read");
  assert.equal(out.tools[1].name, "Bash");
  // Stubs appended with the counterpart's schema cloned.
  const read = out.tools[2];
  const shell = out.tools[3];
  assert.deepEqual(read, {
    type: "function",
    name: "read",
    description: "Read tool",
    parameters: {
      type: "object",
      properties: { path: { type: "string" } },
      required: ["path"],
    },
  });
  assert.equal(shell.name, "shell");
  assert.deepEqual(
    shell.parameters,
    responsesClientTool("Bash").parameters
  );
  assert.deepEqual(ctx.req._opencodeGateAliases, { read: "Read", shell: "Bash" });
}

async function injectsChatShapedStubs() {
  const sent = installFetch();
  const ctx = makeContext();
  const body = {
    model: FREE,
    messages: [{ role: "user", content: "hi" }],
    tools: [chatClientTool("Read"), chatClientTool("Bash")],
  };
  await new OpencodeHeadersTransformer().transformRequestIn(
    structuredClone(body),
    zenChat,
    ctx
  );
  const out = sent[0].body;
  assert.equal(out.stream, true);
  assert.equal(out.tools.length, 4);
  assert.deepEqual(out.tools[2], {
    type: "function",
    function: {
      name: "read",
      description: "Read tool",
      parameters: {
        type: "object",
        properties: { path: { type: "string" } },
      },
    },
  });
  assert.equal(out.tools[3].function.name, "shell");
  assert.deepEqual(ctx.req._opencodeGateAliases, { read: "Read", shell: "Bash" });
}

async function aliasesCursorMockReadFile() {
  // cursor_mock/llm/client.py advertises shell, read_file and edit. The
  // mock dispatches read_file with path/offset/limit, not Claude's Read keys.
  const sent = installFetch();
  const ctx = makeContext();
  const readFile = {
    type: "function",
    name: "read_file",
    description: "Read a file from the filesystem",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string" },
        offset: { type: "integer" },
        limit: { type: "integer" },
        encoding_hint: { type: "string" },
      },
      required: ["path"],
    },
  };
  const shell = responsesClientTool("shell");
  const edit = responsesClientTool("edit");
  await new OpencodeHeadersTransformer().transformRequestIn(
    { model: FREE, input: [], stream: true, tools: [shell, readFile, edit] },
    zenResponses,
    ctx
  );
  assert.deepEqual(sent[0].body.tools.slice(0, 3), [shell, readFile, edit]);
  assert.deepEqual(sent[0].body.tools.map((tool: any) => tool.name), [
    "shell", "read_file", "edit", "read",
  ]);
  assert.deepEqual(sent[0].body.tools[3].parameters, readFile.parameters);
  assert.deepEqual(ctx.req._opencodeGateAliases, { read: "read_file" });

  const args = JSON.stringify({ path: "/tmp/a", offset: 2 });
  const out = await new OpencodeHeadersTransformer().transformResponseOut(
    sseResponse([
      `data: ${JSON.stringify({ type: "response.output_item.done", item: {
        type: "function_call", name: "read", arguments: args,
      } })}`,
    ]),
    { req: ctx.req } as any
  );
  const event = JSON.parse((await out.text()).split("data: ")[1]);
  assert.equal(event.item.name, "read_file");
  assert.equal(event.item.arguments, args);
}

async function aliasesDevinMockExec() {
  // Devin's passthrough profile forwards decoded native read/exec descriptors
  // for Zen free models (devin_mock/chat.py + model_compat/ides/devin.py).
  const sent = installFetch();
  const ctx = makeContext();
  const read = responsesClientTool("read");
  const exec = {
    type: "function",
    name: "exec",
    description: "Execute a command",
    parameters: {
      type: "object",
      properties: { command: { type: "string" } },
      required: ["command"],
    },
  };
  await new OpencodeHeadersTransformer().transformRequestIn(
    { model: FREE, input: [], stream: true, tools: [read, exec] },
    zenResponses,
    ctx
  );
  assert.deepEqual(sent[0].body.tools.slice(0, 2), [read, exec]);
  assert.deepEqual(sent[0].body.tools.map((tool: any) => tool.name), [
    "read", "exec", "shell",
  ]);
  assert.deepEqual(sent[0].body.tools[2].parameters, exec.parameters);
  assert.deepEqual(ctx.req._opencodeGateAliases, { shell: "exec" });

  const args = JSON.stringify({ command: "pwd" });
  const out = await new OpencodeHeadersTransformer().transformResponseOut(
    sseResponse([
      `data: ${JSON.stringify({ type: "response.output_item.done", item: {
        type: "function_call", name: "shell", arguments: args,
      } })}`,
    ]),
    { req: ctx.req } as any
  );
  const event = JSON.parse((await out.text()).split("data: ")[1]);
  assert.equal(event.item.name, "exec");
  assert.equal(event.item.arguments, args);
}

async function mockToolsSurviveResponsesPipeline() {
  const responses = new OpenAIResponsesTransformer();
  for (const tools of [
    [responsesClientTool("shell"), responsesClientTool("read_file"), responsesClientTool("edit")],
    [responsesClientTool("exec"), responsesClientTool("read"), responsesClientTool("apply_patch")],
  ]) {
    const incoming = {
      model: FREE,
      input: [{ role: "user", content: [{ type: "input_text", text: "hi" }] }],
      tools,
      stream: true,
    };
    const unified = await responses.transformRequestOut(incoming, {} as any);
    const outgoing = await responses.transformRequestIn(unified, zenResponses, {});
    assert.deepEqual(
      (outgoing as any).tools.map((tool: any) => ({
        name: tool.name,
        parameters: tool.parameters,
      })),
      tools.map((tool) => ({ name: tool.name, parameters: tool.parameters }))
    );
  }
}

async function replayKeepsClientToolNamesAndResults() {
  const responses = new OpenAIResponsesTransformer();
  const callId = "call_0123456789abcdef01234567";
  const args = '{"command":"pwd"}';
  const incoming = {
    model: FREE,
    input: [
      { role: "user", content: [{ type: "input_text", text: "Run pwd" }] },
      { type: "function_call", name: "bash", call_id: callId, arguments: args },
      { type: "function_call_output", call_id: callId, output: "/tmp" },
    ],
    tools: [responsesClientTool("read"), responsesClientTool("bash")],
    prompt_cache_key: "stable-client-cache-key",
    stream: true,
  };
  const unified = await responses.transformRequestOut(incoming, {} as any);
  const outgoing = await responses.transformRequestIn(unified, zenResponses, {});
  const sent = installFetch();
  await new OpencodeHeadersTransformer().transformRequestIn(
    outgoing,
    zenResponses,
    makeContext()
  );
  const wire = sent[0].body;
  assert.deepEqual(
    wire.input.filter((item: any) => item.type === "function_call"),
    [{ type: "function_call", name: "bash", call_id: callId, arguments: args, status: "completed" }]
  );
  assert.deepEqual(
    wire.input.filter((item: any) => item.type === "function_call_output"),
    [{ type: "function_call_output", call_id: callId, output: "/tmp" }]
  );
  assert.deepEqual(wire.tools.map((tool: any) => tool.name), ["read", "bash", "shell"]);
}

async function aliasesNativeShellVariants() {
  // Pi/oh-my-pi/MiMo/DeepSeek use read+bash; Pi on Windows uses
  // powershell, and DeepSeek's Windows profile uses pwsh.
  for (const name of ["bash", "pwsh", "powershell"]) {
    const sent = installFetch();
    const ctx = makeContext();
    const shell = {
      type: "function",
      name,
      description: `Run ${name}`,
      parameters: {
        type: "object",
        properties: {
          command: { type: "string" },
          description: { type: "string" },
        },
        required: ["command", "description"],
      },
    };
    await new OpencodeHeadersTransformer().transformRequestIn(
      { model: FREE, input: [], stream: true, tools: [responsesClientTool("read"), shell] },
      zenResponses,
      ctx
    );
    assert.deepEqual(sent[0].body.tools.map((tool: any) => tool.name), ["read", name, "shell"]);
    assert.deepEqual(sent[0].body.tools[2].parameters, shell.parameters);
    assert.deepEqual(ctx.req._opencodeGateAliases, { shell: name });
    const out = await new OpencodeHeadersTransformer().transformResponseOut(
      sseResponse([`data: ${JSON.stringify({ type: "response.output_item.done", item: {
        type: "function_call", name: "shell", arguments: '{"command":"pwd","description":"Show directory"}',
      } })}`]),
      { req: ctx.req } as any
    );
    const event = JSON.parse((await out.text()).split("data: ")[1]);
    assert.equal(event.item.name, name);
    assert.equal(JSON.parse(event.item.arguments).command, "pwd");
  }
}

async function aliasesCodeOnlyHarnesses() {
  // MiMo's GPT toolset exposes exec({code}); DeepSeek PTC exposes
  // run_code({code,description}); oh-my-pi code mode exposes
  // eval({language,code}). All can invoke their normal tools from code.
  for (const [name, extra] of [
    ["exec", {}],
    ["run_code", { description: { type: "string" } }],
    ["eval", { language: { type: "string", enum: ["js", "py"] } }],
  ] as const) {
    const sent = installFetch();
    const ctx = makeContext();
    const gateway = {
      type: "function",
      name,
      description: `Run code through ${name}`,
      parameters: {
        type: "object",
        properties: { code: { type: "string" }, ...extra },
        required: ["code", ...Object.keys(extra)],
      },
    };
    await new OpencodeHeadersTransformer().transformRequestIn(
      { model: FREE, input: [], stream: true, tools: [gateway] },
      zenResponses,
      ctx
    );
    assert.deepEqual(sent[0].body.tools.map((tool: any) => tool.name), [name, "read", "shell"]);
    assert.deepEqual(sent[0].body.tools[1].parameters, gateway.parameters);
    assert.deepEqual(sent[0].body.tools[2].parameters, gateway.parameters);
    assert.deepEqual(ctx.req._opencodeGateAliases, { read: name, shell: name });
    for (const stub of ["read", "shell"]) {
      const args = JSON.stringify({ code: "return 1", ...Object.fromEntries(Object.keys(extra).map((key) => [key, key === "language" ? "js" : "Run code"])) });
      const out = await new OpencodeHeadersTransformer().transformResponseOut(
        sseResponse([`data: ${JSON.stringify({ type: "response.output_item.done", item: {
          type: "function_call", name: stub, arguments: args,
        } })}`]),
        { req: ctx.req } as any
      );
      const event = JSON.parse((await out.text()).split("data: ")[1]);
      assert.equal(event.item.name, name);
      assert.equal(event.item.arguments, args);
    }
  }
}

async function createsStubsWhenToolsAbsent() {
  const sent = installFetch();
  const ctx = makeContext();
  await new OpencodeHeadersTransformer().transformRequestIn(
    { model: FREE, messages: [{ role: "user", content: "hi" }] },
    zenChat,
    ctx
  );
  const out = sent[0].body;
  assert.equal(out.stream, true);
  assert.equal(
    out.tools.map((tool: any) => tool.function.name).join(","),
    "read,shell"
  );
  // No client counterpart: generic schema, null aliases (no rewrite).
  assert.deepEqual(out.tools[0].function.parameters, {
    type: "object",
    properties: {},
  });
  assert.match(out.tools[0].function.description, /cannot execute it.*Do not call/);
  assert.deepEqual(ctx.req._opencodeGateAliases, { read: null, shell: null });
}

async function skipsNativeStubsAndPaidModels() {
  const sent = installFetch();
  const ctx = makeContext();
  // Exact lowercase stubs already present: nothing injected, no aliases.
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      input: [],
      stream: true,
      tools: [
        { type: "function", name: "read", description: "r", parameters: {} },
        { type: "function", name: "shell", description: "s", parameters: {} },
      ],
    },
    zenResponses,
    ctx
  );
  assert.equal(sent[0].body.tools.length, 2);
  assert.equal(sent[0].body.stream, true);
  assert.equal(ctx.req._opencodeGateAliases, undefined);

  // Paid model: stream untouched, no stubs, no aliases.
  const ctx2 = makeContext();
  await new OpencodeHeadersTransformer().transformRequestIn(
    { model: "claude-opus-4-6", messages: [] },
    zenChat,
    ctx2
  );
  assert.equal(sent[1].body.stream, undefined);
  assert.equal(sent[1].body.tools, undefined);
  assert.equal(ctx2.req._opencodeGateAliases, undefined);

  // Free model on a non-Zen host: skipped as well.
  const ctx3 = makeContext();
  await new OpencodeHeadersTransformer().transformRequestIn(
    { model: FREE, messages: [] },
    { name: "other", apiKey: "k", baseUrl: "https://example.com/v1" },
    ctx3
  );
  assert.equal(sent[2].body.tools, undefined);
  assert.equal(ctx3.req._opencodeGateAliases, undefined);

  // A lookalike host must not receive the Zen-only body changes.
  const ctx4 = makeContext();
  await new OpencodeHeadersTransformer().transformRequestIn(
    { model: FREE, messages: [] },
    { name: "other", apiKey: "k", baseUrl: "https://opencode.ai.example.com/zen/v1/chat/completions" },
    ctx4
  );
  assert.equal(sent[3].body.tools, undefined);
  assert.equal(sent[3].body.stream, undefined);
}

function sseResponse(events: string[]): Response {
  return new Response(events.join("\n\n") + "\n\n", {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function readStream(response: Response): Promise<string> {
  return await response.text();
}

async function rewritesResponsesStubCalls() {
  const ctx = {
    req: { _opencodeGateAliases: { read: "Read", shell: "Bash" } },
  } as any;
  const callEvent =
    `event: response.output_item.done\ndata: {"type":"response.output_item.done","output_index":0,` +
    `"item":{"id":"fc_1","type":"function_call","name":"read","arguments":"{}"}}`;
  const textEvent =
    `event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"ok"}`;
  const out = await new OpencodeHeadersTransformer().transformResponseOut(
    sseResponse([callEvent, textEvent]),
    ctx
  );
  const text = await readStream(out);
  assert.match(text, /"name":"Read"/);
  assert.doesNotMatch(text, /"name":"read"/);
  assert.match(text, /"delta":"ok"/);
}

async function rewritesChatStubCalls() {
  const ctx = {
    req: { _opencodeGateAliases: { read: "Read", shell: "Bash" } },
  } as any;
  const toolEvent =
    `data: {"id":"x","choices":[{"delta":{"tool_calls":[` +
    `{"id":"c1","type":"function","function":{"name":"shell","arguments":"{}"}}]}}]}`;
  const out = await new OpencodeHeadersTransformer().transformResponseOut(
    sseResponse([toolEvent]),
    ctx
  );
  const text = await readStream(out);
  assert.match(text, /"name":"Bash"/);
  assert.doesNotMatch(text, /"name":"shell"/);
}

async function aliasesSurviveFreshResponseContext() {
  // Mirrors routes.ts: processResponseTransformers builds a FRESH context
  // ({req, signal, protocolContext}) for the response phase. Only props on
  // the shared `req` object ride across; anything stashed on the
  // request-phase context itself is lost (live bug: stub calls reached the
  // client unrenamed).
  const sent = installFetch();
  const ctx = makeContext();
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      input: [],
      stream: true,
      tools: [
        {
          type: "function",
          name: "Bash",
          description: "b",
          parameters: { type: "object", properties: {} },
        },
      ],
    },
    zenResponses,
    ctx
  );
  assert.equal(sent.length, 1);
  // Sanity: shell stub injected (Bash donor), read stub generic.
  const names = sent[0].body.tools.map((tool: any) => tool.name);
  assert.ok(names.includes("shell") && names.includes("read"));

  const freshResponseContext = { req: ctx.req, provider: zenResponses } as any;
  const callEvent =
    `data: {"type":"response.output_item.done","item":` +
    `{"id":"fc_1","type":"function_call","name":"shell","arguments":"{}"}}`;
  const out = await new OpencodeHeadersTransformer().transformResponseOut(
    sseResponse([callEvent]),
    freshResponseContext
  );
  const text = await readStream(out);
  assert.match(text, /"name":"Bash"/);
  assert.doesNotMatch(text, /"name":"shell"/);
}

async function stampsDetailedSummaryOnlyWhenReasoning() {
  // Reasoning active, no summary: stamped.
  let sent = installFetch();
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      input: [],
      stream: true,
      reasoning: { effort: "medium" },
      tools: [{ type: "function", name: "read", description: "r", parameters: {} }],
    },
    zenResponses,
    makeContext()
  );
  assert.deepEqual(sent[0].body.reasoning, {
    effort: "medium",
    summary: "detailed",
  });

  // Explicit client summary (incl. "none") wins.
  sent = installFetch();
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      input: [],
      stream: true,
      reasoning: { effort: "medium", summary: "none" },
      tools: [{ type: "function", name: "read", description: "r", parameters: {} }],
    },
    zenResponses,
    makeContext()
  );
  assert.deepEqual(sent[0].body.reasoning, {
    effort: "medium",
    summary: "none",
  });

  // Disabled reasoning: never invent a summary.
  sent = installFetch();
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      input: [],
      stream: true,
      reasoning: { effort: "none" },
      tools: [{ type: "function", name: "read", description: "r", parameters: {} }],
    },
    zenResponses,
    makeContext()
  );
  assert.deepEqual(sent[0].body.reasoning, { effort: "none" });

  // No reasoning block at all: untouched.
  sent = installFetch();
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      input: [],
      stream: true,
      tools: [{ type: "function", name: "read", description: "r", parameters: {} }],
    },
    zenResponses,
    makeContext()
  );
  assert.equal(sent[0].body.reasoning, undefined);
}

async function cacheKeyEqualsSessionHeader() {
  // Zen goes `response.incomplete` (client-side stall) unless
  // prompt_cache_key equals x-opencode-session (curl A/B 2026-09-24).
  // A foreign key (e.g. CCR's ccr_<sha256>) is replaced on every attempt.
  const sent = installFetch();
  const ctx = makeContext();
  const failingThenOk = [
    () =>
      new Response(
        JSON.stringify({
          error: {
            message:
              "Error from provider (Console): Upstream request failed",
            type: "invalid_request_error",
          },
        }),
        { status: 400 }
      ),
    () =>
      new Response(JSON.stringify({ ok: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  ];
  (globalThis as any).fetch = async (url: any, init: any) => {
    const headers: Record<string, string> = {};
    new Headers(init?.headers).forEach((value, key) => (headers[key] = value));
    sent.push({
      url: String(url),
      headers,
      body: JSON.parse(String(init?.body ?? "{}")),
    });
    return failingThenOk[Math.min(sent.length - 1, 1)]();
  };
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      input: [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "hi" }],
        },
      ],
      stream: true,
      prompt_cache_key: "ccr_foreignkey",
      tools: [
        { type: "function", name: "read", description: "r", parameters: {} },
        { type: "function", name: "shell", description: "s", parameters: {} },
      ],
    },
    zenResponses,
    ctx
  );
  assert.equal(sent.length, 2);
  for (const call of sent) {
    // Same stable session on every attempt, key locked to it.
    assert.equal(call.body.prompt_cache_key, call.headers["x-opencode-session"]);
  }
  assert.equal(
    sent[0].headers["x-opencode-session"],
    sent[1].headers["x-opencode-session"]
  );
  assert.equal(sent[0].body.prompt_cache_key, sent[1].body.prompt_cache_key);

  // Chat wire keeps existing behavior (out of the proven scope).
  const chatSent = installFetch();
  await new OpencodeHeadersTransformer().transformRequestIn(
    {
      model: FREE,
      messages: [{ role: "user", content: "hi" }],
      prompt_cache_key: "ccr_foreignkey",
    },
    zenChat,
    makeContext()
  );
  assert.equal(chatSent[0].body.prompt_cache_key, "ccr_foreignkey");
}

async function sessionAffinitySurvivesToolHistory() {
  const sent = installFetch();
  const user = {
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: `replay ${Math.random()}` }],
  };
  const history = [
    { type: "function_call", name: "bash", call_id: "call_1", arguments: '{"command":"pwd"}' },
    { type: "function_call_output", call_id: "call_1", output: "/tmp" },
  ];
  const clientKey = `client-session-${Math.random()}`;
  for (const cacheKey of [clientKey, undefined]) {
    const first = { model: FREE, input: [user], stream: true, ...(cacheKey ? { prompt_cache_key: cacheKey } : {}) };
    const second = { ...first, input: [user, ...history] };
    for (const body of [first, second]) {
      const ctx = makeContext();
      delete ctx.req.sessionId;
      ctx.req.headers = { "user-agent": "replay-test" };
      await new OpencodeHeadersTransformer().transformRequestIn(
        structuredClone(body), zenResponses, ctx
      );
    }
    const previous = sent.at(-2)!;
    const current = sent.at(-1)!;
    assert.equal(current.headers["x-opencode-session"], previous.headers["x-opencode-session"]);
    assert.equal(current.body.prompt_cache_key, current.headers["x-opencode-session"]);
  }
}

async function leavesNullAliasesAndForeignToolsAlone() {  const ctx = {
    req: { _opencodeGateAliases: { read: null, shell: null } },
  } as any;
  const callEvent =
    `data: {"type":"response.output_item.done","item":` +
    `{"id":"fc_1","type":"function_call","name":"read","arguments":"{}"}}`;
  const out = await new OpencodeHeadersTransformer().transformResponseOut(
    sseResponse([callEvent]),
    ctx
  );
  const text = await readStream(out);
  assert.match(text, /"name":"read"/);

  // No aliases at all: byte-identical passthrough (modulo event framing).
  const out2 = await new OpencodeHeadersTransformer().transformResponseOut(
    sseResponse([`data: {"hello":"read world"}`]),
    {} as any
  );
  assert.match(await readStream(out2), /"hello":"read world"/);
}

async function restoresJsonForNonStreamingResponsesClient() {
  const originalFetch = (globalThis as any).fetch;
  const ctx = makeContext();
  (globalThis as any).fetch = async () => {
    const events = [
      'data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}',
      'data: {"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","model":"muse-spark-1.3-contributor-free","created_at":1,"output":[{"id":"fc_1","type":"function_call","name":"shell","call_id":"call_1","arguments":"{}"}]}}',
    ];
    // Deliberately leave the transport open after the terminal event. A JSON
    // client must finish from response.completed, without waiting for EOF.
    return new Response(
      new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode(events.join("\n\n") + "\n\n"));
        },
      }),
      { headers: { "Content-Type": "text/event-stream" } }
    );
  };
  try {
    const tf = new OpencodeHeadersTransformer();
    const request = await tf.transformRequestIn(
      { model: FREE, input: [], stream: false, tools: [responsesClientTool("Bash")] },
      zenResponses,
      ctx
    );
    assert.equal(request.body.stream, true);
    const restored = await tf.transformResponseOut(
      request.config.__providerResponse,
      { req: ctx.req }
    );
    assert.match(restored.headers.get("content-type") || "", /application\/json/);
    const chat = await new OpenAIResponsesTransformer().transformResponseOut(restored);
    const value: any = await chat.json();
    assert.equal(value.choices[0].message.tool_calls[0].function.name, "Bash");
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

async function terminalCancelCannotBlockJsonResponse() {
  const event =
    'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n' +
    'data: {"type":"response.incomplete","response":{"id":"resp_1","object":"response","status":"incomplete","output":[]}}\n\n';
  const upstream = new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode(event));
      },
      cancel() {
        return new Promise<void>(() => {});
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } }
  );
  const tf = new OpencodeHeadersTransformer();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const restored = await Promise.race([
      tf.transformResponseOut(upstream, {
        req: { _opencodeForcedStream: "responses", _opencodeGateAliases: { shell: "Bash" } },
      } as any),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error("terminal response blocked on stream cancellation")), 250);
      }),
    ]);
    assert.equal(restored.headers.get("content-type"), "application/json");
    assert.equal((await restored.json()).status, "incomplete");
  } finally {
    clearTimeout(timer);
  }
}

async function bufferedTerminalSurvivesDebugAndLatencyTaps() {
  const originalFetch = (globalThis as any).fetch;
  const ctx = makeContext();
  const frames = [
    'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n',
    'data: {"type":"response.in_progress"}\n\n',
    'data: {"type":"response.output_item.added","item":{"type":"reasoning"}}\n\n',
    'data: {"type":"response.incomplete","response":{"id":"resp_1","object":"response","status":"incomplete","output":[]}}\n\n',
  ];
  try {
    (globalThis as any).fetch = async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            for (const frame of frames) {
              const bytes = new TextEncoder().encode(frame);
              for (let offset = 0; offset < bytes.length; offset += 27) {
                controller.enqueue(bytes.slice(offset, offset + 27));
              }
            }
          },
          cancel() {
            return new Promise<void>(() => {});
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } }
      );
    const tf = new OpencodeHeadersTransformer();
    const request = await tf.transformRequestIn(
      {
        model: FREE,
        input: [{ role: "user", content: [{ type: "input_text", text: "hi" }] }],
        stream: false,
        tools: [responsesClientTool("Read"), responsesClientTool("Bash")],
      },
      zenResponses,
      ctx
    );
    const tapped = await tapUpstreamSSEDebug(request.config.__providerResponse, {
      logger: ctx.req.log,
      rawEvents: true,
    });
    const timed = tapResponseFirstByte(tapped, () => {});
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      const result = await Promise.race([
        tf.transformResponseOut(timed, { req: ctx.req } as any),
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => reject(new Error("buffered terminal stalled in response taps")), 250);
        }),
      ]);
      assert.equal((await result.json()).status, "incomplete");
    } finally {
      clearTimeout(timer);
    }
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

async function restoresJsonForNonStreamingChatClient() {
  const originalFetch = (globalThis as any).fetch;
  const ctx = makeContext();
  (globalThis as any).fetch = async () =>
    sseResponse([
      'data: {"id":"chat_1","model":"muse-spark-1.3-contributor-free","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{"}}]},"finish_reason":null}]}',
      'data: {"id":"chat_1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]},"finish_reason":null}]}',
      'data: {"id":"chat_1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
      'data: [DONE]',
    ]);
  try {
    const tf = new OpencodeHeadersTransformer();
    const request = await tf.transformRequestIn(
      { model: FREE, messages: [{ role: "user", content: "hi" }], stream: false, tools: [chatClientTool("Bash")] },
      zenChat,
      ctx
    );
    const restored = await tf.transformResponseOut(
      request.config.__providerResponse,
      { req: ctx.req }
    );
    assert.match(restored.headers.get("content-type") || "", /application\/json/);
    const value: any = await restored.json();
    assert.equal(value.choices[0].message.tool_calls[0].function.name, "Bash");
    assert.equal(value.choices[0].message.tool_calls[0].function.arguments, "{}");
    assert.equal(value.choices[0].finish_reason, "tool_calls");
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

async function firstZenEventMustComplete() {
  const originalFetch = (globalThis as any).fetch;
  const request = {
    model: FREE,
    input: [{ role: "user", content: [{ type: "input_text", text: "hi" }] }],
    stream: true,
    tools: [responsesClientTool("read"), responsesClientTool("shell")],
  };
  try {
    (globalThis as any).fetch = async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('data: {"type":"response.created"'));
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } }
      );
    const stalled = new OpencodeHeadersTransformer();
    (stalled as any).firstEventTimeoutMs = 25;
    await assert.rejects(
      () => stalled.transformRequestIn(structuredClone(request), zenResponses, makeContext()),
      (error: any) => {
        assert.equal(error.statusCode, 504);
        assert.equal(isFallbackEligibleError(error), true);
        assert.match(error.message, /no complete response event/);
        return true;
      }
    );

    (globalThis as any).fetch = async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(
              'data: {"type":"response.created"}\n\n' +
              'data: {"type":"response.in_progress"}\n\n' +
              'data: {"type":"response.output_item.added","item":{"type":"reasoning"}}\n\n'
            ));
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } }
      );
    const noProgress = new OpencodeHeadersTransformer();
    (noProgress as any).firstProgressTimeoutMs = 25;
    await assert.rejects(
      () => noProgress.transformRequestIn(structuredClone(request), zenResponses, makeContext()),
      (error: any) => {
        assert.equal(error.statusCode, 504);
        assert.equal(isFallbackEligibleError(error), true);
        assert.match(error.message, /no output progress/);
        return true;
      }
    );

    const event =
      'data: {"type":"response.created","response":{"id":"resp_1"}}\n\n' +
      'data: {"type":"response.completed","response":{"id":"resp_1"}}\n\n';
    (globalThis as any).fetch = async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            const bytes = new TextEncoder().encode(event);
            controller.enqueue(bytes.slice(0, 19));
            controller.enqueue(bytes.slice(19));
            controller.close();
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } }
      );
    const passed = await new OpencodeHeadersTransformer().transformRequestIn(
      structuredClone(request),
      zenResponses,
      makeContext()
    );
    assert.equal(await passed.config.__providerResponse.text(), event);

    (globalThis as any).fetch = async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode(
              'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
            ));
          },
        }),
        { headers: { "Content-Type": "text/event-stream" } }
      );
    const idle = new OpencodeHeadersTransformer();
    (idle as any).streamIdleTimeoutMs = 25;
    const idleResult = await idle.transformRequestIn(
      structuredClone(request), zenResponses, makeContext()
    );
    await assert.rejects(
      () => idleResult.config.__providerResponse.text(),
      /Zen stream idle/
    );
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

async function main() {
  const originalFetch = (globalThis as any).fetch;
  try {
    await injectsResponsesStubsWithClonedSchemas();
    await injectsChatShapedStubs();
    await aliasesCursorMockReadFile();
    await aliasesDevinMockExec();
    await mockToolsSurviveResponsesPipeline();
    await replayKeepsClientToolNamesAndResults();
    await aliasesNativeShellVariants();
    await aliasesCodeOnlyHarnesses();
    await createsStubsWhenToolsAbsent();
    await skipsNativeStubsAndPaidModels();
    await rewritesResponsesStubCalls();
    await rewritesChatStubCalls();
    await aliasesSurviveFreshResponseContext();
    await stampsDetailedSummaryOnlyWhenReasoning();
    await cacheKeyEqualsSessionHeader();
    await sessionAffinitySurvivesToolHistory();
    await leavesNullAliasesAndForeignToolsAlone();
    await restoresJsonForNonStreamingResponsesClient();
    await terminalCancelCannotBlockJsonResponse();
    await bufferedTerminalSurvivesDebugAndLatencyTaps();
    await restoresJsonForNonStreamingChatClient();
    await firstZenEventMustComplete();
    console.log("opencode-gate-stubs: PASS");
  } finally {
    (globalThis as any).fetch = originalFetch;
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

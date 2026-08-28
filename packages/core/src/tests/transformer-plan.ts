import assert from "node:assert/strict";
import {
  compileTransformerPlan,
  cancelReplacedProviderResponse,
  isExactProtocolResponsePlan,
} from "../utils/transformer-plan";
import { OpencodeHeadersTransformer } from "../transformer/opencode-headers.transformer";
import { ReasoningTransformer } from "../transformer/reasoning.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { AntigravityAuthTransformer } from "../transformer/antigravity-auth.transformer";
import { CursorSdkTransformer } from "../transformer/cursor-sdk.transformer";

async function planDedupesAndOrdersTransportLast() {
  const openai = new OpenAITransformer();
  const reasoning = new ReasoningTransformer();
  const opencode = new OpencodeHeadersTransformer();

  const plan = compileTransformerPlan(
    [openai, opencode],
    [reasoning, opencode]
  );

  assert.equal(plan.request.length, 3);
  assert.equal(plan.request[0]?.name, "OpenAI");
  assert.equal(plan.request[1]?.name, "reasoning");
  assert.equal(plan.request[2]?.name, "opencode-headers");
  assert.equal(plan.transportOwner?.name, "opencode-headers");
  assert.deepEqual(
    plan.response.map((t) => t.name),
    ["opencode-headers", "reasoning", "OpenAI"]
  );
}

async function planRejectsMultipleTransportOwners() {
  assert.throws(
    () =>
      compileTransformerPlan(
        [new OpencodeHeadersTransformer()],
        [new AntigravityAuthTransformer()]
      ),
    /multiple transport owners/
  );
  assert.throws(
    () =>
      compileTransformerPlan(
        [new CursorSdkTransformer(), new OpencodeHeadersTransformer()],
        []
      ),
    /multiple transport owners/
  );
}

async function exactProtocolResponsePlansCoverEveryEndpoint() {
  for (const name of ["Anthropic", "OpenAI", "openai-responses"]) {
    const endpoint = { name };
    const plan = compileTransformerPlan(
      [endpoint, { name: "auth-or-transport-middleware" }],
      []
    );
    assert.equal(isExactProtocolResponsePlan(plan, endpoint, name), true);
    assert.equal(
      isExactProtocolResponsePlan(plan, endpoint, "different-protocol"),
      false
    );
  }
}

async function oneFetchForOpenCodeDeepSeekChain() {
  const calls: Array<{ body: any }> = [];
  (globalThis as any).fetch = async (_url: any, init: any) => {
    calls.push({ body: JSON.parse(String(init?.body || "{}")) });
    return new Response(
      JSON.stringify({
        id: "ok",
        choices: [{ message: { role: "assistant", content: "hi" } }],
      }),
      { status: 200, headers: { "Content-Type": "application/json" } }
    );
  };

  const openai = new OpenAITransformer();
  const reasoning = new ReasoningTransformer();
  const opencode = new OpencodeHeadersTransformer();
  const plan = compileTransformerPlan(
    [openai, opencode],
    [reasoning, opencode]
  );

  const provider = {
    name: "opencode",
    apiKey: "test-key",
    baseUrl: "https://opencode.ai/zen/v1/chat/completions",
    models: ["deepseek-v4-flash-free"],
  };
  const context = {
    req: {
      sessionId: "conv-plan-1",
      id: "req-1",
      log: { warn() {}, info() {}, debug() {} },
      server: { configService: { getHttpsProxy: () => undefined } },
    },
  };

  let requestBody: any = {
    model: "deepseek-v4-flash-free",
    messages: [
      {
        role: "assistant",
        content: "prior",
        reasoning_content: "hidden thought",
      },
      { role: "user", content: "hi" },
    ],
  };
  let config: any = {};

  for (const transformer of plan.request) {
    if (typeof transformer.transformRequestIn !== "function") continue;
    const transformIn = await transformer.transformRequestIn(
      requestBody,
      provider,
      context
    );
    if (transformIn?.body) {
      requestBody = transformIn.body;
      config = { ...config, ...(transformIn.config || {}) };
    } else if (transformIn) {
      requestBody = transformIn;
    }
  }

  assert.equal(calls.length, 1, "exactly one upstream fetch");
  assert.ok(config.__providerResponse);
  // Reasoning must run before the sole transport fetch so DeepSeek thinking
  // flags reach the wire.
  assert.equal(calls[0].body.enable_thinking, true);

  // Response plan installs Zen inspector once.
  const responseOutOwners = plan.response.filter(
    (t) => t.name === "opencode-headers"
  );
  assert.equal(responseOutOwners.length, 1);
}

async function cancelReplacedResponseBody() {
  let cancelled = false;
  const stream = new ReadableStream({
    start(controller) {
      controller.enqueue(new TextEncoder().encode("data: hi\n\n"));
    },
    cancel() {
      cancelled = true;
    },
  });
  const previous = new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
  const next = new Response("ok", { status: 200 });
  cancelReplacedProviderResponse(previous, next);
  // Give the cancel microtask a turn.
  await Promise.resolve();
  assert.equal(cancelled, true);
}

async function zenInspectorForwardsBytesUnchanged() {
  const encoder = new TextEncoder();
  const chunkA = encoder.encode('data: {"choices":[{"delta":{"content":"a"}}]}\n\n');
  const chunkB = encoder.encode('data: {"choices":[{"delta":{"content":"b"}}]}\n\n');

  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(chunkA);
      controller.enqueue(chunkB);
      controller.close();
    },
  });

  const transformer = new OpencodeHeadersTransformer();
  const wrapped = await transformer.transformResponseOut(
    new Response(upstream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    })
  );

  const reader = wrapped.body!.getReader();
  const out: Uint8Array[] = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    out.push(value!);
  }

  assert.equal(out.length, 2);
  assert.deepEqual(Buffer.from(out[0]), Buffer.from(chunkA));
  assert.deepEqual(Buffer.from(out[1]), Buffer.from(chunkB));
}

async function main() {
  await planDedupesAndOrdersTransportLast();
  await planRejectsMultipleTransportOwners();
  await exactProtocolResponsePlansCoverEveryEndpoint();
  await oneFetchForOpenCodeDeepSeekChain();
  await cancelReplacedResponseBody();
  await zenInspectorForwardsBytesUnchanged();
  console.log("transformer-plan: ok");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

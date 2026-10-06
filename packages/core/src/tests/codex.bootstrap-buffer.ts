/**
 * Codex stream bootstrap buffering: handshake hold, in-stream overload
 * detection, budget/timeout release, and byte-exact replay.
 */
import assert from "node:assert/strict";
import {
  bufferCodexBootstrapStream,
  classifyCodexError,
} from "../utils/codex-bootstrap";
import {
  CodexTransformer,
  type CodexTransformerOptions,
} from "../transformer/codex.transformer";
import { ProviderService } from "../services/provider";
import { TransformerService } from "../services/transformer";
import {
  decideScopedError,
  normalizeRequestScopedErrorRules,
} from "../utils/request-scoped-errors";
import { sanitizeErrorForLog } from "../utils/redact";

process.exitCode = 1;

const encoder = new TextEncoder();

function sseResponse(lines: string[]): Response {
  return new Response(encoder.encode(lines.join("\n")), {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function created(seq = 0): string {
  return `data: {"type":"response.created","sequence_number":${seq},"response":{"id":"resp_1"}}`;
}

function inProgress(seq = 1): string {
  return `data: {"type":"response.in_progress","sequence_number":${seq},"response":{"id":"resp_1"}}`;
}

function keepaliveComment(): string {
  return ": keepalive";
}

function itemAdded(seq = 2): string {
  return `data: {"type":"response.output_item.added","sequence_number":${seq},"output_index":0,"item":{"type":"message","id":"msg_1"}}`;
}

function textDelta(text: string, seq = 3): string {
  return `data: {"type":"response.output_text.delta","sequence_number":${seq},"item_id":"msg_1","delta":${JSON.stringify(text)}}`;
}

/**
 * A capacity failure frame. The ChatGPT plan limit is an error `type`
 * (Codex reads `UsageErrorResponse.error.type`); the rest are codes.
 */
function overloadFrame(kind = "server_is_overloaded"): string {
  const field = kind === "usage_limit_reached" ? "type" : "code";
  return `data: {"type":"response.failed","sequence_number":9,"response":{"status":"failed","error":{${JSON.stringify(field)}:${JSON.stringify(kind)},"message":"upstream ${kind}, try again"}}}`;
}

/** Codex CLI's own classification: exact codes (and quota types), no text. */
async function testCodexErrorClassification() {
  assert.equal(classifyCodexError({ code: "server_is_overloaded" }), "overload");
  for (const code of ["rate_limit_exceeded", "slow_down", "flex_unavailable"]) {
    assert.equal(classifyCodexError({ code }), "rate_limit", code);
  }
  for (const code of [
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
    "organization_usage_limit_exceeded",
    "usage_not_included",
  ]) {
    assert.equal(classifyCodexError({ code }), "quota", code);
  }
  for (const type of ["usage_limit_reached", "usage_not_included", "insufficient_quota"]) {
    assert.equal(classifyCodexError({ type }), "quota", type);
  }
  for (const code of [
    "context_length_exceeded",
    "invalid_prompt",
    "cyber_policy",
    "bio_policy",
    "misalignment_policy_violation",
    "some_new_code",
  ]) {
    assert.equal(classifyCodexError({ code, message: "server overloaded, try again" }), undefined, code);
  }
  // Message text alone never classifies.
  assert.equal(classifyCodexError({ message: "Rate limit reached, try again" }), undefined);
}

/**
 * Request failures pass through even when their message says "try again":
 * the next model would fail the same way, and the client keeps the real code.
 */
async function testRequestFailuresPassThrough() {
  for (const error of [
    {
      code: "context_length_exceeded",
      message: "Your input exceeds the context window of this model. Please adjust your input and try again.",
    },
    { code: "invalid_prompt", message: "Invalid prompt, try again." },
    { message: "The server is overloaded, try again later." },
  ]) {
    const lines = [
      created(),
      "",
      `data: ${JSON.stringify({ type: "response.failed", response: { status: "failed", error } })}`,
      "",
    ];
    const out = await bufferCodexBootstrapStream(sseResponse(lines), {});
    assert.equal(out.overloaded, false, JSON.stringify(error));
    assert.equal(await out.response.text(), lines.join("\n"));
    const passed = await new CodexTransformer({ streamBootstrapBuffering: true })
      .transformResponseOut(sseResponse(lines));
    assert.equal(await passed.text(), lines.join("\n"));
  }
}

/** Rate limits fail over as 429 and carry Codex's retry advice. */
async function testRateLimitCarriesRetryAfter() {
  const cases: Array<[Record<string, unknown>, string | undefined]> = [
    [{ code: "rate_limit_exceeded", message: "Rate limit reached. Please try again in 1.5s." }, "2"],
    [{ code: "slow_down", message: "Temporary limit. try again in 250ms" }, "1"],
    [{ code: "rate_limit_exceeded", message: "later", headers: { "Retry-After": "7" } }, "7"],
    [{ code: "flex_unavailable", message: "Flex capacity unavailable" }, undefined],
  ];
  for (const [error, retryAfter] of cases) {
    await assert.rejects(
      new CodexTransformer({ streamBootstrapBuffering: true }).transformResponseOut(
        sseResponse([
          created(),
          "",
          `data: ${JSON.stringify({ type: "response.failed", response: { status: "failed", error } })}`,
          "",
        ])
      ),
      (caught: any) =>
        caught.statusCode === 429 &&
        caught.code === "rate_limit_exceeded" &&
        caught.headers?.["Retry-After"] === retryAfter
    );
  }
  // An `error` event carries the flex code at top level or under `error`.
  await assert.rejects(
    new CodexTransformer({ streamBootstrapBuffering: true }).transformResponseOut(
      sseResponse([created(), "", `data: ${JSON.stringify({ type: "error", code: "flex_unavailable" })}`, ""])
    ),
    (caught: any) => caught.statusCode === 429
  );
}

async function testOverloadBeforeGenerationIsDetected() {
  const res = sseResponse([
    created(),
    "",
    inProgress(),
    "",
    keepaliveComment(),
    "",
    overloadFrame(),
    "",
  ]);
  const out = await bufferCodexBootstrapStream(res, {});
  assert.equal(out.overloaded, true);
  assert.equal(out.overloadKind, "overload");
  assert.equal(out.releasedBy, "overload");
  assert.ok((out.overloadText || "").includes("server_is_overloaded"));
  assert.ok(out.heldFrames >= 3);
}

async function testQuotaOverloadKind() {
  const res = sseResponse([created(), "", overloadFrame("usage_limit_reached"), ""]);
  const out = await bufferCodexBootstrapStream(res, {});
  assert.equal(out.overloaded, true);
  assert.equal(out.overloadKind, "quota");
}

async function testGenerationReleasesWithExactReplay() {
  const lines = [
    created(),
    "",
    inProgress(),
    "",
    itemAdded(),
    "",
    textDelta("Hello"),
    "",
    textDelta(" world"),
    "",
  ];
  const raw = lines.join("\n");
  const out = await bufferCodexBootstrapStream(sseResponse(lines), {});
  assert.equal(out.overloaded, false);
  assert.equal(out.releasedBy, "generated");
  const replayed = await out.response.text();
  assert.equal(replayed, raw);
  assert.ok(out.response.headers.get("Content-Type")?.includes("text/event-stream"));
}

async function testFrameBudgetReleases() {
  const lines = [created(), ""];
  for (let i = 0; i < 10; i += 1) {
    lines.push(itemAdded(10 + i), "");
  }
  const out = await bufferCodexBootstrapStream(sseResponse(lines), {
    maxFrames: 3,
  });
  assert.equal(out.overloaded, false);
  assert.equal(out.releasedBy, "budget");
  const replayed = await out.response.text();
  assert.equal(replayed, lines.join("\n"));
}

async function testTransformerThrowsFallbackEligibleOnOverload() {
  const tf = new CodexTransformer({ streamBootstrapBuffering: true });
  const res = sseResponse([created(), "", overloadFrame(), ""]);
  let caught: any;
  try {
    await tf.transformResponseOut(res, {});
  } catch (error) {
    caught = error;
  }
  assert.ok(caught, "expected overload to throw");
  assert.equal(caught.statusCode, 503);
  assert.equal(caught.code, "server_overloaded");
}

async function testTransformerThrows429OnQuota() {
  const tf = new CodexTransformer({ streamBootstrapBuffering: true });
  const res = sseResponse([
    created(),
    "",
    overloadFrame("usage_limit_reached"),
    "",
  ]);
  let caught: any;
  try {
    await tf.transformResponseOut(res, {});
  } catch (error) {
    caught = error;
  }
  assert.ok(caught, "expected quota exhaustion to throw");
  assert.equal(caught.statusCode, 429);
}

async function testTransformerDisabledPassesStreamThrough() {
  const tf = new CodexTransformer();
  const lines = [created(), "", textDelta("hi"), ""];
  const out = await tf.transformResponseOut(sseResponse(lines), {});
  assert.ok(out.ok);
  const text = await out.text();
  assert.ok(text.includes("response.created"));
  assert.ok(text.includes("response.output_text.delta"));
}

async function testNonSseUntouchedByBootstrap() {
  // JSON bodies skip bootstrap but still take the normal Codex transport path
  // (non-streaming intent unknown here defaults to SSE conversion).
  const tf = new CodexTransformer({ streamBootstrapBuffering: true });
  const out = await tf.transformResponseOut(
    new Response(JSON.stringify({ object: "response", output: [] }), {
      headers: { "Content-Type": "application/json" },
    }),
    {}
  );
  const text = await out.text();
  assert.ok(text.includes("response.completed"));
}

function chunkedResponse(chunks: string[], delayMs = 0): Response {
  let index = 0;
  return new Response(
    new ReadableStream<Uint8Array>({
      async pull(controller) {
        if (index === chunks.length) {
          controller.close();
          return;
        }
        if (delayMs) await new Promise((resolve) => setTimeout(resolve, delayMs));
        controller.enqueue(encoder.encode(chunks[index++]));
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } }
  );
}

async function testProviderOptionRegistration() {
  const errors: unknown[] = [];
  const logger = {
    info() {},
    debug() {},
    warn() {},
    error(error: unknown) { errors.push(error); },
  };
  const providers = [
    {
      name: "plain",
      api_base_url: "https://example.com",
      api_key: "dummy",
      models: ["gpt-5"],
      transformer: { use: ["openai-responses", "codex"] },
    },
    {
      name: "configured",
      api_base_url: "https://example.com",
      api_key: "dummy",
      models: ["gpt-5"],
      transformer: {
        use: ["openai-responses", ["codex", {
          streamBootstrapBuffering: true,
          streamBootstrapMaxFrames: 1,
        }]],
      },
    },
  ];
  const config = {
    get(key: string, fallback?: unknown) {
      return key === "providers" ? providers : fallback;
    },
  };
  const service = new TransformerService(config as any, logger);
  await service.initialize();
  assert.equal(service.getTransformer("codex"), CodexTransformer);
  const providerService = new ProviderService(config as any, service, logger);
  assert.deepEqual(errors, []);
  assert.equal(providerService.getProviders().length, 2);
  for (const name of ["plain", "configured"]) {
    const provider = providerService.getProvider(name)!;
    const codex = provider.transformer?.use?.find((tf) => tf.name === "codex");
    assert.ok(codex instanceof CodexTransformer);
    const response = await codex.transformResponseOut(
      sseResponse([created(), "", overloadFrame(), ""])
    );
    assert.equal(response.status, 200);
    assert.ok((await response.text()).includes("response.failed"));
  }
}

async function testHealthyEventsDoNotTriggerOverload() {
  const cases = [
    [created(), "", textDelta("Please try again; quota_exceeded is an error code."), ""],
    [
      `data: ${JSON.stringify({
        type: "response.created",
        response: { id: "resp_1", instructions: "Explain server_is_overloaded" },
      })}`,
      "",
      textDelta("Hello"),
      "",
    ],
    [
      created(),
      "",
      `data: ${JSON.stringify({
        type: "response.output_item.added",
        item: { type: "function_call", name: "quota_exceeded" },
      })}`,
      "",
      `data: ${JSON.stringify({
        type: "response.function_call_arguments.delta",
        delta: '{"message":"try again"}',
      })}`,
      "",
    ],
    [created(), "", ': keepalive server_is_overloaded'],
    [created(), "", textDelta("server_is_overloaded")],
  ];
  for (const lines of cases) {
    const out = await new CodexTransformer({ streamBootstrapBuffering: true })
      .transformResponseOut(sseResponse(lines));
    assert.equal(await out.text(), lines.join("\n"));
  }
}

async function testStructuredErrorEvents() {
  for (const event of [
    { type: "error", code: "server_is_overloaded", message: "busy" },
    { type: "error", error: { type: "insufficient_quota", message: "exhausted" } },
  ]) {
    const expectedStatus = "error" in event ? 429 : 503;
    await assert.rejects(
      new CodexTransformer({ streamBootstrapBuffering: true }).transformResponseOut(
        sseResponse([created(), "", `data: ${JSON.stringify(event)}`, ""])
      ),
      (error: any) => error.statusCode === expectedStatus
    );
  }
}

async function testChunkBoundariesAndTrailingErrors() {
  const raw = [created(), "", textDelta("Hello"), "", textDelta(" world"), ""].join("\n");
  const out = await bufferCodexBootstrapStream(
    chunkedResponse([raw.slice(0, 11), raw.slice(11, 97), raw.slice(97)])
  );
  assert.equal(out.overloaded, false);
  assert.equal(await out.response.text(), raw);

  const unicodeRaw = [created(), "", textDelta("Hello 世界👋; try again"), ""].join("\n");
  const bytes = encoder.encode(unicodeRaw);
  for (let cut = 1; cut < bytes.length; cut += 1) {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes.slice(0, cut));
        controller.enqueue(bytes.slice(cut));
        controller.close();
      },
    });
    const replay = await bufferCodexBootstrapStream(new Response(body));
    assert.equal(replay.overloaded, false);
    assert.equal(await replay.response.text(), unicodeRaw);
    assert.equal(body.locked, false);
  }

  for (const trailingNewline of ["", "\n\n"]) {
    const errorRaw = [created(), "", overloadFrame("usage_limit_reached")].join("\n") + trailingNewline;
    const result = await bufferCodexBootstrapStream(
      chunkedResponse([errorRaw.slice(0, 19), errorRaw.slice(19, -7), errorRaw.slice(-7)])
    );
    assert.equal(result.overloaded, true);
    assert.equal(result.overloadKind, "quota");
    assert.equal(await result.response.text(), errorRaw);
  }
}

async function testStructuredErrorClassificationProvenance() {
  const marker = "server_is_overloaded";
  const secret = "private-bootstrap-token";
  const event = {
    type: "response.failed",
    response: {
      id: "resp_1",
      metadata: { padding: "x".repeat(3000) },
      error: { code: marker, message: `busy token=${secret}` },
    },
  };
  await assert.rejects(
    new CodexTransformer({ streamBootstrapBuffering: true }).transformResponseOut(
      sseResponse([created(), "", `data: ${JSON.stringify(event)}`, ""])
    ),
    (error: any) => {
      assert.equal(error.statusCode, 503);
      assert.equal(error.message.includes(marker), false);
      const rules = normalizeRequestScopedErrorRules([
        { status: 503, match: [marker], action: "continue" },
      ]);
      assert.equal(decideScopedError(error, rules, []).kind, "continue");
      for (const serialized of [JSON.stringify(error), JSON.stringify(sanitizeErrorForLog(error))]) {
        assert.equal(serialized.includes(marker), false);
        assert.equal(serialized.includes(secret), false);
      }
      return true;
    }
  );
}

async function testPublicBudgetOptions() {
  const options: CodexTransformerOptions[] = [
    { streamBootstrapMaxFrames: 1, maxFrames: 48 },
    { streamBootstrapMaxBytes: 1, maxBytes: 1024 * 1024 },
    { streamBootstrapTimeoutMs: 5, timeoutMs: 0 },
    { maxFrames: 1 },
    { maxBytes: 1 },
    { timeoutMs: 5 },
  ];
  for (const option of options) {
    const chunks = [created() + "\n\n", overloadFrame() + "\n\n"];
    const response = await new CodexTransformer({
      streamBootstrapBuffering: true,
      ...option,
    }).transformResponseOut(chunkedResponse(chunks, 20));
    assert.equal(response.status, 200);
    assert.equal(await response.text(), chunks.join(""));
  }
}

async function testBootstrapReadFailureIsPropagated() {
  const expected = new Error("socket reset");
  const unhandled: unknown[] = [];
  const onUnhandled = (error: unknown) => { unhandled.push(error); };
  process.on("unhandledRejection", onUnhandled);
  try {
    let first = true;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (first) {
          first = false;
          controller.enqueue(encoder.encode(created() + "\n\n"));
        } else {
          controller.error(expected);
        }
      },
    });
    await assert.rejects(
      new CodexTransformer({ streamBootstrapBuffering: true }).transformResponseOut(
        new Response(body, { headers: { "Content-Type": "text/event-stream" } })
      ),
      (error: unknown) => error === expected
    );
    await new Promise((resolve) => setImmediate(resolve));
    assert.deepEqual(unhandled, []);
    assert.equal(body.locked, false);
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }
}

async function testReplayReadFailureIsPropagated() {
  const expected = new Error("socket reset after generation");
  let first = true;
  const body = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (first) {
        first = false;
        controller.enqueue(encoder.encode(textDelta("Hello") + "\n\n"));
      } else {
        controller.error(expected);
      }
    },
  });
  const replay = await bufferCodexBootstrapStream(
    new Response(body, { headers: { "Content-Type": "text/event-stream" } })
  );
  assert.equal(replay.releasedBy, "generated");
  await assert.rejects(replay.response.text(), (error: unknown) => error === expected);
  assert.equal(body.locked, false);
}

async function testOverloadCancelsUpstream() {
  for (const marker of ["server_is_overloaded", "usage_limit_reached"]) {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(overloadFrame(marker) + "\n\n"));
      },
      cancel() {
        cancelled = true;
        return Promise.reject(new Error("cleanup failed"));
      },
    });
    await assert.rejects(
      new CodexTransformer({ streamBootstrapBuffering: true }).transformResponseOut(
        new Response(body, { headers: { "Content-Type": "text/event-stream" } })
      ),
      (error: any) => error.statusCode === (marker === "usage_limit_reached" ? 429 : 503)
    );
    assert.equal(cancelled, true);
    assert.equal(body.locked, false);
  }
}

async function testReplayCancellationAndEofReleaseReader() {
  const reason = new Error("client disconnected");
  let cancellationReason: unknown;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(textDelta("Hello") + "\n\n"));
    },
    cancel(value) { cancellationReason = value; },
  });
  const result = await bufferCodexBootstrapStream(
    new Response(body, { headers: { "Content-Type": "text/event-stream" } })
  );
  await result.response.body!.cancel(reason);
  assert.equal(cancellationReason, reason);
  assert.equal(body.locked, false);

  const ended = sseResponse([created(), ""]);
  const eof = await bufferCodexBootstrapStream(ended);
  assert.equal(ended.body!.locked, false);
  assert.equal(await eof.response.text(), [created(), ""].join("\n"));
}

async function main() {
  await testProviderOptionRegistration();
  await testHealthyEventsDoNotTriggerOverload();
  await testStructuredErrorEvents();
  await testChunkBoundariesAndTrailingErrors();
  await testStructuredErrorClassificationProvenance();
  await testPublicBudgetOptions();
  await testBootstrapReadFailureIsPropagated();
  await testReplayReadFailureIsPropagated();
  await testOverloadCancelsUpstream();
  await testReplayCancellationAndEofReleaseReader();
  await testCodexErrorClassification();
  await testRequestFailuresPassThrough();
  await testRateLimitCarriesRetryAfter();
  await testOverloadBeforeGenerationIsDetected();
  await testQuotaOverloadKind();
  await testGenerationReleasesWithExactReplay();
  await testFrameBudgetReleases();
  await testTransformerThrowsFallbackEligibleOnOverload();
  await testTransformerThrows429OnQuota();
  await testTransformerDisabledPassesStreamThrough();
  await testNonSseUntouchedByBootstrap();
  console.log("codex.bootstrap-buffer: PASS");
}

main().then(() => { process.exitCode = 0; }).catch((err) => {
  console.error(err);
  process.exit(1);
});

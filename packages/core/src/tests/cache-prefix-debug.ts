/**
 * Per-conversation debug diffs of cache-relevant outbound fields.
 * Logs never include message text — only paths, hashes, ids, and breakpoint moves.
 */
import assert from "node:assert/strict";
import {
  __resetCachePrefixSnapshotsForTests,
  attributeDivergenceStage,
  diffCachePrefixSnapshots,
  rememberAndDiffOutboundCachePrefix,
  snapshotOutboundCachePrefix,
} from "../utils/cache-prefix-debug";
import {
  logOutboundCacheStructure,
  tapUpstreamSSEDebug,
} from "../utils/sse-debug-tap";

function anthropicBody(extraUser?: string) {
  const messages: any[] = [
    { role: "user", content: "hi" },
    {
      role: "assistant",
      content: [
        { type: "thinking", thinking: "plan first" },
        { type: "text", text: "ok" },
        { type: "tool_use", id: "call_1", name: "Read", input: {} },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "tool_result",
          tool_use_id: "call_1",
          content: "ok",
          cache_control: { type: "ephemeral" },
        },
      ],
    },
  ];
  if (extraUser) messages.push({ role: "user", content: extraUser });
  return {
    model: "claude-sonnet-4-20250514",
    system: [
      { type: "text", text: "stable system", cache_control: { type: "ephemeral" } },
    ],
    messages,
    tools: [
      {
        name: "Read",
        input_schema: { type: "object" },
        cache_control: { type: "ephemeral" },
      },
    ],
  };
}

function testHealthyAppendKeepsPrefix() {
  __resetCachePrefixSnapshotsForTests();
  const first = rememberAndDiffOutboundCachePrefix("sess-1", anthropicBody());
  assert.equal(first?.firstTurn, true);
  assert.equal(first?.prefixIntact, true);
  assert.equal(first?.change, "none");

  const second = rememberAndDiffOutboundCachePrefix(
    "sess-1",
    anthropicBody("continue")
  );
  assert.equal(second?.firstTurn, false);
  assert.equal(second?.prefixIntact, true);
  assert.equal(second?.change, "appended");
  assert.deepEqual(second?.appendedPaths, ["messages[3]"]);
  assert.ok((second?.unchangedPrefixCount || 0) >= 3);
}

function testRewrittenHistoryReportsFirstDivergence() {
  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("sess-2", anthropicBody());
  const mutated = anthropicBody();
  mutated.messages[0] = { role: "user", content: "hi (rewritten)" };
  mutated.messages.push({ role: "user", content: "continue" });
  const diff = rememberAndDiffOutboundCachePrefix("sess-2", mutated);
  assert.equal(diff?.prefixIntact, false);
  assert.equal(diff?.change, "modified");
  assert.equal(diff?.firstDivergencePath, "messages[0]");
  assert.equal(diff?.firstDivergence?.previous?.role, "user");
  assert.equal(diff?.firstDivergence?.current?.role, "user");
  assert.equal(
    diff?.firstDivergence?.previous?.path,
    diff?.firstDivergence?.current?.path
  );
}

function testReasoningIdChangeIsACacheBreak() {
  __resetCachePrefixSnapshotsForTests();
  const responses = {
    model: "grok-4.6",
    prompt_cache_key: "ccr_same",
    input: [
      { role: "user", content: "hi" },
      {
        type: "reasoning",
        id: "rs_aaa",
        summary: [{ type: "summary_text", text: "plan first" }],
      },
      { type: "function_call", name: "Read", call_id: "call_1", arguments: "{}" },
    ],
  };
  rememberAndDiffOutboundCachePrefix("sess-3", responses);
  const next = structuredClone(responses);
  next.input.push({ role: "user", content: "continue" } as any);
  next.input[1].id = "rs_bbb";
  const diff = rememberAndDiffOutboundCachePrefix("sess-3", next);
  assert.equal(diff?.prefixIntact, false);
  assert.equal(diff?.firstDivergencePath, "input[1]");
  assert.equal(diff?.firstDivergence?.previous?.reasoningId, "rs_aaa");
  assert.equal(diff?.firstDivergence?.current?.reasoningId, "rs_bbb");
}

function testPromptCacheKeyAndAffinity() {
  __resetCachePrefixSnapshotsForTests();
  const body = {
    model: "gpt-5.6-sol",
    prompt_cache_key: "ccr_one",
    input: [{ role: "user", content: "hi" }],
  };
  rememberAndDiffOutboundCachePrefix("sess-4", body, {
    sessionId: "ccr_one",
    threadId: "ccr_one",
  });
  const diff = rememberAndDiffOutboundCachePrefix(
    "sess-4",
    { ...body, prompt_cache_key: "ccr_two" },
    { sessionId: "ccr_two", threadId: "ccr_two" }
  );
  assert.equal(diff?.prompt_cache_keyChanged, true);
  assert.equal(diff?.affinityChanged, true);
  assert.equal(diff?.prefixIntact, false);
  assert.equal(diff?.firstDivergencePath, "prompt_cache_key");
  assert.deepEqual(diff?.prompt_cache_key, {
    previous: "ccr_one",
    current: "ccr_two",
  });
}

function testConversationsAreIsolated() {
  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("a", anthropicBody());
  const other = rememberAndDiffOutboundCachePrefix("b", anthropicBody());
  assert.equal(other?.firstTurn, true);
  const again = rememberAndDiffOutboundCachePrefix("a", anthropicBody("next"));
  assert.equal(again?.firstTurn, false);
  assert.equal(again?.prefixIntact, true);
}

function testSnapshotOmitsMessageText() {
  const snap = snapshotOutboundCachePrefix(anthropicBody());
  assert.ok(snap);
  assert.ok(snap!.segments.some((s) => s.path === "system"));
  assert.ok(snap!.segments.some((s) => s.path === "tools"));
  assert.ok(JSON.stringify(snap).includes("plan first") === false);
  assert.ok(JSON.stringify(snap).includes("stable system") === false);
  assert.ok(snap!.breakpointPaths.includes("system[1]") || snap!.breakpointPaths.some((p) => p.startsWith("system")));
}

function testLogEmitsSearchableType() {
  __resetCachePrefixSnapshotsForTests();
  const records: any[] = [];
  const logger = {
    level: "debug",
    debug(payload: any) {
      records.push(payload);
    },
  };
  logOutboundCacheStructure(anthropicBody(), {
    logger,
    reqId: "r1",
    provider: "claude",
    conversationId: "sess-log",
  });
  logOutboundCacheStructure(anthropicBody("continue"), {
    logger,
    reqId: "r2",
    provider: "claude",
    conversationId: "sess-log",
  });
  const diffs = records.filter((r) => r.type === "cache prefix diff");
  assert.equal(diffs.length, 2);
  assert.equal(diffs[0].firstTurn, true);
  assert.equal(diffs[1].change, "appended");
  assert.equal(diffs[1].prefixIntact, true);
  assert.ok(records.some((r) => r.type === "cache structure"));
}

function testDebugOffDoesNotStore() {
  __resetCachePrefixSnapshotsForTests();
  logOutboundCacheStructure(anthropicBody(), {
    logger: {
      level: "info",
      debug() {
        throw new Error("debug must stay off");
      },
    },
    conversationId: "sess-silent",
  });
  const first = rememberAndDiffOutboundCachePrefix(
    "sess-silent",
    anthropicBody()
  );
  assert.equal(first?.firstTurn, true);
}

function testDiffWithoutStore() {
  const a = snapshotOutboundCachePrefix(anthropicBody())!;
  const b = snapshotOutboundCachePrefix(anthropicBody("continue"))!;
  const diff = diffCachePrefixSnapshots("x", a, b);
  assert.equal(diff.change, "appended");
  assert.equal(diff.prefixIntact, true);
}

function testMissIsPricedInTokens() {
  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("sess-cost", anthropicBody());
  const mutated = anthropicBody("continue");
  mutated.messages[0] = { role: "user", content: "hi (rewritten)" };
  const diff = rememberAndDiffOutboundCachePrefix("sess-cost", mutated);
  assert.equal(diff?.change, "modified");
  assert.ok(
    (diff?.approxPrefixTokensLost || 0) > 0,
    "a broken prefix must report the tokens it costs"
  );

  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("sess-cheap", anthropicBody());
  const appended = rememberAndDiffOutboundCachePrefix(
    "sess-cheap",
    anthropicBody("continue")
  );
  assert.equal(appended?.approxPrefixTokensLost, 0);
  assert.ok((appended?.unchangedPrefixApproxTokens || 0) > 0);
  assert.equal(typeof appended?.msSinceLastTurn, "number");
}

function testRejectedRequestDoesNotBecomeBaseline() {
  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("sess-429", anthropicBody());
  // A body upstream rejected was never cached; it must not shift the baseline.
  rememberAndDiffOutboundCachePrefix(
    "sess-429",
    { ...anthropicBody("continue"), model: "claude-sonnet-4-20250514" },
    undefined,
    { commit: false }
  );
  const retry = rememberAndDiffOutboundCachePrefix(
    "sess-429",
    anthropicBody("continue")
  );
  assert.equal(retry?.change, "appended");
  assert.equal(retry?.prefixIntact, true);
}

function testFingerprintKeepsAnonymousConversationsApart() {
  __resetCachePrefixSnapshotsForTests();
  const a = anthropicBody();
  const b = anthropicBody();
  b.messages[0] = { role: "user", content: "a different conversation" };

  const firstA = rememberAndDiffOutboundCachePrefix(undefined, a);
  assert.equal(firstA?.conversationIdSource, "fingerprint");
  const firstB = rememberAndDiffOutboundCachePrefix(undefined, b);
  assert.equal(firstB?.firstTurn, true, "distinct openings must not collide");

  const secondA = rememberAndDiffOutboundCachePrefix(
    undefined,
    anthropicBody("continue")
  );
  assert.equal(secondA?.firstTurn, false);
  assert.equal(secondA?.prefixIntact, true);
}

function testSnapshotsAreKeyedPerDestination() {
  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("sess-route", anthropicBody(), undefined, {
    provider: "claude",
  });
  // A routed fallback is a real miss, not prefix corruption — report it cold.
  const fallback = rememberAndDiffOutboundCachePrefix(
    "sess-route",
    anthropicBody("continue"),
    undefined,
    { provider: "openrouter" }
  );
  assert.equal(fallback?.firstTurn, true);
}

function testStabilityFlagsNameTheFailure() {
  __resetCachePrefixSnapshotsForTests();
  rememberAndDiffOutboundCachePrefix("sess-flags", anthropicBody());
  const changedTools = anthropicBody("continue");
  changedTools.tools = [
    { name: "Read", input_schema: { type: "object" } },
    { name: "Write", input_schema: { type: "object" } },
  ] as any;
  const diff = rememberAndDiffOutboundCachePrefix("sess-flags", changedTools);
  assert.equal(diff?.toolsHashChanged, true);
  assert.equal(diff?.systemHashChanged, false);
  assert.equal(diff?.breakpointsMoved, true);
  assert.equal(diff?.prefixIntact, false);
}

function testTimestampIdsAreFlagged() {
  __resetCachePrefixSnapshotsForTests();
  const diff = rememberAndDiffOutboundCachePrefix("sess-unstable", {
    model: "gpt-5.6-sol",
    prompt_cache_key: "ccr_x",
    input: [
      { role: "user", content: "hi" },
      { type: "reasoning", id: `rs_${Date.now()}`, summary: [] },
    ],
  });
  assert.equal(diff?.unstableIds?.length, 1);

  const stable = snapshotOutboundCachePrefix({
    input: [{ type: "reasoning", id: "rs_deadbeefdeadbeefdeadbeef", summary: [] }],
  });
  assert.deepEqual(stable?.unstableIds, []);
}

function testDivergenceStageAttribution() {
  const intact = { firstTurn: false, prefixIntact: true } as any;
  const broken = { firstTurn: false, prefixIntact: false } as any;
  assert.equal(attributeDivergenceStage(intact, intact), "none");
  // Client prefix held but the wire prefix moved: our transformers did it.
  assert.equal(attributeDivergenceStage(intact, broken), "wire");
  assert.equal(attributeDivergenceStage(broken, broken), "client");
  assert.equal(attributeDivergenceStage(intact, undefined), undefined);
}

function sseResponse(frames: unknown[]): Response {
  const body = frames
    .map((frame) => `event: message\ndata: ${JSON.stringify(frame)}\n\n`)
    .join("");
  return new Response(body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

async function outcomeFor(
  records: any[],
  logger: any,
  cacheDiff: any,
  frames: unknown[]
) {
  const tapped = await tapUpstreamSSEDebug(sseResponse(frames), {
    logger,
    reqId: "r",
    provider: "p",
    responseStatus: 200,
    cacheDiff,
  });
  await tapped.text();
  for (let i = 0; i < 50; i += 1) {
    const found = records.find((r) => r.type === "cache outcome");
    if (found) return found;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("cache outcome was never logged");
}

async function testCacheOutcomeJoinsPredictionWithUsage() {
  const logger = (records: any[]) => ({
    level: "debug",
    debug: (payload: any) => records.push(payload),
  });

  // Anthropic reports cached tokens outside input_tokens.
  let records: any[] = [];
  let outcome = await outcomeFor(
    records,
    logger(records),
    { firstTurn: false, prefixIntact: true, change: "appended", conversationId: "c", conversationIdSource: "session", approxPrefixTokensLost: 0 },
    [
      {
        type: "message_start",
        message: {
          usage: {
            input_tokens: 10,
            cache_read_input_tokens: 90,
            cache_creation_input_tokens: 0,
          },
        },
      },
      { type: "message_delta", usage: { output_tokens: 5 } },
    ]
  );
  assert.equal(outcome.verdict, "hit");
  assert.equal(outcome.promptTokens, 100);
  assert.equal(outcome.cachedTokens, 90);
  assert.equal(outcome.cacheHitRatio, 0.9);
  assert.equal(outcome.outputTokens, 5);

  // The row worth chasing: prefix held, provider cached nothing.
  records = [];
  outcome = await outcomeFor(
    records,
    logger(records),
    { firstTurn: false, prefixIntact: true, change: "appended", conversationId: "c", conversationIdSource: "session", approxPrefixTokensLost: 0 },
    [
      {
        type: "response.completed",
        response: {
          usage: {
            input_tokens: 100,
            input_tokens_details: { cached_tokens: 0 },
            output_tokens: 7,
          },
        },
      },
    ]
  );
  assert.equal(outcome.verdict, "unexpected-miss");
  assert.equal(outcome.promptTokens, 100);
  assert.equal(outcome.cacheHitRatio, 0);

  // A prefix we already know we broke should not read as a surprise.
  records = [];
  outcome = await outcomeFor(
    records,
    logger(records),
    {
      firstTurn: false,
      prefixIntact: false,
      change: "modified",
      conversationId: "c",
      conversationIdSource: "session",
      firstDivergencePath: "messages[0]",
      approxPrefixTokensLost: 4200,
    },
    [{ usage: { prompt_tokens: 100, prompt_tokens_details: { cached_tokens: 0 } } }]
  );
  assert.equal(outcome.verdict, "expected-miss");
  assert.equal(outcome.firstDivergencePath, "messages[0]");
  assert.equal(outcome.approxPrefixTokensLost, 4200);

  // Gemini folds cached tokens into the prompt total.
  records = [];
  outcome = await outcomeFor(records, logger(records), null, [
    {
      usageMetadata: {
        promptTokenCount: 200,
        cachedContentTokenCount: 50,
        candidatesTokenCount: 9,
      },
    },
  ]);
  assert.equal(outcome.promptTokens, 200);
  assert.equal(outcome.cachedTokens, 50);
  assert.equal(outcome.cacheHitRatio, 0.25);
}

async function testOutcomeStaysBehindDebug() {
  const tapped = await tapUpstreamSSEDebug(
    sseResponse([{ usage: { prompt_tokens: 1 } }]),
    {
      logger: {
        level: "info",
        debug() {
          throw new Error("debug must stay off");
        },
      },
    }
  );
  await tapped.text();
  await new Promise((resolve) => setTimeout(resolve, 20));
}

async function main() {
  testHealthyAppendKeepsPrefix();
  testRewrittenHistoryReportsFirstDivergence();
  testReasoningIdChangeIsACacheBreak();
  testPromptCacheKeyAndAffinity();
  testConversationsAreIsolated();
  testSnapshotOmitsMessageText();
  testLogEmitsSearchableType();
  testDebugOffDoesNotStore();
  testDiffWithoutStore();
  testMissIsPricedInTokens();
  testRejectedRequestDoesNotBecomeBaseline();
  testFingerprintKeepsAnonymousConversationsApart();
  testSnapshotsAreKeyedPerDestination();
  testStabilityFlagsNameTheFailure();
  testTimestampIdsAreFlagged();
  testDivergenceStageAttribution();
  await testCacheOutcomeJoinsPredictionWithUsage();
  await testOutcomeStaysBehindDebug();
  console.log("cache-prefix-debug: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

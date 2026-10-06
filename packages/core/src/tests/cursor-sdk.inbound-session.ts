import "./support/isolate-session-registry";
import assert from "node:assert/strict";
import { readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { resolveInternalCursorSession } from "../cursor-sdk/inbound-session";
import {
  getPersistedSession,
  resetSessionRegistryForTests,
} from "../session-registry";

function resolve(input: {
  protocol?: string;
  request: unknown;
  context?: any;
  model?: string;
  sourceSessionIdentity?: string;
  isActive?: (internalId: string) => boolean;
}) {
  return resolveInternalCursorSession({
    protocol: input.protocol,
    request: input.request,
    context: input.context || {},
    model: input.model || "composer-2",
    sourceSessionIdentity: input.sourceSessionIdentity,
    isActive: input.isActive,
  });
}

function registryFile(): string {
  return join(process.env.CCR_SESSION_REGISTRY_DIR!, "ccr-sessions.json");
}

/** Run `fn` with Date.now() shifted forward by `ms`. */
function later<T>(ms: number, fn: () => T): T {
  const realNow = Date.now;
  const base = realNow();
  Date.now = () => base + ms;
  try {
    return fn();
  } finally {
    Date.now = realNow;
  }
}

const shared = {
  model: "composer-2",
  messages: [
    { role: "user", content: "<datasources>\nazure-customer-1\n</datasources>" },
    { role: "user", content: "What did we spend on Azure in total?" },
  ],
};

const otherQuestion = {
  model: "composer-2",
  messages: [
    { role: "user", content: "<datasources>\nazure-customer-1\n</datasources>" },
    { role: "user", content: "How much VAT did we pay on Azure in June 2026?" },
  ],
};

const followUp = {
  model: "composer-2",
  messages: [
    ...shared.messages,
    { role: "assistant", content: "14,506,468.49 EUR" },
    { role: "user", content: "Break that down by service." },
  ],
};

function testAnonymousOpeningsStayApart() {
  const billed = resolve({
    protocol: "openai_chat_completions",
    request: shared,
  });
  const vat = resolve({
    protocol: "openai_chat_completions",
    request: otherQuestion,
  });
  assert.notEqual(billed.internalId, vat.internalId);
  const billedFollow = resolve({
    protocol: "openai_chat_completions",
    request: followUp,
  });
  assert.equal(billedFollow.internalId, billed.internalId);
}

function testAnonymousFollowUpDoesNotGlueLaterOpenings() {
  const first = resolve({
    protocol: "openai_chat_completions",
    request: {
      messages: [{ role: "user", content: "plan the migration" }],
    },
  });
  const postgres = resolve({
    protocol: "openai_chat_completions",
    request: {
      messages: [
        { role: "user", content: "plan the migration" },
        { role: "assistant", content: "ok" },
        { role: "user", content: "use postgres" },
      ],
    },
  });
  assert.equal(postgres.internalId, first.internalId);
  assert.notEqual(postgres.inboundKey, first.inboundKey);
  const mysql = resolve({
    protocol: "openai_chat_completions",
    request: {
      messages: [
        { role: "user", content: "plan the migration" },
        { role: "assistant", content: "ok" },
        { role: "user", content: "use mysql" },
      ],
    },
  });
  assert.notEqual(mysql.internalId, first.internalId);
  // The longer list keeps the session on its next turn.
  const postgresNext = resolve({
    protocol: "openai_chat_completions",
    request: {
      messages: [
        { role: "user", content: "plan the migration" },
        { role: "assistant", content: "ok" },
        { role: "user", content: "use postgres" },
        { role: "assistant", content: "done" },
        { role: "user", content: "add indexes" },
      ],
    },
  });
  assert.equal(postgresNext.internalId, first.internalId);
}

function testCursorHeaderIsIgnored() {
  const headersA = { "x-ccr-cursor-session": "client-a" };
  const headersB = { "x-ccr-cursor-session": "client-b" };
  const a = resolve({
    protocol: "openai_chat_completions",
    request: { model: "composer-2", messages: [{ role: "user", content: "same" }] },
    context: { req: { headers: headersA } },
  });
  const b = resolve({
    protocol: "openai_chat_completions",
    request: { model: "composer-2", messages: [{ role: "user", content: "same" }] },
    context: { req: { headers: headersB } },
  });
  assert.equal(a.internalId, b.internalId);
  assert.equal(headersA["x-ccr-cursor-session"], undefined);
  assert.equal(headersB["x-ccr-cursor-session"], undefined);
}

function testProtocolIdentities() {
  const anthropic = resolve({
    protocol: "anthropic_messages",
    request: shared,
    sourceSessionIdentity: JSON.stringify({ session_id: "claude-1" }),
  });
  const anthropicAgain = resolve({
    protocol: "anthropic_messages",
    request: otherQuestion,
    context: { req: { sessionId: "claude-1" } },
  });
  assert.equal(anthropic.internalId, anthropicAgain.internalId);

  const taskA = resolve({
    protocol: "anthropic_messages",
    request: { messages: [{ role: "user", content: "task alpha" }] },
    context: {
      protocolContext: { protocol: "anthropic_messages", sessionId: "parent" },
    },
  });
  const taskB = resolve({
    protocol: "anthropic_messages",
    request: { messages: [{ role: "user", content: "task beta" }] },
    context: { req: { sessionId: "parent" } },
  });
  const taskAFollow = resolve({
    protocol: "anthropic_messages",
    request: {
      messages: [
        { role: "user", content: "task alpha" },
        { role: "user", content: "continue alpha" },
      ],
    },
    context: { req: { sessionId: "parent" } },
  });
  assert.notEqual(taskA.internalId, taskB.internalId);
  assert.equal(taskA.internalId, taskAFollow.internalId);

  const openCode = resolve({
    protocol: "openai_chat_completions",
    request: shared,
    context: { req: { headers: { "x-opencode-session": "ses_child" } } },
  });
  const openCodeAgain = resolve({
    protocol: "openai_chat_completions",
    request: followUp,
    context: { req: { headers: { "x-opencode-session": "ses_child" } } },
  });
  assert.equal(openCode.internalId, openCodeAgain.internalId);
  assert.notEqual(openCode.internalId, anthropic.internalId);

  const responses = resolve({
    protocol: "openai_responses",
    request: {
      model: "composer-2",
      prompt_cache_key: "resp-1",
      input: [{ role: "user", content: "hi" }],
    },
  });
  const responsesAgain = resolve({
    protocol: "openai_responses",
    request: {
      model: "composer-2",
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
        { role: "user", content: "later" },
      ],
    },
    context: {
      req: { originalClientBody: { prompt_cache_key: "resp-1" } },
    },
  });
  assert.equal(responses.internalId, responsesAgain.internalId);

  // Responses precedence: a client session header wins over prompt_cache_key.
  const byHeader = resolve({
    protocol: "openai_responses",
    request: { prompt_cache_key: "resp-2", input: [{ role: "user", content: "hi" }] },
    context: { req: { headers: { "x-session-id": "hdr-2" } } },
  });
  const sameHeaderOtherKey = later(5_000, () =>
    resolve({
      protocol: "openai_responses",
      request: { prompt_cache_key: "resp-3", input: [{ role: "user", content: "hi" }] },
      context: { req: { headers: { "x-session-id": "hdr-2" } } },
    })
  );
  assert.equal(sameHeaderOtherKey.internalId, byHeader.internalId);
  assert.match(byHeader.inboundKey, /\0hdr-2\0/);

  const parentOnly = resolve({
    protocol: "openai_chat_completions",
    request: { messages: [{ role: "user", content: "same question" }] },
    context: { req: { headers: { "x-parent-session-id": "parent-a" } } },
  });
  const otherParent = resolve({
    protocol: "openai_chat_completions",
    request: { messages: [{ role: "user", content: "same question" }] },
    context: { req: { headers: { "x-parent-session-id": "parent-b" } } },
  });
  assert.equal(parentOnly.internalId, otherParent.internalId);
}

function testRestartKeepsTheBinding() {
  const first = resolve({
    protocol: "openai_chat_completions",
    request: { conversation_id: "chat-9", messages: [{ role: "user", content: "hi" }] },
  });
  resetSessionRegistryForTests();
  const second = resolve({
    protocol: "openai_chat_completions",
    request: {
      conversation_id: "chat-9",
      messages: [
        { role: "user", content: "hi" },
        { role: "assistant", content: "hello" },
        { role: "user", content: "later" },
      ],
    },
  });
  assert.equal(first.internalId, second.internalId);
}

/**
 * OpenCode sends its title request with the session's prompt_cache_key while
 * the main turn runs. One agent for both lets one supersede the other.
 */
function testSideCallUnderNativeIdStaysApart() {
  const context = { req: { originalClientBody: { prompt_cache_key: "ses_oc" } } };
  const main = resolve({
    protocol: "openai_responses",
    request: { messages: [{ role: "user", content: "Say hi" }] },
    context,
  });
  const title = resolve({
    protocol: "openai_responses",
    request: {
      messages: [
        { role: "system", content: "You are a title generator." },
        { role: "user", content: "Generate a title for this conversation:" },
        { role: "user", content: "Say hi" },
      ],
    },
    context,
  });
  const mainFollow = resolve({
    protocol: "openai_responses",
    request: {
      messages: [
        { role: "user", content: "Say hi" },
        { role: "assistant", content: "Hi." },
        { role: "user", content: "again" },
      ],
    },
    context,
  });
  assert.notEqual(main.internalId, title.internalId);
  assert.equal(mainFollow.internalId, main.internalId);
}

/** A stateless client repeating an opening must not see the earlier run. */
function testRepeatedOpeningGetsFreshAgent() {
  const opening = {
    messages: [{ role: "user", content: "Output a random 6-digit number." }],
  };
  const first = resolve({ protocol: "openai_chat_completions", request: opening });
  // A retry inside the replay window joins or replays the same turn.
  const retry = later(5_000, () =>
    resolve({ protocol: "openai_chat_completions", request: opening })
  );
  assert.equal(retry.internalId, first.internalId);
  // After the window the same opening is a new conversation.
  const repeat = later(61_000, () =>
    resolve({ protocol: "openai_chat_completions", request: opening })
  );
  assert.notEqual(repeat.internalId, first.internalId);

  // Once a binding progressed (tool loop), a repeat is new even within the window.
  const toolOpening = {
    messages: [{ role: "user", content: "Look up the EUR rate." }],
  };
  const toolFirst = resolve({
    protocol: "openai_chat_completions",
    request: toolOpening,
  });
  const toolResult = resolve({
    protocol: "openai_chat_completions",
    request: {
      messages: [
        ...toolOpening.messages,
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "lookup_rate", arguments: "{}" },
            },
          ],
        },
        { role: "tool", tool_call_id: "call_1", content: "1.0873" },
      ],
    },
  });
  assert.equal(toolResult.internalId, toolFirst.internalId);
  const toolRepeat = resolve({
    protocol: "openai_chat_completions",
    request: toolOpening,
  });
  assert.notEqual(toolRepeat.internalId, toolFirst.internalId);

  // Same rule under a native id: a rewind to the opening after progress.
  const context = { req: { headers: { "x-session-id": "native-7" } } };
  const nativeFirst = resolve({
    protocol: "openai_chat_completions",
    request: { messages: [{ role: "user", content: "start" }] },
    context,
  });
  const nativeFollow = resolve({
    protocol: "openai_chat_completions",
    request: {
      messages: [
        { role: "user", content: "start" },
        { role: "assistant", content: "ok" },
        { role: "user", content: "next" },
      ],
    },
    context,
  });
  assert.equal(nativeFollow.internalId, nativeFirst.internalId);
  const nativeRewind = resolve({
    protocol: "openai_chat_completions",
    request: { messages: [{ role: "user", content: "start" }] },
    context,
  });
  assert.notEqual(nativeRewind.internalId, nativeFirst.internalId);
}

function toolTurn(opening: any[], callId: string) {
  return {
    messages: [
      ...opening,
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: callId,
            type: "function",
            function: { name: "lookup", arguments: "{}" },
          },
        ],
      },
      { role: "tool", tool_call_id: callId, content: "result" },
    ],
  };
}

/**
 * A repeated opening must not take over an earlier conversation that is
 * still going: each keeps its own agent on later turns.
 */
function testRepeatedOpeningDoesNotHijackOngoingConversation() {
  for (const context of [
    {},
    // A raw metadata.user_id is a per-user id, shared across conversations.
    { req: { sessionId: "end-user-42" } },
  ]) {
    const opening = [{ role: "user", content: "Check the EUR rate again." }];
    const a = resolve({
      protocol: "anthropic_messages",
      request: { messages: opening },
      context,
    });
    const aTurn2 = resolve({
      protocol: "anthropic_messages",
      request: toolTurn(opening, "toolu_a1"),
      context,
    });
    assert.equal(aTurn2.internalId, a.internalId);
    const b = later(5_000, () =>
      resolve({
        protocol: "anthropic_messages",
        request: { messages: opening },
        context,
      })
    );
    assert.notEqual(b.internalId, a.internalId);
    const aTurn3 = resolve({
      protocol: "anthropic_messages",
      request: toolTurn(opening, "toolu_a1"),
      context,
    });
    assert.equal(aTurn3.internalId, a.internalId);
    const bTurn2 = resolve({
      protocol: "anthropic_messages",
      request: toolTurn(opening, "toolu_b1"),
      context,
    });
    assert.equal(bTurn2.internalId, b.internalId);
  }
}

/**
 * An identical opening that replaces an unclaimed one leaves the follow-up
 * ambiguous: it gets a fresh agent, never the other conversation's.
 */
function testContestedOpeningFollowUpGetsFreshAgent() {
  const opening = [{ role: "user", content: "Name a prime number." }];
  const a = resolve({
    protocol: "openai_chat_completions",
    request: { messages: opening },
  });
  const b = later(120_000, () =>
    resolve({ protocol: "openai_chat_completions", request: { messages: opening } })
  );
  assert.notEqual(b.internalId, a.internalId);
  const follow = later(130_000, () =>
    resolve({
      protocol: "openai_chat_completions",
      request: {
        messages: [
          ...opening,
          { role: "assistant", content: "7" },
          { role: "user", content: "Another one." },
        ],
      },
    })
  );
  assert.notEqual(follow.internalId, a.internalId);
  assert.notEqual(follow.internalId, b.internalId);
}

/**
 * The retry window follows the agent, not the first request: a retry of a
 * long opening turn that is still running (or just finished) joins it.
 */
function testOpeningRetryJoinsLongRunningTurn() {
  const opening = { messages: [{ role: "user", content: "Think for a while." }] };
  const first = resolve({ protocol: "openai_chat_completions", request: opening });
  const active = new Set([first.internalId]);
  const retry = later(10 * 60_000, () =>
    resolve({
      protocol: "openai_chat_completions",
      request: opening,
      isActive: (id) => active.has(id),
    })
  );
  assert.equal(retry.internalId, first.internalId);
  active.clear();
  const repeat = later(11 * 60_000, () =>
    resolve({
      protocol: "openai_chat_completions",
      request: opening,
      isActive: (id) => active.has(id),
    })
  );
  assert.notEqual(repeat.internalId, first.internalId);
}

/** Walk JSON string values (and keys). Skip numbers so `updatedAt` cannot collide. */
function jsonStrings(value: unknown, out: string[] = []): string[] {
  if (typeof value === "string") {
    out.push(value);
    return out;
  }
  if (Array.isArray(value)) {
    for (const item of value) jsonStrings(item, out);
    return out;
  }
  if (value && typeof value === "object") {
    for (const [key, nested] of Object.entries(value)) {
      out.push(key);
      jsonStrings(nested, out);
    }
  }
  return out;
}

/** Conversation text never reaches disk; unchanged turns do not rewrite it. */
function testRegistryStoresHashesAndSkipsUnchangedWrites() {
  const secret = "my secret quarterly numbers 4711";
  const opening = [{ role: "user", content: secret }];
  resolve({ protocol: "openai_chat_completions", request: { messages: opening } });
  const follow = toolTurn(opening, "call_secret");
  resolve({ protocol: "openai_chat_completions", request: follow });
  const raw = readFileSync(registryFile(), "utf-8");
  const persisted = JSON.parse(raw) as unknown;
  assert.equal(
    jsonStrings(persisted).some((value) => value.includes(secret)),
    false
  );
  const before = statSync(registryFile()).mtimeMs;
  const content = readFileSync(registryFile(), "utf-8");
  // Any rewrite inside the refresh interval would change updatedAt.
  for (let i = 1; i <= 5; i += 1) {
    later(i * 60_000, () =>
      resolve({ protocol: "openai_chat_completions", request: follow })
    );
  }
  assert.equal(statSync(registryFile()).mtimeMs, before);
  assert.equal(readFileSync(registryFile(), "utf-8"), content);
}

function testInheritedForksKeepTheirOwnConversation() {
  for (const protocol of ["anthropic_messages", "openai_chat_completions", "openai_responses"]) {
    for (const native of [false, true]) {
      const context = native
        ? { protocolContext: { protocol, sessionId: `fork-parent-${protocol}` } }
        : {};
      const wire = (messages: any[]) => protocol === "openai_responses"
        ? { input: messages }
        : { messages };
      const opening = [{ role: "user", content: `Inspect project ${protocol}-${native}.` }];
      const parent = resolve({ protocol, request: wire(opening), context });
      const inherited = [...opening, { role: "assistant", content: "I will inspect it." }];
      const parentFollow = wire([...inherited, { role: "user", content: "Continue the main task." }]);
      assert.equal(resolve({ protocol, request: parentFollow, context }).internalId, parent.internalId);
      const workerContext = {
        ...context,
        protocolContext: { ...context.protocolContext, nestedAgent: true, claudeCodeSubagent: true },
      };
      const fork = (instruction: string) => [
        ...inherited,
        { role: "user", content: "<fork-boilerplate> You are a worker fork." },
        { role: "user", content: instruction },
      ];
      const alphaMessages = fork("Review only tests.");
      const betaMessages = fork("Review only routing.");
      const alphaRequest = wire(alphaMessages);
      const originalAlphaRequest = structuredClone(alphaRequest);
      const alpha = resolve({ protocol, request: alphaRequest, context: workerContext });
      assert.deepEqual(alphaRequest, originalAlphaRequest, "identity slicing must not modify the SDK request");
      assert.equal(resolve({ protocol, request: alphaRequest, context }).internalId, alpha.internalId, "explicit fork markers work without nested metadata");
      const beta = resolve({ protocol, request: wire(betaMessages), context: workerContext });
      assert.notEqual(alpha.internalId, parent.internalId, "a fork must not claim the parent agent");
      assert.notEqual(beta.internalId, parent.internalId);
      assert.notEqual(alpha.internalId, beta.internalId, "shared boilerplate is not a worker identity");
      assert.equal(resolve({ protocol, request: wire(alphaMessages), context: workerContext }).internalId, alpha.internalId);
      const alphaFollow = [
        ...alphaMessages,
        { role: "assistant", content: "Tests reviewed." },
        { role: "user", content: "Check coverage too." },
      ];
      assert.equal(resolve({ protocol, request: wire(alphaFollow), context: workerContext }).internalId, alpha.internalId);
      assert.equal(resolve({ protocol, request: parentFollow, context }).internalId, parent.internalId);
      const grandchild = resolve({
        protocol,
        request: wire([
          ...alphaFollow,
          { role: "user", content: "You are a worker fork. Review fixtures only." },
        ]),
        context: workerContext,
      });
      assert.notEqual(grandchild.internalId, alpha.internalId, "a nested fork uses its own boundary");
    }
  }
}

function testNestedNamespaceAndHarnessNoise() {
  const protocol = "openai_chat_completions";
  const context = { protocolContext: { sessionId: "nested-namespace", protocol } };
  const opening = [{ role: "user", content: "Inspect the namespace test project." }];
  const first = resolve({ protocol, request: { messages: opening }, context });
  const request = {
    messages: [
      ...opening,
      { role: "assistant", content: "Ready." },
      { role: "user", content: "Review tests." },
    ],
  };
  const parent = resolve({ protocol, request, context });
  assert.equal(parent.internalId, first.internalId);
  const worker = resolve({
    protocol,
    request,
    context: { protocolContext: { ...context.protocolContext, nestedAgent: true } },
  });
  assert.notEqual(worker.internalId, parent.internalId, "nested metadata separates a marker-free worker from its parent");
  for (const noiseContext of [{}, context]) {
    const clean = resolve({ protocol, request: { messages: opening }, context: noiseContext });
    const noisy = resolve({
      protocol,
      request: { messages: [
        ...opening,
        { role: "user", content: "<system-reminder> Documentation mentions <fork-boilerplate> and You are a worker fork." },
      ] },
      context: noiseContext,
    });
    assert.equal(noisy.internalId, clean.internalId, "harness noise cannot become a fork boundary");
  }
}

function testLineageFreeFollowUpKeepsClaimAfterRefresh() {
  const protocol = "openai_chat_completions";
  const context = { req: { sessionId: "lineage-free-refresh" } };
  const opening = [{ role: "user", content: "Start a lineage-free conversation." }];
  const first = resolve({ protocol, request: { messages: opening }, context });
  const follow = {
    messages: [
      ...opening,
      { role: "assistant", content: "" },
      { role: "user", content: "Continue." },
    ],
  };
  const claimed = resolve({ protocol, request: follow, context });
  assert.equal(claimed.internalId, first.internalId);
  assert.equal(getPersistedSession("cursor-inbound", claimed.inboundKey)?.progressed, true);
  later(3600_001, () => {
    assert.equal(resolve({ protocol, request: follow, context }).internalId, first.internalId);
    resetSessionRegistryForTests();
    assert.equal(getPersistedSession("cursor-inbound", claimed.inboundKey)?.progressed, true);
    assert.notEqual(resolve({ protocol, request: { messages: opening }, context }).internalId, first.internalId);
  });
}

testLineageFreeFollowUpKeepsClaimAfterRefresh();
testInheritedForksKeepTheirOwnConversation();
testNestedNamespaceAndHarnessNoise();
testAnonymousOpeningsStayApart();
testAnonymousFollowUpDoesNotGlueLaterOpenings();
testCursorHeaderIsIgnored();
testProtocolIdentities();
testRestartKeepsTheBinding();
testSideCallUnderNativeIdStaysApart();
testRepeatedOpeningGetsFreshAgent();
testRepeatedOpeningDoesNotHijackOngoingConversation();
testContestedOpeningFollowUpGetsFreshAgent();
testOpeningRetryJoinsLongRunningTurn();
testRegistryStoresHashesAndSkipsUnchangedWrites();
console.log("cursor-sdk.inbound-session ok");

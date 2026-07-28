import assert from "node:assert/strict";
import { Cursor } from "@cursor/sdk";
import { runCursor } from "../cursor-sdk/runner";
import { globalSessionManager } from "../cursor-sdk/session";
import {
  CursorTurnRegistry,
  globalCursorTurnRegistry,
} from "../cursor-sdk/turn-output";

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function fakeSession(input: {
  key: string;
  agentId: string;
  send: (prompt: any, options: any) => Promise<any>;
}) {
  const session: any = {
    key: input.key,
    agentId: input.agentId,
    agent: {
      send: input.send,
      close() {},
    },
    mode: "bridge",
    workspaceDir: `/tmp/${input.key}`,
    hostEnv: {
      additionalRoots: [],
      facts: [],
      fingerprint: "none",
      known: false,
    },
    parked: [],
    pendingEmit: [],
    emitWaiters: [],
    pendingSdkMessages: [],
    sdkMessageWaiters: [],
    sendChain: Promise.resolve(),
    hasSentPrompt: false,
    lastActiveAt: Date.now(),
    metrics: {
      builtinToolCallsSeen: 0,
      customToolCalls: 0,
      scratchPathCorrections: 0,
      scratchPathViolations: 0,
    },
    notifyEmit() {
      const waiters = this.emitWaiters.splice(0);
      for (const waiter of waiters) waiter();
    },
    waitForEmit() {
      return new Promise<void>((resolve) => {
        if (this.pendingEmit.length) resolve();
        else this.emitWaiters.push(resolve);
      });
    },
    enqueueSdkMessage(this: any, message: any, runToken?: symbol) {
      this.pendingSdkMessages.push({
        message,
        runToken: runToken || this.activeRunToken,
        source: "delta",
      });
      this.notifySdkMessage();
    },
    notifySdkMessage() {
      const waiters = this.sdkMessageWaiters.splice(0);
      for (const waiter of waiters) waiter();
    },
    waitForSdkMessage() {
      return new Promise<void>((resolve) => {
        if (this.pendingSdkMessages.length) resolve();
        else this.sdkMessageWaiters.push(resolve);
      });
    },
  };
  return session;
}

function requestFor(session: string) {
  return {
    request: {
      model: "composer-2.5",
      stream: true,
      messages: [{ role: "user", content: "Return the shared result." }],
    } as any,
    context: {
      req: { headers: { "x-ccr-cursor-session": session } },
    },
  };
}

async function waitFor(predicate: () => boolean, label: string): Promise<void> {
  const deadline = Date.now() + 2_000;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error(`${label} timed out`);
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}

async function readRemaining(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  first: Uint8Array
): Promise<string> {
  const decoder = new TextDecoder();
  let text = decoder.decode(first, { stream: true });
  while (true) {
    const next = await reader.read();
    if (next.done) break;
    text += decoder.decode(next.value, { stream: true });
  }
  return text + decoder.decode();
}

async function testConcurrentJoinAndReplay() {
  globalCursorTurnRegistry.clear();
  const sendGate = deferred();
  const finishGate = deferred();
  let sendCalls = 0;
  let nextCalls = 0;
  let cancelCalls = 0;

  const session = fakeSession({
    key: "turn-join",
    agentId: "agent-turn-join",
    async send() {
      sendCalls += 1;
      await sendGate.promise;
      let emitted = false;
      return {
        id: "run-turn-join",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  nextCalls += 1;
                  if (!emitted) {
                    emitted = true;
                    return {
                      done: false,
                      value: {
                        type: "assistant",
                        message: {
                          content: [{ type: "text", text: "shared result" }],
                        },
                      },
                    };
                  }
                  await finishGate.promise;
                  return { done: true, value: undefined };
                },
                async return() {
                  return { done: true, value: undefined };
                },
              };
            },
          };
        },
        async cancel() {
          cancelCalls += 1;
        },
      };
    },
  });

  const originalGetOrCreate =
    globalSessionManager.getOrCreate.bind(globalSessionManager);
  let getCalls = 0;
  (globalSessionManager as any).getOrCreate = async () => {
    getCalls += 1;
    return session;
  };

  try {
    const { request, context } = requestFor("turn-join");
    const responseA = runCursor(
      structuredClone(request),
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    await waitFor(() => sendCalls === 1, "leader send");

    const responseB = runCursor(
      structuredClone(request),
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    await Promise.resolve();
    assert.equal(sendCalls, 1);
    assert.equal(getCalls, 1);

    sendGate.resolve();
    const [joinedA, joinedB] = await Promise.all([responseA, responseB]);
    assert.ok(joinedA.body);
    assert.ok(joinedB.body);
    const readerA = joinedA.body.getReader();
    const readerB = joinedB.body.getReader();
    const [firstA, firstB] = await Promise.all([
      readerA.read(),
      readerB.read(),
    ]);
    assert.equal(firstA.done, false);
    assert.equal(firstB.done, false);
    assert.deepEqual(firstA.value, firstB.value);

    await readerA.cancel("one duplicate disconnected");
    assert.equal(
      cancelCalls,
      0,
      "one subscriber must not cancel the shared Cursor run"
    );

    finishGate.resolve();
    const joinedBody = await readRemaining(readerB, firstB.value!);
    assert.match(joinedBody, /shared result/);
    assert.equal(sendCalls, 1);
    assert.equal(nextCalls, 2, "only one producer may consume the SDK iterator");

    const replay = await runCursor(
      structuredClone(request),
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    const replayBody = await replay.text();
    assert.equal(replayBody, joinedBody);
    assert.equal(sendCalls, 1, "completed retry must use bounded replay");
    assert.equal(getCalls, 1);
  } finally {
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    globalCursorTurnRegistry.clear();
  }
}

async function testLeaderAbortBeforeResponseKeepsJoinerAlive() {
  globalCursorTurnRegistry.clear();
  const sendGate = deferred();
  const leaderAbort = new AbortController();
  let sendCalls = 0;
  let cancelCalls = 0;

  const session = fakeSession({
    key: "turn-pre-response-abort",
    agentId: "agent-turn-pre-response-abort",
    async send() {
      sendCalls += 1;
      await sendGate.promise;
      let emitted = false;
      return {
        id: "run-turn-pre-response-abort",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  if (emitted) return { done: true, value: undefined };
                  emitted = true;
                  return {
                    done: false,
                    value: {
                      type: "assistant",
                      message: {
                        content: [{ type: "text", text: "joiner survived" }],
                      },
                    },
                  };
                },
              };
            },
          };
        },
        async cancel() {
          cancelCalls += 1;
        },
      };
    },
  });

  const originalGetOrCreate =
    globalSessionManager.getOrCreate.bind(globalSessionManager);
  (globalSessionManager as any).getOrCreate = async () => session;

  try {
    const { request, context } = requestFor("turn-pre-response-abort");
    const leader = runCursor(
      structuredClone(request),
      { apiKey: "crsr_test" },
      structuredClone(context),
      {
        cursorMode: "bridge",
        abortSignal: leaderAbort.signal,
      }
    );
    const leaderOutcome = leader.then(
      () => undefined,
      (error) => error
    );
    await waitFor(() => sendCalls === 1, "pre-response leader send");

    const joiner = runCursor(
      structuredClone(request),
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    await Promise.resolve();
    leaderAbort.abort("leader disconnected before response");
    sendGate.resolve();

    const [leaderError, joinerResponse] = await Promise.all([
      leaderOutcome,
      joiner,
    ]);
    assert.equal(leaderError?.name, "AbortError");
    assert.match(await joinerResponse.text(), /joiner survived/);
    assert.equal(sendCalls, 1);
    assert.equal(cancelCalls, 0);
  } finally {
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    globalCursorTurnRegistry.clear();
  }
}

async function testAbortedRetryStartsNewGeneration() {
  globalCursorTurnRegistry.clear();
  let sendCalls = 0;
  let oldCancelCalls = 0;
  let oldIteratorReturns = 0;
  const oldCancelGate = deferred();
  const oldIteratorGate = deferred();

  const oldSession = fakeSession({
    key: "turn-abort-retry",
    agentId: "agent-turn-abort-old",
    async send() {
      sendCalls += 1;
      let emitted = false;
      return {
        id: "run-turn-abort-old",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  if (!emitted) {
                    emitted = true;
                    return {
                      done: false,
                      value: {
                        type: "assistant",
                        message: {
                          content: [{ type: "text", text: "partial" }],
                        },
                      },
                    };
                  }
                  return new Promise(() => undefined);
                },
                async return() {
                  oldIteratorReturns += 1;
                  await oldIteratorGate.promise;
                  return { done: true, value: undefined };
                },
              };
            },
          };
        },
        async cancel() {
          oldCancelCalls += 1;
          await oldCancelGate.promise;
        },
      };
    },
  });

  const freshSession = fakeSession({
    key: "turn-abort-retry",
    agentId: "agent-turn-abort-fresh",
    async send() {
      sendCalls += 1;
      let emitted = false;
      return {
        id: "run-turn-abort-fresh",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  if (emitted) return { done: true, value: undefined };
                  emitted = true;
                  return {
                    done: false,
                    value: {
                      type: "assistant",
                      message: {
                        content: [{ type: "text", text: "retry completed" }],
                      },
                    },
                  };
                },
              };
            },
          };
        },
        async cancel() {},
      };
    },
  });

  const originalGetOrCreate =
    globalSessionManager.getOrCreate.bind(globalSessionManager);
  const originalInvalidate =
    globalSessionManager.invalidate.bind(globalSessionManager);
  let getCalls = 0;
  (globalSessionManager as any).getOrCreate = async () => {
    getCalls += 1;
    return getCalls === 1 ? oldSession : freshSession;
  };
  (globalSessionManager as any).invalidate = (session: any, reason: string) => {
    originalInvalidate(session, reason);
  };

  try {
    const { request, context } = requestFor("turn-abort-retry");
    const first = await runCursor(
      structuredClone(request),
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    assert.ok(first.body);
    const reader = first.body.getReader();
    const partial = await reader.read();
    assert.equal(partial.done, false);
    const cancellation = reader.cancel("user aborted");
    await waitFor(
      () => oldCancelCalls === 1 && oldIteratorReturns === 1,
      "old generation cleanup"
    );
    assert.equal(oldCancelCalls, 1);
    assert.equal(oldIteratorReturns, 1);

    let retrySettled = false;
    const retryPromise = runCursor(
      structuredClone(request),
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    ).then((response) => {
      retrySettled = true;
      return response;
    });
    await Promise.resolve();
    assert.equal(getCalls, 1);
    assert.equal(sendCalls, 1);
    assert.equal(retrySettled, false);

    oldIteratorGate.resolve();
    await Promise.resolve();
    assert.equal(getCalls, 1);
    assert.equal(retrySettled, false);

    oldCancelGate.resolve();
    await cancellation;
    const retry = await retryPromise;
    const retryBody = await retry.text();
    assert.match(retryBody, /retry completed/);
    assert.equal(sendCalls, 2);
    assert.equal(getCalls, 2);
  } finally {
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    (globalSessionManager as any).invalidate = originalInvalidate;
    globalCursorTurnRegistry.clear();
  }
}

async function testStrictExtensionReusesIdleAgent() {
  globalCursorTurnRegistry.clear();
  const prompts: string[] = [];
  const freshPrompts: string[] = [];
  const sendOptions: any[] = [];
  let sendCalls = 0;

  const session = fakeSession({
    key: "turn-strict-extension",
    agentId: "agent-turn-strict-extension",
    async send(prompt, options) {
      sendCalls += 1;
      prompts.push(prompt.text);
      sendOptions.push(options);
      const text = sendCalls === 1 ? "first answer" : "second answer";
      let emitted = false;
      return {
        id: `run-${sendCalls}`,
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  if (emitted) return { done: true, value: undefined };
                  emitted = true;
                  return {
                    done: false,
                    value: {
                      type: "assistant",
                      message: { content: [{ type: "text", text }] },
                    },
                  };
                },
              };
            },
          };
        },
        async cancel() {},
      };
    },
  });
  const freshSession = fakeSession({
    key: "turn-strict-extension",
    agentId: "agent-turn-strict-extension-fresh",
    async send(prompt) {
      sendCalls += 1;
      freshPrompts.push(prompt.text);
      let emitted = false;
      return {
        id: "run-fresh-multi-suffix",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  if (emitted) return { done: true, value: undefined };
                  emitted = true;
                  return {
                    done: false,
                    value: {
                      type: "assistant",
                      message: {
                        content: [{ type: "text", text: "fresh answer" }],
                      },
                    },
                  };
                },
              };
            },
          };
        },
        async cancel() {},
      };
    },
  });

  const originalGetOrCreate =
    globalSessionManager.getOrCreate.bind(globalSessionManager);
  let getCalls = 0;
  (globalSessionManager as any).getOrCreate = async () => {
    getCalls += 1;
    return session.poisoned ? freshSession : session;
  };

  try {
    const context = {
      req: {
        headers: { "x-ccr-cursor-session": "turn-strict-extension" },
      },
    };
    const firstRequest = {
      model: "composer-2.5",
      stream: true,
      messages: [{ role: "user", content: "first user" }],
    } as any;
    const first = await runCursor(
      firstRequest,
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    assert.match(await first.text(), /first answer/);

    const secondRequest = {
      ...firstRequest,
      messages: [
        { role: "user", content: "first user" },
        { role: "assistant", content: "first answer" },
        { role: "user", content: "second user" },
      ],
    };
    const second = await runCursor(
      secondRequest,
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    assert.match(await second.text(), /second answer/);

    assert.equal(sendCalls, 2);
    assert.equal(getCalls, 2);
    assert.match(prompts[1], /\[user\]\nsecond user/);
    assert.equal(prompts[1].includes("first user"), false);
    assert.equal(prompts[1].includes("first answer"), false);
    assert.equal(sendOptions[1]?.local?.force, undefined);
    assert.equal(session.poisoned, undefined);

    const multiSuffixRequest = {
      ...firstRequest,
      messages: [
        { role: "user", content: "first user" },
        { role: "assistant", content: "first answer" },
        { role: "user", content: "second user" },
        { role: "assistant", content: "second answer" },
        { role: "user", content: "third user, part one" },
        { role: "user", content: "third user, part two" },
      ],
    };
    const replayed = await runCursor(
      multiSuffixRequest,
      { apiKey: "crsr_test" },
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    assert.match(await replayed.text(), /fresh answer/);
    assert.equal(sendCalls, 3);
    assert.equal(freshPrompts.length, 1);
    assert.match(freshPrompts[0], /third user, part one/);
    assert.match(freshPrompts[0], /third user, part two/);
    assert.equal(
      prompts.length,
      2,
      "a multi-message suffix must not be truncated onto the old agent"
    );
  } finally {
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    globalCursorTurnRegistry.clear();
  }
}

async function testSupersededFingerprintCannotCancelNewerTurn() {
  const makeHangingResponse = (onCancel: () => void) =>
    new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("data"));
        },
        cancel() {
          onCancel();
        },
      })
    );

  const abortedRegistry = new CursorTurnRegistry();
  let healthyCancels = 0;
  const healthyA = await abortedRegistry.admit({
    sessionKey: "generation-pre-aborted",
    fingerprint: "turn-a",
    responseKind: "stream",
  });
  await healthyA.attach(makeHangingResponse(() => healthyCancels++));
  const healthyResponse = await healthyA.response();
  const healthyReader = healthyResponse.body!.getReader();
  await healthyReader.read();
  const alreadyAborted = new AbortController();
  alreadyAborted.abort("request already disconnected");
  await assert.rejects(
    abortedRegistry.admit({
      sessionKey: "generation-pre-aborted",
      fingerprint: "turn-b",
      responseKind: "stream",
      signal: alreadyAborted.signal,
    }),
    (error: any) => error?.name === "AbortError"
  );
  assert.equal(
    healthyCancels,
    0,
    "an already-aborted request must not supersede a healthy turn"
  );
  await healthyReader.cancel();
  abortedRegistry.clear();

  const activeRegistry = new CursorTurnRegistry();
  let activeACancels = 0;
  let activeBCancels = 0;
  const activeA = await activeRegistry.admit({
    sessionKey: "generation-active",
    fingerprint: "turn-a",
    responseKind: "stream",
  });
  await activeA.attach(makeHangingResponse(() => activeACancels++));
  const activeAResponse = await activeA.response();
  const activeAReader = activeAResponse.body!.getReader();
  await activeAReader.read();

  const activeB = await activeRegistry.admit({
    sessionKey: "generation-active",
    fingerprint: "turn-b",
    responseKind: "stream",
  });
  assert.equal(activeACancels, 1);
  await activeB.attach(makeHangingResponse(() => activeBCancels++));
  const activeBResponse = await activeB.response();
  const activeBReader = activeBResponse.body!.getReader();
  await activeBReader.read();

  await assert.rejects(
    activeRegistry.admit({
      sessionKey: "generation-active",
      fingerprint: "turn-a",
      responseKind: "stream",
    }),
    (error: any) => error?.code === "cursor_turn_superseded"
  );
  assert.equal(
    activeBCancels,
    0,
    "a delayed older retry must not cancel the newer active turn"
  );
  await activeBReader.cancel();
  activeRegistry.clear();

  const completedRegistry = new CursorTurnRegistry();
  let completedBCancels = 0;
  const completedA = await completedRegistry.admit({
    sessionKey: "generation-completed",
    fingerprint: "turn-a",
    responseKind: "stream",
  });
  await completedA.attach(new Response("completed-a"));
  const completedAResponse = await completedA.response();
  assert.equal(await completedAResponse.text(), "completed-a");

  const completedB = await completedRegistry.admit({
    sessionKey: "generation-completed",
    fingerprint: "turn-b",
    responseKind: "stream",
  });
  await completedB.attach(makeHangingResponse(() => completedBCancels++));
  const completedBResponse = await completedB.response();
  const completedBReader = completedBResponse.body!.getReader();
  await completedBReader.read();

  await assert.rejects(
    completedRegistry.admit({
      sessionKey: "generation-completed",
      fingerprint: "turn-a",
      responseKind: "stream",
    }),
    (error: any) => error?.code === "cursor_turn_superseded"
  );
  assert.equal(completedBCancels, 0);
  await completedBReader.cancel();
  completedRegistry.clear();
}

async function testOversizedFirstChunkReachesLeader() {
  const registry = new CursorTurnRegistry();
  const firstChunk = new Uint8Array(8 * 1024 * 1024 + 1);
  firstChunk[0] = 37;
  let sourceCancels = 0;

  const leader = await registry.admit({
    sessionKey: "oversized-first-chunk",
    fingerprint: "turn-a",
    responseKind: "stream",
  });
  await leader.attach(
    new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(firstChunk);
        },
        cancel() {
          sourceCancels += 1;
        },
      })
    )
  );

  const response = await leader.response();
  const reader = response.body!.getReader();
  const first = await Promise.race([
    reader.read(),
    new Promise<never>((_, reject) => {
      const timer = setTimeout(
        () => reject(new Error("oversized leader chunk timed out")),
        500
      );
      if (typeof timer.unref === "function") timer.unref();
    }),
  ]);
  assert.equal(first.done, false);
  assert.equal(first.value?.byteLength, firstChunk.byteLength);
  assert.equal(first.value?.[0], 37);

  await reader.cancel("test complete");
  assert.equal(sourceCancels, 1);
  registry.clear();
}

async function testConcurrentSupersessionsStayInAdmissionOrder() {
  const registry = new CursorTurnRegistry();
  const cancelGate = deferred();
  let firstCancelStarted = false;

  const first = await registry.admit({
    sessionKey: "ordered-supersession",
    fingerprint: "turn-a",
    responseKind: "stream",
  });
  await first.attach(
    new Response(
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("turn-a"));
        },
        async cancel() {
          firstCancelStarted = true;
          await cancelGate.promise;
        },
      })
    )
  );
  const firstResponse = await first.response();
  await firstResponse.body!.getReader().read();

  const secondPromise = registry.admit({
    sessionKey: "ordered-supersession",
    fingerprint: "turn-b",
    responseKind: "stream",
  });
  await waitFor(() => firstCancelStarted, "first supersession cancellation");
  const thirdPromise = registry.admit({
    sessionKey: "ordered-supersession",
    fingerprint: "turn-c",
    responseKind: "stream",
  });

  cancelGate.resolve();
  const second = await secondPromise;
  await waitFor(
    () => second.producerSignal.aborted,
    "newest admission superseding the middle turn"
  );
  second.fail(new Error("middle turn superseded before response attachment"));

  const third = await thirdPromise;
  assert.equal(second.producerSignal.aborted, true);
  assert.equal(
    third.producerSignal.aborted,
    false,
    "the newest concurrent admission must remain active"
  );
  third.fail(new Error("test cleanup"));
  registry.clear();
}

async function testSupersededProducerCannotMutateNewerSession() {
  globalCursorTurnRegistry.clear();
  const staleModelGate = deferred();
  const freshFinishGate = deferred();
  let modelListCalls = 0;
  let getCalls = 0;
  let sendCalls = 0;
  let freshCancelCalls = 0;

  const freshSession = fakeSession({
    key: "stale-producer",
    agentId: "agent-newest-generation",
    async send() {
      sendCalls += 1;
      let emitted = false;
      return {
        id: "run-newest-generation",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  if (!emitted) {
                    emitted = true;
                    return {
                      done: false,
                      value: {
                        type: "assistant",
                        message: {
                          content: [{ type: "text", text: "newest survives" }],
                        },
                      },
                    };
                  }
                  await freshFinishGate.promise;
                  return { done: true, value: undefined };
                },
                async return() {
                  return { done: true, value: undefined };
                },
              };
            },
          };
        },
        async cancel() {
          freshCancelCalls += 1;
        },
      };
    },
  });

  const originalModelList = Cursor.models.list;
  const originalGetOrCreate =
    globalSessionManager.getOrCreate.bind(globalSessionManager);
  (Cursor.models as any).list = async () => {
    modelListCalls += 1;
    if (modelListCalls === 1) await staleModelGate.promise;
    return [];
  };
  (globalSessionManager as any).getOrCreate = async () => {
    getCalls += 1;
    return freshSession;
  };

  try {
    const context = {
      req: { headers: { "x-ccr-cursor-session": "stale-producer" } },
    };
    const provider = { apiKey: "crsr_stale_generation_test" };
    const stale = runCursor(
      {
        model: "composer-2.5",
        stream: true,
        messages: [{ role: "user", content: "older request" }],
      } as any,
      provider,
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    await waitFor(() => modelListCalls === 1, "stale model lookup");

    const newest = await runCursor(
      {
        model: "composer-2.5",
        stream: true,
        messages: [{ role: "user", content: "newer request" }],
      } as any,
      provider,
      structuredClone(context),
      { cursorMode: "bridge" }
    );
    const newestReader = newest.body!.getReader();
    const first = await newestReader.read();
    assert.equal(first.done, false);
    assert.equal(sendCalls, 1);
    assert.equal(getCalls, 1);

    staleModelGate.resolve();
    await assert.rejects(
      stale,
      (error: any) => error?.name === "AbortError"
    );
    assert.equal(
      getCalls,
      1,
      "a stale producer must stop before acquiring the newer session"
    );
    assert.equal(freshCancelCalls, 0);
    assert.equal(freshSession.poisoned, undefined);

    freshFinishGate.resolve();
    const newestBody = await readRemaining(newestReader, first.value!);
    assert.match(newestBody, /newest survives/);
  } finally {
    staleModelGate.resolve();
    freshFinishGate.resolve();
    (Cursor.models as any).list = originalModelList;
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    globalCursorTurnRegistry.clear();
  }
}

async function main() {
  const originalModelList = Cursor.models.list;
  (Cursor.models as any).list = async () => [];
  try {
    await testConcurrentJoinAndReplay();
    await testLeaderAbortBeforeResponseKeepsJoinerAlive();
    await testAbortedRetryStartsNewGeneration();
    await testStrictExtensionReusesIdleAgent();
    await testSupersededFingerprintCannotCancelNewerTurn();
    await testOversizedFirstChunkReachesLeader();
    await testConcurrentSupersessionsStayInAdmissionOrder();
    await testSupersededProducerCannotMutateNewerSession();
  } finally {
    (Cursor.models as any).list = originalModelList;
    globalCursorTurnRegistry.clear();
  }
  console.log("cursor-sdk.turn-coordination: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

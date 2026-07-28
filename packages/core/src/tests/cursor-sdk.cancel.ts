import assert from "node:assert/strict";
import { Agent } from "@cursor/sdk";
import { cancelActiveRun, SessionManager, withTimeout } from "../cursor-sdk/session";

async function main() {
  const started = Date.now();
  await assert.rejects(
    withTimeout(new Promise(() => undefined), 25, "timed out"),
    /timed out/
  );
  assert.ok(Date.now() - started < 250);

  const token = Symbol("old-run");
  let rejected = false;
  const session: any = {
    run: {
      status: "running",
      cancel: () => new Promise(() => undefined),
    },
    streamIterator: {
      return: () => new Promise(() => undefined),
    },
    activeRunToken: token,
    parked: [
      {
        reject(error: Error) {
          rejected = /cancelled/.test(error.message);
        },
      },
    ],
    pendingEmit: [{ id: "tool", name: "Bash", args: {}, runToken: token }],
    pendingSdkMessages: [{ message: { type: "thinking", text: "old" }, runToken: token }],
    emitWaiters: [() => undefined],
    sdkMessageWaiters: [() => undefined],
    notifyEmit() {
      this.notified = true;
    },
    notifySdkMessage() {
      this.sdkNotified = true;
    },
  };

  const result = await cancelActiveRun(session, {
    rejectParked: true,
    reason: "cancelled for test",
    timeoutMs: 25,
    onlyRunToken: token,
    poisonOnFailure: true,
  });

  assert.equal(result.failed, true);
  assert.equal(result.timedOut, true);
  assert.equal(result.runCancelFailed, true);
  assert.equal(result.iteratorReturnFailed, true);
  assert.equal(session.run, undefined);
  assert.equal(session.streamIterator, undefined);
  assert.equal(session.activeRunToken, undefined);
  assert.deepEqual(session.pendingEmit, []);
  assert.deepEqual(session.pendingSdkMessages, []);
  assert.equal(rejected, true);
  assert.equal(session.notified, true);
  assert.equal(session.sdkNotified, true);
  assert.equal(session.poisoned, true);

  const newerSession: any = {
    activeRunToken: Symbol("new-run"),
    pendingEmit: [{ id: "new", name: "Bash", args: {} }],
    pendingSdkMessages: [{ message: { type: "thinking", text: "new" } }],
    parked: [],
    notifyEmit() {
      throw new Error("should not cancel newer run");
    },
  };
  const skipped = await cancelActiveRun(newerSession, {
    onlyRunToken: token,
    timeoutMs: 1,
  });
  assert.equal(skipped.skipped, true);
  assert.equal(newerSession.pendingEmit.length, 1);
  assert.equal(newerSession.pendingSdkMessages.length, 1);

  let releaseGeneratorNext!: () => void;
  const generatorNextGate = new Promise<void>((resolve) => {
    releaseGeneratorNext = resolve;
  });
  const blockingIterator = (async function* () {
    await generatorNextGate;
    yield "released";
  })();
  const pendingGeneratorNext = blockingIterator.next();
  let generatorRunCancelCalls = 0;
  const generatorSession: any = {
    run: {
      status: "running",
      async cancel() {
        generatorRunCancelCalls += 1;
        releaseGeneratorNext();
      },
    },
    streamIterator: blockingIterator,
    activeRunToken: Symbol("generator-run"),
    parked: [],
    pendingEmit: [],
    pendingSdkMessages: [],
    notifyEmit() {},
    notifySdkMessage() {},
  };
  const generatorCancellation = await cancelActiveRun(generatorSession, {
    timeoutMs: 100,
    poisonOnFailure: true,
  });
  await pendingGeneratorNext;
  assert.equal(generatorRunCancelCalls, 1);
  assert.equal(generatorCancellation.failed, false);
  assert.equal(generatorCancellation.timedOut, false);
  assert.equal(generatorCancellation.runCancelFailed, false);
  assert.equal(generatorCancellation.iteratorReturnFailed, false);
  assert.equal(generatorSession.poisoned, undefined);

  let closed = false;
  let invalidatedRejected = false;
  const manager = new SessionManager();
  const invalidatedSession: any = {
    key: "poisoned-session",
    agentId: "agent-poisoned",
    agent: {
      close() {
        closed = true;
      },
    },
    pendingEmit: [{ id: "old", name: "Bash", args: {} }],
    pendingSdkMessages: [{ message: { type: "thinking", text: "old" } }],
    parked: [
      {
        reject(error: Error) {
          invalidatedRejected = /unsafe/.test(error.message);
        },
      },
    ],
    notifyEmit() {
      this.invalidatedNotified = true;
    },
    notifySdkMessage() {
      this.invalidatedSdkNotified = true;
    },
  };
  (manager as any).sessions.set("poisoned-session", invalidatedSession);
  manager.invalidate(invalidatedSession, "unsafe for reuse");
  assert.equal(manager.get("poisoned-session"), undefined);
  assert.equal(invalidatedSession.poisoned, true);
  assert.equal(invalidatedSession.run, undefined);
  assert.equal(invalidatedSession.streamIterator, undefined);
  assert.equal(invalidatedSession.activeRunToken, undefined);
  assert.equal(invalidatedSession.pendingEmit.length, 0);
  assert.equal(invalidatedSession.pendingSdkMessages.length, 0);
  assert.equal(invalidatedSession.parked.length, 0);
  assert.equal(invalidatedRejected, true);
  assert.equal(invalidatedSession.invalidatedNotified, true);
  assert.equal(invalidatedSession.invalidatedSdkNotified, true);
  assert.equal(closed, true);

  let retirementClosed = false;
  const retiringSession: any = {
    key: "retiring-session",
    agentId: "agent-retiring",
    agent: {
      close() {
        retirementClosed = true;
      },
    },
    run: { status: "running", cancel: async () => undefined },
    streamIterator: { return: async () => ({ done: true }) },
    activeRunToken: Symbol("retiring-run"),
    pendingEmit: [],
    pendingSdkMessages: [],
    parked: [],
    notifyEmit() {},
    notifySdkMessage() {},
  };
  (manager as any).sessions.set("retiring-session", retiringSession);
  let releaseRetirement!: () => void;
  const retirementGate = new Promise<void>((resolve) => {
    releaseRetirement = resolve;
  });
  let retirementCleanupStarted = false;
  const retirement = manager.retireSession(
    retiringSession,
    "retirement test",
    async () => {
      retirementCleanupStarted = true;
      await retirementGate;
      manager.invalidate(retiringSession, "retirement finished");
      return true;
    }
  );
  assert.equal(retirementCleanupStarted, true);
  assert.equal(manager.get("retiring-session"), undefined);
  assert.ok(retiringSession.run, "retirement must preserve the run for cancellation");
  assert.ok(
    retiringSession.streamIterator,
    "retirement must preserve the iterator for cancellation"
  );
  assert.equal(retirementClosed, false);

  const originalAgentCreate = Agent.create;
  let createCalls = 0;
  let releaseCreation!: () => void;
  const creationGate = new Promise<void>((resolve) => {
    releaseCreation = resolve;
  });
  (Agent as any).create = async () => {
    createCalls += 1;
    await creationGate;
    return {
      agentId: "agent-replacement",
      close() {},
    };
  };

  let reconnectsSettled = false;
  const reconnectOptions = {
      key: "retiring-session",
      apiKey: "crsr_test",
      model: { id: "composer-2.5" },
      mode: "agent" as const,
      cursorCwd: "/tmp/ccr-cursor-retirement-test",
  };
  const reconnectA = manager.getOrCreate(reconnectOptions);
  const reconnectB = manager.getOrCreate(reconnectOptions);
  const reconnects = Promise.all([reconnectA, reconnectB]).then((value) => {
    reconnectsSettled = true;
    return value;
  });

  try {
    await Promise.resolve();
    assert.equal(
      reconnectsSettled,
      false,
      "reconnects must wait for the retirement barrier"
    );

    releaseRetirement();
    assert.equal(await retirement, true);
    for (let i = 0; i < 10 && createCalls === 0; i += 1) {
      await Promise.resolve();
    }
    assert.equal(createCalls, 1, "reconnects must share one Agent.create");
    assert.equal(reconnectsSettled, false);

    releaseCreation();
    const [replacementA, replacementB] = await reconnects;
    assert.equal(replacementA, replacementB);
    assert.equal(replacementA.agentId, "agent-replacement");
    assert.equal(retirementClosed, true);
  } finally {
    (Agent as any).create = originalAgentCreate;
  }

  const sharedKey = "queued-retirements";
  const staleSameKeySession: any = {
    key: sharedKey,
    agentId: "agent-stale-same-key",
  };
  const newerSameKeySession: any = {
    key: sharedKey,
    agentId: "agent-newer-same-key",
  };
  (manager as any).sessions.set(sharedKey, newerSameKeySession);
  let releaseStaleCleanup!: () => void;
  const staleCleanupGate = new Promise<void>((resolve) => {
    releaseStaleCleanup = resolve;
  });
  let releaseNewerCleanup!: () => void;
  const newerCleanupGate = new Promise<void>((resolve) => {
    releaseNewerCleanup = resolve;
  });
  let staleCleanupCalls = 0;
  let duplicateStaleCleanupCalls = 0;
  let newerCleanupCalls = 0;
  const staleRetirement = manager.retireSession(
    staleSameKeySession,
    "stale object cleanup",
    async () => {
      staleCleanupCalls += 1;
      await staleCleanupGate;
      return true;
    }
  );
  const queuedOriginalAgentCreate = Agent.create;
  let queuedCreateCalls = 0;
  (Agent as any).create = async () => {
    queuedCreateCalls += 1;
    return {
      agentId: "agent-after-all-retirements",
      close() {},
    };
  };
  const queuedReconnect = manager.getOrCreate({
    key: sharedKey,
    apiKey: "crsr_test",
    model: { id: "composer-2.5" },
    mode: "agent",
    cursorCwd: "/tmp/ccr-cursor-queued-retirement-test",
  });
  const duplicateStaleRetirement = manager.retireSession(
    staleSameKeySession,
    "duplicate stale object cleanup",
    async () => {
      duplicateStaleCleanupCalls += 1;
      return true;
    }
  );
  const newerRetirement = manager.retireSession(
    newerSameKeySession,
    "newer object cleanup",
    async () => {
      newerCleanupCalls += 1;
      await newerCleanupGate;
      return true;
    }
  );
  assert.equal(
    manager.get(sharedKey),
    undefined,
    "a retirement queued behind another same-key object must detach immediately"
  );
  assert.equal(newerSameKeySession.poisoned, true);
  assert.equal(newerCleanupCalls, 0);
  releaseStaleCleanup();
  assert.equal(await staleRetirement, true);
  assert.equal(await duplicateStaleRetirement, false);
  for (let index = 0; index < 5; index += 1) await Promise.resolve();
  assert.equal(newerCleanupCalls, 1);
  assert.equal(
    queuedCreateCalls,
    0,
    "replacement creation must wait for every queued same-key retirement"
  );
  releaseNewerCleanup();
  assert.equal(await newerRetirement, true);
  assert.equal(staleCleanupCalls, 1);
  assert.equal(
    duplicateStaleCleanupCalls,
    0,
    "duplicate retirement of one session object must share its cleanup"
  );
  assert.equal(
    newerCleanupCalls,
    1,
    "different same-key session objects must each run their own cleanup"
  );
  const afterRetirements = await queuedReconnect;
  assert.equal(queuedCreateCalls, 1);
  assert.equal(afterRetirements.agentId, "agent-after-all-retirements");
  (Agent as any).create = queuedOriginalAgentCreate;

  console.log("cursor-sdk.cancel: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

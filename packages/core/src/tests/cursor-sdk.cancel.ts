import assert from "node:assert/strict";
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

  console.log("cursor-sdk.cancel: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

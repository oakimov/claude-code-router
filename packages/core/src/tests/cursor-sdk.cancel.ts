import assert from "node:assert/strict";
import { cancelActiveRun, withTimeout } from "../cursor-sdk/session";

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
    emitWaiters: [() => undefined],
    notifyEmit() {
      this.notified = true;
    },
  };

  await cancelActiveRun(session, {
    rejectParked: true,
    reason: "cancelled for test",
    timeoutMs: 25,
    onlyRunToken: token,
  });

  assert.equal(session.run, undefined);
  assert.equal(session.streamIterator, undefined);
  assert.equal(session.activeRunToken, undefined);
  assert.deepEqual(session.pendingEmit, []);
  assert.equal(rejected, true);
  assert.equal(session.notified, true);

  const newerSession: any = {
    activeRunToken: Symbol("new-run"),
    pendingEmit: [{ id: "new", name: "Bash", args: {} }],
    parked: [],
    notifyEmit() {
      throw new Error("should not cancel newer run");
    },
  };
  await cancelActiveRun(newerSession, { onlyRunToken: token, timeoutMs: 1 });
  assert.equal(newerSession.pendingEmit.length, 1);

  console.log("cursor-sdk.cancel: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

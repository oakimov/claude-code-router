/**
 * A pending `iterator.next()` is rejected by the Cursor SDK's internal
 * AbortController when `run.cancel()` runs during an interrupt. The event
 * generator can return before it races that promise (client abort or a
 * run-token change), so the rejection must already carry a handler — otherwise
 * Node reports `unhandledRejection: AbortError` followed by
 * `PromiseRejectionHandledWarning` once cancelActiveRun attaches its catch.
 */
import assert from "node:assert/strict";
import { streamSessionEvents } from "../cursor-sdk/runner";

function makeSession(runToken: symbol) {
  let rejectNext!: (error: Error) => void;
  const nextGate = new Promise<never>((_, reject) => {
    rejectNext = reject;
  });
  const session: any = {
    key: "pending-next",
    agentId: "agent-pending-next",
    activeRunToken: runToken,
    parked: [],
    pendingEmit: [],
    pendingSdkMessages: [],
    metrics: { builtinToolCallsSeen: 0 },
    streamIterator: {
      next: () => nextGate,
      return: async () => ({ done: true, value: undefined }),
    },
    waitForEmit: () => new Promise<void>(() => undefined),
    waitForSdkMessage: () => new Promise<void>(() => undefined),
    notifyEmit() {},
    notifySdkMessage() {},
  };
  return { session, rejectNext };
}

async function drainUntilEnd(generator: AsyncGenerator<any>) {
  for await (const event of generator) {
    if (event.kind === "end") return event;
  }
  return undefined;
}

async function main() {
  const unhandled: unknown[] = [];
  const onUnhandled = (reason: unknown) => unhandled.push(reason);
  process.on("unhandledRejection", onUnhandled);

  try {
    // Case 1: the client aborts before the generator ever reaches its race.
    const abortToken = Symbol("abort-run");
    const aborted = makeSession(abortToken);
    const abortController = new AbortController();
    abortController.abort();
    const abortEnd = await drainUntilEnd(
      streamSessionEvents(
        aborted.session,
        "agent" as any,
        abortToken,
        abortController.signal
      )
    );
    assert.equal(abortEnd?.aborted, true);
    // run.cancel() rejects the outstanding next() after the generator returned.
    aborted.rejectNext(
      Object.assign(new Error("This operation was aborted"), {
        name: "AbortError",
      })
    );

    // Case 2: a retirement swaps the run token while next() is still pending.
    const staleToken = Symbol("stale-run");
    const stale = makeSession(staleToken);
    stale.session.activeRunToken = Symbol("newer-run");
    const staleEnd = await drainUntilEnd(
      streamSessionEvents(stale.session, "agent" as any, staleToken)
    );
    assert.equal(staleEnd?.kind, "end");
    stale.rejectNext(
      Object.assign(new Error("This operation was aborted"), {
        name: "AbortError",
      })
    );

    // Let the microtask queue and one macrotask turn settle so any orphaned
    // rejection would have surfaced.
    await new Promise((resolve) => setTimeout(resolve, 10));
    assert.deepEqual(
      unhandled.map((reason: any) => String(reason?.message || reason)),
      [],
      "pending iterator.next() rejections must not reach unhandledRejection"
    );
  } finally {
    process.off("unhandledRejection", onUnhandled);
  }

  console.log("cursor-sdk.pending-next-rejection: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

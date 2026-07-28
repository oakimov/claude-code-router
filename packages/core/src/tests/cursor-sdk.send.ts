import assert from "node:assert/strict";
import {
  isCursorAgentBusyError,
  isCursorSendPoisonError,
  sendCursorPrompt,
} from "../cursor-sdk/send";

async function main() {
  assert.equal(
    isCursorAgentBusyError(
      Object.assign(new Error("Agent abc already has active run"), {
        name: "AgentBusyError",
      })
    ),
    true
  );
  assert.equal(
    isCursorSendPoisonError(new Error("Cursor SDK agent.send timed out")),
    true
  );

  // Both matchers gate retire-and-replay. Substring matches that are too broad
  // silently churn healthy sessions on unrelated failures.
  for (const notBusy of [
    "Agent abc has no active run to cancel",
    "active runtime failure while starting the agent",
  ]) {
    assert.equal(isCursorAgentBusyError(new Error(notBusy)), false, notBusy);
  }
  for (const notPoison of [
    "Failed to access network filesystem path /mnt/share",
    "tool wrote to the network drive",
  ]) {
    assert.equal(isCursorSendPoisonError(new Error(notPoison)), false, notPoison);
  }
  for (const poison of [
    "network error while contacting Cursor",
    "network request failed",
    "network timeout after 30s",
    "network connection reset",
  ]) {
    assert.equal(isCursorSendPoisonError(new Error(poison)), true, poison);
  }

  const sendOptionsSeen: any[] = [];
  const customTools = { tools: [] };
  const fakeRun = {
    status: "running",
    stream() {
      return {
        [Symbol.asyncIterator]() {
          return {
            next: async () => ({ done: true, value: undefined }),
          };
        },
      };
    },
    cancel: async () => undefined,
  };
  const busySession: any = {
    key: "busy-session",
    agentId: "agent-busy",
    agent: {
      async send(_prompt: unknown, options: any) {
        sendOptionsSeen.push(options);
        if (sendOptionsSeen.length === 1) {
          throw Object.assign(new Error("already has active run"), {
            name: "AgentBusyError",
          });
        }
        return fakeRun;
      },
    },
  };

  await assert.rejects(
    sendCursorPrompt(
      busySession,
      { text: "hello" } as any,
      { mode: "agent", local: { customTools } }
    ),
    /already has active run/
  );
  assert.equal(sendOptionsSeen.length, 1);
  assert.deepEqual(sendOptionsSeen[0].local, { customTools });

  let shouldNotSend = false;
  const alreadyAborted = new AbortController();
  alreadyAborted.abort();
  await assert.rejects(
    sendCursorPrompt(
      {
        key: "aborted-session",
        agentId: "agent-aborted",
        agent: {
          async send() {
            shouldNotSend = true;
            return fakeRun;
          },
        },
      } as any,
      { text: "cancelled" } as any,
      { mode: "agent" },
      { abortSignal: alreadyAborted.signal }
    ),
    /aborted before send/
  );
  assert.equal(shouldNotSend, false);

  let releaseSend!: (run: any) => void;
  let cancelledLateRun = false;
  const controller = new AbortController();
  const abortingSession: any = {
    key: "abort-during-send",
    agentId: "agent-abort",
    agent: {
      send() {
        return new Promise((resolve) => {
          releaseSend = resolve;
        });
      },
    },
  };
  const aborted = sendCursorPrompt(
    abortingSession,
    { text: "hi" } as any,
    { mode: "agent" },
    { abortSignal: controller.signal }
  );
  controller.abort();
  await assert.rejects(aborted, /aborted during send/);
  releaseSend({
    ...fakeRun,
    cancel: async () => {
      cancelledLateRun = true;
      throw new Error("late run cancellation failed");
    },
  });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(cancelledLateRun, true);

  console.log("cursor-sdk.send: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

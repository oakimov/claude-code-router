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

  const run = await sendCursorPrompt(
    busySession,
    { text: "hello" } as any,
    { mode: "agent", local: { customTools } }
  );
  assert.equal(run, fakeRun);
  assert.equal(sendOptionsSeen.length, 2);
  assert.deepEqual(sendOptionsSeen[0].local, { customTools });
  assert.deepEqual(sendOptionsSeen[1].local, { customTools, force: true });

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

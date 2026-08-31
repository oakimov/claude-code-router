import assert from "node:assert/strict";
import { Cursor } from "@cursor/sdk";
import { runCursor } from "../cursor-sdk/runner";
import { globalSessionManager } from "../cursor-sdk/session";

function fakeRun(text: string) {
  return {
    status: "running",
    usage: undefined,
    stream() {
      let index = 0;
      const messages = [
        {
          type: "assistant",
          message: {
            content: [{ type: "text", text }],
          },
        },
      ];
      return {
        [Symbol.asyncIterator]() {
          return {
            async next() {
              if (index < messages.length) {
                return { done: false, value: messages[index++] };
              }
              return { done: true, value: undefined };
            },
          };
        },
      };
    },
    cancel: async () => undefined,
  };
}

function fakeSession(input: {
  key: string;
  agentId: string;
  hasSentPrompt: boolean;
  send: (prompt: any, options: any) => Promise<any>;
}) {
  const session: any = {
    key: input.key,
    agentId: input.agentId,
    agent: {
      send: input.send,
      close: () => undefined,
    },
    mode: "bridge",
    workspaceDir: "/tmp/ccr-cursor-test",
    parked: [],
    pendingEmit: [],
    emitWaiters: [],
    pendingSdkMessages: [],
    sdkMessageWaiters: [],
    sendChain: Promise.resolve(),
    hasSentPrompt: input.hasSentPrompt,
    lastActiveAt: Date.now(),
    metrics: { customToolCalls: 0, builtinToolCallsSeen: 0 },
    notifyEmit() {
      const waiters = this.emitWaiters.splice(0, this.emitWaiters.length);
      for (const waiter of waiters) waiter();
    },
    waitForEmit() {
      return new Promise<void>((resolve) => {
        this.emitWaiters.push(resolve);
      });
    },
    enqueueSdkMessage(this: any, message: any, runToken = this.activeRunToken) {
      this.pendingSdkMessages.push({ message, runToken, source: "delta" });
      this.notifySdkMessage();
    },
    notifySdkMessage() {
      const waiters = this.sdkMessageWaiters.splice(0, this.sdkMessageWaiters.length);
      for (const waiter of waiters) waiter();
    },
    waitForSdkMessage() {
      return new Promise<void>((resolve) => {
        if (this.pendingSdkMessages.length) {
          resolve();
          return;
        }
        this.sdkMessageWaiters.push(resolve);
      });
    },
  };
  return session;
}

async function main() {
  const originalGetOrCreate = globalSessionManager.getOrCreate.bind(globalSessionManager);
  const originalInvalidate = globalSessionManager.invalidate.bind(globalSessionManager);
  const originalModelList = Cursor.models.list;

  const oldPrompts: string[] = [];
  const oldOptions: any[] = [];
  const freshPrompts: string[] = [];
  const freshOptions: any[] = [];
  const invalidated: string[] = [];

  const oldSession = fakeSession({
    key: "runner-recovery",
    agentId: "old-agent",
    hasSentPrompt: true,
    async send(prompt, options) {
      oldPrompts.push(prompt.text);
      oldOptions.push(options);
      throw Object.assign(new Error("already has active run"), {
        name: "AgentBusyError",
      });
    },
  });
  const freshSession = fakeSession({
    key: "runner-recovery",
    agentId: "fresh-agent",
    hasSentPrompt: false,
    async send(prompt, options) {
      freshPrompts.push(prompt.text);
      freshOptions.push(options);
      return fakeRun("fresh response");
    },
  });

  let getCalls = 0;
  (Cursor.models as any).list = async () => [];
  (globalSessionManager as any).getOrCreate = async () => {
    getCalls += 1;
    return getCalls === 1 ? oldSession : freshSession;
  };
  (globalSessionManager as any).invalidate = (session: any, reason: string) => {
    invalidated.push(`${session.agentId}:${reason}`);
  };

  try {
    const response = await runCursor(
      {
        model: "composer-2.5",
        stream: true,
        messages: [
          { role: "system", content: "system" },
          { role: "user", content: "first user" },
          { role: "assistant", content: "first answer" },
          { role: "user", content: "second user" },
        ],
      } as any,
      { apiKey: "crsr_test" },
      { req: { headers: { "x-ccr-cursor-session": "runner-recovery" } } },
      { cursorMode: "bridge" }
    );
    const body = await response.text();

    // Soft path: unknown alignment stays sticky and attempts incremental send
    // first. AgentBusyError then triggers hard remint + full replay.
    assert.equal(getCalls, 2);
    assert.equal(oldPrompts.length, 1);
    assert.equal(oldOptions.length, 1);
    assert.doesNotMatch(oldPrompts[0], /\[user\]\nfirst user/);
    assert.match(oldPrompts[0], /\[user\]\nsecond user/);
    assert.equal(invalidated.some((entry) => entry.startsWith("old-agent:")), true);
    assert.equal(freshPrompts.length, 1);
    assert.equal(freshOptions[0]?.local?.force, undefined);
    assert.match(freshPrompts[0], /\[user\]\nfirst user/);
    assert.match(freshPrompts[0], /\[assistant\]\nfirst answer/);
    assert.match(freshPrompts[0], /\[user\]\nsecond user/);
    assert.match(body, /fresh response/);
  } finally {
    (Cursor.models as any).list = originalModelList;
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    (globalSessionManager as any).invalidate = originalInvalidate;
  }

  console.log("cursor-sdk.runner-recovery: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

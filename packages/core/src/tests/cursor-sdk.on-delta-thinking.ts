import assert from "node:assert/strict";
import { Cursor } from "@cursor/sdk";
import { runCursor } from "../cursor-sdk/runner";
import { globalSessionManager } from "../cursor-sdk/session";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

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
    workspaceDir: "/tmp/ccr-cursor-on-delta-thinking",
    parked: [],
    pendingEmit: [],
    emitWaiters: [],
    pendingSdkMessages: [],
    sdkMessageWaiters: [],
    sendChain: Promise.resolve(),
    hasSentPrompt: false,
    lastActiveAt: Date.now(),
    metrics: { customToolCalls: 0, builtinToolCallsSeen: 0 },
    notifyEmit() {
      const waiters = this.emitWaiters.splice(0, this.emitWaiters.length);
      for (const waiter of waiters) waiter();
    },
    waitForEmit() {
      return new Promise<void>((resolve) => {
        if (this.pendingEmit.length) {
          resolve();
          return;
        }
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
  const originalModelList = Cursor.models.list;

  const session = fakeSession({
    key: "on-delta-thinking",
    agentId: "agent-on-delta-thinking",
    async send(_prompt, options) {
      await options.onDelta?.({
        update: { type: "thinking-delta", text: "first thought" },
      });
      await options.onDelta?.({
        update: { type: "thinking-delta", text: " then deeper" },
      });
      return fakeRun("final answer");
    },
  });

  (Cursor.models as any).list = async () => [];
  (globalSessionManager as any).getOrCreate = async () => session;

  try {
    const response = await runCursor(
      {
        model: "composer-2.5",
        stream: true,
        messages: [{ role: "user", content: "show thinking" }],
      } as any,
      { apiKey: "crsr_test" },
      { req: { headers: { "x-ccr-cursor-session": "on-delta-thinking" } } },
      { cursorMode: "bridge" }
    );
    const unifiedBody = await response.text();

    const firstThought = unifiedBody.indexOf('"content":"first thought"');
    const deeperThought = unifiedBody.indexOf('"content":" then deeper"');
    const signature = unifiedBody.indexOf('"signature":"ccr_cursor_');
    const finalAnswer = unifiedBody.indexOf('"content":"final answer"');

    assert.ok(firstThought >= 0);
    assert.ok(deeperThought > firstThought);
    assert.ok(signature > deeperThought);
    assert.ok(finalAnswer > signature);

    const transformer = new AnthropicTransformer();
    transformer.logger = { debug() {}, error() {} };
    const anthropicResponse = await transformer.transformResponseIn(
      new Response(unifiedBody, {
        headers: { "Content-Type": "text/event-stream" },
      }),
      { req: { id: "cursor-on-delta-thinking" } } as any
    );
    const anthropicBody = await anthropicResponse.text();

    const thinkingStart = anthropicBody.indexOf(
      '"content_block":{"type":"thinking","thinking":""}'
    );
    const thinkingDelta = anthropicBody.indexOf(
      '"delta":{"type":"thinking_delta","thinking":"first thought"}'
    );
    const signatureDelta = anthropicBody.indexOf(
      '"delta":{"type":"signature_delta","signature":"ccr_cursor_'
    );
    const textStart = anthropicBody.indexOf(
      '"content_block":{"type":"text","text":""}'
    );

    assert.ok(thinkingStart >= 0);
    assert.ok(thinkingDelta > thinkingStart);
    assert.ok(signatureDelta > thinkingDelta);
    assert.ok(textStart > signatureDelta);
    assert.match(
      anthropicBody,
      /"delta":\{"type":"text_delta","text":"final answer"\}/
    );
  } finally {
    (Cursor.models as any).list = originalModelList;
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
  }

  console.log("cursor-sdk.on-delta-thinking: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

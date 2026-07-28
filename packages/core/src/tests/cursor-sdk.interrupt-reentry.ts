import assert from "node:assert/strict";
import { Cursor } from "@cursor/sdk";
import { runCursor } from "../cursor-sdk/runner";
import { globalSessionManager } from "../cursor-sdk/session";
import {
  createCursorCompatibilityStamp,
  createCursorTranscriptCommit,
} from "../cursor-sdk/turn-identity";
import { hashSessionFingerprint } from "../cursor-sdk/shared";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

function iteratorFromMessages(messages: any[]) {
  let index = 0;
  return {
    returned: false,
    async next() {
      if (index < messages.length) {
        return { done: false, value: messages[index++] };
      }
      return { done: true, value: undefined };
    },
    async return() {
      this.returned = true;
      return { done: true, value: undefined };
    },
  };
}

function fakeRun(text: string) {
  return {
    id: `run-${text}`,
    status: "running",
    usage: undefined,
    stream() {
      return {
        [Symbol.asyncIterator]() {
          return iteratorFromMessages([
            {
              type: "assistant",
              message: { content: [{ type: "text", text }] },
            },
          ]);
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
    workspaceDir: "/tmp/ccr-cursor-interrupt-reentry",
    hostEnv: { known: false, fingerprint: "unknown" },
    parked: [],
    pendingEmit: [],
    emitWaiters: [],
    pendingSdkMessages: [],
    sdkMessageWaiters: [],
    sendChain: Promise.resolve(),
    hasSentPrompt: input.hasSentPrompt,
    lastActiveAt: Date.now(),
    metrics: {
      customToolCalls: 0,
      builtinToolCallsSeen: 0,
      scratchPathViolations: 0,
      scratchPathCorrections: 0,
    },
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
      const waiters = this.sdkMessageWaiters.splice(
        0,
        this.sdkMessageWaiters.length
      );
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

function parkTool(session: any, id: string) {
  let resolvedWith: string | undefined;
  let rejectedWith: string | undefined;
  let resolve!: (value: string) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<string>((res, rej) => {
    resolve = (value) => {
      resolvedWith = value;
      res(value);
    };
    reject = (error) => {
      rejectedWith = error.message;
      rej(error);
    };
  });
  // Avoid an unhandled rejection if the cancellation path rejects this parked call.
  void promise.catch(() => undefined);
  session.parked.push({
    id,
    name: "Bash",
    args: {},
    runToken: session.activeRunToken,
    resolve,
    reject,
    promise,
  });
  return {
    get resolvedWith() {
      return resolvedWith;
    },
    get rejectedWith() {
      return rejectedWith;
    },
  };
}

function anthropicRequest(content: any[]) {
  return {
    model: "composer-2.5",
    max_tokens: 1024,
    stream: true,
    messages: [
      {
        role: "user",
        content: "Earlier user context: preserve sentinel ALPHA-42.",
      },
      {
        role: "assistant",
        content: [
          {
            type: "text",
            text: "Earlier assistant context: acknowledged sentinel BETA-73.",
          },
        ],
      },
      {
        role: "user",
        content: "Build the project with the current dependency versions.",
      },
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: "tool-build",
            name: "Bash",
            input: { command: "npm run build" },
          },
        ],
      },
      {
        role: "user",
        content,
      },
    ],
  };
}

function markSessionAligned(session: any, request: any): void {
  let trailingStart = -1;
  for (let index = request.messages.length - 1; index >= 0; index -= 1) {
    if (request.messages[index]?.role === "assistant") {
      trailingStart = index;
      break;
    }
  }
  session.transcriptCommit = createCursorTranscriptCommit({
    ...request,
    messages: request.messages.slice(0, trailingStart + 1),
  });
  session.compatibilityStamp = createCursorCompatibilityStamp({
    credentialFingerprint: hashSessionFingerprint(["crsr_test"]),
    guidanceFingerprint: "none",
    mode: "bridge",
    model: { id: "composer-2.5" },
    sandboxEnabled: false,
    tools: request.tools,
    workspaceDir: session.workspaceDir,
  });
}

async function withDeadline<T>(promise: Promise<T>, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`${label} exceeded regression deadline`)),
          2_000
        );
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function waitFor(
  predicate: () => boolean,
  label: string
): Promise<void> {
  await withDeadline(
    (async () => {
      while (!predicate()) {
        await new Promise((resolve) => setTimeout(resolve, 1));
      }
    })(),
    label
  );
}

async function main() {
  const originalGetOrCreate =
    globalSessionManager.getOrCreate.bind(globalSessionManager);
  const originalInvalidate =
    globalSessionManager.invalidate.bind(globalSessionManager);
  const originalModelList = Cursor.models.list;
  const anthropic = new AnthropicTransformer();

  const pureToken = Symbol("pure-tool-result");
  let pureSendCalls = 0;
  let pureCancelCalls = 0;
  const pureSession = fakeSession({
    key: "pure-tool-result",
    agentId: "agent-pure",
    hasSentPrompt: true,
    async send() {
      pureSendCalls += 1;
      return fakeRun("unexpected");
    },
  });
  pureSession.activeRunToken = pureToken;
  pureSession.run = {
    id: "run-pure",
    status: "running",
    cancel: async () => {
      pureCancelCalls += 1;
    },
  };
  let pureNextCalls = 0;
  let pureMessageEmitted = false;
  pureSession.streamIterator = {
    async next() {
      pureNextCalls += 1;
      if (pureMessageEmitted) return { done: true, value: undefined };
      pureMessageEmitted = true;
      return {
        done: false,
        value: {
          type: "assistant",
          message: { content: [{ type: "text", text: "continued-old-run" }] },
        },
      };
    },
    async return() {
      return { done: true, value: undefined };
    },
  };
  pureSession.streamNext = pureSession.streamIterator.next();
  pureSession.streamNextRunToken = pureToken;
  const pureParked = parkTool(pureSession, "tool-build");

  let markerSendCalls = 0;
  let markerCancelCalls = 0;
  const markerSession = fakeSession({
    key: "marker-only",
    agentId: "agent-marker-only",
    hasSentPrompt: true,
    async send() {
      markerSendCalls += 1;
      return fakeRun("unexpected-marker-send");
    },
  });
  markerSession.activeRunToken = Symbol("marker-only-run");
  markerSession.run = {
    id: "run-marker-only",
    status: "running",
    cancel: async () => {
      markerCancelCalls += 1;
    },
  };
  markerSession.streamIterator = iteratorFromMessages([
    {
      type: "assistant",
      message: { content: [{ type: "text", text: "marker-only-old-run" }] },
    },
  ]);
  const markerParked = parkTool(markerSession, "tool-build");

  const markerDeadPrompts: string[] = [];
  const markerDeadSession = fakeSession({
    key: "marker-only-dead-run",
    agentId: "agent-marker-only-dead-run",
    hasSentPrompt: true,
    async send(prompt) {
      markerDeadPrompts.push(prompt.text);
      return fakeRun("marker-only-dead-run-recovered");
    },
  });
  markerDeadSession.activeRunToken = Symbol("marker-only-dead-run");
  const markerDeadParked = parkTool(markerDeadSession, "tool-build");
  const markerDeadFreshSession = fakeSession({
    key: "marker-only-dead-run",
    agentId: "agent-marker-only-dead-run-fresh",
    hasSentPrompt: false,
    async send(prompt) {
      markerDeadPrompts.push(prompt.text);
      return fakeRun("marker-only-dead-run-recovered");
    },
  });

  const interruptedToken = Symbol("interrupted-run");
  let interruptedCancelCalls = 0;
  let interruptedIteratorReturns = 0;
  let interruptedSendCalls = 0;
  const interruptedSession = fakeSession({
    key: "interrupted",
    agentId: "agent-interrupted",
    hasSentPrompt: true,
    async send() {
      interruptedSendCalls += 1;
      return fakeRun("unexpected-old-session-send");
    },
  });
  interruptedSession.activeRunToken = interruptedToken;
  interruptedSession.run = {
    id: "run-interrupted",
    status: "running",
    cancel: async () => {
      interruptedCancelCalls += 1;
    },
  };
  interruptedSession.streamIterator = {
    async next() {
      return new Promise(() => undefined);
    },
    async return() {
      interruptedIteratorReturns += 1;
      return { done: true, value: undefined };
    },
  };
  const interruptedParked = parkTool(interruptedSession, "tool-build");

  const freshPrompts: string[] = [];
  const freshOptions: any[] = [];
  const freshSession = fakeSession({
    key: "interrupted",
    agentId: "agent-fresh",
    hasSentPrompt: false,
    async send(prompt, options) {
      freshPrompts.push(prompt.text);
      freshOptions.push(options);
      return fakeRun("post-abort-ok");
    },
  });

  let releaseIteratorReturn!: () => void;
  const iteratorReturnGate = new Promise<void>((resolve) => {
    releaseIteratorReturn = resolve;
  });
  let releaseRunCancel!: () => void;
  const runCancelGate = new Promise<void>((resolve) => {
    releaseRunCancel = resolve;
  });
  let iteratorReturnStarted = false;
  let runCancelStarted = false;
  const cancelSession = fakeSession({
    key: "cancel-barrier",
    agentId: "agent-cancel-barrier",
    hasSentPrompt: false,
    async send() {
      let nextCalls = 0;
      return {
        id: "run-cancel-barrier",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  nextCalls += 1;
                  if (nextCalls === 1) {
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
                  iteratorReturnStarted = true;
                  await iteratorReturnGate;
                  return { done: true, value: undefined };
                },
              };
            },
          };
        },
        async cancel() {
          runCancelStarted = true;
          await runCancelGate;
        },
      };
    },
  });

  let releaseContinuationSend!: () => void;
  const continuationSendGate = new Promise<void>((resolve) => {
    releaseContinuationSend = resolve;
  });
  let progressSendCalls = 0;
  let continuationSendStarted = false;
  let continuationCancelCalls = 0;
  let continuationIteratorReturns = 0;
  const progressSession = fakeSession({
    key: "progress-cancel",
    agentId: "agent-progress-cancel",
    hasSentPrompt: false,
    async send() {
      progressSendCalls += 1;
      if (progressSendCalls === 1) {
        return fakeRun("Checking the project state.");
      }

      continuationSendStarted = true;
      await continuationSendGate;
      return {
        id: "run-progress-continuation",
        status: "running",
        usage: undefined,
        stream() {
          return {
            [Symbol.asyncIterator]() {
              return {
                async next() {
                  return new Promise(() => undefined);
                },
                async return() {
                  continuationIteratorReturns += 1;
                  return { done: true, value: undefined };
                },
              };
            },
          };
        },
        async cancel() {
          continuationCancelCalls += 1;
        },
      };
    },
  });

  const sessions = [
    pureSession,
    markerSession,
    markerDeadSession,
    markerDeadFreshSession,
    interruptedSession,
    freshSession,
    cancelSession,
    progressSession,
  ];
  const invalidated: string[] = [];
  let getCalls = 0;
  (Cursor.models as any).list = async () => [];
  (globalSessionManager as any).getOrCreate = async () => {
    const session = sessions[getCalls];
    getCalls += 1;
    if (!session) throw new Error("unexpected extra session acquisition");
    return session;
  };
  (globalSessionManager as any).invalidate = (session: any, reason: string) => {
    invalidated.push(`${session.agentId}:${reason}`);
    originalInvalidate(session, reason);
  };

  try {
    const pureUnified = await anthropic.transformRequestOut(
      anthropicRequest([
        {
          type: "tool_result",
          tool_use_id: "tool-build",
          content: "build result",
        },
      ])
    );
    markSessionAligned(pureSession, pureUnified);
    const pureResponse = await runCursor(
      pureUnified,
      { apiKey: "crsr_test" },
      { req: { headers: { "x-ccr-cursor-session": "pure" } } },
      { cursorMode: "bridge" }
    );
    const pureBody = await withDeadline(
      pureResponse.text(),
      "pure tool-result continuation"
    );

    assert.match(pureBody, /continued-old-run/);
    assert.equal(pureParked.resolvedWith, "build result");
    assert.equal(pureParked.rejectedWith, undefined);
    assert.equal(pureSendCalls, 0);
    assert.equal(pureCancelCalls, 0);
    assert.equal(
      pureNextCalls,
      2,
      "tool continuation must reuse the outstanding iterator read"
    );

    const markerContext: any = {
      req: { headers: { "x-ccr-cursor-session": "marker-only" } },
    };
    const markerUnified = await anthropic.transformRequestOut(
      anthropicRequest([
        {
          type: "tool_result",
          tool_use_id: "tool-build",
          is_error: true,
          content: "The user rejected this tool. Stop and wait.",
        },
        { type: "text", text: "[Request interrupted by user for tool use]" },
      ]),
      markerContext
    );
    markSessionAligned(markerSession, markerUnified);
    const markerResponse = await runCursor(
      markerUnified,
      { apiKey: "crsr_test" },
      markerContext,
      { cursorMode: "bridge" }
    );
    const markerBody = await withDeadline(
      markerResponse.text(),
      "marker-only tool-result continuation"
    );
    assert.match(markerBody, /marker-only-old-run/);
    assert.equal(
      markerParked.resolvedWith,
      "The user rejected this tool. Stop and wait."
    );
    assert.equal(markerParked.rejectedWith, undefined);
    assert.equal(markerSendCalls, 0);
    assert.equal(markerCancelCalls, 0);

    const markerDeadContext: any = {
      req: {
        headers: { "x-ccr-cursor-session": "marker-only-dead-run" },
      },
    };
    const markerDeadUnified = await anthropic.transformRequestOut(
      anthropicRequest([
        {
          type: "tool_result",
          tool_use_id: "tool-build",
          is_error: true,
          content: "The user rejected this dead-run tool. Stop and wait.",
        },
        { type: "text", text: "[Request interrupted by user for tool use]" },
      ]),
      markerDeadContext
    );
    markSessionAligned(markerDeadSession, markerDeadUnified);
    const markerDeadResponse = await runCursor(
      markerDeadUnified,
      { apiKey: "crsr_test" },
      markerDeadContext,
      { cursorMode: "bridge" }
    );
    const markerDeadBody = await withDeadline(
      markerDeadResponse.text(),
      "marker-only dead-run recovery"
    );
    assert.match(markerDeadBody, /marker-only-dead-run-recovered/);
    assert.equal(markerDeadParked.resolvedWith, undefined);
    assert.match(
      markerDeadParked.rejectedWith || "",
      /dead-parked-run/
    );
    assert.equal(markerDeadPrompts.length, 1);
    assert.match(
      markerDeadPrompts[0],
      /\[assistant tool_call id=tool-build name=Bash\]/
    );
    assert.match(markerDeadPrompts[0], /\[tool_result id=tool-build\]/);
    assert.match(
      markerDeadPrompts[0],
      /\[Request interrupted by user for tool use\]/
    );

    const mixedContext: any = {
      req: { headers: { "x-ccr-cursor-session": "mixed" } },
    };
    const mixedUnified = await anthropic.transformRequestOut(
      anthropicRequest([
        {
          type: "tool_result",
          tool_use_id: "tool-build",
          is_error: true,
          content: "The user rejected this tool. Stop and wait.",
        },
        { type: "text", text: "[Request interrupted by user for tool use]" },
        {
          type: "text",
          text: "Do not downgrade. Use only current versions.",
        },
      ]),
      mixedContext
    );
    assert.deepEqual(
      mixedUnified.messages.map((message) => message.role),
      ["user", "assistant", "user", "assistant", "tool", "user"]
    );

    const mixedResponse = await runCursor(
      mixedUnified,
      { apiKey: "crsr_test" },
      mixedContext,
      { cursorMode: "bridge" }
    );
    const mixedBody = await withDeadline(
      mixedResponse.text(),
      "interrupt replacement turn"
    );

    assert.equal(interruptedParked.resolvedWith, undefined);
    assert.match(
      interruptedParked.rejectedWith || "",
      /parked-turn-has-steering|unknown-context-alignment/
    );
    assert.equal(interruptedIteratorReturns, 1);
    assert.equal(interruptedCancelCalls, 1);
    assert.equal(interruptedSendCalls, 0);
    assert.equal(freshPrompts.length, 1);
    assert.match(freshPrompts[0], /preserve sentinel ALPHA-42/);
    assert.match(freshPrompts[0], /acknowledged sentinel BETA-73/);
    assert.match(freshPrompts[0], /Do not downgrade\. Use only current versions\./);
    assert.match(freshPrompts[0], /\[tool_result id=tool-build\]/);
    assert.equal(freshOptions[0]?.local?.force, undefined);
    assert.match(mixedBody, /post-abort-ok/);
    assert.equal(
      invalidated.some((entry) => entry.startsWith("agent-interrupted:")),
      true
    );

    const cancelResponse = await runCursor(
      {
        model: "composer-2.5",
        stream: true,
        messages: [{ role: "user", content: "stream until cancelled" }],
      } as any,
      { apiKey: "crsr_test" },
      { req: { headers: { "x-ccr-cursor-session": "cancel-barrier" } } },
      { cursorMode: "bridge" }
    );
    assert.ok(cancelResponse.body);
    const cancelReader = cancelResponse.body.getReader();
    const firstChunk = await withDeadline(
      cancelReader.read(),
      "first cancellable stream chunk"
    );
    assert.equal(firstChunk.done, false);

    let cancelSettled = false;
    const cancelPromise = cancelReader.cancel("client interrupted").then(() => {
      cancelSettled = true;
    });
    await waitFor(
      () => iteratorReturnStarted,
      "stream cancellation to enter iterator cleanup"
    );
    assert.equal(cancelSession.poisoned, true);
    assert.equal(cancelSettled, false);

    releaseIteratorReturn();
    await waitFor(() => runCancelStarted, "stream cancellation to reach run.cancel");
    assert.equal(cancelSettled, false);

    releaseRunCancel();
    await withDeadline(cancelPromise, "stream cancellation teardown barrier");
    assert.equal(cancelSettled, true);
    assert.equal(
      invalidated.some((entry) => entry.startsWith("agent-cancel-barrier:")),
      true
    );

    const progressResponse = await runCursor(
      {
        model: "composer-2.5",
        stream: true,
        messages: [{ role: "user", content: "inspect the project" }],
      } as any,
      { apiKey: "crsr_test" },
      { req: { headers: { "x-ccr-cursor-session": "progress-cancel" } } },
      { cursorMode: "bridge" }
    );
    assert.ok(progressResponse.body);
    const progressReader = progressResponse.body.getReader();
    const progressChunk = await withDeadline(
      progressReader.read(),
      "progress-only first chunk"
    );
    assert.equal(progressChunk.done, false);
    await waitFor(
      () => continuationSendStarted,
      "progress-only continuation send to start"
    );

    let progressCancelSettled = false;
    const progressCancel = progressReader.cancel("cancel continuation").then(() => {
      progressCancelSettled = true;
    });
    await waitFor(
      () => progressSession.poisoned === true,
      "progress continuation retirement to claim ownership"
    );
    assert.equal(progressCancelSettled, false);

    releaseContinuationSend();
    await withDeadline(
      progressCancel,
      "progress continuation cancellation cleanup"
    );
    assert.equal(progressCancelSettled, true);
    assert.equal(continuationCancelCalls, 1);
    assert.equal(
      continuationIteratorReturns,
      0,
      "an aborted pending send is cancelled before its iterator is acquired"
    );
    assert.equal(progressSession.run, undefined);
    assert.equal(progressSession.streamIterator, undefined);
    assert.equal(progressSession.activeRunToken, undefined);
    assert.equal(getCalls, 8);
  } finally {
    (Cursor.models as any).list = originalModelList;
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    (globalSessionManager as any).invalidate = originalInvalidate;
  }

  console.log("cursor-sdk.interrupt-reentry: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

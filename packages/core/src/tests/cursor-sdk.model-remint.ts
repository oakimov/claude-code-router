import "./support/isolate-session-registry";
import assert from "node:assert/strict";
import { Cursor } from "@cursor/sdk";
import { runCursor } from "../cursor-sdk/runner";
import { globalSessionManager } from "../cursor-sdk/session";

/**
 * A live session bound to a parameterized selection is reminted only when
 * the catalog says the selection changed — never because the catalog could
 * not be read.
 */

const LOW = { id: "grok-4.7", params: [{ id: "reasoning_effort", value: "low" }] };

const CATALOG = [
  {
    id: "grok-4.7",
    displayName: "Grok 4.7",
    parameters: [
      {
        id: "reasoning_effort",
        values: [{ value: "low" }, { value: "high" }],
      },
    ],
    variants: [
      {
        displayName: "Grok 4.7 Low",
        isDefault: true,
        params: [{ id: "reasoning_effort", value: "low" }],
      },
      {
        displayName: "Grok 4.7 High",
        params: [{ id: "reasoning_effort", value: "high" }],
      },
    ],
  },
];

function fakeRun(text: string) {
  return {
    status: "running",
    usage: undefined,
    stream() {
      let done = false;
      return {
        [Symbol.asyncIterator]() {
          return {
            async next() {
              if (done) return { done: true, value: undefined };
              done = true;
              return {
                done: false,
                value: { type: "assistant", message: { content: [{ type: "text", text }] } },
              };
            },
          };
        },
      };
    },
    cancel: async () => undefined,
  };
}

function fakeSession(agentId: string, model: any, sends: any[]) {
  const session: any = {
    key: "model-remint",
    agentId,
    agent: {
      model,
      async send(prompt: any, options: any) {
        sends.push({ agentId, prompt: prompt.text, options });
        return fakeRun(`reply from ${agentId}`);
      },
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
    hasSentPrompt: false,
    lastActiveAt: Date.now(),
    metrics: { customToolCalls: 0, builtinToolCallsSeen: 0 },
    notifyEmit() {
      for (const waiter of this.emitWaiters.splice(0)) waiter();
    },
    waitForEmit() {
      return new Promise<void>((resolve) => this.emitWaiters.push(resolve));
    },
    enqueueSdkMessage(this: any, message: any, runToken = this.activeRunToken) {
      this.pendingSdkMessages.push({ message, runToken, source: "delta" });
      this.notifySdkMessage();
    },
    notifySdkMessage() {
      for (const waiter of this.sdkMessageWaiters.splice(0)) waiter();
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

async function run(text: string, extra: Record<string, unknown> = {}) {
  const response = await runCursor(
    {
      model: "grok-4.7",
      stream: true,
      messages: [
        { role: "system", content: "system" },
        { role: "user", content: text },
      ],
      ...extra,
    } as any,
    { apiKey: "crsr_remint" },
    { req: { sessionId: `remint-${text}` } },
    { cursorMode: "bridge" }
  );
  return response.text();
}

async function main() {
  const originalGetOrCreate = globalSessionManager.getOrCreate.bind(globalSessionManager);
  const originalRetire = globalSessionManager.retireSession.bind(globalSessionManager);
  const originalList = Cursor.models.list;
  const realNow = Date.now;

  const retired: string[] = [];
  const sends: any[] = [];
  const requested: any[] = [];
  let created = 0;
  let prepare: ((session: any) => void) | undefined;
  (globalSessionManager as any).getOrCreate = async (options: any) => {
    requested.push(options.model);
    created += 1;
    // First session of each run is the live one, bound to the low variant.
    const session = fakeSession(`agent-${created}`, LOW, sends);
    prepare?.(session);
    prepare = undefined;
    return session;
  };
  (globalSessionManager as any).retireSession = async (
    session: any,
    reason: string
  ) => {
    retired.push(`${session.agentId}:${reason}`);
    return true;
  };

  try {
    // 1. No catalog ever: the bare id is not a selection change.
    (Cursor.models as any).list = async () => {
      throw new Error("catalog unavailable");
    };
    let body = await run("cold catalog");
    assert.match(body, /reply from agent-1/);
    assert.deepEqual(retired, []);
    assert.equal(sends.length, 1);
    assert.equal(sends[0].options.model, undefined);

    // 2. Catalog loads: the least-reasoning variant equals the bound one.
    (Cursor.models as any).list = async () => CATALOG;
    body = await run("warm catalog");
    assert.match(body, /reply from agent-2/);
    assert.deepEqual(retired, []);

    // 3. A failed refresh after the TTL keeps serving the cached catalog.
    (Cursor.models as any).list = async () => {
      throw new Error("refresh failed");
    };
    const base = realNow();
    Date.now = () => base + 16 * 60_000;
    body = await run("stale catalog");
    Date.now = realNow;
    assert.match(body, /reply from agent-3/);
    assert.deepEqual(retired, []);
    // New agents still get catalog params, not a bare id.
    assert.deepEqual(requested.at(-1), LOW);

    // 4. A real change (reasoning requested) on an idle session still remints.
    (Cursor.models as any).list = async () => CATALOG;
    body = await run("real change", { reasoning: { effort: "high" } });
    assert.equal(retired.length, 1);
    assert.match(retired[0], /^agent-4:.*model selection changed/);
    assert.deepEqual(sends.at(-1).options.model, {
      id: "grok-4.7",
      params: [{ id: "reasoning_effort", value: "high" }],
    });
    assert.match(body, /reply from agent-5/);

    // 5. The same real change on an in-flight session keeps it: retiring
    //    would cancel the live run. The bound selection serves this turn.
    let priorRunCancelled = false;
    prepare = (session) => {
      session.hasSentPrompt = true;
      session.activeRunToken = Symbol("prior-run");
      session.run = {
        id: "prior-run",
        status: "running",
        cancel: async () => {
          priorRunCancelled = true;
        },
      };
    };
    const sendsBefore = sends.length;
    body = await run("in flight change", { reasoning: { effort: "high" } });
    assert.equal(retired.length, 1);
    assert.match(body, /reply from agent-6/);
    assert.equal(created, 6);
    assert.equal(sends.length, sendsBefore + 1);
    assert.equal(sends.at(-1).agentId, "agent-6");
    assert.equal(sends.at(-1).options.model, undefined);
    // The new turn still supersedes the prior run on the same agent.
    assert.equal(priorRunCancelled, true);

    // 6. A registry-model rejection on an agent created for this request is
    //    final: a fresh agent would get the same selection.
    const invalid = () =>
      Object.assign(
        new Error('AI Model Not Found Invalid parameters for registry model: "grok-4.7"'),
        { name: "ConfigurationError" }
      );
    prepare = (session) => {
      session.agent.send = async () => {
        throw invalid();
      };
    };
    const createdBefore = created;
    await assert.rejects(run("rejected fresh"), /registry model/);
    assert.equal(created, createdBefore + 1);

    // 7. A resumed agent can carry a stale binding: one fresh replay.
    prepare = (session) => {
      session.resumedAgent = true;
      session.agent.send = async () => {
        throw invalid();
      };
    };
    body = await run("rejected resumed");
    assert.equal(created, createdBefore + 3);
    assert.match(body, new RegExp(`reply from agent-${createdBefore + 3}`));
  } finally {
    Date.now = realNow;
    (Cursor.models as any).list = originalList;
    (globalSessionManager as any).getOrCreate = originalGetOrCreate;
    (globalSessionManager as any).retireSession = originalRetire;
  }

  console.log("cursor-sdk.model-remint: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

import assert from "node:assert/strict";
import { toCustomTools } from "../cursor-sdk/tools";
import {
  SessionManager,
  aliasHostToolId,
  resolveHostToolId,
} from "../cursor-sdk/session";

/**
 * `SDKCustomToolContext.toolCallId` arrives as the upstream OpenAI-format call
 * id and the SDK's own tracking id joined by a newline. Emitted unchanged it
 * becomes a `tool_use.id` that Anthropic rejects with
 * `invalid_request_error: String should match pattern '^[a-zA-Z0-9_-]+$'`.
 */
const CURSOR_CONCATENATED_ID =
  "call-901b1ddc-d889-4a6e-8c58-564ad17bc095-3\nfc_b466705e-df33-9395-8d4a-21a95066affe_0";

const ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

function fakeSession() {
  const parked: Array<{ id: string; name: string }> = [];
  const pendingEmit: Array<{ id: string; name: string }> = [];
  const session: any = {
    key: "test",
    workspaceDir: "/tmp/scratch",
    hostEnv: undefined,
    metrics: {
      customToolCalls: 0,
      builtinToolCallsSeen: 0,
      scratchPathViolations: 0,
      scratchPathCorrections: 0,
    },
    toolIdAliases: { byOriginal: new Map(), byAlias: new Map() },
    parked,
    pendingEmit,
    notifyEmit() {},
    // Mirrors the production parkHostTool split: park the original SDK id,
    // emit the host-safe alias.
    parkHostTool: (tool: any) => {
      parked.push({ id: tool.id, name: tool.name });
      pendingEmit.push({
        id: aliasHostToolId(session, tool.id) ?? tool.id,
        name: tool.name,
      });
      return Promise.resolve("host result");
    },
  };
  return session;
}

async function main() {
  const session = fakeSession();
  const tools = toCustomTools(
    {
      messages: [],
      tools: [{ function: { name: "Bash", description: "d" } }],
    } as any,
    session
  );

  await tools.Bash.execute({ command: "ls" } as any, {
    toolCallId: CURSOR_CONCATENATED_ID,
  } as any);

  // Parked keeps the original; the emitted alias is wire-safe.
  assert.equal(session.parked[0].id, CURSOR_CONCATENATED_ID);
  const emittedId = session.pendingEmit[0].id;
  assert.match(emittedId, ID_PATTERN);
  assert.doesNotMatch(emittedId, /\n/);
  assert.ok(emittedId.length <= 64, `length: ${emittedId.length}`);
  assert.equal(resolveHostToolId(session, emittedId), CURSOR_CONCATENATED_ID);

  // A conforming id must survive untouched — the id is the join key used to
  // resolve the parked tool when the host returns its result.
  await tools.Bash.execute({ command: "ls" } as any, {
    toolCallId: "call_abc-123",
  } as any);
  assert.equal(session.parked[1].id, "call_abc-123");
  assert.equal(session.pendingEmit[1].id, "call_abc-123");

  // Missing id still falls back to a generated one.
  await tools.Bash.execute({ command: "ls" } as any, {} as any);
  assert.match(session.parked[2].id, ID_PATTERN);

  // An id consisting only of invalid characters still yields an alias.
  await tools.Bash.execute({ command: "ls" } as any, {
    toolCallId: "\n\n",
  } as any);
  assert.match(session.pendingEmit[3].id, ID_PATTERN);
  assert.ok(session.pendingEmit[3].id.length > 0);

  await toolCallStillResolvesEndToEnd();

  console.log("cursor-sdk.tool-call-id: ok");
}

/**
 * The point of the alias is not just to stop the 400 — the tool call must
 * still work. The result travels back to the SDK through the parked closure,
 * so the emitted alias must translate back to the parked original when the
 * client echoes it.
 */
async function toolCallStillResolvesEndToEnd() {
  const parked: any[] = [];
  const pendingEmit: any[] = [];
  const session: any = {
    key: "roundtrip",
    workspaceDir: "/tmp/scratch",
    hostEnv: undefined,
    metrics: {
      customToolCalls: 0,
      builtinToolCallsSeen: 0,
      scratchPathViolations: 0,
      scratchPathCorrections: 0,
    },
    toolIdAliases: { byOriginal: new Map(), byAlias: new Map() },
    parked,
    pendingEmit,
    notifyEmit() {},
    parkHostTool({ id, name, args }: any) {
      let resolve!: (value: string) => void;
      const promise = new Promise<string>((res) => {
        resolve = res;
      });
      const alias = aliasHostToolId(session, id) ?? id;
      parked.push({ id, name, args, resolve, reject() {}, promise });
      pendingEmit.push({ id: alias, name, args });
      return promise;
    },
  };

  const tools = toCustomTools(
    { messages: [], tools: [{ function: { name: "Bash" } }] } as any,
    session
  );

  // 1. Cursor invokes the tool with its newline-joined id.
  const sdkPromise = tools.Bash.execute({ command: "ls" } as any, {
    toolCallId: CURSOR_CONCATENATED_ID,
  } as any);

  // 2. The client sees the alias, never the raw id.
  const emittedId = session.pendingEmit[0].id;
  assert.match(emittedId, ID_PATTERN);
  assert.ok(emittedId.length <= 64);

  // 3. The client echoes the alias back; translate before matching.
  const echoed = resolveHostToolId(session, emittedId);
  assert.equal(echoed, CURSOR_CONCATENATED_ID);
  const manager = new SessionManager({ warn() {}, debug() {} });
  const resolved = manager.resolveParkedTools(session, [
    { toolCallId: echoed, content: "host output" },
  ]);
  assert.equal(resolved, 1, "translated id must match the parked tool");

  // 4. The SDK's awaited promise settles with the host's result.
  const outcome = await Promise.race([
    sdkPromise,
    new Promise((res) => setTimeout(() => res("<<TIMEOUT>>"), 200)),
  ]);
  assert.equal(
    outcome,
    "host output",
    "the Cursor tool call must receive the host result"
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

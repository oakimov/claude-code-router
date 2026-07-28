import assert from "node:assert/strict";
import { toCustomTools } from "../cursor-sdk/tools";
import { SessionManager } from "../cursor-sdk/session";
import { sanitizeToolCallId } from "../utils/toolCallId";

/**
 * `SDKCustomToolContext.toolCallId` arrives as the upstream OpenAI-format call
 * id and the SDK's own tracking id joined by a newline. Emitted unchanged it
 * becomes a `tool_use.id` that Anthropic rejects with
 * `invalid_request_error: String should match pattern '^[a-zA-Z0-9_-]+$'`.
 */
const CURSOR_CONCATENATED_ID =
  "call-901b1ddc-d889-4a6e-8c58-564ad17bc095-3\nfc_b466705e-df33-9395-8d4a-21a95066affe_0";

function fakeSession() {
  const parked: Array<{ id: string; name: string }> = [];
  return {
    key: "test",
    workspaceDir: "/tmp/scratch",
    hostEnv: undefined,
    metrics: {
      customToolCalls: 0,
      builtinToolCallsSeen: 0,
      scratchPathViolations: 0,
      scratchPathCorrections: 0,
    },
    parked,
    parkHostTool: (tool: any) => {
      parked.push({ id: tool.id, name: tool.name });
      return Promise.resolve("host result");
    },
  } as any;
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

  const parkedId = session.parked[0].id;
  assert.match(
    parkedId,
    /^[a-zA-Z0-9_-]+$/,
    "parked tool id must satisfy Anthropic's tool_use.id pattern"
  );
  assert.doesNotMatch(parkedId, /\n/);

  // A conforming id must survive untouched — the id is the join key used to
  // resolve the parked tool when the host returns its result.
  await tools.Bash.execute({ command: "ls" } as any, {
    toolCallId: "call_abc-123",
  } as any);
  assert.equal(session.parked[1].id, "call_abc-123");

  // Missing id still falls back to a generated one.
  await tools.Bash.execute({ command: "ls" } as any, {} as any);
  assert.match(session.parked[2].id, /^[a-zA-Z0-9_-]+$/);

  // An id consisting only of invalid characters must not become "".
  await tools.Bash.execute({ command: "ls" } as any, {
    toolCallId: "\n\n",
  } as any);
  assert.match(session.parked[3].id, /^[a-zA-Z0-9_-]+$/);
  assert.ok(session.parked[3].id.length > 0);

  await toolCallStillResolvesEndToEnd();

  console.log("cursor-sdk.tool-call-id: ok");
}

/**
 * The point of sanitizing is not just to stop the 400 — the tool call must
 * still work. The result travels back to the SDK through the parked closure,
 * never through the id, so rewriting the id is safe as long as the id we emit
 * is the id we match on when the client echoes it back.
 */
async function toolCallStillResolvesEndToEnd() {
  const parked: any[] = [];
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
    parked,
    pendingEmit: [],
    notifyEmit() {},
    parkHostTool({ id, name, args }: any) {
      let resolve!: (value: string) => void;
      const promise = new Promise<string>((res) => {
        resolve = res;
      });
      parked.push({ id, name, args, resolve, reject() {}, promise });
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

  // 2. The parked id is emitted as tool_calls[].id, then passes through the
  //    Anthropic transformer, which sanitizes again on the way to the client.
  //    If that second pass is not idempotent, the client is told an id we can
  //    no longer match and the tool call hangs forever.
  const parkedId = session.parked[0].id;
  assert.match(parkedId, /^[a-zA-Z0-9_-]+$/);
  const idSeenByClient = sanitizeToolCallId(parkedId);
  assert.equal(
    idSeenByClient,
    parkedId,
    "the id the client sees must equal the id we match on"
  );

  // 3. The client echoes that id back as tool_result.tool_use_id.
  const manager = new SessionManager({ warn() {}, debug() {} });
  const resolved = manager.resolveParkedTools(session, [
    { toolCallId: idSeenByClient!, content: "host output" },
  ]);
  assert.equal(resolved, 1, "sanitized id must still match the parked tool");

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

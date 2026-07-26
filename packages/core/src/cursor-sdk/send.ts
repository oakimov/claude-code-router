import type { SDKUserMessage } from "@cursor/sdk";
import {
  type CursorSdkSession,
  withTimeout,
} from "./session";

export const CURSOR_SEND_TIMEOUT_MESSAGE = "Cursor SDK agent.send timed out";

export function isCursorAgentBusyError(err: unknown): boolean {
  const e = (err || {}) as { name?: unknown; message?: unknown; status?: unknown };
  const name = typeof e.name === "string" ? e.name : "";
  const message = typeof e.message === "string" ? e.message : String(err);
  return (
    name === "AgentBusyError" ||
    e.status === 409 ||
    /already has active run|active run|agent busy/i.test(message)
  );
}

export function isCursorSendPoisonError(err: unknown): boolean {
  const e = (err || {}) as { name?: unknown; message?: unknown; status?: unknown };
  const name = typeof e.name === "string" ? e.name : "";
  const message = typeof e.message === "string" ? e.message : String(err);
  return (
    name === "AgentNotFoundError" ||
    name === "NetworkError" ||
    e.status === 503 ||
    e.status === 504 ||
    /agent not found|request aborted (?:before|during) send|agent\.send timed out|network/i.test(
      message
    )
  );
}

function withLocalForce(sendOptions: Record<string, any>): Record<string, any> {
  return {
    ...sendOptions,
    local: {
      ...(sendOptions.local || {}),
      force: true,
    },
  };
}

function abortRace(abortSignal?: AbortSignal):
  | { promise: Promise<never>; cleanup: () => void }
  | undefined {
  if (!abortSignal) return undefined;
  let onAbort: (() => void) | undefined;
  const promise = new Promise<never>((_, reject) => {
    if (abortSignal.aborted) {
      reject(new Error("cursor-sdk request aborted before send"));
      return;
    }
    onAbort = () => reject(new Error("cursor-sdk request aborted during send"));
    abortSignal.addEventListener("abort", onAbort, { once: true });
  });
  return {
    promise,
    cleanup: () => {
      if (onAbort) abortSignal.removeEventListener("abort", onAbort);
    },
  };
}

async function sendAttempt(
  session: CursorSdkSession,
  prompt: SDKUserMessage,
  sendOptions: Record<string, any>,
  abortSignal?: AbortSignal
) {
  let sendPromise: ReturnType<typeof session.agent.send> | undefined;
  let abort:
    | { promise: Promise<never>; cleanup: () => void }
    | undefined;
  try {
    if (abortSignal?.aborted) {
      throw new Error("cursor-sdk request aborted before send");
    }
    sendPromise = session.agent.send(prompt, sendOptions as any);
    abort = abortRace(abortSignal);
    return await withTimeout(
      abort ? Promise.race([sendPromise, abort.promise]) : sendPromise,
      15_000,
      CURSOR_SEND_TIMEOUT_MESSAGE
    );
  } catch (err) {
    if (sendPromise) {
      void sendPromise
        .then((run) => {
          try {
            void run.cancel();
          } catch {
            // ignore
          }
        })
        .catch(() => undefined);
    }
    throw err;
  } finally {
    abort?.cleanup();
  }
}

export async function sendCursorPrompt(
  session: CursorSdkSession,
  prompt: SDKUserMessage,
  sendOptions: Record<string, any>,
  options: {
    abortSignal?: AbortSignal;
    logger?: any;
  } = {}
) {
  try {
    return await sendAttempt(session, prompt, sendOptions, options.abortSignal);
  } catch (err) {
    if (options.abortSignal?.aborted || !isCursorAgentBusyError(err)) {
      throw err;
    }
    options.logger?.warn?.(
      { err, sessionKey: session.key, agentId: session.agentId },
      "cursor-sdk agent busy; retrying send with local.force"
    );
    return sendAttempt(
      session,
      prompt,
      withLocalForce(sendOptions),
      options.abortSignal
    );
  }
}

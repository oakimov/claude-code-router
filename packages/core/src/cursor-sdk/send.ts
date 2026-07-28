import type { SDKUserMessage } from "@cursor/sdk";
import {
  type CursorSdkSession,
  withTimeout,
} from "./session";

export const CURSOR_SEND_TIMEOUT_MESSAGE = "Cursor SDK agent.send timed out";

export function isCursorAgentBusyError(err: unknown): boolean {
  const e = (err || {}) as { name?: unknown; message?: unknown; status?: unknown };
  const name = typeof e.name === "string" ? e.name : "";
  const message = (
    typeof e.message === "string" ? e.message : String(err)
  ).toLowerCase();
  return (
    name === "AgentBusyError" ||
    e.status === 409 ||
    // Anchored to the SDK's phrasing. A bare "active run" also matches
    // "no active run" and "active runtime …", which are not busy signals.
    message.includes("already has active run") ||
    message.includes("agent busy")
  );
}

export function isCursorSendPoisonError(err: unknown): boolean {
  const e = (err || {}) as { name?: unknown; message?: unknown; status?: unknown };
  const name = typeof e.name === "string" ? e.name : "";
  const message = (
    typeof e.message === "string" ? e.message : String(err)
  ).toLowerCase();
  return (
    name === "AgentNotFoundError" ||
    name === "NetworkError" ||
    e.status === 503 ||
    e.status === 504 ||
    message.includes("agent not found") ||
    message.includes("request aborted before send") ||
    message.includes("request aborted during send") ||
    message.includes("agent.send timed out") ||
    // Only genuine transport failures. A bare "network" also matches messages
    // like "Failed to access network filesystem path", which is not poison.
    message.includes("network error") ||
    message.includes("network request failed") ||
    message.includes("network timeout") ||
    message.includes("network connection")
  );
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
        // A timed-out/aborted send may still resolve with a live run. Chain its
        // cancellation so both synchronous throws and async rejections are
        // observed instead of becoming process-level unhandled rejections.
        .then((run) => run.cancel())
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
  return sendAttempt(session, prompt, sendOptions, options.abortSignal);
}

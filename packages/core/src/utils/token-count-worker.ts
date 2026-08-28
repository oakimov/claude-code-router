/**
 * Off-main-thread token counting for large prompts so one request cannot
 * starve concurrent SSE streams on the event loop.
 */
import { Worker } from "node:worker_threads";
import { cpus } from "node:os";

export type TokenCountWorkerRequest = {
  messages: unknown;
  system: unknown;
  tools: unknown;
};

type Pending = {
  resolve: (n: number) => void;
  reject: (err: Error) => void;
};

/** Offload when serialized payload exceeds this many characters. */
export const TOKEN_COUNT_WORKER_THRESHOLD_CHARS = 200_000;

const MAX_WORKERS = Math.max(1, Math.min(2, (cpus().length || 2) - 1));

let nextId = 1;
const pending = new Map<number, Pending>();
const workers: Worker[] = [];
let rr = 0;

const WORKER_SOURCE = `
const { parentPort } = require("worker_threads");
const { get_encoding } = require("tiktoken");
const enc = get_encoding("cl100k_base");
function encodeSafe(text) {
  try { return enc.encode(text); }
  catch { return enc.encode(String(text).replace(/[^\\u0000-\\uFFFF]/g, "")); }
}
function countTokens(messages, system, tools) {
  let tokenCount = 0;
  if (Array.isArray(messages)) {
    for (const message of messages) {
      if (typeof message?.content === "string") {
        tokenCount += encodeSafe(message.content).length;
      } else if (Array.isArray(message?.content)) {
        for (const part of message.content) {
          if (part?.type === "text") tokenCount += encodeSafe(part.text || "").length;
          else if (part?.type === "tool_use") tokenCount += encodeSafe(JSON.stringify(part.input)).length;
          else if (part?.type === "tool_result") {
            tokenCount += encodeSafe(typeof part.content === "string" ? part.content : JSON.stringify(part.content)).length;
          }
        }
      }
    }
  }
  if (typeof system === "string") tokenCount += encodeSafe(system).length;
  else if (Array.isArray(system)) {
    for (const item of system) {
      if (item?.type !== "text") continue;
      if (typeof item.text === "string") tokenCount += encodeSafe(item.text).length;
      else if (Array.isArray(item.text)) {
        for (const textPart of item.text) tokenCount += encodeSafe(textPart || "").length;
      }
    }
  }
  if (Array.isArray(tools)) {
    for (const tool of tools) {
      if (tool?.description) tokenCount += encodeSafe(String(tool.name || "") + String(tool.description)).length;
      if (tool?.input_schema) tokenCount += encodeSafe(JSON.stringify(tool.input_schema)).length;
    }
  }
  return tokenCount;
}
parentPort.on("message", (msg) => {
  try {
    parentPort.postMessage({ id: msg.id, tokenCount: countTokens(msg.messages, msg.system, msg.tools) });
  } catch (error) {
    parentPort.postMessage({ id: msg.id, error: error?.message || String(error) });
  }
});
`;

function ensureWorker(): Worker {
  if (workers.length < MAX_WORKERS) {
    const worker = new Worker(WORKER_SOURCE, { eval: true });
    worker.on(
      "message",
      (msg: { id: number; tokenCount?: number; error?: string }) => {
        const slot = pending.get(msg.id);
        if (!slot) return;
        pending.delete(msg.id);
        if (msg.error) slot.reject(new Error(msg.error));
        else slot.resolve(msg.tokenCount ?? 0);
      }
    );
    worker.on("error", (err) => {
      for (const [id, slot] of pending) {
        pending.delete(id);
        slot.reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
    workers.push(worker);
    return worker;
  }
  const worker = workers[rr % workers.length]!;
  rr += 1;
  return worker;
}

export function estimateTokenizePayloadChars(
  messages: unknown,
  system: unknown,
  tools: unknown
): number {
  let n = 0;
  const add = (v: unknown) => {
    if (typeof v === "string") n += v.length;
    else if (Array.isArray(v)) {
      for (const item of v) add(item);
    } else if (v && typeof v === "object") {
      for (const val of Object.values(v as Record<string, unknown>)) add(val);
    }
  };
  add(messages);
  add(system);
  add(tools);
  return n;
}

/** Conservative under-estimate: ~4 chars/token. Safe for "below threshold" skips. */
export function estimateTokensFromChars(charCount: number): number {
  return Math.ceil(charCount / 4);
}

export function countTokensInWorker(
  request: TokenCountWorkerRequest
): Promise<number> {
  const id = nextId++;
  const worker = ensureWorker();
  return new Promise<number>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, ...request });
  });
}

export async function closeTokenCountWorkers(): Promise<void> {
  const closing = workers.splice(0, workers.length);
  await Promise.all(
    closing.map((w) =>
      w.terminate().catch(() => {
        /* ignore */
      })
    )
  );
  pending.clear();
}

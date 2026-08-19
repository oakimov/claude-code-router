import { AsyncLocalStorage } from "node:async_hooks";
import { sanitizeHeadersForLog } from "@caeliq/llms";
import type { CapturedLlmExchange, TokenUsage } from "./types";

export interface LlmCaptureStore {
  last?: CapturedLlmExchange;
  patchRequestBody?: (body: Record<string, unknown>, url: string) => Record<string, unknown>;
  captureBody?: Promise<string>;
  signal?: AbortSignal;
}

const MAX_CAPTURE_BODY_BYTES = 10 * 1024 * 1024;

export const llmCaptureAls = new AsyncLocalStorage<LlmCaptureStore>();

const SKIP_HOST_HINTS = [
  "oauth2.googleapis.com",
  "accounts.google.com",
  "platform.claude.com",
  "auth0.openai.com",
  "qwen.aikit.club",
];

function headersFromFetchArgs(
  input: RequestInfo | URL,
  init?: RequestInit
): Headers | Record<string, unknown> | undefined {
  if (init?.headers) return init.headers as Headers | Record<string, unknown>;
  if (input instanceof Request) return input.headers;
  return undefined;
}

function methodFromFetchArgs(input: RequestInfo | URL, init?: RequestInit): string {
  if (init?.method) return init.method;
  if (input instanceof Request) return input.method;
  return "GET";
}

function urlFromFetchArgs(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  if (input instanceof Request) return input.url;
  return String(input);
}

function bodyFromFetchArgs(input: RequestInfo | URL, init?: RequestInit): unknown {
  const raw = init?.body;
  if (raw == null) {
    return input instanceof Request ? undefined : undefined;
  }
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw);
    } catch {
      return raw;
    }
  }
  return "<stream body not captured>";
}

function shouldCapture(url: string, method: string): boolean {
  if (method.toUpperCase() !== "POST") return false;
  const lower = url.toLowerCase();
  if (SKIP_HOST_HINTS.some((hint) => lower.includes(hint))) return false;
  return (
    lower.includes("/v1/") ||
    lower.includes("/chat/completions") ||
    lower.includes("/messages") ||
    lower.includes("/responses")
  );
}

function isStreamingResponse(res: Response, requestBody: unknown): boolean {
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("text/event-stream")) return true;
  if (requestBody && typeof requestBody === "object" && (requestBody as any).stream === true) {
    return true;
  }
  return false;
}

function mergeUsage(a?: TokenUsage, b?: TokenUsage): TokenUsage | undefined {
  if (!a) return b;
  if (!b) return a;
  return {
    input: b.input ?? a.input,
    output: b.output ?? a.output,
    total: b.total ?? a.total,
    cacheRead: b.cacheRead ?? a.cacheRead,
    cacheWrite: b.cacheWrite ?? a.cacheWrite,
    reasoning: b.reasoning ?? a.reasoning,
  };
}

export function parseTokenUsage(body: unknown): TokenUsage | undefined {
  if (!body || typeof body !== "object") return undefined;
  const raw = body as any;
  const usage =
    raw.usage ??
    raw.response?.usage ??
    raw.message?.usage ??
    raw.usageMetadata;
  if (!usage || typeof usage !== "object") return undefined;
  const input =
    usage.prompt_tokens ??
    usage.input_tokens ??
    usage.promptTokenCount;
  const output =
    usage.completion_tokens ??
    usage.output_tokens ??
    usage.candidatesTokenCount;
  const total =
    usage.total_tokens ??
    usage.totalTokenCount ??
    (typeof input === "number" && typeof output === "number" ? input + output : undefined);
  const cacheRead =
    usage.prompt_tokens_details?.cached_tokens ??
    usage.input_tokens_details?.cached_tokens ??
    usage.cache_read_input_tokens ??
    usage.cached_tokens ??
    usage.cachedContentTokenCount;
  const cacheWrite =
    usage.prompt_tokens_details?.cache_write_tokens ??
    usage.input_tokens_details?.cache_write_tokens ??
    usage.cache_creation_input_tokens ??
    usage.cache_write_tokens;
  const reasoning =
    usage.completion_tokens_details?.reasoning_tokens ??
    usage.output_tokens_details?.reasoning_tokens ??
    usage.reasoning_tokens ??
    usage.thoughtsTokenCount;
  if (
    input == null &&
    output == null &&
    total == null &&
    cacheRead == null &&
    cacheWrite == null &&
    reasoning == null
  ) {
    return undefined;
  }
  return { input, output, total, cacheRead, cacheWrite, reasoning };
}

export function parseTokenUsageFromPayload(text: string): TokenUsage | undefined {
  if (!text) return undefined;
  try {
    return parseTokenUsage(JSON.parse(text));
  } catch {
    // SSE or concatenated JSON chunks.
  }
  let merged: TokenUsage | undefined;
  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const payload = trimmed.slice(5).trim();
    if (!payload || payload === "[DONE]") continue;
    try {
      const parsed = JSON.parse(payload);
      const usage = parseTokenUsage(parsed) || parseTokenUsage({ usage: parsed.usage });
      if (usage) merged = mergeUsage(merged, usage);
    } catch {
      // skip malformed SSE data
    }
  }
  return merged;
}

function prettyIfJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

/** Pull thinking text from Chat Completions deltas (Grok, DeepSeek, OpenRouter, …). */
export function reasoningTextFromDelta(delta: unknown): string {
  if (!delta || typeof delta !== "object") return "";
  const rec = delta as Record<string, unknown>;
  if (typeof rec.reasoning_content === "string" && rec.reasoning_content) {
    return rec.reasoning_content;
  }
  if (typeof rec.reasoning === "string" && rec.reasoning) return rec.reasoning;
  const thinking = rec.thinking;
  if (typeof thinking === "string" && thinking) return thinking;
  if (thinking && typeof thinking === "object") {
    const content = (thinking as { content?: unknown }).content;
    if (typeof content === "string" && content) return content;
  }
  return "";
}

/**
 * Copy `thinking.content` onto `reasoning_content` so the AI SDK's
 * OpenAI-compatible parser emits reasoning parts.
 */
export function normalizeChatCompletionReasoningChunk(parsed: unknown): unknown {
  if (!parsed || typeof parsed !== "object") return parsed;
  const rec = parsed as Record<string, any>;
  const choice = rec.choices?.[0];
  const delta = choice?.delta;
  if (!delta || typeof delta !== "object") return parsed;
  if (typeof delta.reasoning_content === "string" && delta.reasoning_content) {
    return parsed;
  }
  const text = reasoningTextFromDelta(delta);
  if (!text) return parsed;
  const choices = Array.isArray(rec.choices) ? [...rec.choices] : [];
  choices[0] = { ...choice, delta: { ...delta, reasoning_content: text } };
  return { ...rec, choices };
}

export function rewriteSseReasoningLine(line: string): string {
  const trimmed = line.trim();
  if (!trimmed.startsWith("data:")) return line;
  const payload = trimmed.slice(5).trim();
  if (!payload || payload === "[DONE]" || payload.startsWith("[DONE]")) return line;
  try {
    const normalized = normalizeChatCompletionReasoningChunk(JSON.parse(payload));
    const prefix = line.slice(0, line.indexOf("data:"));
    return `${prefix}data: ${JSON.stringify(normalized)}`;
  } catch {
    return line;
  }
}

/**
 * OpenCode Zen appends a cost trailer in the same SSE event as `data: [DONE]`.
 * Close `[DONE]` as its own event and drop the trailer — Chat Completions
 * streams end there, and JSON.parse of `[DONE] {…}` fails.
 */
export function splitChatCompletionsDoneLine(line: string): string[] {
  const idx = line.indexOf("data:");
  if (idx < 0) return [line];
  const prefix = line.slice(0, idx);
  const payload = line.slice(idx + 5).trim();
  if (!payload.startsWith("[DONE]")) return [line];
  return [`${prefix}data: [DONE]`, ""];
}

export function createReasoningNormalizeTransform(): TransformStream<Uint8Array, Uint8Array> {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  let buffer = "";
  let sawDone = false;
  return new TransformStream({
    transform(chunk, controller) {
      if (sawDone) return;
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split(/\n/);
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (sawDone) return;
        for (const piece of splitChatCompletionsDoneLine(rewriteSseReasoningLine(line))) {
          if (sawDone) return;
          if (/^data:\s*\[DONE\]\s*$/.test(piece)) {
            sawDone = true;
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            return;
          }
          controller.enqueue(encoder.encode(`${piece}\n`));
        }
      }
    },
    flush(controller) {
      if (sawDone || !buffer) return;
      for (const piece of splitChatCompletionsDoneLine(rewriteSseReasoningLine(buffer))) {
        if (sawDone) return;
        if (/^data:\s*\[DONE\]\s*$/.test(piece)) {
          sawDone = true;
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
          return;
        }
        controller.enqueue(encoder.encode(`${piece}\n`));
      }
    },
  });
}

export function redactCapturedHeaders(
  headers: Headers | Record<string, unknown> | undefined
): Record<string, string> {
  return sanitizeHeadersForLog(headers);
}

function rewriteJsonInit(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
  body: Record<string, unknown>
): { input: RequestInfo | URL; init: RequestInit } {
  const headers = new Headers();
  const existing = headersFromFetchArgs(input, init);
  if (existing instanceof Headers) {
    existing.forEach((value, key) => headers.set(key, value));
  } else if (existing && typeof existing === "object") {
    for (const [key, value] of Object.entries(existing)) {
      if (value == null) continue;
      headers.set(key, Array.isArray(value) ? value.join(", ") : String(value));
    }
  }
  if (!headers.has("content-type")) {
    headers.set("content-type", "application/json");
  }
  const next: RequestInit = {
    ...(init || {}),
    method: methodFromFetchArgs(input, init),
    headers,
    body: JSON.stringify(body),
  };
  if (input instanceof Request) {
    next.signal = init?.signal ?? input.signal;
    next.credentials = init?.credentials ?? input.credentials;
    return { input: input.url, init: next };
  }
  return { input, init: next };
}

async function finalizeCapturedBody(store: LlmCaptureStore): Promise<CapturedLlmExchange | undefined> {
  if (store.captureBody && store.last) {
    try {
      const text = await store.captureBody;
      store.last.responseBody = prettyIfJson(text);
      store.last.usage = parseTokenUsageFromPayload(text) ?? store.last.usage;
    } catch {
      // Keep status/headers even if the tee copy failed.
    }
  }
  return store.last;
}

export async function readCapturedBody(
  stream: ReadableStream<Uint8Array>,
  signal?: AbortSignal
): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let bytes = 0;
  const onAbort = () => {
    void reader.cancel(signal?.reason).catch(() => {});
  };
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    if (signal?.aborted) throw signal.reason ?? new Error("Capture aborted");
    while (true) {
      const { done, value } = await reader.read();
      if (signal?.aborted) throw signal.reason ?? new Error("Capture aborted");
      if (done) return text + decoder.decode();
      bytes += value.byteLength;
      if (bytes > MAX_CAPTURE_BODY_BYTES) {
        await reader.cancel("Debug capture body limit reached");
        return `${text}\n<body capture truncated at ${MAX_CAPTURE_BODY_BYTES} bytes>`;
      }
      text += decoder.decode(value, { stream: true });
    }
  } finally {
    signal?.removeEventListener("abort", onAbort);
    try {
      reader.releaseLock();
    } catch {
      // Stream already released.
    }
  }
}

export async function runWithLlmCapture<T>(
  fn: () => Promise<T>,
  options?: Pick<LlmCaptureStore, "patchRequestBody" | "signal">
): Promise<{
  result?: T;
  last?: CapturedLlmExchange;
  finalize: () => Promise<CapturedLlmExchange | undefined>;
  error?: unknown;
}> {
  const store: LlmCaptureStore = {
    patchRequestBody: options?.patchRequestBody,
    signal: options?.signal,
  };
  try {
    const result = await llmCaptureAls.run(store, fn);
    return {
      result,
      last: store.last,
      finalize: () => finalizeCapturedBody(store),
    };
  } catch (error) {
    return {
      last: store.last,
      finalize: () => finalizeCapturedBody(store),
      error,
    };
  }
}

let captureInstalled = false;

/**
 * Wrap global.fetch so debug-chat can record the last upstream LLM exchange.
 * Must run after the server's logging interceptor so this wrap is outermost.
 */
export function installLlmCaptureFetch(): void {
  if (captureInstalled) return;
  captureInstalled = true;
  const inner = global.fetch;
  global.fetch = async (...args: Parameters<typeof fetch>) => {
    const store = llmCaptureAls.getStore();
    if (!store) {
      return inner(...args);
    }

    const inputArg = args[0] as RequestInfo | URL;
    const initArg = args[1];
    let input = inputArg;
    let init = initArg;
    const url = urlFromFetchArgs(input);
    const method = methodFromFetchArgs(input, init);
    if (!shouldCapture(url, method)) {
      return inner(...args);
    }

    let requestBody = bodyFromFetchArgs(input, init);
    if (
      store.patchRequestBody &&
      requestBody &&
      typeof requestBody === "object" &&
      !Array.isArray(requestBody)
    ) {
      requestBody = store.patchRequestBody(
        { ...(requestBody as Record<string, unknown>) },
        url
      );
      const rewritten = rewriteJsonInit(input, init, requestBody as Record<string, unknown>);
      input = rewritten.input;
      init = rewritten.init;
    }
    const requestHeaders = redactCapturedHeaders(headersFromFetchArgs(input, init));
    const res = await inner(input as RequestInfo | URL, init);
    const streaming = isStreamingResponse(res, requestBody);
    const responseHeaders = redactCapturedHeaders(res.headers);

    let responseBody = "";
    let usage: TokenUsage | undefined;
    let outbound: Response = res;
    if (!streaming) {
      try {
        responseBody = await res.clone().text();
        usage = parseTokenUsageFromPayload(responseBody);
        responseBody = prettyIfJson(responseBody);
      } catch {
        responseBody = "<body not captured>";
      }
    } else if (res.body) {
      const [live, copy] = res.body.tee();
      store.captureBody = readCapturedBody(copy, store.signal);
      outbound = new Response(live.pipeThrough(createReasoningNormalizeTransform()), {
        status: res.status,
        statusText: res.statusText,
        headers: res.headers,
      });
    }

    store.last = {
      url,
      method: method.toUpperCase(),
      requestHeaders,
      requestBody,
      status: res.status,
      responseHeaders,
      responseBody,
      streaming,
      usage,
    };
    return outbound;
  };
}

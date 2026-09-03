import type { UnifiedChatRequest } from "@/types/llm";

export type OpenAiUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens?: number };
};

export type CursorUsageCounters = {
  inputTokens: number;
  outputTokens: number;
  cacheRead: number;
  cacheWrite: number;
  reasoningTokens: number;
};

/**
 * Rough char→token estimate for mid-turn usage before Cursor emits a usage
 * event (and as the host-facing request usage even when it does).
 * Mirrors cursor-opencode-provider: never report Cursor cumulative counters as
 * the current Claude Code / OpenCode request usage.
 */
export function estimateTokens(chars: number): number {
  if (!Number.isFinite(chars) || chars <= 0) return 0;
  return Math.ceil(chars / 4);
}

function contentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  try {
    return JSON.stringify(content) ?? "";
  } catch {
    return String(content);
  }
}

/** Estimate current-request prompt tokens from the Claude Code → CCR payload. */
export function estimateRequestPromptTokens(
  request: UnifiedChatRequest
): number {
  const parts: string[] = [];

  if (request.system) parts.push(contentToText(request.system));

  for (const msg of request.messages || []) {
    parts.push(contentToText(msg.content));
    if (msg.thinking?.content) parts.push(String(msg.thinking.content));
    if (msg.reasoning_content) parts.push(String(msg.reasoning_content));
    if (msg.tool_calls?.length) parts.push(JSON.stringify(msg.tool_calls));
  }

  if (request.tools?.length) parts.push(JSON.stringify(request.tools));

  return estimateTokens(parts.join("\n").length);
}

/** Build OpenAI-style usage for a single CCR request (not Cursor session totals). */
export function requestUsageFromEstimate(
  promptTokens: number,
  outputChars: number,
  cacheReadTokens = 0
): OpenAiUsage {
  const prompt_tokens = Math.max(0, Math.trunc(promptTokens) || 0);
  const completion_tokens = estimateTokens(outputChars);
  const cached_tokens = Math.min(
    prompt_tokens,
    Math.max(0, Math.trunc(cacheReadTokens) || 0)
  );
  return {
    prompt_tokens,
    completion_tokens,
    total_tokens: prompt_tokens + completion_tokens,
    prompt_tokens_details: { cached_tokens },
  };
}

function tokenValue(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

function truncNonNegative(value: unknown): number {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? Math.trunc(n) : 0;
}

export function cursorUsageCountersFromSdk(
  raw: OpenAiUsage | undefined
): CursorUsageCounters | undefined {
  if (!raw) return undefined;
  // OpenAiUsage here is the SDK TokenUsage projected onto flat fields; also
  // accept the SDK's native TokenUsage shape when passed directly.
  const anyRaw: any = raw as any;
  const inputTokens =
    truncNonNegative(anyRaw.prompt_tokens ?? anyRaw.inputTokens ?? anyRaw.input_tokens);
  const outputTokens =
    truncNonNegative(anyRaw.completion_tokens ?? anyRaw.outputTokens ?? anyRaw.output_tokens);
  const cacheRead = Math.min(
    inputTokens,
    truncNonNegative(
      anyRaw.prompt_tokens_details?.cached_tokens ??
        anyRaw.cacheReadTokens ??
        anyRaw.cache_read_input_tokens ??
        anyRaw.cacheRead
    )
  );
  const cacheWrite = Math.min(
    Math.max(0, inputTokens - cacheRead),
    truncNonNegative(
      anyRaw.cacheWriteTokens ??
        anyRaw.cache_write_input_tokens ??
        anyRaw.cacheWrite ??
        // usageFromSdk projects SDK write onto this diagnostic field
        anyRaw._cacheWriteTokens
    )
  );
  const reasoningTokens = Math.min(
    outputTokens,
    truncNonNegative(anyRaw.reasoningTokens ?? anyRaw.reasoning_tokens)
  );
  if (!inputTokens && !outputTokens && !cacheRead && !cacheWrite && !reasoningTokens) {
    return undefined;
  }
  return { inputTokens, outputTokens, cacheRead, cacheWrite, reasoningTokens };
}

/**
 * Cursor SDK usage is per SDK turn, while Claude Code needs per-CCR-request
 * usage. In bridge mode one SDK turn can span several parked host-tool request
 * cycles. Use a terminal SDK witness only as a cache-read ratio for the CCR
 * request that observes it; never attribute it to an earlier parked response.
 *
 * Kept for tests; prefer buildAccurateUsageFromSdk for turn-end reporting.
 */
export function cacheReadFromSdkDelta(
  current: OpenAiUsage | undefined,
  previous: OpenAiUsage | undefined,
  promptTokens: number
): number {
  const counters = cursorUsageCountersFromSdk(current);
  const prior = cursorUsageCountersFromSdk(previous);
  if (!counters) return 0;
  return accurateCacheReadForPrompt(counters, promptTokens, prior?.inputTokens);
}

function accurateCacheReadForPrompt(
  counters: CursorUsageCounters,
  promptTokens: number,
  priorInputTokens?: number
): number {
  const prompt = Math.max(0, Math.trunc(promptTokens) || 0);
  if (!prompt || !counters.inputTokens) return 0;
  const rawInput = Math.max(0, Math.trunc(counters.inputTokens));
  const rawCacheRead = Math.min(rawInput, Math.max(0, Math.trunc(counters.cacheRead)));
  if (!rawCacheRead) return 0;
  // Port of cursor-opencode-provider usage.ts: preserve cache proportions while
  // normalizing to the per-request prompt size. priorInputTokens is the
  // previous turn's occupancy — when the cache read covers that window, do not
  // dilute the hit by multi-step TurnEnded aggregates.
  const proportionalRead = Math.min(prompt, Math.round(prompt * rawCacheRead / rawInput));
  const prefixRead =
    typeof priorInputTokens === "number" &&
    Number.isFinite(priorInputTokens) &&
    priorInputTokens > 0 &&
    rawCacheRead >= priorInputTokens
      ? Math.min(prompt, Math.trunc(priorInputTokens))
      : 0;
  return Math.min(prompt, Math.max(proportionalRead, prefixRead));
}

/**
 * Build per-request OpenAI usage from the SDK turn-end usage message,
 * normalized to the per-request prompt estimate but preserving Cursor's cache
 * proportions (same math as cursor-opencode-provider
 * buildLanguageModelV3UsageFromCounters). Falls back to chars/4-based
 * completion tokens when SDK output is absent.
 *
 * When the runtime reports no usage for the turn (the SDK emits the usage
 * message only "when the runtime reported usage"), prompt_tokens_details is
 * omitted entirely so the cache-outcome tap reports "unknown" instead of a
 * bogus "unexpected-miss" from a zero that was never measured.
 */
export function buildAccurateUsageFromSdk(
  sdkRaw: OpenAiUsage | undefined,
  promptTokens: number,
  outputChars: number,
  priorRaw?: OpenAiUsage | undefined
): OpenAiUsage {
  const prompt_tokens = Math.max(0, Math.trunc(promptTokens) || 0);
  const fallbackCompletion = estimateTokens(outputChars);
  const counters = cursorUsageCountersFromSdk(sdkRaw);
  if (!counters) {
    return {
      prompt_tokens,
      completion_tokens: fallbackCompletion,
      total_tokens: prompt_tokens + fallbackCompletion,
    };
  }
  const cacheRead = accurateCacheReadForPrompt(counters, prompt_tokens, cursorUsageCountersFromSdk(priorRaw)?.inputTokens);
  const rawCacheWrite = Math.min(
    Math.max(0, counters.inputTokens - counters.cacheRead),
    Math.max(0, counters.cacheWrite)
  );
  const cacheWrite =
    counters.inputTokens > 0
      ? Math.min(prompt_tokens - cacheRead, Math.round(prompt_tokens * rawCacheWrite / counters.inputTokens))
      : 0;
  const completion_tokens =
    counters.outputTokens > 0 ? Math.min(fallbackCompletion || counters.outputTokens, counters.outputTokens) || fallbackCompletion : fallbackCompletion;
  // Keep total as prompt+completion; expose cached via details. reasoning stays in completion.
  return {
    prompt_tokens,
    completion_tokens,
    total_tokens: prompt_tokens + completion_tokens,
    prompt_tokens_details: { cached_tokens: cacheRead },
    // Carry cacheWrite for diagnostics without changing the OpenAI shape
    ...(cacheWrite ? { _cacheWriteTokens: cacheWrite } as any : {}),
  };
}

/** Map SDK TokenUsage / usage message into OpenAI shape (diagnostics only). */
export function usageFromSdk(message: any): OpenAiUsage | undefined {
  const u = message?.usage ?? message;
  if (!u || typeof u !== "object") return undefined;
  const input = u.inputTokens ?? u.input_tokens ?? u.prompt_tokens;
  const output = u.outputTokens ?? u.output_tokens ?? u.completion_tokens;
  if (input == null && output == null) return undefined;
  const prompt_tokens = Number(input) || 0;
  const completion_tokens = Number(output) || 0;
  const cacheRead =
    Number(u.cacheReadTokens ?? u.cache_read_input_tokens ?? u.cacheRead ?? 0) || 0;
  const cacheWrite =
    Number(u.cacheWriteTokens ?? u.cache_write_input_tokens ?? u.cacheWrite ?? 0) || 0;
  const total =
    Number(u.totalTokens ?? u.total_tokens ?? prompt_tokens + completion_tokens) || 0;
  return {
    prompt_tokens,
    completion_tokens,
    total_tokens: total || prompt_tokens + completion_tokens,
    prompt_tokens_details: { cached_tokens: cacheRead },
    ...(cacheWrite ? { _cacheWriteTokens: cacheWrite } as any : {}),
  };
}

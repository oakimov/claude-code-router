import type { UnifiedChatRequest } from "@/types/llm";

export type OpenAiUsage = {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prompt_tokens_details?: { cached_tokens: number };
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
  outputChars: number
): OpenAiUsage {
  const prompt_tokens = Math.max(0, Math.trunc(promptTokens) || 0);
  const completion_tokens = estimateTokens(outputChars);
  return {
    prompt_tokens,
    completion_tokens,
    total_tokens: prompt_tokens + completion_tokens,
    prompt_tokens_details: { cached_tokens: 0 },
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
    Number(u.cacheReadTokens ?? u.cache_read_input_tokens ?? 0) || 0;
  return {
    prompt_tokens,
    completion_tokens,
    total_tokens:
      Number(u.totalTokens ?? u.total_tokens ?? prompt_tokens + completion_tokens) ||
      0,
    prompt_tokens_details: { cached_tokens: cacheRead },
  };
}

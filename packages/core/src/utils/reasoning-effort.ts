import type { ThinkLevel, UnifiedChatRequest } from "@/types/llm";

type UnifiedReasoning = UnifiedChatRequest["reasoning"];

/** Normalize effort tokens at the protocol boundary without rejecting extensions. */
export function normalizeReasoningEffort(
  value: unknown
): ThinkLevel | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  return normalized ? (normalized as ThinkLevel) : undefined;
}

/** `none` and an explicit enabled:false are the canonical off signals. */
export function isReasoningDisabled(
  reasoning: UnifiedReasoning | undefined,
  thinking?: UnifiedChatRequest["thinking"]
): boolean {
  return (
    thinking?.type === "disabled" ||
    reasoning?.enabled === false ||
    normalizeReasoningEffort(reasoning?.effort) === "none"
  );
}

/** Build the canonical reasoning state used by every inbound protocol. */
export function canonicalReasoning(
  effortValue: unknown,
  enabledWhenEffortAbsent?: boolean
): UnifiedReasoning | undefined {
  const effort = normalizeReasoningEffort(effortValue);
  if (!effort && enabledWhenEffortAbsent === undefined) return undefined;
  return {
    ...(effort ? { effort } : {}),
    enabled: effort === "none" ? false : enabledWhenEffortAbsent ?? true,
  };
}

/** Serialize Unified reasoning onto the Chat Completions wire shape in place. */
export function applyOpenAIChatReasoning(
  request: UnifiedChatRequest
): UnifiedChatRequest {
  if (!request.reasoning) return request;
  const effort = isReasoningDisabled(request.reasoning, request.thinking)
    ? "none"
    : normalizeReasoningEffort(request.reasoning.effort);
  if (effort) request.reasoning_effort = effort;
  delete request.reasoning;
  return request;
}

/** Anthropic accepts low..max, while CCR/OpenAI may additionally emit minimal/ultra/none. */
export function toAnthropicReasoningEffort(
  effortValue: unknown
): Exclude<ThinkLevel, "none" | "minimal" | "ultra"> | undefined {
  const effort = normalizeReasoningEffort(effortValue);
  if (!effort || effort === "none") return undefined;
  if (effort === "minimal") return "low";
  if (effort === "ultra") return "max";
  return effort as Exclude<ThinkLevel, "none" | "minimal" | "ultra">;
}

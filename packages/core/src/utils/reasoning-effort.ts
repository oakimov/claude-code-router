import type { ThinkLevel, UnifiedChatRequest } from "@/types/llm";

type UnifiedReasoning = UnifiedChatRequest["reasoning"];

/** Readable-reasoning request levels shared across Responses / Codex / config. */
export type ReasoningSummaryLevel = "auto" | "detailed" | "concise";

const REASONING_SUMMARY_LEVELS = new Set<string>([
  "auto",
  "detailed",
  "concise",
]);

/** Normalize effort tokens at the protocol boundary without rejecting extensions. */
export function normalizeReasoningEffort(
  value: unknown
): ThinkLevel | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  return normalized ? (normalized as ThinkLevel) : undefined;
}

/**
 * Parse `REASONING_AUTO_SUMMARY` / provider `reasoningSummary`.
 * `true` → `"detailed"` (LiteLLM-compatible). `"none"` / false / unset → off.
 */
export function resolveReasoningAutoSummary(
  value: unknown
): ReasoningSummaryLevel | undefined {
  if (value === true || value === 1) return "detailed";
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (
      normalized === "true" ||
      normalized === "1" ||
      normalized === "yes" ||
      normalized === "on"
    ) {
      return "detailed";
    }
    if (REASONING_SUMMARY_LEVELS.has(normalized)) {
      return normalized as ReasoningSummaryLevel;
    }
  }
  return undefined;
}

function normalizeReasoningSummaryToken(
  value: unknown
): ReasoningSummaryLevel | "none" | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  if (normalized === "none") return "none";
  if (REASONING_SUMMARY_LEVELS.has(normalized)) {
    return normalized as ReasoningSummaryLevel;
  }
  return undefined;
}

/** True when reasoning is on (effort present and not none, or enabled:true). */
export function isReasoningActive(
  reasoning: UnifiedReasoning | undefined,
  thinking?: UnifiedChatRequest["thinking"]
): boolean {
  if (isReasoningDisabled(reasoning, thinking)) return false;
  if (!reasoning) return false;
  const effort = normalizeReasoningEffort(reasoning.effort);
  if (effort && effort !== "none") return true;
  return reasoning.enabled === true;
}

/**
 * Opt-in: when config asks for auto-summary and the client enabled reasoning
 * without an explicit `reasoning.summary`, stamp the Unified field so every
 * destination protocol can request readable thinking the same way.
 */
export function applyReasoningAutoSummary(
  request: UnifiedChatRequest,
  configValue: unknown
): UnifiedChatRequest {
  const summary = resolveReasoningAutoSummary(configValue);
  if (!summary || !request.reasoning) return request;
  if (!isReasoningActive(request.reasoning, request.thinking)) return request;
  const existing = normalizeReasoningSummaryToken(request.reasoning.summary);
  // Explicit client value (including "none") wins over the config default.
  if (existing) return request;
  request.reasoning.summary = summary;
  return request;
}

/**
 * Outbound precedence: Unified `reasoning.summary` → provider.reasoningSummary.
 * `"none"` means do not request a readable summary.
 */
export function resolveOutboundReasoningSummary(
  request: Pick<UnifiedChatRequest, "reasoning" | "thinking">,
  provider?: { reasoningSummary?: unknown } | null
): ReasoningSummaryLevel | undefined {
  if (isReasoningDisabled(request.reasoning, request.thinking)) {
    return undefined;
  }
  const fromRequest = normalizeReasoningSummaryToken(request.reasoning?.summary);
  if (fromRequest === "none") return undefined;
  if (fromRequest) return fromRequest;
  const fromProvider = normalizeReasoningSummaryToken(provider?.reasoningSummary);
  if (fromProvider === "none") return undefined;
  if (fromProvider) return fromProvider;
  return undefined;
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

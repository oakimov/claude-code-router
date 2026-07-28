/**
 * Request-local semantics recovered by a wire-protocol transformer.
 *
 * This state belongs on TransformerContext, not UnifiedChatRequest: the latter
 * is also the OpenAI-compatible outbound body and is serialized for providers.
 */
export type UnifiedTrailingToolResult = {
  toolCallId: string;
  content: string;
  isError: boolean;
};

export type UnifiedTurnIntent = {
  source: "anthropic" | "unified-fallback";
  trailingToolResults: UnifiedTrailingToolResult[];
  interruption: "none" | "synthetic_client_interrupt";
  steering: "none" | "meaningful";
};

export type UnifiedRequestRuntime = {
  source: "anthropic";
  sourceSessionIdentity?: string;
  turnIntent: UnifiedTurnIntent;
};

export const ANTHROPIC_SYNTHETIC_INTERRUPT_MARKERS = [
  "[Request interrupted by user]",
  "[Request interrupted by user for tool use]",
] as const;

function isSyntheticInterruptText(value: unknown): boolean {
  if (typeof value !== "string") return false;
  const normalized = value.trim();
  return ANTHROPIC_SYNTHETIC_INTERRUPT_MARKERS.some(
    (marker) => normalized === marker
  );
}

function toolResultContentToText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const text = content
      .filter(
        (part: any) =>
          part?.type === "text" &&
          typeof part.text === "string" &&
          part.text.length > 0
      )
      .map((part: any) => part.text)
      .join("\n");
    if (text) return text;
  }
  return JSON.stringify(content) ?? "";
}

function hasMeaningfulNonTextPart(part: any): boolean {
  if (!part || typeof part !== "object") return false;
  if (part.type === "image") return Boolean(part.source);
  // Unknown non-tool user blocks are conservatively treated as steering.
  return part.type !== "tool_result" && part.type !== "text";
}

export function classifyAnthropicTurnIntent(
  requestMessages: unknown
): UnifiedTurnIntent {
  const intent: UnifiedTurnIntent = {
    source: "anthropic",
    trailingToolResults: [],
    interruption: "none",
    steering: "none",
  };
  if (!Array.isArray(requestMessages) || requestMessages.length === 0) {
    return intent;
  }

  const trailingMessage = requestMessages[requestMessages.length - 1] as any;
  if (trailingMessage?.role !== "user") return intent;

  if (typeof trailingMessage.content === "string") {
    if (trailingMessage.content.trim()) intent.steering = "meaningful";
    return intent;
  }
  if (!Array.isArray(trailingMessage.content)) return intent;

  const hasToolResult = trailingMessage.content.some(
    (part: any) =>
      part?.type === "tool_result" &&
      typeof part.tool_use_id === "string" &&
      part.tool_use_id.length > 0
  );

  for (const part of trailingMessage.content) {
    if (
      part?.type === "tool_result" &&
      typeof part.tool_use_id === "string" &&
      part.tool_use_id.length > 0
    ) {
      intent.trailingToolResults.push({
        toolCallId: part.tool_use_id,
        content: toolResultContentToText(part.content),
        isError: part.is_error === true,
      });
      continue;
    }

    if (part?.type === "text" && typeof part.text === "string") {
      // The marker is protocol metadata only when Claude Code places it beside
      // a tool_result in the same raw user block. The identical text typed as
      // an ordinary user message remains meaningful input.
      if (hasToolResult && isSyntheticInterruptText(part.text)) {
        intent.interruption = "synthetic_client_interrupt";
      } else if (part.text.trim()) {
        intent.steering = "meaningful";
      }
      continue;
    }

    if (hasMeaningfulNonTextPart(part)) {
      intent.steering = "meaningful";
    }
  }

  return intent;
}

export function extractAnthropicSourceSessionIdentity(
  metadata: unknown
): string | undefined {
  const userId =
    metadata &&
    typeof metadata === "object" &&
    typeof (metadata as any).user_id === "string"
      ? (metadata as any).user_id
      : undefined;
  if (!userId) return undefined;

  try {
    const parsed = JSON.parse(userId);
    if (parsed && typeof parsed.session_id === "string" && parsed.session_id) {
      return parsed.session_id;
    }
  } catch {
    // Non-JSON metadata.user_id values are valid and remain useful identities.
  }

  const sessionSuffix = userId.split("_session_")[1];
  return sessionSuffix || userId;
}

export function buildAnthropicRequestRuntime(
  request: Record<string, any>
): UnifiedRequestRuntime {
  return {
    source: "anthropic",
    sourceSessionIdentity: extractAnthropicSourceSessionIdentity(
      request.metadata
    ),
    turnIntent: classifyAnthropicTurnIntent(request.messages),
  };
}

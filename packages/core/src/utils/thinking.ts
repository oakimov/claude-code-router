import { ThinkLevel } from "@/types/llm";

export const getThinkLevel = (thinking_budget: number): ThinkLevel => {
  if (thinking_budget <= 0) return "none";
  if (thinking_budget <= 1024) return "low";
  if (thinking_budget <= 8192) return "medium";
  return "high";
};

/**
 * Accumulator for streaming reasoning/thinking content.
 * Used to buffer reasoning chunks across SSE events and emit
 * a final thinking block with signature when reasoning completes.
 */
export interface ReasoningAccumulator {
  content: string;
  isComplete: boolean;
  hasContent: boolean;
}

export function createReasoningAccumulator(): ReasoningAccumulator {
  return { content: "", isComplete: false, hasContent: false };
}

export function accumulateReasoning(
  accumulator: ReasoningAccumulator,
  text: string
): void {
  accumulator.content += text;
  accumulator.hasContent = true;
}

export function finalizeReasoning(accumulator: ReasoningAccumulator): {
  content: string;
  signature: string;
} {
  accumulator.isComplete = true;
  return {
    content: accumulator.content,
    signature: Date.now().toString(),
  };
}

/**
 * Build a chat.completion.chunk with a thinking delta.
 * Used when emitting accumulated reasoning content as a thinking block.
 */
export function buildThinkingChunk(
  baseData: any,
  thinking: { content?: string; signature?: string }
): any {
  return {
    ...baseData,
    choices: [
      {
        ...(baseData.choices?.[0] || {}),
        delta: {
          ...(baseData.choices?.[0]?.delta || {}),
          content: null,
          thinking,
        },
      },
    ],
  };
}

/**
 * Extract reasoning text from various provider-specific delta fields.
 * Supports: delta.reasoning_content (DeepSeek/Mistral), delta.reasoning (OpenRouter)
 */
export function extractReasoningText(delta: any): string | null {
  if (delta.reasoning_content) {
    const rc = delta.reasoning_content;
    return typeof rc === "string"
      ? rc
      : typeof rc?.text === "string"
        ? rc.text
        : null;
  }
  if (delta.reasoning) {
    const r = delta.reasoning;
    return typeof r === "string" ? r : null;
  }
  return null;
}

/**
 * Remove reasoning-specific fields from a delta after extraction.
 */
export function cleanReasoningFields(delta: any): void {
  delete delta.reasoning_content;
  delete delta.reasoning;
}

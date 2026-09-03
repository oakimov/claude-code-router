import { encodeSSEData } from "@/utils/stream";
import { coerceThinkingText } from "./shared";

export type SseEmitter = {
  enqueueChunk: (chunk: Record<string, unknown>) => void;
  enqueueDone: () => void;
};

export function createSseHelpers(model: string, encoder: TextEncoder) {
  const created = Math.floor(Date.now() / 1000);
  let id = `chatcmpl-cursor-${Date.now()}`;

  const base = () => ({
    id,
    object: "chat.completion.chunk",
    created,
    model,
  });

  return {
    setId(next: string) {
      if (next) id = next;
    },
    content(text: string) {
      return {
        ...base(),
        choices: [
          {
            index: 0,
            delta: { role: "assistant", content: text },
            finish_reason: null,
          },
        ],
      };
    },
    thinking(text: string) {
      return {
        ...base(),
        choices: [
          {
            index: 0,
            delta: { thinking: { content: coerceThinkingText(text) } },
            finish_reason: null,
          },
        ],
      };
    },
    /**
     * Claude Code / Anthropic expecting extended-thinking streams require a
     * signature_delta before the thinking block is closed. Cursor SDK has no
     * provider signature, so emit a synthetic one (same pattern as reasoning /
     * forcereasoning transformers). Without this, thinking_deltas are on the
     * wire but the UI often never surfaces them.
     */
    thinkingSignature(signature = `ccr_cursor_${Date.now()}`) {
      return {
        ...base(),
        choices: [
          {
            index: 0,
            delta: { thinking: { signature } },
            finish_reason: null,
          },
        ],
      };
    },
    toolCall(tool: { id: string; name: string; args: Record<string, unknown> }, index = 0) {
      return {
        ...base(),
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index,
                  id: tool.id,
                  type: "function",
                  function: {
                    name: tool.name,
                    arguments: JSON.stringify(tool.args ?? {}),
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    },
    finish(
      reason: "stop" | "tool_calls",
      usage?: {
        prompt_tokens?: number;
        completion_tokens?: number;
        total_tokens?: number;
        prompt_tokens_details?: { cached_tokens?: number };
      }
    ) {
      const prompt_tokens = Number(usage?.prompt_tokens) || 0;
      const completion_tokens = Number(usage?.completion_tokens) || 0;
      const total_tokens =
        Number(usage?.total_tokens) || prompt_tokens + completion_tokens;
      const cached = Number(usage?.prompt_tokens_details?.cached_tokens);
      return {
        ...base(),
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: reason,
          },
        ],
        // Always attach usage so AnthropicTransformer message_delta never
        // reports input_tokens:0 solely because the finish chunk omitted it.
        // prompt_tokens_details is attached only when the runtime reported a
        // cache count: omitting it lets the cache-outcome tap report "unknown"
        // instead of a bogus miss from an unmeasured zero.
        usage: {
          prompt_tokens,
          completion_tokens,
          total_tokens,
          ...(Number.isFinite(cached)
            ? { prompt_tokens_details: { cached_tokens: cached } }
            : {}),
        },
      };
    },
    encode(chunk: Record<string, unknown>) {
      return encodeSSEData(JSON.stringify(chunk), encoder);
    },
    encodeDone() {
      return encodeSSEData("[DONE]", encoder);
    },
  };
}

export function accumulateChatCompletion(
  model: string,
  chunks: Array<Record<string, unknown>>
): Record<string, unknown> {
  let content = "";
  let thinking = "";
  let thinkingSignature = "";
  const toolCalls: any[] = [];
  let finishReason: string | null = "stop";
  let usage: any = null;
  let id = `chatcmpl-cursor-${Date.now()}`;
  const created = Math.floor(Date.now() / 1000);

  for (const chunk of chunks) {
    if (typeof chunk.id === "string") id = chunk.id;
    const choice = (chunk.choices as any)?.[0];
    if (!choice) continue;
    if (choice.finish_reason) finishReason = choice.finish_reason;
    const delta = choice.delta || {};
    if (typeof delta.content === "string") content += delta.content;
    if (delta.thinking?.content) thinking += coerceThinkingText(delta.thinking.content);
    if (typeof delta.thinking?.signature === "string" && delta.thinking.signature) {
      thinkingSignature = delta.thinking.signature;
    }
    if (Array.isArray(delta.tool_calls)) {
      for (const tc of delta.tool_calls) {
        const idx = tc.index ?? toolCalls.length;
        if (!toolCalls[idx]) {
          toolCalls[idx] = {
            id: tc.id,
            type: "function",
            function: { name: tc.function?.name || "", arguments: "" },
          };
        }
        if (tc.id) toolCalls[idx].id = tc.id;
        if (tc.function?.name) toolCalls[idx].function.name = tc.function.name;
        if (typeof tc.function?.arguments === "string") {
          toolCalls[idx].function.arguments += tc.function.arguments;
        }
      }
    }
    if (chunk.usage) usage = chunk.usage;
  }

  const message: Record<string, unknown> = { role: "assistant", content: content || null };
  if (thinking) {
    // Non-empty signature keeps AnthropicTransformer request replay from
    // dropping the thinking block (`c.type === "thinking" && c.signature`).
    message.thinking = {
      content: thinking,
      signature: thinkingSignature || `ccr_cursor_${created}`,
    };
  }
  if (toolCalls.length) {
    // Index-based assignment can leave sparse holes; JSON would serialize them as null.
    message.tool_calls = toolCalls.filter(Boolean);
    finishReason = "tool_calls";
  }

  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: finishReason,
      },
    ],
    usage,
  };
}
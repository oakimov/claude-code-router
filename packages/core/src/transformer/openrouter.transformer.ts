import { UnifiedChatRequest } from "@/types/llm";
import { Transformer, TransformerOptions } from "../types/transformer";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "../utils/stream";
import { stripMessagesCacheControl } from "../utils/cacheControl";
import {
  createReasoningAccumulator,
  accumulateReasoning,
  finalizeReasoning,
  buildThinkingChunk,
  extractReasoningText,
  cleanReasoningFields,
} from "../utils/thinking";
import { v4 as uuidv4 } from "uuid";

export class OpenrouterTransformer implements Transformer {
  static TransformerName = "openrouter";

  constructor(private readonly options?: TransformerOptions) {}

  async transformRequestIn(
    request: UnifiedChatRequest
  ): Promise<UnifiedChatRequest> {
    if (!request.model.includes("claude")) {
      request.messages = stripMessagesCacheControl(request.messages);

      // Handle non-HTTP image URLs for non-Claude models
      request.messages.forEach((msg) => {
        if (Array.isArray(msg.content)) {
          msg.content.forEach((item: any) => {
            if (item.type === "image_url") {
              if (!item.image_url.url.startsWith("http")) {
                item.image_url.url = `${item.image_url.url}`;
              }
              delete item.media_type;
            }
          });
        }
      });
    } else {
      request.messages.forEach((msg) => {
        if (Array.isArray(msg.content)) {
          msg.content.forEach((item: any) => {
            if (item.type === "image_url") {
              if (!item.image_url.url.startsWith("http")) {
                item.image_url.url = `data:${item.media_type};base64,${item.image_url.url}`;
              }
              delete item.media_type;
            }
          });
        }
      });
    }
    Object.assign(request, this.options || {});
    return request;
  }

  async transformResponseOut(response: Response): Promise<Response> {
    if (response.headers.get("Content-Type")?.includes("application/json")) {
      const jsonResponse = await response.json();
      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } else if (response.headers.get("Content-Type")?.includes("stream")) {
      if (!response.body) return response;

      const accumulator = createReasoningAccumulator();
      let hasTextContent = false;
      let hasToolCall = false;

      return createSSEStreamReader(response, (line: string, ctx: StreamContext) => {
        if (!line.trim()) {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          return;
        }

        if (!line.startsWith("data: ") || line.trim() === "data: [DONE]") {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          return;
        }

        try {
          const jsonStr = line.slice(6);
          const data = JSON.parse(jsonStr);

          if (data.usage) {
            this.logger?.debug(
              { usage: data.usage, hasToolCall },
              "usage"
            );
            data.choices[0].finish_reason = hasToolCall
              ? "tool_calls"
              : "stop";
          }

          if (data.choices?.[0]?.finish_reason === "error") {
            ctx.controller.enqueue(
              encodeSSEData(
                JSON.stringify({ error: data.choices?.[0]?.error }),
                ctx.encoder
              )
            );
          }

          if (data.choices?.[0]?.delta?.content && !hasTextContent) {
            hasTextContent = true;
          }

          const delta = data.choices?.[0]?.delta;
          const reasoningText = delta ? extractReasoningText(delta) : null;

          if (reasoningText) {
            accumulateReasoning(accumulator, reasoningText);
            const thinkingChunk = buildThinkingChunk(data, {
              content: reasoningText,
            });
            cleanReasoningFields(thinkingChunk.choices[0].delta);
            ctx.controller.enqueue(encodeSSEData(JSON.stringify(thinkingChunk), ctx.encoder));
            return;
          }

          if (
            delta?.content &&
            accumulator.hasContent &&
            !accumulator.isComplete
          ) {
            const { content, signature } = finalizeReasoning(accumulator);
            const thinkingChunk = buildThinkingChunk(data, {
              content,
              signature,
            });
            cleanReasoningFields(thinkingChunk.choices[0].delta);
            thinkingChunk.choices[0].delta.content = null;
            ctx.controller.enqueue(encodeSSEData(JSON.stringify(thinkingChunk), ctx.encoder));
          }

          if (delta?.reasoning) {
            delete delta.reasoning;
          }

          if (
            delta?.tool_calls?.length &&
            !Number.isNaN(parseInt(delta.tool_calls[0].id, 10))
          ) {
            delta.tool_calls.forEach((tool: any) => {
              tool.id = `call_${uuidv4()}`;
            });
          }

          if (delta?.tool_calls?.length && !hasToolCall) {
            hasToolCall = true;
          }

          if (delta?.tool_calls?.length && hasTextContent) {
            if (typeof data.choices[0].index === "number") {
              data.choices[0].index += 1;
            } else {
              data.choices[0].index = 1;
            }
          }

          ctx.controller.enqueue(encodeSSEData(JSON.stringify(data), ctx.encoder));
        } catch {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
        }
      });
    }

    return response;
  }
}

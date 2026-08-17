import { UnifiedChatRequest, MessageContent, TextContent } from "../types/llm";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "./stream";
import {
  buildThinkingChunk,
  extractReasoningText,
  cleanReasoningFields,
} from "./thinking";
import { normalizeToolParameters } from "./schema";
import { deriveCacheSessionKey } from "./cacheControl";

function thinkingTextFromPart(part: any): string {
  if (!part || part.type !== "thinking") return "";
  const parts = Array.isArray(part.thinking) ? part.thinking : [part.thinking];
  return parts
    .map((piece: any) => {
      if (typeof piece === "string") return piece;
      if (piece && typeof piece.text === "string") return piece.text;
      return piece == null ? "" : JSON.stringify(piece);
    })
    .join("");
}

function thinkingTextFromMessage(msg: any): string {
  const fromThinking =
    typeof msg?.thinking?.content === "string" ? msg.thinking.content : "";
  const fromReasoning =
    typeof msg?.reasoning_content === "string" ? msg.reasoning_content : "";
  let fromContent = "";
  if (Array.isArray(msg?.content)) {
    for (const part of msg.content) {
      fromContent += thinkingTextFromPart(part);
    }
  }
  return fromThinking || fromReasoning || fromContent;
}

function mistralThinkingBlock(text: string): Record<string, unknown> {
  return {
    type: "thinking",
    thinking: [{ type: "text", text }],
  };
}

function stripPartCacheControl(part: any): any {
  if (!part || typeof part !== "object" || !("cache_control" in part)) {
    return part;
  }
  const { cache_control: _cache_control, ...rest } = part;
  return rest;
}

/**
 * Flatten array content, fold Unified/Chat thinking into Mistral ThinkChunks,
 * and drop Unified-only fields Mistral rejects.
 */
function transformMessage(msg: any): any {
  const clonedMsg = { ...msg };
  const thinkingText = thinkingTextFromMessage(clonedMsg);
  const contentArray = Array.isArray(clonedMsg.content)
    ? (clonedMsg.content as any[])
    : undefined;
  const imageParts = (contentArray || [])
    .filter((part) => part?.type === "image_url")
    .map(stripPartCacheControl);
  const plainText = contentArray
    ? contentArray
        .filter((part): part is TextContent => part?.type === "text")
        .map((part) => part.text)
        .filter((text) => text && text.length > 0)
        .join("\n")
    : typeof clonedMsg.content === "string"
      ? clonedMsg.content
      : "";

  if (clonedMsg.role === "assistant" && thinkingText) {
    const content: any[] = [];
    content.push(mistralThinkingBlock(thinkingText));
    if (plainText) content.push({ type: "text", text: plainText });
    content.push(...imageParts);
    clonedMsg.content = content;
  } else if (contentArray) {
    if (imageParts.length > 0) {
      clonedMsg.content = contentArray
        .filter((part) => part?.type === "text" || part?.type === "image_url")
        .map(stripPartCacheControl);
    } else {
      clonedMsg.content = plainText;
    }
  }

  delete clonedMsg.thinking;
  delete clonedMsg.reasoning_content;
  delete clonedMsg.cache_control;
  return clonedMsg;
}

/**
 * Helper to transform tool_choice to Mistral-compatible format
 */
function transformToolChoice(toolChoice: UnifiedChatRequest["tool_choice"]): any {
  if (toolChoice === "auto" || toolChoice === "none") {
    return toolChoice;
  }

  if (toolChoice === "required") {
    return "any";
  }

  if (typeof toolChoice === "object" && toolChoice.function?.name) {
    return {
      type: "function",
      function: { name: toolChoice.function.name },
    };
  }

  return toolChoice;
}

/**
 * Helper to transform reasoning parameter to Mistral's reasoning_effort format.
 * Mistral only supports "low" | "medium" | "high", so out-of-range effort
 * levels are clamped to the nearest boundary.
 */
function transformReasoning(reasoning: any): string | undefined {
  const effort = reasoning.effort?.toLowerCase();
  if (!effort || effort === "none") return undefined;

  if (effort === "minimal") return "low";

  if (effort === "low" || effort === "medium" || effort === "high") {
    return effort;
  }
  // Map Claude-level efforts beyond Mistral's range to its highest level
  return "high";
}

const NON_REASONING = new Set([
  "mistral-small-2506",
  "mistral-medium-2505",
  "mistral-medium-2508",
]);

/**
 * Transform incoming request to Mistral-compatible format
 */
export function buildRequestBody(
  request: UnifiedChatRequest,
  context?: any,
  _provider?: any
): Record<string, any> {
  const req = { ...request };

  // 1. Process messages
  if (Array.isArray(req.messages)) {
    req.messages = req.messages.map((msg) => transformMessage(msg));
  }

  // 2. Defaults
  if (req.stream === undefined) {
    req.stream = true;
  }

  // Mistral prompt caching is keyed at request level. Inline cache_control
  // markers are removed below after the native key has been derived.
  const cacheKey = deriveCacheSessionKey(context, request);
  if (cacheKey) {
    (req as any).prompt_cache_key = cacheKey;
  }

  // 3. Tool Choice
  if (req.tool_choice) {
    req.tool_choice = transformToolChoice(req.tool_choice);
  }

  // 4. Tool Cleanup - normalize schemas, remove $schema and cache_control.
  // Mistral rejects unknown fields; cache_control rides on tool definitions
  // after the Anthropic → Unified round-trip, so strip it here alongside the
  // message-level cleanup above.
  if (Array.isArray(req.tools)) {
    req.tools = req.tools.map((tool) => {
      const { cache_control: _cache_control, ...rest } = tool as any;
      if (rest?.function?.parameters) {
        return {
          ...rest,
          function: {
            ...rest.function,
            parameters: normalizeToolParameters(rest.function.parameters),
          },
        };
      }
      return rest;
    });
  }

  // 5. Reasoning conversion
  if (req.reasoning && req.model) {
    const modelId = req.model;
    const supportsReasoning =
      modelId.startsWith("magistral-") ||
      modelId.startsWith("labs-leanstral-") ||
      modelId === "mistral-vibe-cli-fast" ||
      (modelId.startsWith("mistral-small-") && !NON_REASONING.has(modelId)) ||
      (modelId.startsWith("mistral-medium-") && !NON_REASONING.has(modelId));

    if (supportsReasoning) {
      req.reasoning_effort = transformReasoning(req.reasoning);
    }
    delete req.reasoning;
  }

  // Request-level Unified / Anthropic control fields are not Mistral Chat
  // properties. Thinking history already lives on messages as ThinkChunks.
  delete (req as any).thinking;
  delete (req as any).enable_thinking;
  delete (req as any).anthropic_thinking;
  delete (req as any).anthropic_output_config;
  delete (req as any).anthropic_metadata;
  delete (req as any).anthropic_stop_sequences;

  return req;
}

/**
 * Transform a Mistral provider request back into a UnifiedChatRequest
 */
export async function transformRequestOut(request: any): Promise<UnifiedChatRequest> {
  return request as UnifiedChatRequest;
}

/**
 * Transform response back — convert Mistral's content-array thinking format
 * to the delta.thinking / delta.content shape expected by @caeliq/llms.
 */
export async function transformResponseOut(
  response: Response,
  providerName: string,
  logger?: any
): Promise<Response> {
  const contentType = response.headers.get("Content-Type") ?? "";

  if (contentType.includes("application/json")) {
    const jsonResponse = await response.json();
    logger?.debug({ response: jsonResponse }, `${providerName} response:`);

    const choice = jsonResponse.choices?.[0];
    if (choice?.message) {
      const message = choice.message;
      let thinkingText = "";

      if (message.reasoning_content) {
        const rc = message.reasoning_content;
        thinkingText += typeof rc === "string" ? rc : (typeof rc?.text === "string" ? rc.text : JSON.stringify(rc));
        delete message.reasoning_content;
      }

      if (Array.isArray(message.content)) {
        let plainText = "";
        for (const block of message.content) {
          if (block.type === "thinking") {
            const parts = Array.isArray(block.thinking) ? block.thinking : [block.thinking];
            thinkingText += parts.map((p: any) => {
              if (typeof p === "string") return p;
              if (p && typeof p.text === "string") return p.text;
              return JSON.stringify(p);
            }).join("");
          } else if (block.type === "text") {
            plainText += typeof block.text === "string" ? block.text : JSON.stringify(block.text ?? "");
          }
        }
        message.content = plainText;
      }

      if (thinkingText) {
        jsonResponse.thinking = { content: thinkingText };
        message.thinking = { content: thinkingText };
      }
    }

    return new Response(JSON.stringify(jsonResponse), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  } else if (contentType.includes("stream")) {
    if (!response.body) return response;

    return createSSEStreamReader(response, (line: string, ctx: StreamContext) => {
      if (!line.trim()) {
        ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
        return;
      }

      if (!line.startsWith("data:") || line.trim() === "data: [DONE]") {
        ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
        return;
      }

      try {
        const rawDataStr = line.slice(5).trim();
        const data = JSON.parse(rawDataStr);

        const delta = data.choices?.[0]?.delta;
        if (!delta) {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          return;
        }

        const reasoningText = extractReasoningText(delta);

        if (reasoningText) {
          const thinkingChunk = buildThinkingChunk(data, { content: reasoningText });
          cleanReasoningFields(thinkingChunk.choices[0].delta);

          // Handle content array thinking format
          const deltaContent = delta.content;
          if (Array.isArray(deltaContent)) {
            let arrThinkingText = "";
            const plainText = deltaContent
              .filter((b: any) => b.type === "text")
              .map((b: any) => typeof b.text === "string" ? b.text : JSON.stringify(b.text ?? ""))
              .join("");

            for (const block of deltaContent) {
              if (block.type === "thinking") {
                const parts = Array.isArray(block.thinking) ? block.thinking : [block.thinking];
                arrThinkingText += parts.map((p: any) => {
                  if (typeof p === "string") return p;
                  if (p && typeof p.text === "string") return p.text;
                  return JSON.stringify(p);
                }).join("");
              }
            }

            delete thinkingChunk.choices[0].delta.content;
            if (arrThinkingText || reasoningText) {
              thinkingChunk.choices[0].delta.thinking = { content: arrThinkingText || reasoningText };
            }
            if (plainText) thinkingChunk.choices[0].delta.content = plainText;
          }

          ctx.controller.enqueue(encodeSSEData(JSON.stringify(thinkingChunk), ctx.encoder));
          return;
        }

        // Handle content array thinking format without reasoning_content
        if (Array.isArray(delta.content)) {
          let thinkingFromArr = "";
          const plainText = delta.content
            .filter((b: any) => b.type === "text")
            .map((b: any) => typeof b.text === "string" ? b.text : JSON.stringify(b.text ?? ""))
            .join("");

          for (const block of delta.content) {
            if (block.type === "thinking") {
              const parts = Array.isArray(block.thinking) ? block.thinking : [block.thinking];
              thinkingFromArr += parts.map((p: any) => {
                if (typeof p === "string") return p;
                if (p && typeof p.text === "string") return p.text;
                return JSON.stringify(p);
              }).join("");
            }
          }

          delete delta.content;
          if (thinkingFromArr) delta.thinking = { content: thinkingFromArr };
          if (plainText) delta.content = plainText;
        }

        ctx.controller.enqueue(encodeSSEData(JSON.stringify(data), ctx.encoder));
      } catch {
        ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
      }
    });
  }

  return response;
}

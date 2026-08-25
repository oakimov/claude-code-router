import { UnifiedChatRequest, UnifiedMessage, UnifiedTool } from "../types/llm";
import { createSSEStreamReader } from "./stream";
import {
  mapRole,
  extractImageParts,
  processImageContent,
  consolidateMessages,
  normalizeTool
} from "./google.util";
import { applyRawAnthropicPromptCaching } from "./cacheControl";

// Vertex Claude message interface
interface ClaudeMessage {
  role: "user" | "assistant";
  content: Array<{
    type: "text" | "image" | "tool_use" | "tool_result";
    text?: string;
    source?: {
      type: "base64";
      media_type: string;
      data: string;
    };
    id?: string;
    name?: string;
    input?: Record<string, any>;
    tool_use_id?: string;
    content?: string | Array<any>;
  }>;
}

// Vertex Claude tool interface
interface ClaudeTool {
  name: string;
  description: string;
  input_schema: {
    type: string;
    properties: Record<string, any>;
    required?: string[];
    additionalProperties?: boolean;
    $schema?: string;
  };
}

// Vertex Claude request interface
interface VertexClaudeRequest {
  anthropic_version: "vertex-2023-10-16";
  messages: ClaudeMessage[];
  max_tokens: number;
  stream?: boolean;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  tools?: ClaudeTool[];
  tool_choice?: "auto" | "none" | { type: "tool"; name: string };
}

// Vertex Claude response interface
interface VertexClaudeResponse {
  content: Array<{
    type: "text";
    text: string;
  }>;
  id: string;
  model: string;
  role: "assistant";
  stop_reason: string;
  stop_sequence: null;
  type: "message";
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
  };
  tool_use?: Array<{
    id: string;
    name: string;
    input: Record<string, any>;
  }>;
}

function toOpenAIUsage(usage: any): Record<string, any> {
  const input = usage?.input_tokens || 0;
  const cached = usage?.cache_read_input_tokens || 0;
  const written = usage?.cache_creation_input_tokens || 0;
  const output = usage?.output_tokens || 0;
  return {
    completion_tokens: output,
    prompt_tokens: input + cached + written,
    prompt_tokens_details: {
      cached_tokens: cached,
      cache_write_tokens: written,
    },
    total_tokens: input + cached + written + output,
  };
}

function anthropicStopReasonToOpenAI(reason: unknown): string | null {
  switch (reason) {
    case "tool_use":
      return "tool_calls";
    case "max_tokens":
    case "model_context_window_exceeded":
      return "length";
    case "refusal":
      return "content_filter";
    case "end_turn":
    case "stop_sequence":
    case "pause_turn":
      return "stop";
    default:
      return typeof reason === "string" && reason ? "stop" : null;
  }
}

export function buildRequestBody(
  request: UnifiedChatRequest
): VertexClaudeRequest {
  const rawMessages: any[] = [];

  for (let i = 0; i < request.messages.length; i++) {
    const message = request.messages[i];
    const role = mapRole(message.role, { assistant: "assistant" });

    const content: any[] = [];

    if (message.role === "tool") {
      let resultText = message.content;
      if (Array.isArray(resultText)) {
        resultText = resultText
          .filter((part: any) => part.type === "text")
          .map((part: any) => part.text)
          .join("\n");
      } else if (typeof resultText === "object" && resultText !== null) {
        resultText = JSON.stringify(resultText);
      }
      content.push({
        type: "tool_result",
        tool_use_id: message.tool_call_id,
        content: resultText as string,
        ...(message.cache_control
          ? { cache_control: message.cache_control }
          : {}),
      });
    } else {
      if (typeof message.content === "string") {
        content.push({
          type: "text",
          text: message.content,
          ...(message.cache_control
            ? { cache_control: message.cache_control }
            : {}),
        });
      } else if (Array.isArray(message.content)) {
        // Text parts
        message.content.forEach((item) => {
          if (item.type === "text") {
            content.push({
              type: "text",
              text: item.text || "",
              ...((item as any).cache_control
                ? { cache_control: (item as any).cache_control }
                : {}),
            });
          }
        });
        // Image parts
        const images = extractImageParts(message.content);
        images.forEach((img) => {
          content.push(processImageContent(img, "claude"));
        });
      }

      if (message.tool_calls && message.tool_calls.length > 0) {
        message.tool_calls.forEach(toolCall => {
          content.push({
            type: "tool_use",
            id: toolCall.id,
            name: toolCall.function.name,
            input: JSON.parse(toolCall.function.arguments || "{}"),
          });
        });
      }
    }

    if (content.length > 0) {
      rawMessages.push({ role, content });
    }
  }

  const messages = consolidateMessages(rawMessages, 'content');

  const requestBody: VertexClaudeRequest = {
    anthropic_version: "vertex-2023-10-16",
    messages,
    max_tokens: request.max_tokens || 1000,
    stream: request.stream || false,
    ...(request.temperature && { temperature: request.temperature }),
  };

  // Handle tool definitions
  if (request.tools && request.tools.length > 0) {
    requestBody.tools = request.tools.map((tool: UnifiedTool) => {
      const normalized = normalizeTool(tool);
      return {
        name: normalized.name,
        description: normalized.description,
        input_schema: normalized.parameters,
        ...(tool.cache_control ? { cache_control: tool.cache_control } : {}),
      };
    });
  }

  // Handle tool choice
  if (request.tool_choice) {
    if (request.tool_choice === "auto" || request.tool_choice === "none") {
      requestBody.tool_choice = request.tool_choice;
    } else if (typeof request.tool_choice === "string") {
      // If tool_choice is a string, assume it's the tool name
      requestBody.tool_choice = {
        type: "tool",
        name: request.tool_choice,
      };
    }
  }

  return applyRawAnthropicPromptCaching(requestBody) as VertexClaudeRequest;
}

export function transformRequestOut(
  request: Record<string, any>
): UnifiedChatRequest {
  const vertexRequest = request as VertexClaudeRequest;

  const messages: UnifiedMessage[] = vertexRequest.messages.map((msg) => {
    const content = msg.content.map((item) => {
      if (item.type === "text") {
        return {
          type: "text" as const,
          text: item.text || "",
        };
      } else if (item.type === "image" && item.source) {
        return {
          type: "image_url" as const,
          image_url: {
            url: item.source.data,
          },
          media_type: item.source.media_type,
        };
      }
      return {
        type: "text" as const,
        text: "",
      };
    });

    return {
      role: msg.role,
      content,
    };
  });

  const result: UnifiedChatRequest = {
    messages,
    model: request.model || "claude-sonnet-4@20250514",
    max_tokens: vertexRequest.max_tokens,
    temperature: vertexRequest.temperature,
    stream: vertexRequest.stream,
  };

  // Handle tool definitions
  if (vertexRequest.tools && vertexRequest.tools.length > 0) {
    result.tools = vertexRequest.tools.map((tool) => ({
      type: "function" as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: {
          type: "object" as const,
          properties: tool.input_schema.properties,
          required: tool.input_schema.required,
          additionalProperties: tool.input_schema.additionalProperties,
          $schema: tool.input_schema.$schema,
        },
      },
    }));
  }

  // Handle tool choice
  if (vertexRequest.tool_choice) {
    if (typeof vertexRequest.tool_choice === "string") {
      result.tool_choice = vertexRequest.tool_choice;
    } else if (vertexRequest.tool_choice.type === "tool") {
      result.tool_choice = vertexRequest.tool_choice.name;
    }
  }

  return result;
}

export async function transformResponseOut(
  response: Response,
  providerName: string,
  logger?: any
): Promise<Response> {
  if (response.headers.get("Content-Type")?.includes("application/json")) {
    const jsonResponse = (await response.json()) as any;

    // Prefer standard Anthropic content[] blocks (text / tool_use / thinking).
    // Fall back to the Vertex-only top-level tool_use array when present.
    const contentBlocks: any[] = Array.isArray(jsonResponse.content)
      ? jsonResponse.content
      : [];
    const textParts = contentBlocks
      .filter((c) => c?.type === "text" && typeof c.text === "string")
      .map((c) => c.text);
    const thinkingBlock = contentBlocks.find(
      (c) => c?.type === "thinking" && typeof c.thinking === "string"
    );

    let tool_calls: any[] | undefined;
    const contentToolUses = contentBlocks.filter((c) => c?.type === "tool_use");
    if (contentToolUses.length > 0) {
      tool_calls = contentToolUses.map((tool) => ({
        id: tool.id,
        type: "function" as const,
        function: {
          name: tool.name,
          arguments: JSON.stringify(tool.input ?? {}),
        },
      }));
    } else if (
      Array.isArray(jsonResponse.tool_use) &&
      jsonResponse.tool_use.length > 0
    ) {
      tool_calls = jsonResponse.tool_use.map((tool: any) => ({
        id: tool.id,
        type: "function" as const,
        function: {
          name: tool.name,
          arguments: JSON.stringify(tool.input ?? {}),
        },
      }));
    }

    const finishReason = anthropicStopReasonToOpenAI(
      jsonResponse.stop_reason
    );

    const message: Record<string, any> = {
      role: "assistant",
      content: textParts.length > 0 ? textParts.join("") : null,
      ...(tool_calls && { tool_calls }),
    };
    if (thinkingBlock) {
      message.thinking = {
        content: thinkingBlock.thinking,
        signature: thinkingBlock.signature,
      };
    }

    const res = {
      id: jsonResponse.id,
      choices: [
        {
          finish_reason: finishReason,
          index: 0,
          message,
        },
      ],
      created: parseInt(new Date().getTime() / 1000 + "", 10),
      model: jsonResponse.model,
      object: "chat.completion",
      usage: toOpenAIUsage(jsonResponse.usage),
    };

    return new Response(JSON.stringify(res), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  } else if (response.headers.get("Content-Type")?.includes("stream")) {
    // Handle streaming response
    if (!response.body) {
      return response;
    }

    let streamInputUsage: Record<string, any> = {};
    let streamId = "";
    let streamModel = "";
    const toolBlockIndexes = new Map<number, number>();
    const processLine = (
      line: string,
      ctx: { controller: ReadableStreamDefaultController, encoder: TextEncoder }
    ) => {
      const { controller, encoder } = ctx;
      if (line.startsWith("data: ")) {
        const chunkStr = line.slice(6).trim();
        if (chunkStr) {
          logger?.debug({ chunkStr }, `${providerName} chunk:`);
          try {
            const chunk = JSON.parse(chunkStr);

            // Handle Anthropic native format streaming response
            if (chunk.type === "message_start") {
              streamInputUsage = chunk.message?.usage || chunk.usage || {};
              streamId = chunk.message?.id || chunk.id || streamId;
              streamModel = chunk.message?.model || chunk.model || streamModel;
              const res = {
                choices: [
                  {
                    delta: { role: "assistant" },
                    finish_reason: null,
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: streamId,
                model: streamModel,
                object: "chat.completion.chunk",
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            } else if (
              chunk.type === "content_block_delta" &&
              chunk.delta?.type === "text_delta"
            ) {
              // This is Anthropic native format, need to convert to OpenAI format
              const res = {
                choices: [
                  {
                    delta: {
                      role: "assistant",
                      content: chunk.delta.text || "",
                    },
                    finish_reason: null,
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.id || streamId,
                model: chunk.model || streamModel,
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                ...(chunk.usage
                  ? { usage: toOpenAIUsage(chunk.usage) }
                  : {}),
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            } else if (
              chunk.type === "content_block_delta" &&
              chunk.delta?.type === "input_json_delta"
            ) {
              // Handle tool call argument delta
              const res = {
                choices: [
                  {
                    delta: {
                      tool_calls: [
                        {
                          index:
                            toolBlockIndexes.get(Number(chunk.index || 0)) ??
                            Number(chunk.index || 0),
                          function: {
                            arguments: chunk.delta.partial_json || "",
                          },
                        },
                      ],
                    },
                    finish_reason: null,
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.id || streamId,
                model: chunk.model || streamModel,
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                ...(chunk.usage
                  ? { usage: toOpenAIUsage(chunk.usage) }
                  : {}),
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            } else if (
              chunk.type === "content_block_delta" &&
              chunk.delta?.type === "thinking_delta"
            ) {
              const thinkingText = chunk.delta.thinking || "";
              // display:"omitted" (and similar) yields empty thinking_delta —
              // skip rather than emit a useless Unified thinking chunk.
              if (!thinkingText) return;
              const res = {
                choices: [
                  {
                    delta: {
                      thinking: {
                        content: thinkingText,
                      },
                    },
                    finish_reason: null,
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.id || streamId,
                model: chunk.model || streamModel,
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                ...(chunk.usage
                  ? { usage: toOpenAIUsage(chunk.usage) }
                  : {}),
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            } else if (
              chunk.type === "content_block_delta" &&
              chunk.delta?.type === "signature_delta"
            ) {
              const res = {
                choices: [
                  {
                    delta: {
                      thinking: {
                        signature: chunk.delta.signature || "",
                      },
                    },
                    finish_reason: null,
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.id || streamId,
                model: chunk.model || streamModel,
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                ...(chunk.usage
                  ? { usage: toOpenAIUsage(chunk.usage) }
                  : {}),
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            } else if (
              chunk.type === "content_block_start" &&
              chunk.content_block?.type === "tool_use"
            ) {
              // Handle tool call start
              const blockIndex = Number(chunk.index || 0);
              const toolIndex = toolBlockIndexes.size;
              toolBlockIndexes.set(blockIndex, toolIndex);
              const res = {
                choices: [
                  {
                    delta: {
                      tool_calls: [
                        {
                          index: toolIndex,
                          id: chunk.content_block.id,
                          type: "function",
                          function: {
                            name: chunk.content_block.name,
                            arguments: "",
                          },
                        },
                      ],
                    },
                    finish_reason: null,
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.id || streamId,
                model: chunk.model || streamModel,
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                ...(chunk.usage
                  ? { usage: toOpenAIUsage(chunk.usage) }
                  : {}),
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            } else if (chunk.type === "message_delta") {
              // Handle message end
              const res = {
                choices: [
                  {
                    delta: {},
                    finish_reason:
                      anthropicStopReasonToOpenAI(
                        chunk.delta?.stop_reason
                      ) || "stop",
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.id || streamId,
                model: chunk.model || streamModel,
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                usage: toOpenAIUsage({
                  ...streamInputUsage,
                  ...chunk.usage,
                }),
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            } else if (chunk.type === "message_stop") {
              // Send end marker
              controller.enqueue(encoder.encode(`data: [DONE]\n\n`));
            } else if (
              chunk.type === "content_block_start" ||
              chunk.type === "content_block_stop" ||
              chunk.type === "ping"
            ) {
              // Lifecycle-only Anthropic events have no Chat delta payload.
              return;
            } else {
              // Handle other format responses (keep original logic as fallback)
              const res = {
                choices: [
                  {
                    delta: {
                      role: "assistant",
                      content: chunk.content?.[0]?.text || "",
                    },
                    finish_reason: anthropicStopReasonToOpenAI(
                      chunk.stop_reason
                    ),
                    index: 0,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.id || streamId,
                model: chunk.model || streamModel,
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                ...(chunk.usage
                  ? { usage: toOpenAIUsage(chunk.usage) }
                  : {}),
              };
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );
            }
          } catch (error: any) {
            logger?.error(
              `Error parsing ${providerName} stream chunk`,
              chunkStr,
              error.message
            );
          }
        }
      }
    };

    return createSSEStreamReader(response, processLine);
  }
  return response;
}

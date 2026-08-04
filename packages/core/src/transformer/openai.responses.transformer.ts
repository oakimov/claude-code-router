import { UnifiedChatRequest, MessageContent } from "@/types/llm";
import { Transformer, TransformerContext } from "@/types/transformer";
import { createApiError } from "@/api/middleware";
import { sanitizeUpstreamErrorText } from "@/utils/redact";
import { sanitizeResponsesCallId } from "@/utils/toolCallId";
import {
  applyOpenAIChatCaching,
  validateOpenAIToolCalls,
  openAIContentCacheBreakpoint,
} from "../utils/openai.util";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "../utils/stream";
import {
  createCallIdMap,
  createResponsesStreamState,
  finalizeResponsesStream,
  responsesFailedEvent,
  responsesRequestToUnified,
  unifiedChunkToResponsesEvents,
  unifiedResponseToResponses,
  type ResponsesCallIdMap,
} from "../utils/openai.responses.util";

interface ResponsesAPIOutputItem {
  type: string;
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
	  content?: Array<{
	    type: string;
	    text?: string;
	    image_url?: string;
	    mime_type?: string;
	    image_base64?: string;
	    annotations?: Array<{
	      url?: string;
	      title?: string;
	      start_index?: number;
	      end_index?: number;
	    }>;
	  }>;
  reasoning?: string;
}

interface ResponsesAPIPayload {
  id: string;
  object: string;
  model: string;
  created_at: number;
  output: ResponsesAPIOutputItem[];
  usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    input_tokens_details?: {
      cached_tokens?: number;
      cache_write_tokens?: number;
    };
  };
}

interface ResponsesStreamEvent {
  type: string;
  item_id?: string;
  output_index?: number;
  delta?:
    | string
    | {
        url?: string;
        b64_json?: string;
        mime_type?: string;
      };
  item?: {
    id?: string;
    type?: string;
    call_id?: string;
    name?: string;
    content?: Array<{
      type: string;
      text?: string;
      image_url?: string;
    }>;
    reasoning?: string;
  };
  response?: {
    id?: string;
    model?: string;
    status?: string;
    error?: {
      message?: string;
      type?: string;
      code?: string;
    };
    output?: Array<{
      type: string;
    }>;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
      input_tokens_details?: {
        cached_tokens?: number;
        cache_write_tokens?: number;
      };
    };
  };
  reasoning_summary?: string;
  annotation?: {
    url?: string;
    title?: string;
    start_index?: number;
    end_index?: number;
  };
  part?: any;
}

export class OpenAIResponsesTransformer implements Transformer {
  logger?: any;
  name = "openai-responses";
  endPoint = "/v1/responses";

  /**
   * Client → Unified: validate Responses MVP and project to Chat Completions shape.
   * Call-id mapping is stored on the transformer context for the response path.
   */
  async transformRequestOut(
    request: any,
    context?: TransformerContext
  ): Promise<UnifiedChatRequest> {
    const callIdMap = createCallIdMap();
    if (context) {
      (context as any).responsesCallIdMap = callIdMap;
      if ((context as any).protocolContext) {
        (context as any).protocolContext.responsesCallIdMap = callIdMap;
      }
    }
    return responsesRequestToUnified(request, callIdMap);
  }

  /**
   * Unified → client Responses: JSON object or SSE lifecycle with mandatory
   * content_part events (Codex requires these or streamed text is discarded).
   */
  async transformResponseIn(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    const callIdMap: ResponsesCallIdMap =
      (context as any)?.responsesCallIdMap ||
      (context as any)?.protocolContext?.responsesCallIdMap ||
      (context as any)?.req?.protocolContext?.responsesCallIdMap ||
      createCallIdMap();
    const originalModel =
      (context as any)?.protocolContext?.originalModel ||
      (context as any)?.req?.protocolContext?.originalModel;

    const contentType = response.headers.get("Content-Type") || "";
    if (contentType.includes("application/json")) {
      const json: any = await response.json();
      // Already a Responses object (same-protocol passthrough).
      if (json?.object === "response") {
        return new Response(JSON.stringify(json), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }
      const responsesBody = unifiedResponseToResponses(json, {
        originalModel,
        callIdMap,
      });
      return new Response(JSON.stringify(responsesBody), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    if (contentType.includes("text/event-stream")) {
      return this.convertUnifiedStreamToResponses(response, {
        callIdMap,
        originalModel,
      });
    }

    return response;
  }

  private convertUnifiedStreamToResponses(
    response: Response,
    options: { callIdMap: ResponsesCallIdMap; originalModel?: string }
  ): Response {
    if (!response.body) return response;

    const state = createResponsesStreamState({
      model: options.originalModel,
      callIdMap: options.callIdMap,
    });

    return createSSEStreamReader(
      response,
      (line: string, ctx: StreamContext) => {
        if (state.finished) return;
        if (!line.trim()) return;
        if (line.startsWith("event: ")) return;

        if (!line.startsWith("data: ")) {
          // SSE comments are transport keepalives. Any other non-data payload
          // is malformed for the supported Responses compatibility stream.
          if (!line.startsWith(":")) {
            const failed = responsesFailedEvent(
              "Malformed upstream SSE event",
              state
            );
            ctx.controller.enqueue(
              encodeSSEData(JSON.stringify(failed), ctx.encoder)
            );
            state.finished = true;
          }
          return;
        }

        const dataStr = line.slice(5).trim();
        if (dataStr === "[DONE]") {
          if (!state.finished) {
            for (const event of finalizeResponsesStream(state)) {
              ctx.controller.enqueue(
                encodeSSEData(JSON.stringify(event), ctx.encoder)
              );
            }
          }
          // Responses streams do not require Chat Completions' [DONE].
          return;
        }

        try {
          const data = JSON.parse(dataStr);

          // Upstream already speaking Responses — pass through.
          if (typeof data?.type === "string" && data.type.startsWith("response.")) {
            ctx.controller.enqueue(
              encodeSSEData(JSON.stringify(data), ctx.encoder)
            );
            if (data.type === "response.completed" || data.type === "response.failed") {
              state.finished = true;
            }
            return;
          }

          // Provider error object mid-stream.
          if (data?.error && !data?.choices) {
            const failed = responsesFailedEvent(
              sanitizeUpstreamErrorText(
                String(data.error.message || data.error)
              ) || "Upstream stream failed",
              state
            );
            ctx.controller.enqueue(
              encodeSSEData(JSON.stringify(failed), ctx.encoder)
            );
            state.finished = true;
            return;
          }

          const events = unifiedChunkToResponsesEvents(data, state);
          for (const event of events) {
            ctx.controller.enqueue(
              encodeSSEData(JSON.stringify(event), ctx.encoder)
            );
          }
        } catch {
          const failed = responsesFailedEvent(
            "Malformed JSON in upstream SSE event",
            state
          );
          ctx.controller.enqueue(
            encodeSSEData(JSON.stringify(failed), ctx.encoder)
          );
          state.finished = true;
        }
      },
      {
        onComplete: (ctx: StreamContext) => {
          if (!state.finished) {
            for (const event of finalizeResponsesStream(state)) {
              ctx.controller.enqueue(
                encodeSSEData(JSON.stringify(event), ctx.encoder)
              );
            }
          }
        },
        onError: (error: unknown, ctx: StreamContext) => {
          if (!state.finished) {
            const message =
              error instanceof Error ? error.message : String(error);
            const failed = responsesFailedEvent(
              sanitizeUpstreamErrorText(message) || "Upstream stream failed",
              state
            );
            ctx.controller.enqueue(
              encodeSSEData(JSON.stringify(failed), ctx.encoder)
            );
            state.finished = true;
          }
          return true;
        },
        logger: this.logger,
      }
    );
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider?: any,
    context?: any
  ): Promise<UnifiedChatRequest> {
    request = structuredClone(request);
    const tokenLimit =
      request.max_tokens ?? (request as any).max_completion_tokens;
    if (tokenLimit != null) {
      (request as any).max_output_tokens = tokenLimit;
    }
    delete request.max_tokens;
    delete (request as any).max_completion_tokens;

    if (request.reasoning) {
      request.reasoning = {
        ...(request.reasoning.effort
          ? { effort: request.reasoning.effort }
          : {}),
        ...((request.reasoning as any).summary
          ? { summary: (request.reasoning as any).summary }
          : {}),
      };
    }
    if ((request as any).stop !== undefined) {
      throw createApiError(
        "stop cannot be represented by the Responses API provider adapter",
        400,
        "unsupported_stop",
        "invalid_request_error"
      );
    }
    // Chat's include_usage request is unnecessary for Responses: terminal
    // response events carry usage when the provider reports it.
    delete (request as any).stream_options;

    const model = request.model || "";
    request = applyOpenAIChatCaching(request, provider, context);
    const messages = validateOpenAIToolCalls(request.messages);
    request.messages = messages;

    const input: any[] = [];
    let lastWasTool = false;


    const systemMessages = request.messages.filter(
      (msg) => msg.role === "system"
    );
    systemMessages.forEach((systemMessage, systemIndex) => {
      if (Array.isArray(systemMessage.content)) {
        systemMessage.content.forEach((item) => {
          let text = "";
          if (typeof item === "string") {
            text = item;
          } else if (item && typeof item === "object" && "text" in item) {
            text = (item as { text: string }).text;
          }
          input.push({
            role: "system",
            content: [
              {
                type: "input_text",
                text,
                ...openAIContentCacheBreakpoint(item, model),
              },
            ],
          });
        });
      } else if (systemIndex === 0) {
        (request as any).instructions = systemMessage.content;
      } else {
        input.push({
          role: "system",
          content: [
            { type: "input_text", text: String(systemMessage.content || "") },
          ],
        });
      }
    });

    request.messages.forEach((message) => {
      if (message.role === "system") return;

      if (Array.isArray(message.content)) {
        const convertedContent = message.content
          .map((content) =>
            this.normalizeRequestContent(content, message.role, model)
          )
          .filter(
            (content): content is Record<string, unknown> => content !== null
          );

        if (convertedContent.length > 0) {
          (message as any).content = convertedContent;
        } else {
          delete (message as any).content;
        }
      }

      if (message.role === "tool") {
        const toolMessage: any = { ...message };
        toolMessage.type = "function_call_output";
        toolMessage.call_id =
          sanitizeResponsesCallId(message.tool_call_id) ?? message.tool_call_id;
        toolMessage.output = message.content;
        delete toolMessage.cache_control;
        delete toolMessage.role;
        delete toolMessage.tool_call_id;
        delete toolMessage.content;
        input.push(toolMessage);
        lastWasTool = true;
        return;
      }

      if (message.role === "assistant" && Array.isArray(message.tool_calls)) {
        const hasContent = message.content &&
          (typeof message.content === "string" ||
            (Array.isArray(message.content) && message.content.length > 0));
        lastWasTool = false;
        if (hasContent) {
          const contentMessage: any = { ...message };
          delete contentMessage.tool_calls;
          input.push(contentMessage);
        }
        message.tool_calls.forEach((tool) => {
          input.push({
            type: "function_call",
            arguments: tool.function.arguments,
            name: tool.function.name,
            call_id: sanitizeResponsesCallId(tool.id) ?? tool.id,
          });
        });

        return;
      }

      // If a user message follows a tool output, insert a dummy assistant message
      if (lastWasTool && message.role === "user") {
        input.push({
          role: "assistant",
          content: "",
        });
      }
      lastWasTool = false;
      input.push(message);
    });

    (request as any).input = input;
    delete (request as any).messages;

    if (Array.isArray(request.tools)) {
      const webSearch = request.tools.find(
        (tool: any) => tool?.type === "web_search"
      );

      (request as any).tools = request.tools
        .filter((tool: any) => tool?.type !== "web_search")
        .map((tool: any) => {
          if (tool.function.name === "WebSearch") {
            if (tool.function.parameters?.properties) {
              delete tool.function.parameters.properties.allowed_domains;
            }
          }
          if (tool.function.name === "Edit") {
            return {
              type: tool.type,
              name: tool.function.name,
              description: tool.function.description,
              parameters: {
                ...tool.function.parameters,
                required: [
                  "file_path",
                  "old_string",
                  "new_string",
                  "replace_all",
                ],
              },
              strict: true,
            };
          }
          return {
            type: tool.type,
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters,
          };
        });

      if (webSearch) {
        (request as any).tools.push({
          ...webSearch,
          type: "web_search",
        });
      }
    }

    if (
      request.tool_choice &&
      typeof request.tool_choice === "object" &&
      request.tool_choice.type === "function"
    ) {
      (request as any).tool_choice = {
        type: "function",
        name: request.tool_choice.function?.name,
      };
    }

    return request;
  }

  async transformResponseOut(response: Response): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";

    if (contentType.includes("application/json")) {
      const jsonResponse: any = await response.json();

      if (jsonResponse.object === "response" && jsonResponse.output) {
        const chatResponse = this.convertResponseToChat(jsonResponse);
        return new Response(JSON.stringify(chatResponse), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }

      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } else if (contentType.includes("text/event-stream")) {
      if (!response.body) {
        return response;
      }

      // Stream state scoped to this one upstream response. Tool identity is
      // keyed by the Responses item id (falling back to output_index), never
      // inferred from event-type transitions, so interleaved parallel calls
      // keep stable Chat tool indexes.
      const toolIndexByKey = new Map<string, number>();
      let nextToolIndex = 0;
      let terminated = false;

      const toolIndexFor = (
        data: ResponsesStreamEvent
      ): number => {
        const key =
          data.item_id ||
          data.item?.id ||
          (typeof data.output_index === "number"
            ? `output:${data.output_index}`
            : undefined);
        if (key !== undefined) {
          const existing = toolIndexByKey.get(key);
          if (existing !== undefined) return existing;
          const index = nextToolIndex++;
          toolIndexByKey.set(key, index);
          return index;
        }
        return nextToolIndex++;
      };

      const terminate = () => {
        terminated = true;
        toolIndexByKey.clear();
      };

      return createSSEStreamReader(
        response,
        (line: string, ctx: StreamContext) => {
          if (terminated) return;
          if (!line.trim()) return;

          if (line.startsWith("event: ")) return;

          if (line.startsWith("data: ")) {
            const dataStr = line.slice(5).trim();
            if (dataStr === "[DONE]") {
              ctx.controller.enqueue(encodeSSEData("[DONE]", ctx.encoder));
              terminate();
              return;
            }

            try {
              const data: ResponsesStreamEvent = JSON.parse(dataStr);

              // Terminal provider failure: emit one protocol-shaped streamed
              // error, then [DONE]. Partial output already delivered stays
              // delivered, but the stream must never end looking like a
              // successful empty completion.
              if (data.type === "response.failed") {
                terminate();
                const failed = data.response?.error ?? {};
                const message =
                  sanitizeUpstreamErrorText(
                    String(failed.message || "Upstream response failed")
                  ) || "Upstream response failed";
                ctx.controller.enqueue(
                  encodeSSEData(
                    JSON.stringify({
                      error: {
                        message,
                        type:
                          typeof failed.type === "string"
                            ? failed.type
                            : "api_error",
                        code:
                          typeof failed.code === "string" ? failed.code : null,
                      },
                    }),
                    ctx.encoder
                  )
                );
                ctx.controller.enqueue(encodeSSEData("[DONE]", ctx.encoder));
                return;
              }

              const chunk = this.convertStreamEvent(data, toolIndexFor);
              if (chunk) {
                ctx.controller.enqueue(encodeSSEData(JSON.stringify(chunk), ctx.encoder));
              }

              // Responses upstreams end after response.completed without a
              // [DONE]; Chat consumers rely on the terminator, so add it.
              if (data.type === "response.completed") {
                ctx.controller.enqueue(encodeSSEData("[DONE]", ctx.encoder));
                terminate();
              }
            } catch {
              ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
            }
          } else {
            ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          }
        }
      );
    }

    return response;
  }

  /**
   * Convert one Responses stream event to a Chat chunk. `choices[0].index` is
   * always 0 — parallel-call identity lives in `delta.tool_calls[n].index`,
   * allocated per Responses item by `toolIndexFor`.
   */
  private convertStreamEvent(data: ResponsesStreamEvent, toolIndexFor: (data: ResponsesStreamEvent) => number): any | null {
    if (data.type === "response.output_text.delta") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: 0,
            delta: {
              content: data.delta || "",
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_item.added" && data.item?.type === "function_call") {
      return {
        id: data.item.call_id || data.item.id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model || "gpt-5-codex-",
        choices: [
          {
            index: 0,
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index: toolIndexFor(data),
                  id:
                    sanitizeResponsesCallId(
                      data.item.call_id || data.item.id
                    ) || data.item.call_id || data.item.id,
                  function: {
                    name: data.item.name || "",
                    arguments: "",
                  },
                  type: "function",
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_item.added" && data.item?.type === "message") {
      const contentItems: MessageContent[] = [];
      (data.item.content || []).forEach((item: any) => {
        if (item.type === "output_text") {
          contentItems.push({
            type: "text",
            text: item.text || "",
          });
        }
      });

      const delta: any = { role: "assistant" };
      if (contentItems.length === 1 && contentItems[0].type === "text") {
        delta.content = contentItems[0].text;
      } else if (contentItems.length > 0) {
        delta.content = contentItems;
      }
      if (delta.content) {
        return {
          id: data.item.id || "chatcmpl-" + Date.now(),
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: data.response?.model,
          choices: [
            {
              index: 0,
              delta,
              finish_reason: null,
            },
          ],
        };
      }
      return null;
    }

    if (data.type === "response.output_text.annotation.added") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model || "gpt-5-codex",
        choices: [
          {
            index: 0,
            delta: {
              annotations: [
                {
                  type: "url_citation",
                  url_citation: {
                    url: data.annotation?.url || "",
                    title: data.annotation?.title || "",
                    content: "",
                    start_index: data.annotation?.start_index || 0,
                    end_index: data.annotation?.end_index || 0,
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.function_call_arguments.delta") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model || "gpt-5-codex-",
        choices: [
          {
            index: 0,
            delta: {
              tool_calls: [
                {
                  index: toolIndexFor(data),
                  function: {
                    arguments: data.delta || "",
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.completed") {
      const finishReason = data.response?.output?.some(
        (item: any) => item.type === "function_call"
      )
        ? "tool_calls"
        : "stop";

      const chunk: any = {
        id: data.response?.id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model || "gpt-5-codex-",
        choices: [
          {
            index: 0,
            delta: {},
            finish_reason: finishReason,
          },
        ],
      };

      if (data.response?.usage) {
        chunk.usage = {
          prompt_tokens: data.response.usage.input_tokens || 0,
          completion_tokens: data.response.usage.output_tokens || 0,
          total_tokens: data.response.usage.total_tokens || 0,
          prompt_tokens_details: {
            cached_tokens:
              data.response.usage.input_tokens_details?.cached_tokens || 0,
            cache_write_tokens:
              data.response.usage.input_tokens_details?.cache_write_tokens || 0,
          },
        };
      }

      return chunk;
    }

    if (data.type === "response.reasoning_summary_text.delta") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: 0,
            delta: {
              thinking: {
                content: data.delta || "",
              },
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.reasoning_summary_part.done" && data.part) {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: 0,
            delta: {
              thinking: {
                signature: data.item_id,
              },
            },
            finish_reason: null,
          },
        ],
      };
    }

    return null;
  }

  private normalizeRequestContent(content: any, role: string | undefined, model: string) {
    // cache_control is already converted to prompt_cache_breakpoint by
    // applyOpenAIChatCaching, so no explicit strip is needed here.
    if (content.type === "text") {
      return {
        type: role === "assistant" ? "output_text" : "input_text",
        text: content.text,
        ...openAIContentCacheBreakpoint(content, model),
      };
    }

    if (content.type === "image_url") {
      this.logger?.debug(content);
      const imagePayload: Record<string, unknown> = {
        type: role === "assistant" ? "output_image" : "input_image",
      };

      if (typeof content.image_url?.url === "string") {
        imagePayload.image_url = content.image_url.url;
      }

      return {
        ...imagePayload,
        ...openAIContentCacheBreakpoint(content, model),
      };
    }

    return null;
  }

  private convertResponseToChat(responseData: ResponsesAPIPayload): any {
    const messageOutput = responseData.output?.find(
      (item) => item.type === "message"
    );
    // Every function_call output survives, in source order — parallel calls
    // must not collapse into the first one found.
    const functionCallOutputs = (responseData.output ?? []).filter(
      (item) => item.type === "function_call"
    );
    let annotations;
    if (
      messageOutput?.content?.length &&
      messageOutput?.content[0].annotations
    ) {
      annotations = messageOutput.content[0].annotations.map((item) => {
        return {
          type: "url_citation",
          url_citation: {
            url: item.url || "",
            title: item.title || "",
            content: "",
            start_index: item.start_index || 0,
            end_index: item.end_index || 0,
          },
        };
      });
    }

    this.logger?.debug?.({
      data: annotations,
      type: "url_citation",
    });

    let messageContent: string | MessageContent[] | null = null;
    let toolCalls = null;
    let thinking = null;

    if (messageOutput && messageOutput.reasoning) {
      thinking = {
        content: messageOutput.reasoning,
      };
    }

    if (messageOutput && messageOutput.content) {
      const textParts: string[] = [];
      const imageParts: MessageContent[] = [];

      messageOutput.content.forEach((item: any) => {
        if (item.type === "output_text") {
          textParts.push(item.text || "");
        } else if (item.type === "output_image") {
          const imageContent = this.buildImageContent({
            url: item.image_url,
            mime_type: item.mime_type,
          });
          if (imageContent) {
            imageParts.push(imageContent);
          }
        } else if (item.type === "output_image_base64") {
          const imageContent = this.buildImageContent({
            b64_json: item.image_base64,
            mime_type: item.mime_type,
          });
          if (imageContent) {
            imageParts.push(imageContent);
          }
        }
      });

      if (imageParts.length > 0) {
        const contentArray: MessageContent[] = [];
        if (textParts.length > 0) {
          contentArray.push({
            type: "text",
            text: textParts.join(""),
          });
        }
        contentArray.push(...imageParts);
        messageContent = contentArray;
      } else {
        messageContent = textParts.join("");
      }
    }

    if (functionCallOutputs.length > 0) {
      toolCalls = functionCallOutputs.map((call) => ({
        id:
          sanitizeResponsesCallId(call.call_id || call.id) ||
          call.call_id ||
          call.id,
        function: {
          name: call.name,
          arguments: call.arguments,
        },
        type: "function",
      }));
    }

    return {
      id: responseData.id || "chatcmpl-" + Date.now(),
      object: "chat.completion",
      created: responseData.created_at,
      model: responseData.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: messageContent || null,
            tool_calls: toolCalls,
            thinking: thinking,
            annotations: annotations,
          },
          logprobs: null,
          finish_reason: toolCalls ? "tool_calls" : "stop",
        },
      ],
      usage: responseData.usage
        ? {
            prompt_tokens: responseData.usage.input_tokens || 0,
            completion_tokens: responseData.usage.output_tokens || 0,
            total_tokens: responseData.usage.total_tokens || 0,
            prompt_tokens_details: {
              cached_tokens:
                responseData.usage.input_tokens_details?.cached_tokens || 0,
              cache_write_tokens:
                responseData.usage.input_tokens_details?.cache_write_tokens ||
                0,
            },
          }
        : null,
    };
  }

  private buildImageContent(source: {
    url?: string;
    b64_json?: string;
    mime_type?: string;
  }): MessageContent | null {
    if (!source) return null;

    if (source.url || source.b64_json) {
      return {
        type: "image_url",
        image_url: {
          url: source.url || "",
          b64_json: source.b64_json,
        },
        media_type: source.mime_type,
      } as MessageContent;
    }

    return null;
  }
}

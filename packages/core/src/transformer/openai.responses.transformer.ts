import { UnifiedChatRequest, MessageContent } from "@/types/llm";
import { Transformer, TransformerContext } from "@/types/transformer";
import { sanitizeUpstreamErrorText } from "@/utils/redact";
import { sanitizeResponsesCallId } from "@/utils/toolCallId";
import {
  applyOpenAIChatCaching,
  validateOpenAIToolCalls,
  openAIContentCacheBreakpoint,
} from "../utils/openai.util";
import {
  isReasoningDisabled,
  normalizeReasoningEffort,
  resolveOutboundReasoningSummary,
} from "@/utils/reasoning-effort";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "../utils/stream";
import {
  CUSTOM_TOOL_INPUT_KEY,
  assistantTurnHasText,
  canonicalAssistantTurn,
  createCallIdMap,
  createResponsesStreamState,
  finalizeResponsesStream,
  normalizeResponsesInclude,
  recordReasoningSummaryDelta,
  responsesFailedEvent,
  responsesReasoningItemFromThinking,
  responsesRequestToUnified,
  responsesTextFormatFromResponseFormat,
  thinkingForLateReasoningItem,
  thinkingFromResponsesReasoningItem,
  thinkingFromUnifiedAssistant,
  unifiedChunkToResponsesEvents,
  unifiedResponseToResponses,
  uniquifyReasoningItemIds,
  type ResponsesCallIdMap,
} from "../utils/openai.responses.util";
import {
  assistantMessageFromResponsesOutput,
  createEncryptedReasoningStreamRecorder,
  hasEncryptedReasoningContext,
  prepareEncryptedReasoningReplay,
  recordEncryptedReasoningResponseMessage,
} from "../utils/responses.encrypted-content-cache";

interface ResponsesAPIOutputItem {
  type: string;
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  input?: string;
  summary?: Array<{ type?: string; text?: string } | string>;
  encrypted_content?: string;
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
  arguments?: string;
  input?: string;
  item?: {
    id?: string;
    type?: string;
    call_id?: string;
    name?: string;
    arguments?: string;
    input?: string;
    summary?: Array<{ type?: string; text?: string } | string>;
    encrypted_content?: string;
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
      id?: string;
      call_id?: string;
      name?: string;
      arguments?: string;
      input?: string;
      content?: Array<{
        type: string;
        text?: string;
      }>;
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
  text?: string;
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
    const customToolNames = new Set<string>();
    if (context) {
      (context as any).responsesCallIdMap = callIdMap;
      (context as any).responsesCustomToolNames = customToolNames;
      if ((context as any).protocolContext) {
        (context as any).protocolContext.responsesCallIdMap = callIdMap;
        (context as any).protocolContext.responsesCustomToolNames = customToolNames;
      }
    }
    return responsesRequestToUnified(request, callIdMap, customToolNames);
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
    const customToolNames: Set<string> =
      (context as any)?.responsesCustomToolNames ||
      (context as any)?.protocolContext?.responsesCustomToolNames ||
      (context as any)?.req?.protocolContext?.responsesCustomToolNames ||
      new Set<string>();
    // `/v1/responses` inbound is the Codex CLI path; Grok-via-Chat recovery
    // of `{cmd:…}` / apply_patch heredocs depends on isolate conventions.

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
        customToolNames,
        codexIsolateConventions: true,
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
      customToolNames,
      codexIsolateConventions: true,
    });
    }

    return response;
  }

  private convertUnifiedStreamToResponses(
    response: Response,
    options: {
      callIdMap: ResponsesCallIdMap;
      originalModel?: string;
      customToolNames: Set<string>;
      codexIsolateConventions?: boolean;
    }
  ): Response {
    if (!response.body) return response;

    const state = createResponsesStreamState({
      model: options.originalModel,
      callIdMap: options.callIdMap,
      customToolNames: options.customToolNames,
      codexIsolateConventions: options.codexIsolateConventions,
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
      // Responses enforces max_output_tokens >= 16; Anthropic clients may
      // legitimately send smaller max_tokens values, so clamp to the floor.
      const MIN_OUTPUT_TOKENS = 16;
      (request as any).max_output_tokens = Math.max(
        MIN_OUTPUT_TOKENS,
        Number(tokenLimit)
      );
    }
    delete request.max_tokens;
    delete (request as any).max_completion_tokens;

    if (request.reasoning) {
      const effort = isReasoningDisabled(request.reasoning, request.thinking)
        ? "none"
        : normalizeReasoningEffort(request.reasoning.effort);
      const summary =
        effort !== "none"
          ? resolveOutboundReasoningSummary(request, provider)
          : undefined;
      request.reasoning = {
        ...(effort ? { effort } : {}),
        ...(summary ? { summary } : {}),
      };
    }
    // Chat/Anthropic stop sequences have no Responses equivalent. Omit
    // rather than reject so a Claude Code or Chat client is not failed
    // for a field the destination cannot express.
    delete (request as any).stop;
    // Chat's include_usage request is unnecessary for Responses: terminal
    // response events carry usage when the provider reports it.
    delete (request as any).stream_options;

    // Chat Completions `response_format` is the Unified stand-in for Responses
    // `text.format` (see responsesRequestToUnified). Restore the native field
    // for every Responses destination — xAI and OpenAI both accept it — then
    // drop the Chat-only field so it is not forwarded as an unknown property.
    const format = responsesTextFormatFromResponseFormat(
      (request as any).response_format
    );
    if (format) {
      (request as any).text = { ...(request as any).text, format };
    }
    delete (request as any).response_format;

    const model = request.model || "";
    request = applyOpenAIChatCaching(request, provider, context);
    // Anthropic/Chat cannot round-trip Responses ciphertext. Request
    // reasoning.encrypted_content and restore prior-turn ciphertext onto
    // assistant tool turns before the Responses input is built.
    prepareEncryptedReasoningReplay(request, provider, context);
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

    const pushReasoningFromMessage = (message: any) => {
      const reasoningItem = responsesReasoningItemFromThinking(
        thinkingFromUnifiedAssistant(message)
      );
      if (reasoningItem) input.push(reasoningItem);
    };

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

      if (message.role === "assistant") {
        const turn = canonicalAssistantTurn(message);
        if (turn.thinking || turn.toolCalls.length || assistantTurnHasText(turn) || turn.images.length) {
          lastWasTool = false;
          if (turn.thinking) pushReasoningFromMessage(message);
          if (assistantTurnHasText(turn) || turn.images.length) {
            const contentMessage: any = { ...message };
            delete contentMessage.tool_calls;
            delete contentMessage.thinking;
            delete contentMessage.reasoning_content;
            if (turn.texts.length === 1 && turn.images.length === 0) {
              contentMessage.content = turn.texts[0].text;
            }
            input.push(contentMessage);
          }
          for (const tool of turn.toolCalls) {
            input.push({
              type: "function_call",
              arguments: tool.function.arguments,
              name: tool.function.name,
              call_id: sanitizeResponsesCallId(tool.id) ?? tool.id,
            });
          }
          return;
        }
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
    uniquifyReasoningItemIds(input);
    delete (request as any).messages;

    if (Array.isArray(request.tools)) {
      // Unified carries web_search as a function tool (Responses→Unified
      // projects hosted tools onto functions). Re-emit the Responses hosted
      // shape when talking to a Responses backend.
      const isWebSearch = (tool: any) =>
        tool?.type === "web_search" ||
        tool?.function?.name === "web_search";
      const webSearch = request.tools.find(isWebSearch);

      (request as any).tools = request.tools
        .filter((tool: any) => !isWebSearch(tool))
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
    } else if (request.tool_choice === "required") {
      (request as any).tool_choice = "required";
    } else if (request.tool_choice === "none") {
      (request as any).tool_choice = "none";
    } else if (request.tool_choice === "auto") {
      (request as any).tool_choice = "auto";
    }

    if (request.parallel_tool_calls !== undefined) {
      (request as any).parallel_tool_calls = request.parallel_tool_calls;
    }

    // Client-driven Responses hints, plus encrypted_content for Anthropic/Chat
    // (prepareEncryptedReasoningReplay already merged that include when needed).
    const include = normalizeResponsesInclude(request.include);
    if (include) {
      request.include = include;
    } else {
      delete (request as any).include;
    }
    if (request.store === false) {
      request.store = false;
    } else {
      delete (request as any).store;
    }

    return request;
  }

  async transformResponseOut(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";
    const shouldCacheEncrypted = hasEncryptedReasoningContext(context);

    if (contentType.includes("application/json")) {
      const jsonResponse: any = await response.json();

      if (jsonResponse.object === "response" && jsonResponse.output) {
        if (shouldCacheEncrypted) {
          recordEncryptedReasoningResponseMessage(
            assistantMessageFromResponsesOutput(jsonResponse.output),
            context
          );
        }
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
      const textByItemId = new Map<string, string>();
      const thinkingByItemId = new Map<string, string>();
      const encryptedRecorder = shouldCacheEncrypted
        ? createEncryptedReasoningStreamRecorder()
        : undefined;
      let nextToolIndex = 0;
      let terminated = false;
      let completed = false;

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

      const terminate = (successful = false) => {
        completed = completed || successful;
        terminated = true;
        toolIndexByKey.clear();
        encryptedRecorder?.discard();
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
              encryptedRecorder?.observe(data);

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

              const chunks = this.convertStreamEvent(
                data,
                toolIndexFor,
                textByItemId,
                thinkingByItemId
              );
              for (const chunk of chunks) {
                ctx.controller.enqueue(
                  encodeSSEData(JSON.stringify(chunk), ctx.encoder)
                );
              }

              // Responses upstreams end after response.completed without a
              // [DONE]; Chat consumers rely on the terminator, so add it.
              // Ciphertext often arrives only on the completed reasoning item.
              if (data.type === "response.completed") {
                if (shouldCacheEncrypted) {
                  recordEncryptedReasoningResponseMessage(
                    assistantMessageFromResponsesOutput(
                      encryptedRecorder?.completedOutput()
                    ),
                    context
                  );
                }
                ctx.controller.enqueue(encodeSSEData("[DONE]", ctx.encoder));
                terminate(true);
              }
            } catch {
              encryptedRecorder?.discard();
              ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
            }
          } else {
            ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          }
        },
        {
          onComplete: () => {
            if (!completed) encryptedRecorder?.discard();
          },
          onError: () => {
            encryptedRecorder?.discard();
            return false;
          },
          logger: this.logger,
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
  private convertStreamEvent(
    data: ResponsesStreamEvent,
    toolIndexFor: (data: ResponsesStreamEvent) => number,
    textByItemId: Map<string, string>,
    thinkingByItemId: Map<string, string>
  ): any[] {
    const textFromItem = (item: any): string =>
      (item?.content || [])
        .filter((part: any) => part?.type === "output_text")
        .map((part: any) => part.text || "")
        .join("");
    const asArray = (chunk: any): any[] => (chunk ? [chunk] : []);
    // output_item.added carries item.id (no top-level item_id); deltas and
    // output_text.done carry item_id. Using only one of those keys made a
    // later terminal copy look unseen and re-emitted the full string.
    const itemKey = (item?: any) =>
      item?.id || data.item?.id || data.item_id || "text";

    // Record every text fragment the provider emits — deltas, message items
    // opened with content, and terminal copies — keyed by item id. The
    // "remainder" logic in the output_text.done / output_item.done /
    // response.completed branches must see the already-delivered text or it
    // re-emits the full text on top of the deltas (xAI: "hello"+" x"+"ai" then
    // output_text.done "hello xai" produced "hello xaihello xai" at the client).
    const recordDelta = (text: string) => {
      const id = itemKey();
      textByItemId.set(
        id,
        (textByItemId.get(id) || "") + (typeof text === "string" ? text : "")
      );
    };
    if (
      data.type === "response.output_text.delta" &&
      typeof data.delta === "string"
    ) {
      recordDelta(data.delta);
    } else if (
      data.type === "response.output_item.added" &&
      data.item?.type === "message"
    ) {
      // Some hosts open a message item already carrying text; record it so a
      // later terminal copy isn't mistaken for unsent content.
      recordDelta(textFromItem(data.item));
    }

    if (
      data.type === "response.reasoning_summary_text.delta" &&
      typeof data.delta === "string"
    ) {
      recordReasoningSummaryDelta(thinkingByItemId, data.item_id, data.delta);
    }

    if (data.type === "response.output_text.done" && typeof data.text === "string") {
      const id = itemKey();
      const previous = textByItemId.get(id) || "";
      const remainder = data.text.startsWith(previous)
        ? data.text.slice(previous.length)
        : data.text;
      textByItemId.set(id, data.text);
      return remainder ? asArray({
        id,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [{ index: 0, delta: { content: remainder }, finish_reason: null }],
      }) : [];
    }

    if (data.type === "response.output_item.done" && data.item?.type === "message") {
      const id = itemKey(data.item);
      const text = textFromItem(data.item);
      const previous = textByItemId.get(id) || "";
      const remainder = text.startsWith(previous) ? text.slice(previous.length) : text;
      textByItemId.set(id, text);
      return remainder ? asArray({
        id,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [{ index: 0, delta: { content: remainder }, finish_reason: null }],
      }) : [];
    }

    if (
      data.type === "response.output_item.done" &&
      data.item?.type === "reasoning"
    ) {
      const thinking = thinkingForLateReasoningItem(
        data.item,
        thinkingByItemId
      );
      if (!thinking) return [];
      return asArray({
        id: data.item.id || data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: 0,
            delta: { thinking },
            finish_reason: null,
          },
        ],
      });
    }

    if (data.type === "response.completed") {
      const chunks: any[] = [];
      const terminalOutput = data.response?.output || [];
      for (const item of terminalOutput) {
        if (item.type !== "message") continue;
        const id = item.id || "text";
        const text = textFromItem(item);
        if (text && !textByItemId.has(id)) textByItemId.set(id, "");
      }

      // Terminal Responses payloads sometimes contain the only copy of text;
      // emit it before the Chat finish chunk so downstream clients cannot see
      // a successful empty completion.
      for (const item of data.response?.output || []) {
        if (item.type !== "message") continue;
        const id = item.id || "text";
        const text = textFromItem(item);
        const previous = textByItemId.get(id) || "";
        const remainder = text.startsWith(previous) ? text.slice(previous.length) : text;
        textByItemId.set(id, text);
        if (remainder) chunks.push({
          id,
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: data.response?.model,
          choices: [{ index: 0, delta: { content: remainder }, finish_reason: null }],
        });
      }
      // Ciphertext often arrives only on the completed reasoning item — emit
      // replay metadata (and content only if summary deltas never streamed it)
      // before the finish chunk so Unified history can replay it.
      for (const item of data.response?.output || []) {
        if (item.type !== "reasoning") continue;
        const thinking = thinkingForLateReasoningItem(item, thinkingByItemId);
        if (!thinking) continue;
        chunks.push({
          id: item.id || data.response?.id || "chatcmpl-" + Date.now(),
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: data.response?.model,
          choices: [{ index: 0, delta: { thinking }, finish_reason: null }],
        });
      }
      // Reuse the core finish chunk so token usage stays attached. This
      // interceptor only exists to emit unseen terminal text first.
      const finish = this.convertStreamEventCore(data, toolIndexFor);
      if (finish) chunks.push(finish);
      return chunks;
    }

    return asArray(this.convertStreamEventCore(data, toolIndexFor));
  }

  private convertStreamEventCore(
    data: ResponsesStreamEvent,
    toolIndexFor: (data: ResponsesStreamEvent) => number
  ): any | null {
    if (data.type === "response.created") {
      return {
        id: data.response?.id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: 0,
            delta: { role: "assistant" },
            finish_reason: null,
          },
        ],
      };
    }

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
        (item: any) => item.type === "function_call" || item.type === "custom_tool_call"
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
                ...(typeof data.item_id === "string" && data.item_id
                  ? { id: data.item_id }
                  : {}),
              },
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.reasoning_summary_part.done" && data.part) {
      // Preserve the reasoning item id for replay. Never store it as
      // thinking.signature — that field used to be copied into
      // encrypted_content and Codex rejected the forged ciphertext.
      if (typeof data.item_id !== "string" || !data.item_id) return null;
      return {
        id: data.item_id,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: 0,
            delta: {
              thinking: {
                id: data.item_id,
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
      // Responses input_image.detail (auto|low|high|original) — preserve when
      // clients/SDKs set it on Chat-shaped Unified image_url parts.
      if (
        typeof content.image_url?.detail === "string" &&
        content.image_url.detail
      ) {
        imagePayload.detail = content.image_url.detail;
      }

      return {
        ...imagePayload,
        ...openAIContentCacheBreakpoint(content, model),
      };
    }

    if (content.type === "file") {
      const filePayload: Record<string, unknown> = {
        type: "input_file",
      };
      if (typeof content.filename === "string" && content.filename) {
        filePayload.filename = content.filename;
      }
      if (typeof content.file_data === "string" && content.file_data) {
        filePayload.file_data = content.file_data;
      }
      if (typeof content.file_url === "string" && content.file_url) {
        filePayload.file_url = content.file_url;
      }
      return {
        ...filePayload,
        ...openAIContentCacheBreakpoint(content, model),
      };
    }

    return null;
  }

  private convertResponseToChat(responseData: ResponsesAPIPayload): any {
    const messageOutput = responseData.output?.find(
      (item) => item.type === "message"
    );
    const reasoningOutput = responseData.output?.find(
      (item) => item.type === "reasoning"
    );
    // Every function_call output survives, in source order — parallel calls
    // must not collapse into the first one found.
    const functionCallOutputs = (responseData.output ?? []).filter(
      (item) => item.type === "function_call" || item.type === "custom_tool_call"
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
    let thinking = thinkingFromResponsesReasoningItem(reasoningOutput) || null;

    if (!thinking && messageOutput && messageOutput.reasoning) {
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
          arguments:
            call.type === "custom_tool_call"
              ? JSON.stringify({
                  [CUSTOM_TOOL_INPUT_KEY]: call.input || "",
                })
              : call.arguments,
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

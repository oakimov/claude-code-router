import { randomUUID } from "crypto";
import { UnifiedChatRequest, MessageContent } from "@/types/llm";
import { Transformer } from "@/types/transformer";
import { validateOpenAIToolCalls, injectPromptCaching } from "../utils/openai.util";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "../utils/stream";
import { stripCacheControl } from "../utils/cacheControl";
import { getValidAccessToken } from "../utils/codex-auth";

// Module-level cache for PAT whoami results — keyed by PAT value.
// Auto-invalidates when the user changes api_key in the provider config.
const whoamiCache = new Map<string, { accountId: string; isFedramp: boolean }>();

interface WhoamiResponse {
  chatgpt_account_id?: string;
  chatgpt_account_is_fedramp?: boolean;
}

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
  };
}

interface ResponsesStreamEvent {
  type: string;
  item_id?: string;
  output_index?: number;
  text?: string;
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
    output?: Array<{
      type: string;
    }>;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
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

interface CodexTurnMetadata {
  session_id: string;
  thread_id: string;
  turn_id: string;
  window_id: string;
  request_kind: "turn";
  turn_started_at_unix_ms: number;
}

const CODEX_TURN_METADATA_HEADER = "x-codex-turn-metadata";
const CODEX_WINDOW_ID_HEADER = "x-codex-window-id";
const CODEX_PARENT_THREAD_ID_HEADER = "x-codex-parent-thread-id";
const CODEX_SESSION_ID_KEY = "session_id";
const CODEX_THREAD_ID_KEY = "thread_id";
const CODEX_TURN_ID_KEY = "turn_id";
const CODEX_WINDOW_ID_KEY = "window_id";

function buildCodexTurnMetadata(sessionId: string): CodexTurnMetadata {
  return {
    session_id: sessionId,
    thread_id: sessionId,
    turn_id: randomUUID(),
    window_id: `${sessionId}:0`,
    request_kind: "turn",
    turn_started_at_unix_ms: Date.now(),
  };
}

function buildCodexClientMetadata(turnMetadata: CodexTurnMetadata): Record<string, string> {
  return {
    [CODEX_SESSION_ID_KEY]: turnMetadata.session_id,
    [CODEX_THREAD_ID_KEY]: turnMetadata.thread_id,
    [CODEX_TURN_ID_KEY]: turnMetadata.turn_id,
    [CODEX_WINDOW_ID_KEY]: turnMetadata.window_id,
    [CODEX_TURN_METADATA_HEADER]: JSON.stringify(turnMetadata),
  };
}

function mergeClientMetadata(
  existing: unknown,
  additions: Record<string, string>
): Record<string, string> {
  if (!existing || typeof existing !== "object" || Array.isArray(existing)) {
    return { ...additions };
  }

  return {
    ...(existing as Record<string, string>),
    ...additions,
  };
}

function buildCodexCompatibilityHeaders(turnMetadata: CodexTurnMetadata): Record<string, string> {
  return {
    [CODEX_WINDOW_ID_HEADER]: turnMetadata.window_id,
    [CODEX_TURN_METADATA_HEADER]: JSON.stringify(turnMetadata),
    [CODEX_PARENT_THREAD_ID_HEADER]: turnMetadata.thread_id,
  };
}

function inferSessionIdFromMetadata(request: UnifiedChatRequest): string | undefined {
  const userId = (request as any)?.metadata?.user_id;
  if (typeof userId !== "string" || !userId) {
    return undefined;
  }

  const parts = userId.split("_session_");
  if (parts.length > 1 && parts[1]) {
    return parts[1];
  }

  try {
    const parsed = JSON.parse(userId);
    if (parsed && typeof parsed.session_id === "string" && parsed.session_id) {
      return parsed.session_id;
    }
  } catch {
    // Ignore non-JSON user_id formats.
  }

  return undefined;
}

function getCodexSessionId(context: any, request: UnifiedChatRequest): string | undefined {
  return context?.req?.sessionId || inferSessionIdFromMetadata(request);
}

function applyCodexTurnMetadata(request: UnifiedChatRequest, headers: Record<string, string>, sessionId?: string): void {
  if (!sessionId) {
    return;
  }

  const turnMetadata = buildCodexTurnMetadata(sessionId);
  (request as any).client_metadata = mergeClientMetadata(
    (request as any).client_metadata,
    buildCodexClientMetadata(turnMetadata)
  );
  Object.assign(headers, buildCodexCompatibilityHeaders(turnMetadata));
}

export class CodexTransformer implements Transformer {
  name = "codex";
  logger?: any;

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any,
    context?: any
  ): Promise<Record<string, any>> {
    // Codex API requires streaming — propagate to original request body so formatResponse streams
    if (context?.req?.body) {
      context.req.body.stream = true;
    }
    // Determine auth method: api_key starting with "at-" → PAT, anything else → OAuth
    const apiKey = typeof provider?.apiKey === "string"
      ? provider.apiKey.trim()
      : provider?.apiKey;
    const isPat = typeof apiKey === "string" && apiKey.startsWith("at-");

    let token: string;
    let accountId: string | undefined;
    let isFedramp = false;

    if (isPat) {
      token = apiKey;
      const patInfo = await this.resolvePatAuth(apiKey);
      accountId = patInfo.accountId;
      isFedramp = patInfo.isFedramp;
    } else {
      const tokenData = await getValidAccessToken();
      token = tokenData.access_token;
      accountId = tokenData.account_id;
    }

    delete request.temperature;
    delete request.max_tokens;

    // Effort comes from the unified reasoning field (populated by anthropic.transformer
    // from the user's /effort setting via output_config.effort or request.effort).
    const effort = (request as any).reasoning?.effort || provider?.reasoningEffort;
    if (effort) {
      const reasoning: Record<string, any> = { effort };
      const VALID_SUMMARIES = ["auto", "detailed", "none"];
      const summary = provider?.reasoningSummary;
      if (summary && VALID_SUMMARIES.includes(summary)) {
        if (summary !== "none") {
          reasoning.summary = summary;
        }
      } else {
        // Default to detailed reasoning summaries when effort is enabled so
        // Claude Code reliably receives visible thinking unless the provider
        // explicitly disables them with reasoningSummary: "none".
        reasoning.summary = "detailed";
      }
      (request as any).reasoning = reasoning;
    } else {
      delete (request as any).reasoning;
    }

    // Pass through verbosity when configured (Codex models accept low|medium|high)
    const VALID_VERBOSITIES = ["low", "medium", "high"];
    if (provider?.verbosity && VALID_VERBOSITIES.includes(provider.verbosity)) {
      (request as any).verbosity = provider.verbosity;
    }

    const model = request.model || "";
    let messages = validateOpenAIToolCalls(request.messages);
    messages = injectPromptCaching(messages, model);
    request.messages = messages;

    const input: any[] = [];
    let lastWasTool = false;

    const systemMessages = request.messages.filter(
      (msg) => msg.role === "system"
    );
    if (systemMessages.length > 0) {
      const firstSystem = systemMessages[0];
      let instructionsText = "";
      if (Array.isArray(firstSystem.content)) {
        instructionsText = firstSystem.content
          .map((item) => {
            if (typeof item === "string") return item;
            if (item && typeof item === "object" && "text" in item)
              return (item as { text: string }).text;
            return "";
          })
          .filter(Boolean)
          .join("\n");
      } else {
        instructionsText = firstSystem.content as string;
      }
      if (instructionsText) {
        (request as any).instructions = instructionsText;
      }
    }

    // Codex API requires instructions — provide a default if none set
    if (!(request as any).instructions) {
      (request as any).instructions = "";
    }

    request.messages.forEach((message) => {
      if (message.role === "system") return;

      if (Array.isArray(message.content)) {
        const convertedContent = message.content
          .map((content) => this.normalizeRequestContent(content, message.role))
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
        toolMessage.call_id = message.tool_call_id;
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
        lastWasTool = false;
        message.tool_calls.forEach((tool) => {
          input.push({
            type: "function_call",
            arguments: tool.function.arguments,
            name: tool.function.name,
            call_id: tool.id,
          });
        });
        return;
      }

      if (lastWasTool && message.role === "user") {
        input.push({
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "" }],
        });
      }
      lastWasTool = false;

      if (typeof message.content === "string") {
        input.push({
          type: "message",
          role: message.role,
          content: [
            {
              type: message.role === "assistant" ? "output_text" : "input_text",
              text: message.content,
            },
          ],
        });
      } else {
        input.push({
          type: "message",
          role: message.role,
          content: message.content,
        });
      }
    });

    (request as any).input = input;
    delete (request as any).messages;

    // Codex API requires store to be false
    (request as any).store = false;

    if (Array.isArray(request.tools)) {
      const webSearch = request.tools.find(
        (tool) => tool.function.name === "web_search"
      );

      (request as any).tools = request.tools
        .filter((tool) => tool.function.name !== "web_search")
        .map((tool) => {
          if (tool.function.name === "WebSearch") {
            delete tool.function.parameters.properties.allowed_domains;
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

    // Default to serial tool calls for safety; provider can opt-in to parallel via config
    request.parallel_tool_calls = provider?.parallelToolCalls === true;
    (request as any).stream = true;

    const headers: Record<string, string> = {
      Authorization: `Bearer ${token}`,
      "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    };

    if (accountId) {
      headers["ChatGPT-Account-ID"] = accountId;
    }
    if (isFedramp) {
      headers["X-OpenAI-Fedramp"] = "true";
    }

    applyCodexTurnMetadata(request, headers, getCodexSessionId(context, request));

    const baseUrl = provider?.baseUrl || "https://chatgpt.com/backend-api/codex";

    return {
      body: request,
      config: {
        url: `${baseUrl}/responses`,
        headers,
      },
    };
  }

  async auth(
    _request: any,
    provider: any
  ): Promise<any> {
    const apiKey = typeof provider?.apiKey === "string"
      ? provider.apiKey.trim()
      : provider?.apiKey;
    const isPat = typeof apiKey === "string" && apiKey.startsWith("at-");

    let token: string;
    let accountId: string | undefined;
    let isFedramp = false;

    if (isPat) {
      token = apiKey;
      const patInfo = await this.resolvePatAuth(apiKey);
      accountId = patInfo.accountId;
      isFedramp = patInfo.isFedramp;
    } else {
      const tokenData = await getValidAccessToken();
      token = tokenData.access_token;
      accountId = tokenData.account_id;
    }

    const headers: Record<string, string> = {
      Authorization: `Bearer ${token}`,
      "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
    };

    if (accountId) {
      headers["ChatGPT-Account-ID"] = accountId;
    }
    if (isFedramp) {
      headers["X-OpenAI-Fedramp"] = "true";
    }

    const baseUrl = provider?.baseUrl || "https://chatgpt.com/backend-api/codex";

    return {
      config: {
        url: `${baseUrl}/responses`,
        headers,
      },
    };
  }

  private async resolvePatAuth(pat: string): Promise<{ accountId: string; isFedramp: boolean }> {
    const cached = whoamiCache.get(pat);
    if (cached) return cached;

    const response = await fetch(
      "https://auth.openai.com/api/accounts/v1/user-auth-credential/whoami",
      {
        headers: { Authorization: `Bearer ${pat}` },
      }
    );

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(
        `Codex PAT whoami failed (${response.status}): ${text}. ` +
        "Verify the api_key contains a valid PAT, or remove it to use OAuth authentication."
      );
    }

    const data: WhoamiResponse = await response.json();
    if (!data.chatgpt_account_id) {
      throw new Error(
        "Codex PAT whoami response missing chatgpt_account_id. " +
        "The PAT may not have Codex access. Run `ccr codex-auth` for OAuth as a fallback."
      );
    }

    const result = {
      accountId: data.chatgpt_account_id,
      isFedramp: data.chatgpt_account_is_fedramp === true,
    };
    whoamiCache.set(pat, result);
    return result;
  }

  async transformResponseOut(response: Response): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";
    const codexTransformer = this;

    // Cloudflare strips Content-Type from SSE responses — treat missing content-type as SSE
    if (!contentType || contentType.includes("text/event-stream")) {
      if (!response.body) {
        return response;
      }

      let isStreamEnded = false;
      let currentIndex = -1;
      let lastEventType = "";

      const getCurrentIndex = (eventType: string) => {
        if (eventType !== lastEventType) {
          currentIndex++;
          lastEventType = eventType;
        }
        return currentIndex;
      };

      return createSSEStreamReader(
        response,
        (line: string, ctx: StreamContext) => {
          if (!line.trim()) return;

          if (line.startsWith("event: ")) return;

          if (line.startsWith("data: ")) {
            const dataStr = line.slice(5).trim();
            if (dataStr === "[DONE]") {
              isStreamEnded = true;
              ctx.controller.enqueue(encodeSSEData("[DONE]", ctx.encoder));
              return;
            }

            try {
              const data: ResponsesStreamEvent = JSON.parse(dataStr);
              const chunk = codexTransformer.convertStreamEvent(data, getCurrentIndex);
              if (chunk) {
                ctx.controller.enqueue(encodeSSEData(JSON.stringify(chunk), ctx.encoder));
              }
            } catch {
              ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
            }
          } else {
            ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          }
        }
      );
    } else if (contentType.includes("application/json")) {
      const jsonResponse: any = await response.json();

      // Streaming response: JSON array of events
      if (Array.isArray(jsonResponse)) {
        const encoder = new TextEncoder();
        let index = -1;
        let lastEventType = "";
        const getCurrentIndex = (eventType: string) => {
          if (eventType !== lastEventType) {
            index++;
            lastEventType = eventType;
          }
          return index;
        };

        const stream = new ReadableStream({
          start(controller) {
            for (const event of jsonResponse) {
              const chunk = codexTransformer.convertStreamEvent(event, getCurrentIndex);
              if (chunk) {
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
                );
              }
            }
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          },
        });

        return new Response(stream, {
          status: response.status,
          statusText: response.statusText,
          headers: { "Content-Type": "text/event-stream" },
        });
      }

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
    }

    return response;
  }

  private convertStreamEvent(data: ResponsesStreamEvent, getCurrentIndex: (type: string) => number): any | null {
    if (data.type === "response.output_text.delta") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: getCurrentIndex(data.type),
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
            index: getCurrentIndex(data.type),
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index: 0,
                  id: data.item.call_id || data.item.id,
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
              index: getCurrentIndex(data.type),
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
            index: getCurrentIndex(data.type),
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
            index: getCurrentIndex(data.type),
            delta: {
              tool_calls: [
                {
                  index: 0,
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
            index: getCurrentIndex(data.type),
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

    if (data.type === "response.reasoning_summary_text.done") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: getCurrentIndex(data.type),
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

    if (data.type === "response.output_text.done") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {},
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_item.done") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: data.response?.model,
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {},
            finish_reason: null,
          },
        ],
      };
    }

    return null;
  }

  private normalizeRequestContent(content: any, role: string | undefined) {
    const clone = stripCacheControl(content);

    if (content.type === "text") {
      return {
        type: role === "assistant" ? "output_text" : "input_text",
        text: content.text,
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

      return imagePayload;
    }

    return null;
  }

  private convertResponseToChat(responseData: ResponsesAPIPayload): any {
    const messageOutput = responseData.output?.find(
      (item) => item.type === "message"
    );
    const functionCallOutput = responseData.output?.find(
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

    this.logger.debug({
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

    if (functionCallOutput) {
      toolCalls = [
        {
          id: functionCallOutput.call_id || functionCallOutput.id,
          function: {
            name: functionCallOutput.name,
            arguments: functionCallOutput.arguments,
          },
          type: "function",
        },
      ];
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

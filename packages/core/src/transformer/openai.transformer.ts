import { Transformer, TransformerContext } from "@/types/transformer";
import { UnifiedChatRequest } from "@/types/llm";
import { applyProviderNativeChatCaching } from "../utils/openai.util";
import { createApiError } from "@/api/middleware";
import {
  isChatCompletionsDoneLine,
  pushChatCompletionsDone,
  splitChatCompletionsDoneLine,
} from "@/utils/sse/done-boundary";
import {
  applyOpenAIChatReasoning,
  canonicalReasoning,
  normalizeReasoningEffort,
} from "@/utils/reasoning-effort";

/**
 * Server-side route handler for the OpenAI Chat Completions API.
 *
 * The Unified format IS the OpenAI Chat Completions format. Client inbound
 * normalization validates the MVP subset and passes through already-correct
 * Chat bodies. Provider-side transformRequestIn applies cache policy for
 * OpenAI-compatible upstreams.
 *
 * ## Full request pipeline (for context)
 *
 *     Client → POST /v1/messages
 *       → AnthropicTransformer.transformRequestOut()        // Anthropic → Unified (OpenAI)
 *       → provider.transformer.use[].transformRequestIn()   // provider middleware
 *       → sendRequestToProvider()                           // HTTP call upstream
 *       → provider.transformer.use[].transformResponseOut() // provider middleware (reversed)
 *       → AnthropicTransformer.transformResponseIn()        // Unified (OpenAI) → Anthropic
 *       → Client
 *
 * For inbound Chat Completions clients:
 *
 *     Client → POST /v1/chat/completions
 *       → OpenAITransformer.transformRequestOut()           // validate → Unified
 *       → provider.transformer.use[].transformRequestIn()
 *       → … → OpenAITransformer.transformResponseIn()       // identity / light normalize
 */
export class OpenAITransformer implements Transformer {
  name = "OpenAI";
  endPoint = "/v1/chat/completions";

  /**
   * Client → Unified: validate the Chat Completions MVP subset.
   * Unified is already Chat-shaped, so this is validation + light normalization.
   */
  async transformRequestOut(
    request: any,
    _context?: TransformerContext
  ): Promise<UnifiedChatRequest> {
    return validateAndNormalizeChatRequest(request);
  }

  /**
   * Provider-side: apply OpenAI-native cache policy to a Unified Chat body.
   */
  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any,
    context: any
  ): Promise<UnifiedChatRequest> {
    request = structuredClone(request);
    applyOpenAIChatReasoning(request);
    request.messages = applyChatReasoningHistory(request.messages);
    return applyProviderNativeChatCaching(request, provider, context);
  }

  /**
   * Unified → client Chat Completions: pass through already-correct Chat JSON/SSE.
   * Ensures Content-Type and that streaming responses terminate with [DONE] when
   * the upstream already speaks Chat Completions.
   */
  async transformResponseIn(
    response: Response,
    _context?: TransformerContext
  ): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";
    if (contentType.includes("text/event-stream")) {
      return ensureChatStreamDone(response, { aliasThinking: true });
    }
    if (contentType.includes("application/json")) {
      const json = await response.json();
      return new Response(JSON.stringify(applyChatThinkingToCompletion(json)), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }
    return response;
  }
}

function validateAndNormalizeChatRequest(body: any): UnifiedChatRequest {
  if (!body || typeof body !== "object") {
    throw createApiError(
      "Invalid Chat Completions body",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }
  if (typeof body.model !== "string" || !body.model.trim()) {
    throw createApiError(
      "Chat Completions requires model",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }
  if (!Array.isArray(body.messages)) {
    throw createApiError(
      "Chat Completions requires messages[]",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }
  if (body.stream !== undefined && typeof body.stream !== "boolean") {
    throw createApiError(
      "stream must be a boolean",
      400,
      "invalid_stream",
      "invalid_request_error"
    );
  }

  if (body.n !== undefined && body.n !== 1) {
    throw createApiError(
      "n values other than 1 are not supported",
      400,
      "unsupported_n",
      "invalid_request_error"
    );
  }
  if (body.logprobs === true || body.top_logprobs !== undefined) {
    throw createApiError(
      "logprobs/top_logprobs are not supported",
      400,
      "unsupported_logprobs",
      "invalid_request_error"
    );
  }
  if (Array.isArray(body.modalities)) {
    const nonText = body.modalities.filter(
      (m: unknown) => m !== "text" && m !== undefined
    );
    if (nonText.length > 0) {
      throw createApiError(
        "Non-text output modalities are not supported",
        400,
        "unsupported_modalities",
        "invalid_request_error"
      );
    }
  }
  if (body.audio !== undefined) {
    throw createApiError(
      "Audio input/output is not supported",
      400,
      "unsupported_audio",
      "invalid_request_error"
    );
  }
  if (body.store === true || body.metadata !== undefined) {
    throw createApiError(
      "Stored Chat Completions and metadata are not supported",
      400,
      "unsupported_state",
      "invalid_request_error"
    );
  }
  for (const field of [
    "prediction",
    "web_search_options",
    "moderation",
    "service_tier",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "user",
    "verbosity",
    "safety_identifier",
    "functions",
    "function_call",
  ]) {
    if (body[field] !== undefined && body[field] !== null) {
      throw createApiError(
        `Chat Completions field '${field}' is not supported`,
        400,
        "unsupported_field",
        "invalid_request_error"
      );
    }
  }
  if (body.response_format !== undefined) {
    const rf = body.response_format;
    const type = typeof rf === "object" && rf ? rf.type : rf;
    if (type && type !== "text" && type !== "json_schema" && type !== "json_object") {
      throw createApiError(
        "response_format structured output is not supported without a verified provider mapping",
        400,
        "unsupported_response_format",
        "invalid_request_error"
      );
    }
  }

  if (body.stream_options != null && body.stream !== true) {
    throw createApiError(
      "stream_options requires stream: true",
      400,
      "invalid_stream_options",
      "invalid_request_error"
    );
  }
  if (body.stream_options != null) {
    if (
      typeof body.stream_options !== "object" ||
      Array.isArray(body.stream_options)
    ) {
      throw createApiError(
        "stream_options must be an object",
        400,
        "invalid_stream_options",
        "invalid_request_error"
      );
    }
    const unsupportedStreamOption = Object.keys(body.stream_options).find(
      (key) => key !== "include_usage"
    );
    if (unsupportedStreamOption) {
      throw createApiError(
        `Unsupported stream option '${unsupportedStreamOption}'`,
        400,
        "invalid_stream_options",
        "invalid_request_error"
      );
    }
    if (
      body.stream_options.include_usage !== undefined &&
      typeof body.stream_options.include_usage !== "boolean"
    ) {
      throw createApiError(
        "stream_options.include_usage must be a boolean",
        400,
        "invalid_stream_options",
        "invalid_request_error"
      );
    }
  }

  const hasMaxTokens = body.max_tokens != null;
  const hasMaxCompletion = body.max_completion_tokens != null;
  if (
    hasMaxTokens &&
    hasMaxCompletion &&
    Number(body.max_tokens) !== Number(body.max_completion_tokens)
  ) {
    throw createApiError(
      "max_tokens and max_completion_tokens both present and differ",
      400,
      "conflicting_token_limits",
      "invalid_request_error"
    );
  }

  for (const [field, present] of [
    ["max_tokens", hasMaxTokens],
    ["max_completion_tokens", hasMaxCompletion],
  ] as const) {
    if (!present) continue;
    const value = Number(body[field]);
    if (!Number.isInteger(value) || value <= 0) {
      throw createApiError(
        `${field} must be a positive integer`,
        400,
        "invalid_token_limit",
        "invalid_request_error"
      );
    }
  }

  validateChatTools(body.tools, body.tool_choice);

  if (
    body.parallel_tool_calls !== undefined &&
    typeof body.parallel_tool_calls !== "boolean"
  ) {
    throw createApiError(
      "parallel_tool_calls must be a boolean",
      400,
      "invalid_parallel_tool_calls",
      "invalid_request_error"
    );
  }

  const messages = body.messages.map((msg: any) => normalizeChatMessage(msg));

  const unified: UnifiedChatRequest = {
    model: body.model,
    messages,
    stream: body.stream === true,
  };

  if (body.stream_options) {
    unified.stream_options = body.stream_options;
  }
  if (hasMaxTokens) {
    unified.max_tokens = Number(body.max_tokens);
  }
  if (hasMaxCompletion) {
    unified.max_completion_tokens = Number(body.max_completion_tokens);
    if (!hasMaxTokens) {
      unified.max_tokens = Number(body.max_completion_tokens);
    }
  }
  if (body.temperature !== undefined) unified.temperature = body.temperature;
  if (body.top_p !== undefined) unified.top_p = body.top_p;
  if (body.stop !== undefined) (unified as any).stop = body.stop;
  if (body.tools !== undefined) unified.tools = body.tools;
  if (body.tool_choice !== undefined) unified.tool_choice = body.tool_choice;
  if (body.parallel_tool_calls !== undefined) {
    unified.parallel_tool_calls = body.parallel_tool_calls;
  }
  if (body.response_format !== undefined) {
    const rf = body.response_format;
    const type = typeof rf === "object" && rf ? rf.type : rf;
    if (type === "json_schema" || type === "json_object") {
      (unified as any).response_format = rf;
    }
  }
  const chatEffort = normalizeReasoningEffort(
    body.reasoning_effort ?? body.reasoning?.effort
  );
  if (body.reasoning !== undefined || chatEffort) {
    unified.reasoning = {
      ...(body.reasoning && typeof body.reasoning === "object"
        ? body.reasoning
        : {}),
      ...(canonicalReasoning(
        chatEffort,
        body.reasoning?.enabled
      ) || {}),
    };
  }
  if (body.thinking !== undefined) unified.thinking = body.thinking;

  return unified;
}

function normalizeChatMessage(msg: any): any {
  if (!msg || typeof msg !== "object") {
    throw createApiError(
      "Invalid message in messages[]",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }
  const role = msg.role;
  if (!["developer", "system", "user", "assistant", "tool"].includes(role)) {
    throw createApiError(
      `Unsupported message role '${role}'`,
      400,
      "unsupported_role",
      "invalid_request_error"
    );
  }
  if (msg.audio !== undefined) {
    throw createApiError(
      "Audio input/output is not supported",
      400,
      "unsupported_audio",
      "invalid_request_error"
    );
  }
  if (msg.name !== undefined) {
    throw createApiError(
      "Named messages are not supported by the cross-protocol compatibility tier",
      400,
      "unsupported_message_name",
      "invalid_request_error"
    );
  }
  if (Array.isArray(msg.content)) {
    const allowedPartsByRole: Record<string, Set<string>> = {
      developer: new Set(["text"]),
      system: new Set(["text"]),
      user: new Set(["text", "image_url"]),
      assistant: new Set(["text", "refusal"]),
      tool: new Set(["text"]),
    };
    for (const part of msg.content) {
      if (!part || typeof part !== "object") {
        throw createApiError(
          `Invalid content part for role '${role}'`,
          400,
          "unsupported_content_part",
          "invalid_request_error"
        );
      }
      if (part?.type === "input_audio" || part?.type === "audio") {
        throw createApiError(
          "Audio input/output is not supported",
          400,
          "unsupported_audio",
          "invalid_request_error"
        );
      }
      if (part?.type === "file") {
        throw createApiError(
          "Provider-hosted file inputs are not supported",
          400,
          "unsupported_file",
          "invalid_request_error"
        );
      }
      if (!allowedPartsByRole[role]?.has(part?.type)) {
        throw createApiError(
          `Unsupported content part '${part?.type || "unknown"}' for role '${role}'`,
          400,
          "unsupported_content_part",
          "invalid_request_error"
        );
      }
      if (
        (part.type === "text" || part.type === "refusal") &&
        typeof part.text !== "string" &&
        typeof part.refusal !== "string"
      ) {
        throw createApiError(
          `Content part '${part.type}' requires text`,
          400,
          "invalid_content_part",
          "invalid_request_error"
        );
      }
      if (
        part.type === "image_url" &&
        (typeof part.image_url !== "object" ||
          typeof part.image_url?.url !== "string" ||
          !part.image_url.url)
      ) {
        throw createApiError(
          "image_url content requires image_url.url",
          400,
          "invalid_image",
          "invalid_request_error"
        );
      }
    }
  } else if (
    typeof msg.content !== "string" &&
    !(role === "assistant" && msg.content == null)
  ) {
    throw createApiError(
      `Message role '${role}' requires string or array content`,
      400,
      "invalid_message_content",
      "invalid_request_error"
    );
  }
  if (role === "tool") {
    if (typeof msg.tool_call_id !== "string" || !msg.tool_call_id) {
      throw createApiError(
        "Tool messages require tool_call_id",
        400,
        "invalid_tool_call_id",
        "invalid_request_error"
      );
    }
  } else if (msg.tool_call_id !== undefined) {
    throw createApiError(
      "tool_call_id is only valid on tool messages",
      400,
      "invalid_tool_call_id",
      "invalid_request_error"
    );
  }
  if (msg.tool_calls !== undefined) {
    if (role !== "assistant" || !Array.isArray(msg.tool_calls)) {
      throw createApiError(
        "tool_calls must be an array on assistant messages",
        400,
        "invalid_tool_calls",
        "invalid_request_error"
      );
    }
    for (const toolCall of msg.tool_calls) {
      if (
        toolCall?.type !== "function" ||
        typeof toolCall?.id !== "string" ||
        !toolCall.id ||
        typeof toolCall?.function?.name !== "string" ||
        !toolCall.function.name ||
        typeof toolCall?.function?.arguments !== "string"
      ) {
        throw createApiError(
          "Invalid function call in assistant tool_calls",
          400,
          "invalid_tool_calls",
          "invalid_request_error"
        );
      }
    }
  }
  // developer → system (Unified system content)
  const normalized = role === "developer" ? { ...msg, role: "system" } : msg;
  if (normalized.role === "assistant") {
    syncAssistantThinkingFields(normalized);
  }
  return normalized;
}

/** Chat-native `reasoning_content` and Unified `thinking` are the same history. */
function syncAssistantThinkingFields(message: any): void {
  if (!message || typeof message !== "object") return;
  const thinkingText =
    typeof message.thinking?.content === "string" ? message.thinking.content : "";
  const reasoningText =
    typeof message.reasoning_content === "string" ? message.reasoning_content : "";
  const content = thinkingText || reasoningText;
  if (
    !content &&
    !message.thinking?.signature &&
    !message.thinking?.encrypted_content &&
    !message.thinking?.id
  ) {
    return;
  }
  if (content) message.reasoning_content = content;
  if (!message.thinking) message.thinking = { content };
  else if (!thinkingText && content) message.thinking.content = content;
}

function applyChatReasoningHistory(messages: any[] | undefined): any[] {
  return (messages || []).map((message) => {
    if (message?.role !== "assistant") return message;
    const next = { ...message };
    if (next.thinking) next.thinking = { ...next.thinking };
    syncAssistantThinkingFields(next);
    // Chat Completions providers speak reasoning_content. Drop the Unified
    // thinking object so it is not forwarded as an unknown message field.
    delete next.thinking;
    return next;
  });
}

function applyChatThinkingToCompletion(payload: any): any {
  if (!payload || typeof payload !== "object") return payload;
  const message = payload.choices?.[0]?.message;
  if (message) syncAssistantThinkingFields(message);
  const delta = payload.choices?.[0]?.delta;
  if (delta) syncAssistantThinkingFields(delta);
  return payload;
}

function validateChatTools(tools: any, toolChoice: any): void {
  if (tools !== undefined && !Array.isArray(tools)) {
    throw createApiError(
      "tools must be an array",
      400,
      "invalid_tools",
      "invalid_request_error"
    );
  }
  for (const tool of tools || []) {
    if (
      tool?.type !== "function" ||
      typeof tool?.function?.name !== "string" ||
      !tool.function.name
    ) {
      throw createApiError(
        "Only function tools are supported by Chat Completions inbound routes",
        400,
        "unsupported_tool",
        "invalid_request_error"
      );
    }
  }
  if (toolChoice == null) return;
  if (["auto", "none", "required"].includes(toolChoice)) return;
  if (
    toolChoice?.type === "function" &&
    typeof toolChoice?.function?.name === "string" &&
    toolChoice.function.name
  ) {
    return;
  }
  throw createApiError(
    "Unsupported tool_choice",
    400,
    "unsupported_tool_choice",
    "invalid_request_error"
  );
}

/**
 * Ensure Chat Completions SSE streams end with data: [DONE] when the upstream
 * already produced Chat-shaped events but omitted the terminator.
 */
function ensureChatStreamDone(
  response: Response,
  options?: { aliasThinking?: boolean }
): Response {
  if (!response.body) return response;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const encoder = new TextEncoder();
  let buffer = "";
  let sawDone = false;
  const aliasThinking = options?.aliasThinking === true;

  const rewriteDataLine = (line: string): string => {
    if (!aliasThinking || !line.startsWith("data:")) return line;
    const payload = line.slice(5).trim();
    if (!payload || payload === "[DONE]") return line;
    try {
      return `data: ${JSON.stringify(applyChatThinkingToCompletion(JSON.parse(payload)))}`;
    } catch {
      return line;
    }
  };

  const emitText = (text: string, flush = false): string => {
    buffer += text;
    const lines = buffer.split(/\r?\n/);
    buffer = flush ? "" : lines.pop() || "";
    const out: string[] = [];
    const pushLine = (line: string) => {
      if (sawDone) return;
      for (const piece of splitChatCompletionsDoneLine(line)) {
        if (sawDone) return;
        if (isChatCompletionsDoneLine(piece)) {
          sawDone = true;
          pushChatCompletionsDone(out);
          return;
        }
        out.push(rewriteDataLine(piece));
      }
    };
    for (const line of lines) pushLine(line);
    if (flush && buffer) {
      pushLine(buffer);
      buffer = "";
    }
    return out.length ? `${out.join("\n")}\n` : "";
  };

  const stream = new ReadableStream({
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (done) {
          const tail = emitText(decoder.decode(), true);
          if (tail) controller.enqueue(encoder.encode(tail));
          if (!sawDone) {
            controller.enqueue(encoder.encode("\ndata: [DONE]\n\n"));
          }
          controller.close();
          return;
        }
        const rewritten = emitText(decoder.decode(value, { stream: true }));
        if (rewritten) controller.enqueue(encoder.encode(rewritten));
      } catch {
        if (!sawDone) {
          controller.enqueue(
            encoder.encode(
              `data: ${JSON.stringify({
                error: {
                  message: "Upstream stream failed",
                  type: "api_error",
                  param: null,
                  code: "provider_response_error",
                },
              })}\n\n`
            )
          );
          controller.enqueue(encoder.encode("\ndata: [DONE]\n\n"));
        }
        controller.close();
      }
    },
    cancel(reason) {
      return reader.cancel(reason);
    },
  });

  return new Response(stream, {
    status: response.status,
    statusText: response.statusText,
    headers: response.headers,
  });
}

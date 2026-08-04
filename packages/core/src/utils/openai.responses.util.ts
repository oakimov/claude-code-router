import { UnifiedChatRequest } from "@/types/llm";
import { createApiError } from "@/api/middleware";
import { sanitizeResponsesCallId } from "@/utils/toolCallId";

export interface ResponsesCallIdMap {
  /** Original client call_id → sanitized id (and reverse). */
  forward: Map<string, string>;
  reverse: Map<string, string>;
}

export function createCallIdMap(): ResponsesCallIdMap {
  return { forward: new Map(), reverse: new Map() };
}

/** Sanitize and remember a stable per-turn mapping for function call correlation. */
export function mapCallId(
  map: ResponsesCallIdMap,
  id: unknown,
  direction: "client_to_unified" | "unified_to_client" = "client_to_unified"
): string | undefined {
  if (typeof id !== "string" || !id) return undefined;
  if (direction === "unified_to_client") {
    const alreadyAssigned = sanitizeResponsesCallId(id) ?? id;
    if (map.reverse.has(alreadyAssigned)) {
      return alreadyAssigned;
    }
  }
  const existing = map.forward.get(id);
  if (existing) return existing;

  // Both directions must satisfy the client/provider Responses call_id
  // contract. A valid client id can equal another invalid id's sanitized form,
  // so resolve collisions per turn instead of relying on the hash suffix alone.
  let candidate = sanitizeResponsesCallId(id) ?? id;
  let collisionIndex = 0;
  while (map.reverse.has(candidate) && map.reverse.get(candidate) !== id) {
    collisionIndex += 1;
    candidate =
      sanitizeResponsesCallId(`${id}_${collisionIndex}`) ||
      `call_${collisionIndex}`;
  }
  map.forward.set(id, candidate);
  map.reverse.set(candidate, id);
  return candidate;
}

/**
 * Client Responses wire → Unified (Chat Completions shape).
 * Supports the Responses MVP subset; rejects CCR-unsupported stateful fields.
 */
export function responsesRequestToUnified(
  body: any,
  callIdMap: ResponsesCallIdMap = createCallIdMap()
): UnifiedChatRequest {
  if (!body || typeof body !== "object") {
    throw createApiError("Invalid Responses body", 400, "invalid_body");
  }
  if (typeof body.model !== "string" || !body.model.trim()) {
    throw createApiError(
      "Responses requires model",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }

  rejectUnsupportedResponsesState(body);
  if (body.stream !== undefined && typeof body.stream !== "boolean") {
    throw createApiError(
      "stream must be a boolean",
      400,
      "invalid_stream",
      "invalid_request_error"
    );
  }
  if (
    body.instructions !== undefined &&
    typeof body.instructions !== "string"
  ) {
    throw createApiError(
      "instructions must be a string",
      400,
      "invalid_instructions",
      "invalid_request_error"
    );
  }
  if (body.max_tokens !== undefined) {
    throw createApiError(
      "Responses uses max_output_tokens, not max_tokens",
      400,
      "unsupported_field",
      "invalid_request_error"
    );
  }
  if (body.max_output_tokens != null) {
    const maxOutputTokens = Number(body.max_output_tokens);
    if (!Number.isInteger(maxOutputTokens) || maxOutputTokens <= 0) {
      throw createApiError(
        "max_output_tokens must be a positive integer",
        400,
        "invalid_token_limit",
        "invalid_request_error"
      );
    }
  }
  if (
    body.reasoning !== undefined &&
    (body.reasoning === null ||
      typeof body.reasoning !== "object" ||
      Array.isArray(body.reasoning))
  ) {
    throw createApiError(
      "reasoning must be an object",
      400,
      "invalid_reasoning",
      "invalid_request_error"
    );
  }
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
  if (
    body.prompt_cache_key !== undefined &&
    typeof body.prompt_cache_key !== "string"
  ) {
    throw createApiError(
      "prompt_cache_key must be a string",
      400,
      "invalid_prompt_cache_key",
      "invalid_request_error"
    );
  }
  if (body.text?.format && body.text.format.type !== "text") {
    throw createApiError(
      "Structured text formats are not supported without a verified mapping",
      400,
      "unsupported_response_format",
      "invalid_request_error"
    );
  }
  for (const field of [
    "metadata",
    "max_tool_calls",
    "service_tier",
    "safety_identifier",
    "top_logprobs",
    "truncation",
  ]) {
    if (body[field] !== undefined && body[field] !== null) {
      throw createApiError(
        `Responses field '${field}' is not supported`,
        400,
        "unsupported_field",
        "invalid_request_error"
      );
    }
  }

  const messages: any[] = [];
  if (typeof body.instructions === "string" && body.instructions) {
    messages.push({ role: "system", content: body.instructions });
  }

  const input = body.input;
  if (typeof input === "string") {
    messages.push({ role: "user", content: input });
  } else if (Array.isArray(input)) {
    if (input.length === 0) {
      throw createApiError(
        "Responses input[] must not be empty",
        400,
        "invalid_input",
        "invalid_request_error"
      );
    }
    for (const item of input) {
      appendInputItem(messages, item, callIdMap);
    }
  } else {
    throw createApiError(
      "Responses requires string input or input[]",
      400,
      "invalid_input",
      "invalid_request_error"
    );
  }

  if (messages.length === 0) {
    throw createApiError(
      "Responses input did not contain a supported item",
      400,
      "invalid_input",
      "invalid_request_error"
    );
  }

  const tools = normalizeResponsesTools(body.tools);
  const toolChoice = normalizeResponsesToolChoice(body.tool_choice);

  const reasoning =
    body.reasoning && typeof body.reasoning === "object"
      ? {
          enabled: true,
          effort: body.reasoning.effort,
          summary: body.reasoning.summary,
        }
      : undefined;

  const unified: UnifiedChatRequest = {
    model: body.model,
    messages,
    tools,
    tool_choice: toolChoice,
    stream: body.stream === true,
    temperature: body.temperature,
    top_p: body.top_p,
    max_tokens: body.max_output_tokens ?? body.max_tokens,
    reasoning,
    parallel_tool_calls: body.parallel_tool_calls,
  } as UnifiedChatRequest;

  // Opaque same-protocol hint — preserved on Unified for passthrough destinations.
  if (typeof body.prompt_cache_key === "string") {
    (unified as any).prompt_cache_key = body.prompt_cache_key;
  }

  return unified;
}

function rejectUnsupportedResponsesState(body: any): void {
  if (body.store === true) {
    throw createApiError(
      "store: true is not supported; CCR does not own Responses state",
      400,
      "unsupported_store",
      "invalid_request_error"
    );
  }
  if (body.previous_response_id) {
    throw createApiError(
      "previous_response_id is not supported",
      400,
      "unsupported_previous_response_id",
      "invalid_request_error"
    );
  }
  if (body.conversation) {
    throw createApiError(
      "conversation is not supported",
      400,
      "unsupported_conversation",
      "invalid_request_error"
    );
  }
  if (body.background === true) {
    throw createApiError(
      "background: true is not supported",
      400,
      "unsupported_background",
      "invalid_request_error"
    );
  }

  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      const type = tool?.type;
      if (typeof type !== "string" || !type) {
        throw createApiError(
          "Responses tools require a type",
          400,
          "invalid_tool",
          "invalid_request_error"
        );
      }
      if (
        type !== "function" &&
        type !== "web_search" &&
        type !== "web_search_preview"
      ) {
        throw createApiError(
          `Hosted tool type '${type}' is not supported on cross-protocol routes`,
          400,
          "unsupported_hosted_tool",
          "invalid_request_error"
        );
      }
    }
  }
}

function appendInputItem(
  messages: any[],
  item: any,
  callIdMap: ResponsesCallIdMap
): void {
  if (!item || typeof item !== "object") {
    throw createApiError(
      "Invalid Responses input item",
      400,
      "invalid_input_item",
      "invalid_request_error"
    );
  }

  if (item.type === "function_call_output") {
    if (typeof item.call_id !== "string" || !item.call_id) {
      throw createApiError(
        "function_call_output requires call_id",
        400,
        "invalid_call_id",
        "invalid_request_error"
      );
    }
    messages.push({
      role: "tool",
      tool_call_id: mapCallId(callIdMap, item.call_id) ?? item.call_id,
      content:
        typeof item.output === "string"
          ? item.output
          : JSON.stringify(item.output ?? ""),
    });
    return;
  }

  if (item.type === "function_call") {
    const rawCallId = item.call_id || item.id;
    if (typeof rawCallId !== "string" || !rawCallId) {
      throw createApiError(
        "function_call requires call_id",
        400,
        "invalid_call_id",
        "invalid_request_error"
      );
    }
    if (typeof item.name !== "string" || !item.name) {
      throw createApiError(
        "function_call requires name",
        400,
        "invalid_function_call",
        "invalid_request_error"
      );
    }
    messages.push({
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: mapCallId(callIdMap, rawCallId) ?? rawCallId,
          type: "function",
          function: {
            name: item.name,
            arguments:
              typeof item.arguments === "string"
                ? item.arguments
                : JSON.stringify(item.arguments ?? {}),
          },
        },
      ],
    });
    return;
  }

  if (item.type === "message" || item.role) {
    const role =
      item.role === "assistant"
        ? "assistant"
        : item.role === "system" || item.role === "developer"
          ? "system"
          : item.role === "user"
            ? "user"
            : undefined;
    if (!role) {
      throw createApiError(
        `Unsupported Responses message role '${item.role}'`,
        400,
        "unsupported_role",
        "invalid_request_error"
      );
    }
    if (
      typeof item.content !== "string" &&
      !Array.isArray(item.content)
    ) {
      throw createApiError(
        "Responses message items require string or array content",
        400,
        "invalid_content",
        "invalid_request_error"
      );
    }
    messages.push({
      role,
      content: flattenResponsesContent(item.content),
    });
    return;
  }

  if (item.type === "input_text" || item.type === "output_text") {
    messages.push({
      role: item.type === "output_text" ? "assistant" : "user",
      content: String(item.text ?? ""),
    });
    return;
  }

  if (item.type === "input_image") {
    if (item.file_id) {
      throw createApiError(
        "Provider-bound file_id inputs are not supported",
        400,
        "unsupported_file_id",
        "invalid_request_error"
      );
    }
    const imageUrl = item.image_url || item.url;
    if (typeof imageUrl !== "string" || !imageUrl) {
      throw createApiError(
        "input_image requires image_url",
        400,
        "invalid_image",
        "invalid_request_error"
      );
    }
    messages.push({
      role: "user",
      content: [
        {
          type: "image_url",
          image_url: {
            url: imageUrl,
            ...(item.detail ? { detail: item.detail } : {}),
          },
        },
      ],
    });
    return;
  }

  throw createApiError(
    `Unsupported Responses input item type '${item.type || "unknown"}'`,
    400,
    "unsupported_input_item",
    "invalid_request_error"
  );
}

function flattenResponsesContent(content: unknown): any {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    return content == null ? "" : JSON.stringify(content);
  }

  const parts: any[] = [];
  for (const part of content) {
    if (typeof part === "string") {
      parts.push({ type: "text", text: part });
      continue;
    }
    if (!part || typeof part !== "object") continue;
    if (part.type === "input_text" || part.type === "output_text" || part.type === "text") {
      parts.push({ type: "text", text: String(part.text ?? "") });
    } else if (part.type === "input_image" || part.type === "image_url") {
      if ((part as any).file_id) {
        throw createApiError(
          "Provider-bound file_id inputs are not supported",
          400,
          "unsupported_file_id",
          "invalid_request_error"
        );
      }
      const imageUrl = part.image_url?.url || part.image_url || part.url;
      if (typeof imageUrl !== "string" || !imageUrl) {
        throw createApiError(
          "input_image requires image_url",
          400,
          "invalid_image",
          "invalid_request_error"
        );
      }
      parts.push({
        type: "image_url",
        image_url: {
          url: imageUrl,
          ...((part as any).detail
            ? { detail: (part as any).detail }
            : {}),
        },
      });
    } else {
      throw createApiError(
        `Unsupported Responses content part '${part.type || "unknown"}'`,
        400,
        "unsupported_content_part",
        "invalid_request_error"
      );
    }
  }

  if (parts.length === 1 && parts[0].type === "text") {
    return parts[0].text;
  }
  return parts.length > 0 ? parts : "";
}

function normalizeResponsesTools(tools: any): UnifiedChatRequest["tools"] | undefined {
  if (tools == null) return undefined;
  if (!Array.isArray(tools)) {
    throw createApiError(
      "Responses tools must be an array",
      400,
      "invalid_tools",
      "invalid_request_error"
    );
  }
  return tools.map((t: any) => {
    if (t?.type === "function") {
      const name = t.name || t.function?.name;
      if (typeof name !== "string" || !name) {
        throw createApiError(
          "Responses function tools require a name",
          400,
          "invalid_tool",
          "invalid_request_error"
        );
      }
      return {
        type: "function" as const,
        function: {
          name,
          description: t.description || t.function?.description,
          parameters:
            t.parameters ||
            t.function?.parameters ||
            { type: "object", properties: {} },
        },
      };
    }
    if (t?.type === "web_search" || t?.type === "web_search_preview") {
      return { ...t, type: "web_search" } as any;
    }
    throw createApiError(
      `Unsupported Responses tool type '${t?.type || "unknown"}'`,
      400,
      "unsupported_hosted_tool",
      "invalid_request_error"
    );
  });
}

function normalizeResponsesToolChoice(toolChoice: any): any {
  if (toolChoice == null) return undefined;
  if (typeof toolChoice === "string") {
    if (["auto", "none", "required"].includes(toolChoice)) {
      return toolChoice;
    }
    throw createApiError(
      `Unsupported Responses tool_choice '${toolChoice}'`,
      400,
      "unsupported_tool_choice",
      "invalid_request_error"
    );
  }
  if (toolChoice?.type === "function" && toolChoice?.name) {
    return {
      type: "function",
      function: { name: toolChoice.name },
    };
  }
  if (toolChoice?.type === "allowed_tools") {
    throw createApiError(
      "allowed_tools tool_choice is not supported",
      400,
      "unsupported_tool_choice",
      "invalid_request_error"
    );
  }
  throw createApiError(
    "Unsupported Responses tool_choice",
    400,
    "unsupported_tool_choice",
    "invalid_request_error"
  );
}

/** Unified Chat JSON → Responses API non-stream response. */
export function unifiedResponseToResponses(
  chat: any,
  options?: {
    originalModel?: string;
    callIdMap?: ResponsesCallIdMap;
  }
): any {
  const callIdMap = options?.callIdMap ?? createCallIdMap();
  const choice = chat?.choices?.[0];
  const message = choice?.message || {};
  const output: any[] = [];

  const text =
    typeof message.content === "string"
      ? message.content
      : Array.isArray(message.content)
        ? message.content
            .filter((p: any) => p?.type === "text")
            .map((p: any) => p.text)
            .join("")
        : "";

  if (text || !message.tool_calls?.length) {
    const itemId = `msg_${chat.id || Date.now()}`;
    output.push({
      type: "message",
      id: itemId,
      status: "completed",
      role: "assistant",
      content: [
        {
          type: "output_text",
          text: text || "",
          annotations: [],
          logprobs: message?.logprobs?.content || [],
        },
      ],
    });
  }

  if (Array.isArray(message.tool_calls)) {
    for (const tc of message.tool_calls) {
      if (!tc?.function) continue;
      const callId =
        mapCallId(callIdMap, tc.id, "unified_to_client") ?? tc.id;
      output.push({
        type: "function_call",
        id: callId,
        call_id: callId,
        name: tc.function.name,
        arguments:
          typeof tc.function.arguments === "string"
            ? tc.function.arguments
            : JSON.stringify(tc.function.arguments ?? {}),
        status: "completed",
      });
    }
  }

  const usage = responsesUsageFromChat(chat?.usage);

  return {
    id: chat?.id || `resp_${Date.now()}`,
    object: "response",
    created_at: chat?.created || Math.floor(Date.now() / 1000),
    status: "completed",
    model: options?.originalModel || chat?.model,
    output,
    usage,
  };
}

export interface ResponsesStreamState {
  responseId: string;
  model?: string;
  textItemId: string;
  textStarted: boolean;
  textOutputIndex?: number;
  textContent: string;
  toolCalls: Map<
    number,
    {
      id: string;
      name: string;
      arguments: string;
      added: boolean;
      outputIndex: number;
      emittedArgumentsLength: number;
    }
  >;
  nextOutputIndex: number;
  finished: boolean;
  created: boolean;
  sequenceNumber: number;
  finishReasonSeen: boolean;
  usage?: any;
  callIdMap: ResponsesCallIdMap;
}

export function createResponsesStreamState(
  options?: { model?: string; callIdMap?: ResponsesCallIdMap }
): ResponsesStreamState {
  const id = `resp_${Date.now()}`;
  return {
    responseId: id,
    model: options?.model,
    textItemId: `msg_${id}`,
    textStarted: false,
    textContent: "",
    toolCalls: new Map(),
    nextOutputIndex: 0,
    finished: false,
    created: false,
    sequenceNumber: 0,
    finishReasonSeen: false,
    callIdMap: options?.callIdMap ?? createCallIdMap(),
  };
}

function sequenceEvents(state: ResponsesStreamState, events: any[]): any[] {
  return events.map((event) => ({
    ...event,
    sequence_number: state.sequenceNumber++,
  }));
}

function appendCreatedEvent(
  state: ResponsesStreamState,
  events: any[]
): void {
  if (state.created) return;
  state.created = true;
  events.push({
    type: "response.created",
    response: skeletonResponse(state, "in_progress"),
  });
}

/**
 * Convert one Unified Chat Completions chunk into zero or more Responses
 * stream events, including the mandatory content_part lifecycle for Codex.
 */
export function unifiedChunkToResponsesEvents(
  chunk: any,
  state: ResponsesStreamState
): any[] {
  if (!chunk || typeof chunk !== "object") return [];
  if (state.finished) return [];

  const events: any[] = [];
  const choice = chunk.choices?.[0];
  const delta = choice?.delta || {};

  if (!state.model && chunk.model) {
    state.model = chunk.model;
  }
  if (chunk.id) {
    state.responseId = chunk.id.startsWith("resp_")
      ? chunk.id
      : `resp_${chunk.id}`;
  }
  if (chunk.usage) {
    state.usage = chunk.usage;
  }

  // First useful event: response.created
  if (
    !state.textStarted &&
    state.toolCalls.size === 0 &&
    (typeof delta.content === "string" ||
      Array.isArray(delta.tool_calls) ||
      choice?.finish_reason)
  ) {
    appendCreatedEvent(state, events);
  }

  if (typeof delta.content === "string" && delta.content.length > 0) {
    if (!state.textStarted) {
      state.textStarted = true;
      state.textOutputIndex = state.nextOutputIndex++;
      events.push({
        type: "response.output_item.added",
        output_index: state.textOutputIndex,
        item: {
          type: "message",
          id: state.textItemId,
          status: "in_progress",
          role: "assistant",
          content: [],
        },
      });
      events.push({
        type: "response.content_part.added",
        item_id: state.textItemId,
        output_index: state.textOutputIndex,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      });
    }
    state.textContent += delta.content;
    events.push({
      type: "response.output_text.delta",
      item_id: state.textItemId,
      output_index: state.textOutputIndex,
      content_index: 0,
      delta: delta.content,
      logprobs: [],
    });
  }

  if (Array.isArray(delta.tool_calls)) {
    for (const tc of delta.tool_calls) {
      const index = typeof tc.index === "number" ? tc.index : 0;
      let entry = state.toolCalls.get(index);
      if (!entry) {
        const callId =
          mapCallId(
            state.callIdMap,
            tc.id || `call_${state.responseId}_${index}`,
            "unified_to_client"
          ) || `call_${index}`;
        entry = {
          id: callId,
          name: tc.function?.name || "",
          arguments: "",
          added: false,
          outputIndex: state.nextOutputIndex++,
          emittedArgumentsLength: 0,
        };
        state.toolCalls.set(index, entry);
      }
      if (tc.id && !entry.added) {
        entry.id =
          mapCallId(state.callIdMap, tc.id, "unified_to_client") || entry.id;
      }
      if (tc.function?.name) entry.name = tc.function.name;
      if (typeof tc.function?.arguments === "string") {
        entry.arguments += tc.function.arguments;
      }

      if (!entry.added && entry.name) {
        entry.added = true;
        events.push({
          type: "response.output_item.added",
          output_index: entry.outputIndex,
          item: {
            type: "function_call",
            id: entry.id,
            call_id: entry.id,
            name: entry.name,
            arguments: "",
            status: "in_progress",
          },
        });
      }

      if (
        entry.added &&
        entry.arguments.length > entry.emittedArgumentsLength
      ) {
        const unreported = entry.arguments.slice(entry.emittedArgumentsLength);
        entry.emittedArgumentsLength = entry.arguments.length;
        events.push({
          type: "response.function_call_arguments.delta",
          item_id: entry.id,
          output_index: entry.outputIndex,
          delta: unreported,
        });
      }
    }
  }

  if (choice?.finish_reason) {
    // Chat Completions may report usage in a separate choices:[] chunk after
    // the finish-reason chunk. Wait for [DONE]/upstream close so the terminal
    // Responses object includes that provider-reported usage.
    state.finishReasonSeen = true;
  }

  return sequenceEvents(state, events);
}

export function finalizeResponsesStream(
  state: ResponsesStreamState,
  usage?: any
): any[] {
  if (state.finished) return [];
  state.finished = true;
  const events: any[] = [];
  appendCreatedEvent(state, events);

  if (state.textStarted) {
    events.push({
      type: "response.output_text.done",
      item_id: state.textItemId,
      output_index: state.textOutputIndex,
      content_index: 0,
      text: state.textContent,
      logprobs: [],
    });
    events.push({
      type: "response.content_part.done",
      item_id: state.textItemId,
      output_index: state.textOutputIndex,
      content_index: 0,
      part: {
        type: "output_text",
        text: state.textContent,
        annotations: [],
        logprobs: [],
      },
    });
    events.push({
      type: "response.output_item.done",
      output_index: state.textOutputIndex,
      item: {
        type: "message",
        id: state.textItemId,
        status: "completed",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text: state.textContent,
            annotations: [],
            logprobs: [],
          },
        ],
      },
    });
  }

  for (const entry of state.toolCalls.values()) {
    events.push({
      type: "response.function_call_arguments.done",
      item_id: entry.id,
      output_index: entry.outputIndex,
      name: entry.name,
      arguments: entry.arguments,
    });
    events.push({
      type: "response.output_item.done",
      output_index: entry.outputIndex,
      item: {
        type: "function_call",
        id: entry.id,
        call_id: entry.id,
        name: entry.name,
        arguments: entry.arguments,
        status: "completed",
      },
    });
  }

  const completed = skeletonResponse(state, "completed", usage || state.usage);
  events.push({
    type: "response.completed",
    response: completed,
  });

  return sequenceEvents(state, events);
}

function skeletonResponse(
  state: ResponsesStreamState,
  status: string,
  usage?: any
): any {
  const indexedOutput: Array<{ index: number; item: any }> = [];
  if (state.textStarted || state.textContent) {
    indexedOutput.push({
      index: state.textOutputIndex ?? 0,
      item: {
        type: "message",
        id: state.textItemId,
        status: status === "completed" ? "completed" : "in_progress",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text: state.textContent,
            annotations: [],
            logprobs: [],
          },
        ],
      },
    });
  }
  for (const entry of state.toolCalls.values()) {
    indexedOutput.push({
      index: entry.outputIndex,
      item: {
        type: "function_call",
        id: entry.id,
        call_id: entry.id,
        name: entry.name,
        arguments: entry.arguments,
        status: status === "completed" ? "completed" : "in_progress",
      },
    });
  }
  const output = indexedOutput
    .sort((left, right) => left.index - right.index)
    .map(({ item }) => item);

  return {
    id: state.responseId,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    status,
    model: state.model,
    output,
    usage: responsesUsageFromChat(usage),
  };
}

function responsesUsageFromChat(usage: any): any | undefined {
  if (!usage || typeof usage !== "object") return undefined;
  const result: any = {
    input_tokens: usage.prompt_tokens ?? 0,
    output_tokens: usage.completion_tokens ?? 0,
    total_tokens: usage.total_tokens ?? 0,
  };
  const details = usage.prompt_tokens_details;
  if (details && typeof details === "object") {
    result.input_tokens_details = {};
    if (details.cached_tokens != null) {
      result.input_tokens_details.cached_tokens = details.cached_tokens;
    }
    if (details.cache_write_tokens != null) {
      result.input_tokens_details.cache_write_tokens =
        details.cache_write_tokens;
    }
  }
  return result;
}

export function responsesFailedEvent(
  message: string,
  state?: ResponsesStreamState
): any {
  const event = {
    type: "response.failed",
    response: {
      id: state?.responseId || `resp_${Date.now()}`,
      object: "response",
      status: "failed",
      error: {
        message,
        type: "api_error",
        code: "provider_response_error",
      },
      output: [],
    },
  };
  if (state) {
    return { ...event, sequence_number: state.sequenceNumber++ };
  }
  return event;
}

import { UnifiedChatRequest } from "@/types/llm";
import { createHash } from "crypto";
import { createApiError } from "@/api/middleware";
import { sanitizeResponsesCallId } from "@/utils/toolCallId";
import { canonicalReasoning } from "@/utils/reasoning-effort";

function shouldLogResponsesPassthrough(): boolean {
  const env =
    process.env.LOG_RESPONSES_PASSTHROUGH ??
    process.env.CCR_LOG_RESPONSES_PASSTHROUGH ??
    process.env.CCR_DEBUG_RESPONSES_PASSTHROUGH;
  return env === "1" || env === "true";
}

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
 * Enforce the Responses call_id contract without rebuilding an exact-wire body.
 *
 * Same-protocol wire keep deliberately skips the Responses owner's full
 * transformRequestIn so images/files/reasoning/cache fields remain byte-faithful.
 * Call ids are still a provider validation boundary, though: Cursor-style
 * composite ids can exceed 64 characters. Rewrite only the identity field on
 * call/output items and reuse the normalization map so paired items and hash
 * collisions resolve identically in both directions.
 */
export function sanitizeResponsesWireCallIds(
  body: any,
  callIdMap: ResponsesCallIdMap = createCallIdMap()
): any {
  if (!body || typeof body !== "object" || !Array.isArray(body.input)) {
    return body;
  }

  let changed = false;
  const input = body.input.map((item: any) => {
    if (!item || typeof item !== "object") return item;

    const isCall =
      item.type === "function_call" || item.type === "custom_tool_call";
    const isOutput =
      item.type === "function_call_output" ||
      item.type === "custom_tool_call_output";
    const rawCallId = isCall
      ? item.call_id || item.id
      : isOutput
        ? item.call_id
        : undefined;
    const mapped = mapCallId(callIdMap, rawCallId);
    if (!mapped || item.call_id === mapped) return item;

    // Calls may use id as a client-side fallback, but provider-bound Responses
    // input requires call_id. Keep the opaque item id and add/repair call_id.
    changed = true;
    return { ...item, call_id: mapped };
  });

  return changed ? { ...body, input } : body;
}

/**
 * Client Responses wire → Unified (Chat Completions shape).
 * Supports the Responses MVP subset; rejects CCR-unsupported stateful fields.
 */
export function responsesRequestToUnified(
  body: any,
  callIdMap: ResponsesCallIdMap = createCallIdMap(),
  customToolNames: Set<string> = new Set()
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
  // Convert into the Chat Completions `response_format` shape (Unified IS
  // Chat Completions) — a verified mapping for the two structured formats
  // real backends (OpenAI/`codex`) actually support. Genuinely unknown
  // format types still reject rather than risk a silent wrong conversion.
  // Responses destinations reconstruct this into a native `text.format`
  // on the outbound request (OpenAIResponsesTransformer and Codex). See
  // customToolNames just below for the restore-only-for-codex pattern
  // applied to freeform tools.
  let responseFormat: any;
  if (body.text?.format && body.text.format.type !== "text") {
    const format = body.text.format;
    if (format.type === "json_schema") {
      responseFormat = {
        type: "json_schema",
        json_schema: {
          name: format.name,
          schema: format.schema,
          ...(format.strict !== undefined ? { strict: format.strict } : {}),
        },
      };
    } else if (format.type === "json_object") {
      responseFormat = { type: "json_object" };
    } else {
      throw createApiError(
        `Structured text formats are not supported without a verified mapping (received text.format.type=${JSON.stringify(
          format.type
        )})`,
        400,
        "unsupported_response_format",
        "invalid_request_error"
      );
    }
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

  const tools = normalizeResponsesTools(body.tools, customToolNames);
  const toolChoice = normalizeResponsesToolChoice(body.tool_choice);

  const reasoning =
    body.reasoning && typeof body.reasoning === "object"
      ? {
          ...canonicalReasoning(body.reasoning.effort, true),
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

  // Opaque same-protocol hints — preserved on Unified for Responses→Responses.
  // Do not invent these for Chat/Anthropic inbound; only the Responses client
  // can express them.
  if (typeof body.prompt_cache_key === "string") {
    (unified as any).prompt_cache_key = body.prompt_cache_key;
  }
  const include = normalizeResponsesInclude(body.include);
  if (include) {
    (unified as any).include = include;
  }
  // store:true is rejected above; only an explicit false is a client signal
  // (AI SDK / OpenCode pair it with include: reasoning.encrypted_content).
  if (body.store === false) {
    (unified as any).store = false;
  }

  if (responseFormat) {
    (unified as any).response_format = responseFormat;
  }

  return unified;
}

/** Responses `include` is a string list; drop non-strings rather than invent values. */
export function normalizeResponsesInclude(
  include: unknown
): string[] | undefined {
  if (!Array.isArray(include)) return undefined;
  const values = include.filter(
    (entry): entry is string => typeof entry === "string" && entry.length > 0
  );
  return values.length > 0 ? values : undefined;
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

  // Tool-type allowlisting lives in normalizeResponsesTools: every inbound
  // Responses tool (function, custom, web_search, and other hosted types) is
  // projected onto a Unified Chat Completions function tool. Reject only
  // tools that lack a type string so conversion has something to key on.
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
    }
  }
}

/**
 * Chat Completions (and DeepSeek) require one assistant message carrying every
 * parallel tool_call plus any same-turn text. Responses emits those as separate
 * items (text then function_call, or the reverse), so fold both onto the
 * trailing assistant. A tool result ends the turn; the next function_call
 * starts a new assistant.
 */
function assistantContentIsEmpty(message: any): boolean {
  return (
    message?.content === null ||
    message?.content === undefined ||
    message?.content === ""
  );
}

function appendAssistantToolCall(messages: any[], toolCall: any): void {
  const last = messages[messages.length - 1];
  if (last && last.role === "assistant") {
    if (Array.isArray(last.tool_calls)) {
      last.tool_calls.push(toolCall);
    } else {
      last.tool_calls = [toolCall];
    }
    return;
  }
  messages.push({
    role: "assistant",
    content: null,
    tool_calls: [toolCall],
  });
}

function appendAssistantContent(messages: any[], content: any): boolean {
  const last = messages[messages.length - 1];
  if (
    last?.role === "assistant" &&
    Array.isArray(last.tool_calls) &&
    last.tool_calls.length > 0 &&
    assistantContentIsEmpty(last)
  ) {
    last.content = content;
    return true;
  }
  return false;
}

function reasoningSummaryText(item: any): string {
  if (typeof item?.content === "string" && item.content) return item.content;
  if (typeof item?.reasoning === "string" && item.reasoning) return item.reasoning;
  if (!Array.isArray(item?.summary)) return "";
  return item.summary
    .map((part: any) =>
      typeof part === "string" ? part : typeof part?.text === "string" ? part.text : ""
    )
    .filter(Boolean)
    .join("\n");
}

/** Responses/Codex reasoning item ids look like `rs_<hex>` or `rs_<epoch>`. */
const RESPONSES_REASONING_ITEM_ID = /^rs_[A-Za-z0-9]+$/;

export function isResponsesReasoningItemId(value: unknown): value is string {
  return typeof value === "string" && RESPONSES_REASONING_ITEM_ID.test(value);
}

/**
 * Ciphertext Codex/OpenAI will verify. Item ids and Anthropic/Gemini
 * signatures are not encrypted_content — replaying them 400s with
 * `invalid_encrypted_content`.
 */
export function responsesEncryptedContentFrom(
  value: unknown
): string | undefined {
  if (typeof value !== "string" || !value) return undefined;
  if (isResponsesReasoningItemId(value)) return undefined;
  return value;
}

export interface UnifiedAssistantThinking {
  content: string;
  signature?: string;
  encrypted_content?: string;
  id?: string;
}

/** Anthropic/Gemini thinking.signature — never a Responses item id. */
export function anthropicThinkingSignatureFrom(
  thinking: { signature?: string } | undefined
): string | undefined {
  const signature =
    typeof thinking?.signature === "string" ? thinking.signature : "";
  if (!signature || isResponsesReasoningItemId(signature)) return undefined;
  return signature;
}

/** Responses `reasoning` item → Unified assistant.thinking. */
export function thinkingFromResponsesReasoningItem(
  item: any
): UnifiedAssistantThinking | undefined {
  if (!item || typeof item !== "object") return undefined;
  const content = reasoningSummaryText(item);
  const encrypted_content = responsesEncryptedContentFrom(item.encrypted_content);
  const id =
    isUsableReasoningItemId(item.id)
      ? item.id
      : isResponsesReasoningItemId(item.encrypted_content)
        ? item.encrypted_content
        : undefined;
  if (!content && !encrypted_content && !id) return undefined;
  return {
    content,
    ...(encrypted_content ? { encrypted_content } : {}),
    ...(id ? { id } : {}),
  };
}

/**
 * Record streamed reasoning summary text for one item. Late handlers
 * (`output_item.done`, `response.completed`) consult this map so they can
 * emit ciphertext / id without re-sending content — Unified
 * `delta.thinking.content` is additive in every downstream consumer.
 */
export function recordReasoningSummaryDelta(
  deliveredContentByItemId: Map<string, string>,
  itemId: unknown,
  delta: unknown
): void {
  if (typeof itemId !== "string" || !itemId) return;
  if (typeof delta !== "string" || !delta) return;
  deliveredContentByItemId.set(
    itemId,
    (deliveredContentByItemId.get(itemId) || "") + delta
  );
}

/**
 * Late Responses reasoning handlers exist to rescue `encrypted_content` /
 * item id after summary deltas have already streamed the text. If this
 * item's content was already delivered on the current stream, return
 * replay metadata only. A terminal-only reasoning item (no summary
 * deltas) still delivers content exactly once.
 */
export function thinkingForLateReasoningItem(
  item: any,
  deliveredContentByItemId: Map<string, string>
): UnifiedAssistantThinking | undefined {
  const thinking = thinkingFromResponsesReasoningItem(item);
  if (!thinking) return undefined;
  const id =
    (typeof item?.id === "string" && item.id) || thinking.id;
  const alreadyDelivered = !!(id && deliveredContentByItemId.get(id));
  if (alreadyDelivered) {
    if (!thinking.encrypted_content && !thinking.id) return undefined;
    return {
      content: "",
      ...(thinking.encrypted_content
        ? { encrypted_content: thinking.encrypted_content }
        : {}),
      ...(thinking.id ? { id: thinking.id } : {}),
    };
  }
  if (id && thinking.content) {
    deliveredContentByItemId.set(id, thinking.content);
  }
  return thinking;
}

/**
 * Inverse of the inbound `text.format` → Chat Completions `response_format`
 * mapping in `responsesRequestToUnified`. Shared by Codex and generic
 * Responses outbound so the two reconstruct sites cannot drift.
 */
export function responsesTextFormatFromResponseFormat(
  responseFormat: any
): { type: string; name?: string; schema?: any; strict?: boolean } | undefined {
  if (!responseFormat || typeof responseFormat !== "object") return undefined;
  if (responseFormat.type === "json_schema") {
    return {
      type: "json_schema",
      name: responseFormat.json_schema?.name,
      schema: responseFormat.json_schema?.schema,
      ...(responseFormat.json_schema?.strict !== undefined
        ? { strict: responseFormat.json_schema.strict }
        : {}),
    };
  }
  if (typeof responseFormat.type === "string" && responseFormat.type) {
    return { type: responseFormat.type };
  }
  return undefined;
}

/** Unified assistant.thinking → Responses `reasoning` input/output item. */
export function thinkingFromUnifiedAssistant(
  message: any
): UnifiedAssistantThinking | undefined {
  if (!message || typeof message !== "object") return undefined;
  const fromThinking =
    typeof message.thinking?.content === "string" ? message.thinking.content : "";
  const fromReasoningContent =
    typeof message.reasoning_content === "string" ? message.reasoning_content : "";
  const fromReasoning =
    typeof message.reasoning === "string" ? message.reasoning : "";
  const content = fromThinking || fromReasoningContent || fromReasoning;
  const rawSignature =
    typeof message.thinking?.signature === "string" && message.thinking.signature
      ? message.thinking.signature
      : undefined;
  // Never copy thinking.signature into encrypted_content: Anthropic/Gemini
  // signatures and Responses item ids are not Codex ciphertext.
  const encrypted_content = responsesEncryptedContentFrom(
    message.thinking?.encrypted_content
  );
  const id =
    (typeof message.thinking?.id === "string" && message.thinking.id) ||
    (isResponsesReasoningItemId(rawSignature) ? rawSignature : undefined) ||
    (isResponsesReasoningItemId(message.thinking?.encrypted_content)
      ? message.thinking.encrypted_content
      : undefined);
  const signature =
    rawSignature && !isResponsesReasoningItemId(rawSignature)
      ? rawSignature
      : undefined;
  if (!content && !signature && !encrypted_content && !id) return undefined;
  return {
    content,
    ...(signature ? { signature } : {}),
    ...(encrypted_content ? { encrypted_content } : {}),
    ...(id ? { id } : {}),
  };
}

/**
 * Fixed assistant-turn order for every protocol:
 * thinking → text → images → tool calls.
 * Reordering here keeps cache prefixes stable across Anthropic, Chat,
 * Responses, Gemini, and Mistral.
 */
export interface CanonicalAssistantTurn {
  thinking?: UnifiedAssistantThinking;
  texts: Array<{ text: string; cache_control?: any }>;
  images: any[];
  toolCalls: any[];
}

export function canonicalAssistantTurn(message: any): CanonicalAssistantTurn {
  const texts: Array<{ text: string; cache_control?: any }> = [];
  const images: any[] = [];
  if (typeof message?.content === "string" && message.content) {
    texts.push({ text: message.content });
  } else if (Array.isArray(message?.content)) {
    for (const part of message.content) {
      if (!part || typeof part !== "object") continue;
      if (
        (part.type === "text" ||
          part.type === "output_text" ||
          part.type === "input_text") &&
        part.text
      ) {
        texts.push({
          text: part.text,
          ...(part.cache_control ? { cache_control: part.cache_control } : {}),
        });
      } else if (part.type === "image_url" || part.type === "input_image" || part.type === "output_image") {
        images.push(part);
      }
    }
  }
  return {
    thinking: thinkingFromUnifiedAssistant(message),
    texts,
    images,
    toolCalls: Array.isArray(message?.tool_calls) ? message.tool_calls : [],
  };
}

export function assistantTurnHasText(turn: CanonicalAssistantTurn): boolean {
  return turn.texts.some((part) => part.text.length > 0);
}

/** Placeholder CCR used to emit for ciphertext-only items; Zen rejects dupes. */
const REASONING_ID_PLACEHOLDER = "rs_anon";

function isUsableReasoningItemId(id: unknown): id is string {
  return typeof id === "string" && !!id && id !== REASONING_ID_PLACEHOLDER;
}

function mintReasoningItemId(seed: string): string {
  return `rs_${createHash("sha256").update(seed).digest("hex").slice(0, 24)}`;
}

export function responsesReasoningItemFromThinking(
  thinking:
    | {
        content?: string;
        signature?: string;
        encrypted_content?: string;
        id?: string;
      }
    | undefined,
  id?: string
): any | null {
  if (!thinking || typeof thinking !== "object") return null;
  const content = typeof thinking.content === "string" ? thinking.content : "";
  const encrypted_content = responsesEncryptedContentFrom(
    thinking.encrypted_content
  );
  // Only an id carried on the thinking object itself (or real ciphertext /
  // summary text) justifies emitting a reasoning item. The optional `id`
  // argument is a preferred label for that item, not a reason to invent one
  // — otherwise skeletonResponse always minting `rs_${responseId}` would
  // prepend an empty reasoning item on every completed stream.
  const thinkingId = isUsableReasoningItemId(thinking.id)
    ? thinking.id
    : undefined;
  const preferredId = isUsableReasoningItemId(id) ? id : undefined;
  if (!content && !encrypted_content && !thinkingId) return null;
  const item: any = {
    type: "reasoning",
    // Date.now() ids rewrite the whole Responses/Codex prefix on every
    // turn and bust prompt cache. Prefer a carried id, then a caller
    // label, then a stable hash of ciphertext / summary. Never emit the
    // shared `rs_anon` placeholder — Zen 400s on duplicate input ids.
    id:
      thinkingId ||
      preferredId ||
      (encrypted_content
        ? mintReasoningItemId(encrypted_content)
        : content
          ? mintReasoningItemId(content)
          : mintReasoningItemId(`reasoning:${thinkingId || preferredId || ""}`)),
    summary: content ? [{ type: "summary_text", text: content }] : [],
  };
  if (encrypted_content) item.encrypted_content = encrypted_content;
  return item;
}

/**
 * Zen rejects Requests whose `input` repeats the same reasoning item id.
 * Rewrite placeholders / collisions using ciphertext (or summary) as seed.
 */
export function uniquifyReasoningItemIds(items: any[] | undefined): void {
  if (!Array.isArray(items)) return;
  const seen = new Set<string>();
  for (let index = 0; index < items.length; index++) {
    const item = items[index];
    if (!item || item.type !== "reasoning") continue;
    let id = isUsableReasoningItemId(item.id) ? item.id : "";
    if (!id || seen.has(id)) {
      const summary = reasoningSummaryText(item);
      const encrypted =
        typeof item.encrypted_content === "string" ? item.encrypted_content : "";
      const seed =
        encrypted || summary || `reasoning:${index}:${id || "missing"}`;
      id = mintReasoningItemId(seen.has(id) ? `${seed}#${index}` : seed);
      item.id = id;
    }
    seen.add(id);
  }
}

function attachThinkingToAssistant(
  messages: any[],
  thinking: UnifiedAssistantThinking
): void {
  const last = messages[messages.length - 1];
  if (last && last.role === "assistant") {
    last.thinking = thinking;
    return;
  }
  messages.push({
    role: "assistant",
    content: null,
    thinking,
  });
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

  if (item.type === "reasoning") {
    const thinking = thinkingFromResponsesReasoningItem(item);
    if (thinking) attachThinkingToAssistant(messages, thinking);
    return;
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
      content: functionCallOutputToUnified(item.output),
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
    // Chat Completions (and DeepSeek) require one assistant message carrying
    // every parallel tool_call; Responses emits them as consecutive items.
    appendAssistantToolCall(messages, {
      id: mapCallId(callIdMap, rawCallId) ?? rawCallId,
      type: "function",
      function: {
        name: item.name,
        arguments:
          typeof item.arguments === "string"
            ? item.arguments
            : JSON.stringify(item.arguments ?? {}),
      },
    });
    return;
  }

  if (item.type === "custom_tool_call_output") {
    if (typeof item.call_id !== "string" || !item.call_id) {
      throw createApiError(
        "custom_tool_call_output requires call_id",
        400,
        "invalid_call_id",
        "invalid_request_error"
      );
    }
    messages.push({
      role: "tool",
      tool_call_id: mapCallId(callIdMap, item.call_id) ?? item.call_id,
      content: functionCallOutputToUnified(item.output),
    });
    return;
  }

  if (item.type === "custom_tool_call") {
    const rawCallId = item.call_id || item.id;
    if (typeof rawCallId !== "string" || !rawCallId) {
      throw createApiError(
        "custom_tool_call requires call_id",
        400,
        "invalid_call_id",
        "invalid_request_error"
      );
    }
    if (typeof item.name !== "string" || !item.name) {
      throw createApiError(
        "custom_tool_call requires name",
        400,
        "invalid_function_call",
        "invalid_request_error"
      );
    }
    if (typeof item.input !== "string") {
      throw createApiError(
        "custom_tool_call requires string input",
        400,
        "invalid_custom_tool_input",
        "invalid_request_error"
      );
    }
    // Mirror the synthetic single-string-param shape normalizeResponsesTools
    // gave this tool on the way out, so replayed history round-trips.
    appendAssistantToolCall(messages, {
      id: mapCallId(callIdMap, rawCallId) ?? rawCallId,
      type: "function",
      function: {
        name: item.name,
        arguments: JSON.stringify({
          [CUSTOM_TOOL_INPUT_KEY]: item.input,
        }),
      },
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
    let rawContent = item.content;
    if (
      typeof rawContent !== "string" &&
      !Array.isArray(rawContent)
    ) {
      if (shouldLogResponsesPassthrough()) {
        try {
          // eslint-disable-next-line no-console
          console.debug(
            `[responses passthrough] message ${role} with non-string/array content ${String(typeof rawContent)} ${JSON.stringify(rawContent)?.slice(0, 400) ?? ""}`
          );
        } catch {}
      }
      if (rawContent == null) {
        rawContent = "";
      } else if (typeof rawContent === "object" && !Array.isArray(rawContent)) {
        rawContent = [rawContent as any];
      } else {
        rawContent = String(rawContent);
      }
    }
    const content = flattenResponsesContent(rawContent);
    if (role === "assistant" && appendAssistantContent(messages, content)) {
      return;
    }
    messages.push({
      role,
      content,
    });
    return;
  }

  if (item.type === "input_text" || item.type === "output_text") {
    const content = String(item.text ?? "");
    const role = item.type === "output_text" ? "assistant" : "user";
    if (role === "assistant" && appendAssistantContent(messages, content)) {
      return;
    }
    messages.push({
      role,
      content,
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

  if (item.type === "input_file") {
    if (item.file_id) {
      throw createApiError(
        "Provider-bound file_id inputs are not supported",
        400,
        "unsupported_file_id",
        "invalid_request_error"
      );
    }
    const fileData = item.file_data;
    const fileUrl = item.file_url;
    if (typeof fileData !== "string" && typeof fileUrl !== "string") {
      throw createApiError(
        "input_file requires file_data or file_url",
        400,
        "invalid_file",
        "invalid_request_error"
      );
    }
    messages.push({
      role: "user",
      content: [
        {
          type: "file",
          ...(typeof item.filename === "string"
            ? { filename: item.filename }
            : {}),
          ...(typeof fileData === "string" ? { file_data: fileData } : {}),
          ...(typeof fileUrl === "string" ? { file_url: fileUrl } : {}),
          ...(typeof item.mime_type === "string"
            ? { media_type: item.mime_type }
            : {}),
        },
      ],
    });
    return;
  }

  // Hosted calls (web_search_call, file_search_call, etc.) are ChatGPT-backend
  // state. With store:false Codex replays them verbatim like reasoning
  // encrypted_content. Keep forwards the original input[] unchanged, but this
  // Unified projection is only for routing — so passthrough (drop from Unified)
  // and log rather than 400 or synthesizing an assistant artifact.
  if (
    item.type === "web_search_call" ||
    (typeof item.type === "string" &&
      item.type.endsWith("_call") &&
      item.type !== "function_call" &&
      item.type !== "custom_tool_call")
  ) {
    const action = (item as any).action;
    const hint =
      (typeof action?.query === "string" && action.query) ||
      (Array.isArray(action?.queries) && action.queries[0]) ||
      (typeof action?.url === "string" && action.url) ||
      "";
    // Only when explicitly enabled via config/env — avoids noise on every
    // Codex turn with web_search. Keep wire already logs the full input[]
    // when body part capture is on.
    if (shouldLogResponsesPassthrough()) {
      try {
        const id = (item as any).id || (item as any).call_id || "";
        // eslint-disable-next-line no-console
        console.debug(
          `[responses passthrough] ${item.type}${id ? ` ${id}` : ""}${hint ? ` — ${String(hint).slice(0, 200)}` : ""}`
        );
      } catch {
        // ignore logging failures
      }
    }
    return;
  }

  throw createApiError(
    `Unsupported Responses input item type '${item.type || "unknown"}'`,
    400,
    "unsupported_input_item",
    "invalid_request_error"
  );
}

/**
 * Responses `function_call_output.output` / `custom_tool_call_output.output`
 * may be a plain string or an OutputContentList (text + images + files).
 * OpenCode / @ai-sdk/openai put webfetch image attachments in that list.
 * JSON.stringifying the list destroys `input_image` and Zen/Meta reject the
 * replayed string as invalid parameters — keep structured parts as Unified
 * `text` / `image_url` so Responses outbound can re-emit `input_image`.
 */
function functionCallOutputToUnified(output: unknown): string | any[] {
  if (typeof output === "string") return output;
  if (Array.isArray(output)) return flattenResponsesContent(output);
  return JSON.stringify(output ?? "");
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
    } else if (part.type === "input_file") {
      if ((part as any).file_id) {
        throw createApiError(
          "Provider-bound file_id inputs are not supported",
          400,
          "unsupported_file_id",
          "invalid_request_error"
        );
      }
      const fileData = part.file_data;
      const fileUrl = part.file_url;
      if (typeof fileData !== "string" && typeof fileUrl !== "string") {
        throw createApiError(
          "input_file requires file_data or file_url",
          400,
          "invalid_file",
          "invalid_request_error"
        );
      }
      parts.push({
        type: "file",
        ...(typeof part.filename === "string" ? { filename: part.filename } : {}),
        ...(typeof fileData === "string" ? { file_data: fileData } : {}),
        ...(typeof fileUrl === "string" ? { file_url: fileUrl } : {}),
        ...(typeof part.mime_type === "string"
          ? { media_type: part.mime_type }
          : {}),
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

/** Synthetic argument key used to carry a `custom` tool's freeform text
 * through the Unified/Chat Completions function-call shape. */
export const CUSTOM_TOOL_INPUT_KEY = "input";

/**
 * Custom/freeform tools have no native concept on models that were never
 * trained on OpenAI's Responses `type: "custom"` grammar tool (only OpenAI's
 * own backend and a handful of Responses-compatible hosts support it — see
 * openai/codex#19416, where Codex itself falls back to a plain function tool
 * for Bedrock-hosted models for the same reason). Proxying through a single
 * JSON string parameter (CUSTOM_TOOL_INPUT_KEY) works, but a model
 * unfamiliar with the convention sometimes reaches for the nearest thing it
 * does know: invoking the tool as if it were a shell command, wrapping its
 * freeform text in a bash heredoc. Unwrap that here so the client (Codex
 * CLI's own apply_patch parser, etc.) receives clean freeform text instead
 * of a heredoc-wrapped shell fragment. Mirrors opencode's stripHeredoc
 * (packages/opencode/src/patch/index.ts) — purely syntactic: only fires when
 * the *entire* input matches the wrapper pattern, so genuine patch/tool
 * content is never touched.
 */
function stripHeredocWrapper(text: string): string {
  const match = text.match(/^(?:cat\s+)?<<['"]?(\w+)['"]?\s*\n([\s\S]*?)\n\1\s*$/);
  return match ? match[2] : text;
}

/**
 * Codex's apply_patch/apply_update grammar accepts only the exact markers
 * `*** Begin Patch` and `*** End Patch` (no trailing asterisk triplet) as the
 * first/last patch lines — anything else is rejected with "The first line of
 * the patch must be '*** Begin Patch'". Models not trained on that grammar
 * (e.g. Grok reaches for Claude's trailing-asterisk variant) emit
 * `*** Begin Patch ***` / `*** End Patch ***`. Normalize those to the Codex
 * form on the client-facing emission, scoped by the marker token itself so a
 * rewrite can only ever hit an apply_patch-style payload regardless of the
 * declared tool name.
 */
export function normalizeCodexPatchMarkers(text: string): string {
  if (!text.includes("*** Begin Patch ***") && !text.includes("*** End Patch ***")) {
    return text;
  }
  return text
    .split("*** Begin Patch ***").join("*** Begin Patch")
    .split("*** End Patch ***").join("*** End Patch");
}

/**
 * Codex's `exec` tool runs raw JavaScript in a V8 isolate; a model that was
 * never trained on that convention (Grok) sometimes fills the freeform input
 * with a JSON shell-envelope instead — e.g. `{"cmd": "ls -la /tmp"}`. That is
 * not valid JS (`{"cmd":…}` parses as a block with a string label →
 * `SyntaxError: Unexpected token ':'`), so the shell call fails and the model
 * wastes a turn retrying as `await tools.exec_command({cmd: …})`.
 * Rewrite the envelope into exactly that retry shape on the client-facing
 * emission. Scoped by the tool name and the whole-input-is-a-shell-object
 * shape so a freeform tool that legitimately takes a JSON blob is never
 * touched. Grok also alternates the key between `cmd` and `command`, and
 * `exec_command` only accepts `cmd`, so normalize either to `cmd`.
 */
export function normalizeExecCommandEnvelope(
  name: string | undefined,
  text: string
): string {
  if (name !== "exec") return text;
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return text;
  let parsed: any;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return text;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return text;
  }
  const commandKey = "cmd" in parsed ? "cmd" : "command" in parsed ? "command" : null;
  if (!commandKey) return text;
  const command = parsed[commandKey];
  if (typeof command !== "string" || !command) return text;
  delete parsed[commandKey];
  parsed.cmd = command;
  return `await tools.exec_command(${JSON.stringify(parsed)});`;
}

/**
 * A model that reaches for the patch tool through a shell hereditary habit
 * wraps the patch in a heredoc instead of JS:
 *   apply_patch << 'PATCH'\n*** Begin Patch … *** End Patch\nPATCH
 * That is a shell command, not the raw JS `exec` runs, so it fails with a
 * syntax error and the model wastes a turn retrying as
 * `await tools.apply_patch(…)`. Normalize the heredoc invocation into the JS
 * call form. Whole-value heredoc match only; anything else passes through.
 */
export function normalizeExecApplyPatchHeredoc(
  name: string | undefined,
  text: string
): string {
  if (name !== "exec") return text;
  const match = text.match(
    /^apply_patch\s*<<\s*['"]?(\w+)['"]?\s*\n([\s\S]*?)\n\1\s*$/
  );
  if (!match) return text;
  return `await tools.apply_patch(${JSON.stringify(match[2])});`;
}

/**
 * `exec` may cross the wire as a plain function tool (arguments stay the JSON
 * wrapper `{"input":"…"}`) rather than a custom tool (whose freeform input was
 * unwrapped already). The envelope/heredoc normalizers above operate on the
 * unwrapped inner text, so when only the wrapper is available, descend into it
 * and normalize its inner `input` value with the same rules.
 */
export function normalizeExecFunctionArguments(
  name: string | undefined,
  rawArguments: string,
  options?: CodexIsolateConventionsOptions
): string {
  if (!applyCodexIsolateConventions(options)) return rawArguments;
  if (name !== "exec") return rawArguments;
  try {
    const parsed = JSON.parse(rawArguments);
    if (parsed && typeof parsed === "object" && typeof parsed.input === "string") {
      parsed.input = normalizeExecApplyPatchHeredoc(
        name,
        normalizeExecCommandEnvelope(name, parsed.input)
      );
      return JSON.stringify(parsed);
    }
  } catch {
    // Not the {"input": …} wrapper; leave untouched.
  }
  return rawArguments;
}

/** Undo the CUSTOM_TOOL_INPUT_KEY wrapping applied in normalizeResponsesTools.
 * Falls back to the raw text so a malformed/empty call still round-trips
 * instead of vanishing. Also strips a whole-value shell heredoc wrapper
 * (see stripHeredocWrapper). That strip is intentional on both paths this
 * helper serves: client emission (so the Responses client sees clean
 * freeform text) and CodexTransformer history replay (so the backend is
 * not re-fed a wrapper the client never kept). Replay is therefore
 * normalized, not byte-faithful to the model's original heredoc. */
export function unwrapCustomToolInput(rawArguments: string): string {
  if (!rawArguments) return "";
  try {
    const parsed = JSON.parse(rawArguments);
    const value = parsed?.[CUSTOM_TOOL_INPUT_KEY];
    if (typeof value === "string") return stripHeredocWrapper(value);
  } catch {
    // Fall through to the raw text below.
  }
  return stripHeredocWrapper(rawArguments);
}

/**
 * Codex V8 isolate calling conventions (`await tools.exec_command` /
 * `await tools.apply_patch`). Default on: CCR's `/v1/responses` inbound
 * is the Codex CLI path, and Grok-via-Chat recovery depends on it.
 * Pass `false` for a generic Responses destination whose `exec` tool
 * legitimately accepts `{"cmd": …}` JSON.
 */
export interface CodexIsolateConventionsOptions {
  codexIsolateConventions?: boolean;
}

function applyCodexIsolateConventions(
  options?: CodexIsolateConventionsOptions
): boolean {
  return options?.codexIsolateConventions !== false;
}

/** Client-facing custom-tool input: unwrap the Unified JSON wrapper, then
 * apply the exec/patch normalizers in one place so stream finalize, the
 * completed skeleton, and the non-stream JSON path cannot drift. */
export function normalizeClientCustomToolInput(
  name: string | undefined,
  rawArguments: string,
  options?: CodexIsolateConventionsOptions
): string {
  let text = unwrapCustomToolInput(rawArguments);
  if (applyCodexIsolateConventions(options)) {
    text = normalizeExecApplyPatchHeredoc(
      name,
      normalizeExecCommandEnvelope(name, text)
    );
  }
  return normalizeCodexPatchMarkers(text);
}

function normalizeResponsesTools(
  tools: any,
  customToolNames?: Set<string>
): UnifiedChatRequest["tools"] | undefined {
  if (tools == null) return undefined;
  if (!Array.isArray(tools)) {
    throw createApiError(
      "Responses tools must be an array",
      400,
      "invalid_tools",
      "invalid_request_error"
    );
  }
  // Unified is OpenAI Chat Completions: only `function` tools are valid.
  // Project every inbound Responses tool (function, custom/MCP, web_search,
  // and any other hosted type) onto a function so Chat Completions backends
  // never see Responses-only hosted-tool variants.
  return tools.map((t: any) => {
    const type = typeof t?.type === "string" ? t.type : "";
    // function / custom always need an explicit client name — do not fall
    // back to the type string ("custom") or the tool is silently unusable.
    let name: string | undefined = t?.name || t?.function?.name;
    if (type === "web_search" || type === "web_search_preview") {
      // Stable name so hasWebSearchTool / provider routing still match.
      name = "web_search";
    } else if (!name && type && type !== "function" && type !== "custom") {
      // Other hosted tools (file_search, computer_use, …) use type as name.
      name = type;
    }
    if (typeof name !== "string" || !name) {
      throw createApiError(
        "Responses function tools require a name",
        400,
        "invalid_tool",
        "invalid_request_error"
      );
    }

    if (type === "custom") {
      // `custom` tools take unconstrained freeform text (no JSON schema —
      // see CustomToolParam), unlike `function` tools. Chat Completions has
      // no freeform-tool concept, so proxy it through a single required
      // string parameter instead of an empty object schema: an empty schema
      // gives the model no signal about what to put where, so it hallucinates
      // JSON-shaped arguments the tool never asked for. The response side
      // (unifiedChunkToResponsesEvents/unifiedResponseToResponses) unwraps
      // this key and re-emits a genuine `custom_tool_call` item.
      //
      // A model that was never trained on this convention (anything but
      // OpenAI's own — see the codex/Bedrock precedent in
      // stripHeredocWrapper's comment above) has to solve two problems at
      // once: get the tool's own freeform grammar right, *and* figure out
      // this JSON-wrapping detour. The property/function descriptions below
      // spell out the wrapping explicitly so the model only has to solve the
      // first problem — this is prompt guidance, not a content rewrite, so
      // it's as safe as the heredoc strip: it only changes what the model is
      // told, never what a well-formed call's content means.
      customToolNames?.add(name);
      const baseDescription = t.description || t.function?.description || "";
      // Provider-neutral contract only. Codex V8 calling conventions
      // (`await tools.exec_command`, `await tools.apply_patch`) stay off
      // this description so Anthropic and Chat Completions backends are
      // not prompted as if they were Codex's isolate. Recovery of
      // `{cmd:…}` / heredoc / trailing-asterisk markers happens on the
      // Responses client emission (normalizeClientCustomToolInput).
      return {
        type: "function" as const,
        function: {
          name,
          description: baseDescription
            ? `${baseDescription}\n\nNote: this tool takes freeform text, not JSON. Call it with a single "${CUSTOM_TOOL_INPUT_KEY}" argument whose value is the complete text described above, as a plain string — do not add markdown code fences, a shell heredoc (e.g. "<<EOF"), or any other wrapping around it.`
            : baseDescription,
          parameters: {
            type: "object",
            properties: {
              [CUSTOM_TOOL_INPUT_KEY]: {
                type: "string",
                description:
                  "The complete freeform text/code input for this tool, exactly as described in the tool description — as a plain string, with no markdown code fences or shell heredoc wrapping.",
              },
            },
            required: [CUSTOM_TOOL_INPUT_KEY],
          },
        },
      };
    }

    return {
      type: "function" as const,
      function: {
        name,
        description: t.description || t.function?.description,
        parameters:
          t.parameters ||
          t.schema ||
          t.function?.parameters ||
          { type: "object", properties: {} },
      },
    };
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
  if (
    toolChoice?.name &&
    (toolChoice?.type === "function" || toolChoice?.type === "custom")
  ) {
    // `custom` tools are projected onto function tools, so a forced choice on
    // one resolves to the same function name.
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
    customToolNames?: Set<string>;
    codexIsolateConventions?: boolean;
  }
): any {
  const callIdMap = options?.callIdMap ?? createCallIdMap();
  const isolate: CodexIsolateConventionsOptions = {
    codexIsolateConventions: options?.codexIsolateConventions,
  };
  const choice = chat?.choices?.[0];
  const message = choice?.message || {};
  const output: any[] = [];

  const reasoningItem = responsesReasoningItemFromThinking(
    thinkingFromUnifiedAssistant(message),
    chat?.id ? `rs_${chat.id}` : undefined
  );
  if (reasoningItem) output.push(reasoningItem);

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
      const rawArguments =
        typeof tc.function.arguments === "string"
          ? tc.function.arguments
          : JSON.stringify(tc.function.arguments ?? {});
      if (options?.customToolNames?.has(tc.function.name)) {
        output.push({
          type: "custom_tool_call",
          id: callId,
          call_id: callId,
          name: tc.function.name,
          input: normalizeClientCustomToolInput(
            tc.function.name,
            rawArguments,
            isolate
          ),
        });
        continue;
      }
      output.push({
        type: "function_call",
        id: callId,
        call_id: callId,
        name: tc.function.name,
        arguments: normalizeCodexPatchMarkers(
          normalizeExecFunctionArguments(tc.function.name, rawArguments, isolate)
        ),
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
  textClosed: boolean;
  textOutputIndex?: number;
  textContent: string;
  closedTextItems: Array<{
    id: string;
    outputIndex: number;
    content: string;
  }>;
  toolCalls: Map<
    number,
    {
      id: string;
      name: string;
      arguments: string;
      added: boolean;
      outputIndex: number;
      emittedArgumentsLength: number;
      isCustom: boolean;
    }
  >;
  nextOutputIndex: number;
  finished: boolean;
  created: boolean;
  sequenceNumber: number;
  finishReasonSeen: boolean;
  usage?: any;
  callIdMap: ResponsesCallIdMap;
  customToolNames: Set<string>;
  thinkingContent: string;
  thinkingEncryptedContent?: string;
  thinkingId?: string;
  thinkingStarted: boolean;
  thinkingClosed: boolean;
  thinkingOutputIndex?: number;
  codexIsolateConventions: boolean;
}

export function createResponsesStreamState(
  options?: {
    model?: string;
    callIdMap?: ResponsesCallIdMap;
    customToolNames?: Set<string>;
    codexIsolateConventions?: boolean;
  }
): ResponsesStreamState {
  const id = `resp_${Date.now()}`;
  return {
    responseId: id,
    model: options?.model,
    textItemId: `msg_${id}`,
    textStarted: false,
    textClosed: false,
    textContent: "",
    closedTextItems: [],
    toolCalls: new Map(),
    nextOutputIndex: 0,
    finished: false,
    created: false,
    sequenceNumber: 0,
    finishReasonSeen: false,
    callIdMap: options?.callIdMap ?? createCallIdMap(),
    customToolNames: options?.customToolNames ?? new Set<string>(),
    thinkingContent: "",
    thinkingStarted: false,
    thinkingClosed: false,
    codexIsolateConventions: options?.codexIsolateConventions !== false,
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
 * Close the text (message) item's Responses lifecycle — idempotent via
 * textClosed. Must run before a tool call's `output_item.added` whenever a
 * text preamble preceded it: Responses items are an ordered array by
 * output_index, and a client (e.g. the Codex/ChatGPT UI) that renders items
 * as they complete expects item N to close before item N+1 opens. Previously
 * this only ran at finalizeResponsesStream (end of stream), so a tool call
 * arriving after a text preamble got its `added` event emitted live while
 * the text item was still open, and the text item's `.done` events only
 * followed at the very end, after the tool call's own added/delta/done
 * sequence had already interleaved with it — a real client can silently
 * fail to render the tool call as a distinct block when items close out of
 * their own start order.
 */
/**
 * Open the Responses reasoning item lifecycle. OpenCode / @ai-sdk/openai only
 * materialize thinking from streamed `reasoning_summary_*` events (plus
 * `output_item.added` for reasoning) — landing the summary solely on
 * `response.completed.output` leaves the client with no thinking parts.
 */
function ensureThinkingItem(state: ResponsesStreamState, events: any[]): void {
  if (state.thinkingStarted || state.thinkingClosed) return;
  state.thinkingStarted = true;
  state.thinkingOutputIndex = state.nextOutputIndex++;
  if (!state.thinkingId) {
    state.thinkingId = `rs_${state.responseId}`;
  }
  events.push({
    type: "response.output_item.added",
    output_index: state.thinkingOutputIndex,
    item: {
      type: "reasoning",
      id: state.thinkingId,
      status: "in_progress",
      summary: [],
    },
  });
  events.push({
    type: "response.reasoning_summary_part.added",
    item_id: state.thinkingId,
    output_index: state.thinkingOutputIndex,
    summary_index: 0,
    part: { type: "summary_text", text: "" },
  });
}

function closeThinkingItem(state: ResponsesStreamState, events: any[]): void {
  if (!state.thinkingStarted || state.thinkingClosed) return;
  state.thinkingClosed = true;
  const id = state.thinkingId || `rs_${state.responseId}`;
  const item = responsesReasoningItemFromThinking(
    {
      content: state.thinkingContent,
      encrypted_content: state.thinkingEncryptedContent,
      id,
    },
    id
  );
  events.push({
    type: "response.reasoning_summary_text.done",
    item_id: id,
    output_index: state.thinkingOutputIndex,
    summary_index: 0,
    text: state.thinkingContent,
  });
  events.push({
    type: "response.reasoning_summary_part.done",
    item_id: id,
    output_index: state.thinkingOutputIndex,
    summary_index: 0,
    part: { type: "summary_text", text: state.thinkingContent },
  });
  events.push({
    type: "response.output_item.done",
    output_index: state.thinkingOutputIndex,
    item: item
      ? { ...item, status: "completed" }
      : {
          type: "reasoning",
          id,
          status: "completed",
          summary: state.thinkingContent
            ? [{ type: "summary_text", text: state.thinkingContent }]
            : [],
        },
  });
}

function closeTextItem(state: ResponsesStreamState, events: any[]): void {
  if (!state.textStarted || state.textClosed) return;
  state.textClosed = true;
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
  state.closedTextItems.push({
    id: state.textItemId,
    outputIndex: state.textOutputIndex ?? 0,
    content: state.textContent,
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

  // Chat providers emit reasoning_content; Unified thinking is the same
  // history. Prefer thinkingFromUnifiedAssistant so a Responses client of a
  // Chat Completions upstream still sees a reasoning item.
  const thinking = thinkingFromUnifiedAssistant(delta);

  // Preserve upstream lifecycle starts instead of waiting for generated content.
  if (
    !state.textStarted &&
    state.toolCalls.size === 0 &&
    (delta.role === "assistant" ||
      typeof delta.content === "string" ||
      Array.isArray(delta.tool_calls) ||
      thinking?.content ||
      thinking?.encrypted_content ||
      thinking?.id ||
      choice?.finish_reason)
  ) {
    appendCreatedEvent(state, events);
  }

  if (thinking?.content) {
    state.thinkingContent += thinking.content;
  }
  const encrypted_content = responsesEncryptedContentFrom(
    thinking?.encrypted_content
  );
  if (encrypted_content) {
    state.thinkingEncryptedContent = encrypted_content;
  }
  // Lock the reasoning item id on first sight — rewriting after
  // output_item.added desyncs @ai-sdk/openai's activeReasoning map.
  // Reject the legacy `rs_anon` placeholder so we mint a unique id instead.
  if (!state.thinkingId) {
    if (
      typeof thinking?.id === "string" &&
      thinking.id &&
      thinking.id !== "rs_anon"
    ) {
      state.thinkingId = thinking.id;
    } else if (isResponsesReasoningItemId(delta.thinking?.signature)) {
      state.thinkingId = delta.thinking.signature;
    }
  }

  // Stream reasoning the way OpenAI / Zen do: clients (OpenCode via
  // @ai-sdk/openai) ignore summaries that only appear on response.completed.
  if (
    thinking?.content ||
    encrypted_content ||
    (typeof thinking?.id === "string" && thinking.id)
  ) {
    ensureThinkingItem(state, events);
    if (thinking?.content) {
      events.push({
        type: "response.reasoning_summary_text.delta",
        item_id: state.thinkingId,
        output_index: state.thinkingOutputIndex,
        summary_index: 0,
        delta: thinking.content,
      });
    }
  }

  if (typeof delta.content === "string" && delta.content.length > 0) {
    closeThinkingItem(state, events);
    if (state.textClosed) {
      // The previous message item already completed (a tool call opened
      // after the preamble). Trailing text is a new output item — emitting
      // more output_text.delta against the closed item leaves a live
      // lifecycle the Codex/ChatGPT UI never sees.
      state.textStarted = false;
      state.textClosed = false;
      state.textContent = "";
      state.textItemId = `msg_${state.responseId}_${state.nextOutputIndex}`;
      state.textOutputIndex = undefined;
    }
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

  if (Array.isArray(delta.tool_calls) && delta.tool_calls.length > 0) {
    // Reasoning and text must fully close (in item-index order) before a
    // tool call's own added/delta/done sequence begins — see closeTextItem.
    closeThinkingItem(state, events);
    closeTextItem(state, events);
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
        const name = tc.function?.name || "";
        entry = {
          id: callId,
          name,
          arguments: "",
          added: false,
          outputIndex: state.nextOutputIndex++,
          emittedArgumentsLength: 0,
          isCustom: state.customToolNames.has(name),
        };
        state.toolCalls.set(index, entry);
      }
      if (tc.id && !entry.added) {
        entry.id =
          mapCallId(state.callIdMap, tc.id, "unified_to_client") || entry.id;
      }
      if (tc.function?.name) {
        entry.name = tc.function.name;
        entry.isCustom = state.customToolNames.has(entry.name);
      }
      if (typeof tc.function?.arguments === "string") {
        entry.arguments += tc.function.arguments;
      }

      if (!entry.added && entry.name) {
        entry.added = true;
        events.push({
          type: "response.output_item.added",
          output_index: entry.outputIndex,
          item: entry.isCustom
            ? {
                type: "custom_tool_call",
                id: entry.id,
                call_id: entry.id,
                name: entry.name,
                input: "",
              }
            : {
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
        // Function arguments arrive incrementally as JSON. For custom tools the
        // client expects deltas of the unwrapped freeform input, not JSON
        // fragments. Wait until finalization to emit the complete custom input;
        // function tools keep their normal streaming deltas.
        if (!entry.isCustom) {
          const unreported = entry.arguments.slice(entry.emittedArgumentsLength);
          events.push({
            type: "response.function_call_arguments.delta",
            item_id: entry.id,
            output_index: entry.outputIndex,
            delta: unreported,
          });
        }
        entry.emittedArgumentsLength = entry.arguments.length;
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
  const isolate: CodexIsolateConventionsOptions = {
    codexIsolateConventions: state.codexIsolateConventions,
  };

  closeThinkingItem(state, events);
  closeTextItem(state, events);

  for (const entry of state.toolCalls.values()) {
    if (entry.isCustom) {
      const input = normalizeClientCustomToolInput(
        entry.name,
        entry.arguments,
        isolate
      );
      events.push({
        type: "response.custom_tool_call_input.delta",
        item_id: entry.id,
        output_index: entry.outputIndex,
        delta: input,
      });
      events.push({
        type: "response.custom_tool_call_input.done",
        item_id: entry.id,
        output_index: entry.outputIndex,
        input,
      });
      events.push({
        type: "response.output_item.done",
        output_index: entry.outputIndex,
        item: {
          type: "custom_tool_call",
          id: entry.id,
          call_id: entry.id,
          name: entry.name,
          input,
        },
      });
      continue;
    }
    const clientArguments = normalizeCodexPatchMarkers(
      normalizeExecFunctionArguments(entry.name, entry.arguments, isolate)
    );
    events.push({
      type: "response.function_call_arguments.done",
      item_id: entry.id,
      output_index: entry.outputIndex,
      name: entry.name,
      arguments: clientArguments,
    });
    events.push({
      type: "response.output_item.done",
      output_index: entry.outputIndex,
      item: {
        type: "function_call",
        id: entry.id,
        call_id: entry.id,
        name: entry.name,
        arguments: clientArguments,
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
  for (const closed of state.closedTextItems) {
    indexedOutput.push({
      index: closed.outputIndex,
      item: {
        type: "message",
        id: closed.id,
        status: status === "completed" ? "completed" : "in_progress",
        role: "assistant",
        content: [
          {
            type: "output_text",
            text: closed.content,
            annotations: [],
            logprobs: [],
          },
        ],
      },
    });
  }
  if (
    (state.textStarted || state.textContent) &&
    !state.closedTextItems.some((item) => item.id === state.textItemId)
  ) {
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
    const isolate: CodexIsolateConventionsOptions = {
      codexIsolateConventions: state.codexIsolateConventions,
    };
    indexedOutput.push({
      index: entry.outputIndex,
      item: entry.isCustom
        ? {
            type: "custom_tool_call",
            id: entry.id,
            call_id: entry.id,
            name: entry.name,
            input: normalizeClientCustomToolInput(
              entry.name,
              entry.arguments,
              isolate
            ),
          }
        : {
            type: "function_call",
            id: entry.id,
            call_id: entry.id,
            name: entry.name,
            arguments: normalizeCodexPatchMarkers(
              normalizeExecFunctionArguments(entry.name, entry.arguments, isolate)
            ),
            status: status === "completed" ? "completed" : "in_progress",
          },
    });
  }
  const output = indexedOutput
    .sort((left, right) => left.index - right.index)
    .map(({ item }) => item);
  const reasoningItem = responsesReasoningItemFromThinking(
    {
      content: state.thinkingContent,
      encrypted_content: state.thinkingEncryptedContent,
      id: state.thinkingId,
    },
    state.thinkingId || `rs_${state.responseId}`
  );
  if (reasoningItem) output.unshift(reasoningItem);

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

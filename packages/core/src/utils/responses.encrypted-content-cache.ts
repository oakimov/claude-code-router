import { createHash } from "node:crypto";
import { LLMProvider, UnifiedChatRequest, UnifiedMessage } from "@/types/llm";
import { TransformerContext } from "@/types/transformer";
import { responsesEncryptedContentFrom } from "./openai.responses.util";

type ToolCallLike = NonNullable<UnifiedMessage["tool_calls"]>[number];

type MessageLike = Pick<
  UnifiedMessage,
  "role" | "content" | "thinking" | "tool_calls" | "tool_call_id" | "reasoning_content"
> & {
  name?: string;
};

export type EncryptedReasoningPayload = {
  encrypted_content: string;
  content?: string;
  id?: string;
};

type RequestContextState = {
  enabled: true;
  namespace: string;
  requestScope: string;
};

type CacheEntry = {
  payload: EncryptedReasoningPayload;
  createdAt: number;
};

const CACHE_TTL_MS = 6 * 60 * 60 * 1000;
const CACHE_MAX_ENTRIES = 4096;
const CONTEXT_KEY = "__ccrResponsesEncryptedReasoning";
const ENCRYPTED_INCLUDE = "reasoning.encrypted_content";

class EncryptedReasoningCache {
  private readonly cache = new Map<string, CacheEntry>();

  get(key: string): EncryptedReasoningPayload | undefined {
    const entry = this.cache.get(key);
    if (!entry) return undefined;
    if (Date.now() - entry.createdAt > CACHE_TTL_MS) {
      this.cache.delete(key);
      return undefined;
    }
    this.cache.delete(key);
    this.cache.set(key, entry);
    return entry.payload;
  }

  put(key: string, payload: EncryptedReasoningPayload): void {
    if (!payload?.encrypted_content) return;
    this.cache.delete(key);
    this.cache.set(key, {
      payload,
      createdAt: Date.now(),
    });
    this.prune();
  }

  private prune(): void {
    for (const [key, entry] of this.cache) {
      if (Date.now() - entry.createdAt > CACHE_TTL_MS) {
        this.cache.delete(key);
      }
    }

    while (this.cache.size > CACHE_MAX_ENTRIES) {
      const oldestKey = this.cache.keys().next().value;
      if (!oldestKey) break;
      this.cache.delete(oldestKey);
    }
  }
}

const encryptedReasoningCache = new EncryptedReasoningCache();

function stableSort(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(stableSort);
  }

  if (value && typeof value === "object") {
    return Object.keys(value as Record<string, unknown>)
      .sort()
      .reduce<Record<string, unknown>>((result, key) => {
        result[key] = stableSort((value as Record<string, unknown>)[key]);
        return result;
      }, {});
  }

  return value;
}

function stableStringify(value: unknown): string {
  return JSON.stringify(stableSort(value));
}

function hash(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function normalizeContent(content: MessageLike["content"]): string {
  if (typeof content === "string") return content;
  if (content == null) return "";
  if (Array.isArray(content)) {
    return content
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object" && "text" in item) {
          return String((item as { text?: string }).text || "");
        }
        return stableStringify(item);
      })
      .join("");
  }
  return stableStringify(content);
}

function normalizeToolCall(toolCall: ToolCallLike | Record<string, unknown>) {
  const functionData = (toolCall as any).function ?? {};
  const argumentsValue =
    typeof functionData.arguments === "string"
      ? functionData.arguments
      : stableStringify(functionData.arguments ?? "");

  return {
    id: (toolCall as any).id || "",
    type: (toolCall as any).type || "function",
    function: {
      name: functionData.name || "",
      arguments: argumentsValue,
    },
  };
}

function toolCallIds(message: MessageLike): string[] {
  return (message.tool_calls || [])
    .map((toolCall) => toolCall.id)
    .filter((id): id is string => typeof id === "string" && id.length > 0);
}

function toolCallSignature(toolCall: ToolCallLike): string {
  const normalized = normalizeToolCall(toolCall);
  return hash(
    stableStringify({
      ...normalized,
      id: undefined,
    })
  );
}

function messageSignature(message: MessageLike): string {
  return hash(
    stableStringify({
      content: normalizeContent(message.content),
      tool_calls: (message.tool_calls || []).map((toolCall) =>
        normalizeToolCall(toolCall)
      ),
    })
  );
}

function canonicalScopeMessage(message: MessageLike) {
  const canonical: Record<string, unknown> = {
    role: message.role,
    content: normalizeContent(message.content),
  };

  if (message.name) canonical.name = message.name;
  if (message.tool_call_id) canonical.tool_call_id = message.tool_call_id;
  if (message.tool_calls?.length) {
    canonical.tool_calls = message.tool_calls.map((toolCall) =>
      normalizeToolCall(toolCall)
    );
  }

  return canonical;
}

function conversationScope(messages: MessageLike[], namespace: string): string {
  const scopeMessages = messages
    .filter((message) => message.role !== "system")
    .map((message) => canonicalScopeMessage(message));

  return hash(
    stableStringify({
      namespace,
      messages: scopeMessages,
    })
  );
}

function assistantNeedsEncryptedReasoning(
  message: MessageLike,
  priorMessages: MessageLike[]
): boolean {
  if (message.tool_calls?.length) return true;

  for (let i = priorMessages.length - 1; i >= 0; i--) {
    const prior = priorMessages[i];
    if (prior.role === "tool") return true;
    if (prior.role === "user" || prior.role === "system") return false;
  }

  return false;
}

function getInlineEncryptedPayload(
  message: MessageLike
): EncryptedReasoningPayload | undefined {
  const encrypted_content = responsesEncryptedContentFrom(
    message.thinking?.encrypted_content
  );
  if (!encrypted_content) return undefined;
  const content =
    typeof message.thinking?.content === "string"
      ? message.thinking.content
      : typeof message.reasoning_content === "string"
        ? message.reasoning_content
        : undefined;
  const id =
    typeof message.thinking?.id === "string" && message.thinking.id
      ? message.thinking.id
      : undefined;
  return {
    encrypted_content,
    ...(content ? { content } : {}),
    ...(id ? { id } : {}),
  };
}

function storeAssistantEncryptedReasoning(
  message: MessageLike,
  scope: string
): number {
  if (message.role !== "assistant") return 0;

  const payload = getInlineEncryptedPayload(message);
  if (!payload) return 0;

  const keys = [`scope:${scope}:signature:${messageSignature(message)}`];
  keys.push(
    ...toolCallIds(message).map(
      (toolCallId) => `scope:${scope}:tool_call:${toolCallId}`
    )
  );
  keys.push(
    ...(message.tool_calls || []).map(
      (toolCall) =>
        `scope:${scope}:tool_call_signature:${toolCallSignature(toolCall)}`
    )
  );

  keys.forEach((key) => encryptedReasoningCache.put(key, payload));
  return keys.length;
}

function lookupEncryptedReasoning(
  message: MessageLike,
  scope: string
): EncryptedReasoningPayload | undefined {
  const bySignature = encryptedReasoningCache.get(
    `scope:${scope}:signature:${messageSignature(message)}`
  );
  if (bySignature) return bySignature;

  for (const toolCallId of toolCallIds(message)) {
    const byId = encryptedReasoningCache.get(
      `scope:${scope}:tool_call:${toolCallId}`
    );
    if (byId) return byId;
  }

  for (const toolCall of message.tool_calls || []) {
    const byToolSignature = encryptedReasoningCache.get(
      `scope:${scope}:tool_call_signature:${toolCallSignature(toolCall)}`
    );
    if (byToolSignature) return byToolSignature;
  }

  return undefined;
}

function applyEncryptedPayload(
  message: MessageLike,
  payload: EncryptedReasoningPayload
): void {
  const existingThinking =
    message.thinking && typeof message.thinking === "object"
      ? (message.thinking as {
          content?: string;
          signature?: string;
          encrypted_content?: string;
          id?: string;
        })
      : undefined;
  const content =
    (typeof existingThinking?.content === "string" && existingThinking.content) ||
    payload.content ||
    "";
  const id = payload.id || existingThinking?.id;
  message.thinking = {
    content,
    encrypted_content: payload.encrypted_content,
    ...(typeof existingThinking?.signature === "string"
      ? { signature: existingThinking.signature }
      : {}),
    ...(id ? { id } : {}),
  };
}

export function isCrossProtocolResponsesClient(
  context?: TransformerContext
): boolean {
  const protocol =
    (context as any)?.clientProtocol ||
    (context as any)?.protocolContext?.protocol ||
    (context as any)?.req?.protocolContext?.protocol ||
    (context as any)?.req?.clientProtocol;
  return protocol === "anthropic_messages" || protocol === "openai_chat_completions";
}

export function buildEncryptedReasoningCacheNamespace(
  request: Pick<UnifiedChatRequest, "model" | "thinking" | "reasoning">,
  provider?: Pick<LLMProvider, "name" | "baseUrl">
): string {
  return hash(
    stableStringify({
      provider: provider?.name || "",
      baseUrl: provider?.baseUrl || "",
      model: request.model || "",
      thinking: request.thinking || null,
      reasoning: request.reasoning || null,
    })
  );
}

/**
 * Anthropic/Chat clients cannot round-trip Responses `encrypted_content`.
 * For those inbound protocols, request ciphertext from the destination and
 * restore it onto assistant tool turns from a local cache keyed like DeepSeek's
 * reasoning replay cache.
 */
export function prepareEncryptedReasoningReplay(
  request: UnifiedChatRequest,
  provider: Pick<LLMProvider, "name" | "baseUrl"> | undefined,
  context?: TransformerContext
): { restoredFromCache: number; includeRequested: boolean } {
  if (!isCrossProtocolResponsesClient(context)) {
    if (context?.req) {
      delete (context.req as any)[CONTEXT_KEY];
    }
    return { restoredFromCache: 0, includeRequested: false };
  }

  const namespace = buildEncryptedReasoningCacheNamespace(request, provider);
  const priorMessages: MessageLike[] = [];
  let restoredFromCache = 0;

  for (const message of request.messages as MessageLike[]) {
    if (message.role === "assistant") {
      const scope = conversationScope(priorMessages, namespace);
      const needsReplay = assistantNeedsEncryptedReasoning(
        message,
        priorMessages
      );

      if (needsReplay) {
        const inline = getInlineEncryptedPayload(message);
        if (!inline) {
          const restored = lookupEncryptedReasoning(message, scope);
          if (restored) {
            applyEncryptedPayload(message, restored);
            restoredFromCache++;
          }
        }
      }

      storeAssistantEncryptedReasoning(message, scope);
    }

    priorMessages.push(message);
  }

  // Always ask Responses destinations for ciphertext when the client cannot
  // carry it. Merge with any existing include list rather than inventing other
  // include values.
  const existing = Array.isArray((request as any).include)
    ? ((request as any).include as unknown[]).filter(
        (entry): entry is string => typeof entry === "string" && entry.length > 0
      )
    : [];
  if (!existing.includes(ENCRYPTED_INCLUDE)) {
    (request as any).include = [...existing, ENCRYPTED_INCLUDE];
  }

  if (context?.req) {
    (context.req as any)[CONTEXT_KEY] = {
      enabled: true,
      namespace,
      requestScope: conversationScope(
        request.messages as MessageLike[],
        namespace
      ),
    } satisfies RequestContextState;
  }

  return { restoredFromCache, includeRequested: true };
}

function getRequestContextState(
  context?: TransformerContext
): RequestContextState | undefined {
  return (context?.req as any)?.[CONTEXT_KEY];
}

export function hasEncryptedReasoningContext(
  context?: TransformerContext
): boolean {
  return getRequestContextState(context)?.enabled === true;
}

export function recordEncryptedReasoningResponseMessage(
  message: MessageLike | null | undefined,
  context?: TransformerContext
): number {
  if (!message) return 0;

  const state = getRequestContextState(context);
  if (!state) return 0;

  return storeAssistantEncryptedReasoning(message, state.requestScope);
}

/**
 * Build a cacheable assistant message from a Responses `output` array.
 * Ciphertext often arrives only on the terminal reasoning item.
 */
export function assistantMessageFromResponsesOutput(
  output: any[] | undefined
): MessageLike | null {
  if (!Array.isArray(output)) return null;

  const reasoningItem = output.find((item) => item?.type === "reasoning");
  const encrypted_content = responsesEncryptedContentFrom(
    reasoningItem?.encrypted_content
  );
  if (!encrypted_content) return null;

  const summaryText = Array.isArray(reasoningItem?.summary)
    ? reasoningItem.summary
        .map((part: any) =>
          typeof part === "string"
            ? part
            : typeof part?.text === "string"
              ? part.text
              : ""
        )
        .filter(Boolean)
        .join("\n")
    : typeof reasoningItem?.content === "string"
      ? reasoningItem.content
      : "";

  const toolCalls = output
    .filter(
      (item) =>
        item?.type === "function_call" || item?.type === "custom_tool_call"
    )
    .map((call) => ({
      id: call.call_id || call.id || "",
      type: "function" as const,
      function: {
        name: call.name || "",
        arguments:
          call.type === "custom_tool_call"
            ? JSON.stringify({ input: call.input || "" })
            : typeof call.arguments === "string"
              ? call.arguments
              : JSON.stringify(call.arguments ?? {}),
      },
    }))
    .filter(
      (toolCall) =>
        typeof toolCall.id === "string" &&
        toolCall.id.length > 0 &&
        typeof toolCall.function.name === "string" &&
        toolCall.function.name.length > 0
    );

  const messageTexts: string[] = [];
  for (const item of output) {
    if (item?.type !== "message" || !Array.isArray(item.content)) continue;
    for (const part of item.content) {
      if (part?.type === "output_text" && typeof part.text === "string") {
        messageTexts.push(part.text);
      }
    }
  }

  return {
    role: "assistant",
    content: messageTexts.join("") || null,
    tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
    thinking: {
      content: summaryText,
      encrypted_content,
      ...(typeof reasoningItem?.id === "string" && reasoningItem.id
        ? { id: reasoningItem.id }
        : {}),
    },
  };
}

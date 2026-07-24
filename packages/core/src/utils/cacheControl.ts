import { createHash } from "crypto";
import {
  UnifiedChatRequest,
  UnifiedMessage,
  MessageContent,
  TextContent,
  UnifiedTool,
} from "../types/llm";

const ANTHROPIC_CACHE_CONTROL = { type: "ephemeral" } as const;

type CacheableLocation =
  | { kind: "message"; messageIndex: number }
  | { kind: "content"; messageIndex: number; contentIndex: number }
  | { kind: "tool"; toolIndex: number };

export interface CacheIntent {
  sessionKey?: string;
  breakpoints: CacheableLocation[];
}

/**
 * Strip cache_control from a single object (shallow clone).
 */
export function stripCacheControl<T extends Record<string, any>>(obj: T): T {
  const clone = { ...obj };
  if ("cache_control" in clone) {
    delete clone.cache_control;
  }
  return clone;
}

/**
 * Strip cache_control from all messages and their content items.
 */
export function stripMessagesCacheControl(
  messages: UnifiedMessage[]
): UnifiedMessage[] {
  return messages.map((msg) => {
    const cloned = { ...msg };

    if (Array.isArray(cloned.content)) {
      cloned.content = (cloned.content as MessageContent[]).map((item) => {
        if ((item as TextContent).cache_control) {
          const { cache_control, ...rest } = item as TextContent;
          return rest as MessageContent;
        }
        return item;
      });
    }

    if ((cloned as any).cache_control) {
      delete (cloned as any).cache_control;
    }
    if (Array.isArray(cloned.tool_calls)) {
      cloned.tool_calls = cloned.tool_calls.map((toolCall: any) => {
        const { cache_control, ...rest } = toolCall;
        return rest;
      });
    }

    return cloned;
  });
}

/**
 * Strip cache_control from tool definitions. Anthropic → Unified preserves
 * cache_control on tools so the Anthropic round-trip keeps prompt-cache hits,
 * but non-Anthropic providers reject it on tool definitions.
 */
export function stripToolsCacheControl(
  tools: UnifiedTool[] | undefined
): UnifiedTool[] | undefined {
  if (!Array.isArray(tools)) return tools;
  return tools.map((tool) => {
    if (!(tool as any).cache_control) return tool;
    const { cache_control, ...rest } = tool as any;
    return rest as UnifiedTool;
  });
}

function hashCacheKey(value: string): string {
  return `ccr_${createHash("sha256").update(value).digest("hex").slice(0, 48)}`;
}

function metadataSessionId(metadataUserId: unknown): string | undefined {
  if (typeof metadataUserId !== "string" || !metadataUserId) {
    return undefined;
  }

  const parts = metadataUserId.split("_session_");
  if (parts.length > 1 && parts[1]) {
    return parts[1];
  }

  try {
    const parsed = JSON.parse(metadataUserId);
    if (parsed && typeof parsed.session_id === "string" && parsed.session_id) {
      return parsed.session_id;
    }
  } catch {
    // Non-JSON metadata.user_id is allowed.
  }

  return metadataUserId;
}

function contentText(content: UnifiedMessage["content"]): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part: any) => {
      if (typeof part === "string") return part;
      if (part?.type === "text" && typeof part.text === "string") {
        return part.text;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

export function deriveCacheSessionKey(
  context: any,
  request: UnifiedChatRequest
): string | undefined {
  const raw =
    context?.req?.sessionId ||
    metadataSessionId((request as any)?.metadata?.user_id);

  if (typeof raw === "string" && raw) {
    return hashCacheKey(raw);
  }

  const firstSystemOrUser = (request.messages || []).find(
    (msg) => msg.role === "system" || msg.role === "user"
  );
  const fallbackText = firstSystemOrUser
    ? contentText(firstSystemOrUser.content)
    : "";
  const fallback = [request.model || "", fallbackText].join("\n");
  return fallback.trim() ? hashCacheKey(fallback) : undefined;
}

function messageHasCacheControl(message: UnifiedMessage): boolean {
  if ((message as any).cache_control) return true;
  if (!Array.isArray(message.content)) return false;
  return message.content.some((part: any) => Boolean(part?.cache_control));
}

function countCacheBreakpoints(
  messages: UnifiedMessage[],
  tools?: UnifiedTool[]
): number {
  const messageCount = messages.reduce((count, message) => {
    const messageMarker = (message as any).cache_control ? 1 : 0;
    const contentMarkers = Array.isArray(message.content)
      ? message.content.reduce(
          (sum: number, part: any) => sum + (part?.cache_control ? 1 : 0),
          0
        )
      : 0;
    return count + messageMarker + contentMarkers;
  }, 0);
  const toolCount = (tools || []).reduce(
    (count, tool: any) => count + (tool?.cache_control ? 1 : 0),
    0
  );
  return messageCount + toolCount;
}

function trimUnifiedCacheBreakpoints(
  messages: UnifiedMessage[],
  tools: UnifiedTool[] | undefined,
  maxBreakpoints: number
): void {
  let seen = 0;
  const keep = (): boolean => {
    seen += 1;
    return seen <= maxBreakpoints;
  };

  for (const message of messages) {
    if ((message as any).cache_control && !keep()) {
      delete (message as any).cache_control;
    }
    if (!Array.isArray(message.content)) continue;
    for (const part of message.content as any[]) {
      if (part?.cache_control && !keep()) {
        delete part.cache_control;
      }
    }
  }

  for (const tool of tools || []) {
    if ((tool as any).cache_control && !keep()) {
      delete (tool as any).cache_control;
    }
  }
}

function hasCacheableText(message: UnifiedMessage): boolean {
  if (typeof message.content === "string") {
    return message.content.length > 0;
  }
  if (!Array.isArray(message.content)) {
    return false;
  }
  return message.content.some(
    (part: any) => part?.type === "text" && typeof part.text === "string" && part.text.length > 0
  );
}

function firstTextContentIndex(message: UnifiedMessage): number | undefined {
  if (!Array.isArray(message.content)) return undefined;
  const index = message.content.findIndex(
    (part: any) => part?.type === "text" && typeof part.text === "string" && part.text.length > 0
  );
  return index >= 0 ? index : undefined;
}

export function selectCacheBreakpoints(
  request: UnifiedChatRequest,
  options: { maxBreakpoints: number; includeTools?: boolean }
): CacheIntent {
  const breakpoints: CacheableLocation[] = [];
  const max = Math.max(0, options.maxBreakpoints);
  if (max === 0) return { breakpoints };

  for (let i = 0; i < request.messages.length && breakpoints.length < max; i += 1) {
    const message = request.messages[i];
    if (message.role !== "system" || messageHasCacheControl(message)) continue;
    const contentIndex = firstTextContentIndex(message);
    breakpoints.push(
      contentIndex === undefined
        ? { kind: "message", messageIndex: i }
        : { kind: "content", messageIndex: i, contentIndex }
    );
  }

  if (
    options.includeTools &&
    Array.isArray(request.tools) &&
    request.tools.length > 0 &&
    breakpoints.length < max
  ) {
    let toolIndex = -1;
    for (let i = request.tools.length - 1; i >= 0; i -= 1) {
      if (!(request.tools[i] as any)?.cache_control) {
        toolIndex = i;
        break;
      }
    }
    if (toolIndex >= 0) {
      breakpoints.push({ kind: "tool", toolIndex });
    }
  }

  for (
    let i = request.messages.length - 1;
    i >= 0 && breakpoints.length < max;
    i -= 1
  ) {
    const message = request.messages[i];
    if (message.role === "system") continue;
    if (messageHasCacheControl(message) || !hasCacheableText(message)) continue;
    const contentIndex = firstTextContentIndex(message);
    breakpoints.push(
      contentIndex === undefined
        ? { kind: "message", messageIndex: i }
        : { kind: "content", messageIndex: i, contentIndex }
    );
  }

  return { breakpoints };
}

function cloneMessages(messages: UnifiedMessage[]): UnifiedMessage[] {
  return messages.map((message) => ({
    ...message,
    content: Array.isArray(message.content)
      ? message.content.map((part: any) => ({ ...part }))
      : message.content,
    tool_calls: Array.isArray(message.tool_calls)
      ? message.tool_calls.map((toolCall) => ({
          ...toolCall,
          function: { ...toolCall.function },
        }))
      : message.tool_calls,
    thinking: message.thinking ? { ...message.thinking } : message.thinking,
  }));
}

function cloneTools(tools?: UnifiedTool[]): UnifiedTool[] | undefined {
  if (!Array.isArray(tools)) return tools;
  return tools.map((tool) => ({
    ...tool,
    function: {
      ...tool.function,
      parameters: { ...tool.function.parameters },
    },
  }));
}

function ensureMessageContentArray(message: UnifiedMessage): MessageContent[] {
  if (Array.isArray(message.content)) {
    return message.content;
  }
  const text = typeof message.content === "string" ? message.content : "";
  const content = [{ type: "text" as const, text }];
  message.content = content;
  return content;
}

function applyCacheLocation(
  messages: UnifiedMessage[],
  tools: UnifiedTool[] | undefined,
  location: CacheableLocation
): void {
  if (location.kind === "tool") {
    if (tools?.[location.toolIndex]) {
      (tools[location.toolIndex] as any).cache_control = ANTHROPIC_CACHE_CONTROL;
    }
    return;
  }

  const message = messages[location.messageIndex];
  if (!message) return;

  if (location.kind === "message") {
    if (message.role === "tool") {
      (message as any).cache_control = ANTHROPIC_CACHE_CONTROL;
      return;
    }
    const content = ensureMessageContentArray(message);
    const index = firstTextContentIndex(message) ?? 0;
    if (content[index]) {
      (content[index] as any).cache_control = ANTHROPIC_CACHE_CONTROL;
    }
    return;
  }

  const content = ensureMessageContentArray(message);
  if (content[location.contentIndex]) {
    (content[location.contentIndex] as any).cache_control =
      ANTHROPIC_CACHE_CONTROL;
  }
}

export function applyAnthropicPromptCaching(
  request: UnifiedChatRequest,
  options: { maxBreakpoints?: number; includeTools?: boolean } = {}
): UnifiedChatRequest {
  const maxBreakpoints = options.maxBreakpoints ?? 4;
  const messages = cloneMessages(request.messages || []);
  const tools = cloneTools(request.tools);
  trimUnifiedCacheBreakpoints(messages, tools, maxBreakpoints);
  const existing = countCacheBreakpoints(messages, tools);
  const remaining = Math.max(0, maxBreakpoints - existing);

  if (remaining > 0) {
    const intent = selectCacheBreakpoints(
      { ...request, messages, tools },
      { maxBreakpoints: remaining, includeTools: options.includeTools ?? true }
    );
    for (const location of intent.breakpoints) {
      applyCacheLocation(messages, tools, location);
    }
  }

  return {
    ...request,
    messages,
    ...(tools ? { tools } : {}),
  };
}

export function applyRawAnthropicPromptCaching<T extends Record<string, any>>(
  request: T,
  options: { maxBreakpoints?: number } = {}
): T {
  const maxBreakpoints = options.maxBreakpoints ?? 4;
  const clone: any = JSON.parse(JSON.stringify(request));
  const explicitMarkers: any[] = [];
  const collectMarker = (value: any): void => {
    if (value && typeof value === "object" && value.cache_control) {
      explicitMarkers.push(value);
    }
  };

  (Array.isArray(clone.tools) ? clone.tools : []).forEach(collectMarker);
  if (Array.isArray(clone.system)) {
    clone.system.forEach(collectMarker);
  } else {
    collectMarker(clone.system);
  }
  (Array.isArray(clone.messages) ? clone.messages : []).forEach(
    (message: any) => {
      collectMarker(message);
      if (Array.isArray(message?.content)) {
        message.content.forEach(collectMarker);
      }
    }
  );

  const hasAutomatic = Boolean(clone.cache_control);
  const explicitLimit = Math.max(0, maxBreakpoints - (hasAutomatic ? 1 : 0));
  if (explicitMarkers.length > explicitLimit) {
    for (const marker of explicitMarkers.slice(
      0,
      explicitMarkers.length - explicitLimit
    )) {
      delete marker.cache_control;
    }
  }

  const remainingExplicit = Math.min(explicitMarkers.length, explicitLimit);
  if (!hasAutomatic && remainingExplicit < maxBreakpoints) {
    const latestControl =
      explicitMarkers[explicitMarkers.length - 1]?.cache_control;
    clone.cache_control = latestControl?.ttl
      ? { type: "ephemeral", ttl: latestControl.ttl }
      : ANTHROPIC_CACHE_CONTROL;
  }

  return clone as T;
}

export function applyQwenPromptCaching(
  request: UnifiedChatRequest
): UnifiedChatRequest {
  // stripMessagesCacheControl already clones each message (shallow) and
  // clones content items carrying cache_control.  For items without
  // cache_control the reference is shared with the original, so we create a
  // new object when mutating instead of deep-cloning everything upfront.
  const messages = stripMessagesCacheControl(request.messages || []);
  const tools = stripToolsCacheControl(request.tools);

  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (typeof message.content === "string" && message.content) {
      message.content = [
        {
          type: "text",
          text: message.content,
          cache_control: ANTHROPIC_CACHE_CONTROL,
        },
      ];
      break;
    }
    if (!Array.isArray(message.content)) continue;
    for (let j = message.content.length - 1; j >= 0; j -= 1) {
      const part = message.content[j] as any;
      if (part?.type === "text" && part.text) {
        // Create a new object so the original request is not mutated.
        message.content[j] = { ...part, cache_control: ANTHROPIC_CACHE_CONTROL };
        return { ...request, messages, ...(tools ? { tools } : {}) };
      }
    }
  }

  return { ...request, messages, ...(tools ? { tools } : {}) };
}

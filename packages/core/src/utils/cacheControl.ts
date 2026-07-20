import { UnifiedMessage, MessageContent, TextContent, UnifiedTool } from "../types/llm";

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

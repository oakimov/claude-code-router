import { UnifiedMessage, MessageContent, TextContent } from "../types/llm";

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

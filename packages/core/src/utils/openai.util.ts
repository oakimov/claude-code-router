import { UnifiedChatRequest, UnifiedMessage } from "../types/llm";
import {
  applyQwenPromptCaching,
  deriveCacheSessionKey,
  stripMessagesCacheControl,
  stripToolsCacheControl,
} from "./cacheControl";

/**
 * Validates OpenAI format messages to ensure complete tool_calls/tool message pairing.
 * Requires tool messages to immediately follow assistant messages with tool_calls.
 * Enforces strict immediate following sequence between tool_calls and tool messages.
 */
export function validateOpenAIToolCalls(messages: any[]): any[] {
  const validatedMessages: any[] = [];

  for (let i = 0; i < messages.length; i++) {
    const currentMessage = { ...messages[i] };

    // Process assistant messages with tool_calls
    if (currentMessage.role === "assistant" && currentMessage.tool_calls) {
      const validToolCalls: any[] = [];
      const removedToolCallIds: string[] = [];

      // Collect all immediately following tool messages
      const immediateToolMessages: any[] = [];
      let j = i + 1;
      while (j < messages.length && messages[j].role === "tool") {
        immediateToolMessages.push(messages[j]);
        j++;
      }

      // For each tool_call, check if there's an immediately following tool message
      currentMessage.tool_calls.forEach((toolCall: any) => {
        const hasImmediateToolMessage = immediateToolMessages.some(toolMsg =>
          toolMsg.tool_call_id === toolCall.id
        );

        if (hasImmediateToolMessage) {
          validToolCalls.push(toolCall);
        } else {
          removedToolCallIds.push(toolCall.id);
        }
      });

      // Update the assistant message
      if (validToolCalls.length > 0) {
        currentMessage.tool_calls = validToolCalls;
      } else {
        delete currentMessage.tool_calls;
      }

      // Only include message if it has content or valid tool_calls
      if (currentMessage.content || currentMessage.tool_calls) {
        validatedMessages.push(currentMessage);
      }
    }

    // Process tool messages
    else if (currentMessage.role === "tool") {
      let hasImmediateToolCall = false;

      // Check if the immediately preceding assistant message has matching tool_call
      if (i > 0) {
        const prevMessage = messages[i - 1];
        if (prevMessage.role === "assistant" && prevMessage.tool_calls) {
          hasImmediateToolCall = prevMessage.tool_calls.some((toolCall: any) =>
            toolCall.id === currentMessage.tool_call_id
          );
        } else if (prevMessage.role === "tool") {
          // Check for assistant message before the sequence of tool messages
          for (let k = i - 1; k >= 0; k--) {
            if (messages[k].role === "tool") continue;
            if (messages[k].role === "assistant" && messages[k].tool_calls) {
              hasImmediateToolCall = messages[k].tool_calls.some((toolCall: any) =>
                toolCall.id === currentMessage.tool_call_id
              );
            }
            break;
          }
        }
      }

      if (hasImmediateToolCall) {
        validatedMessages.push(currentMessage);
      }
    }

    // For all other message types, include as-is
    else {
      validatedMessages.push(currentMessage);
    }
  }

  return validatedMessages;
}

/**
 * Injects prompt caching hints into messages for Anthropic models.
 * Adds cache_control: { type: "ephemeral" } to system messages when model is Claude.
 */
export function injectPromptCaching(messages: any[], model: string): any[] {
  return messages.map((msg) => {
    // Add cache_control to system messages for Claude models
    if (msg.role === "system" && model.includes("claude")) {
      if (Array.isArray(msg.content)) {
        return {
          ...msg,
          content: msg.content.map((item: any) => ({
            ...item,
            cache_control: { type: "ephemeral" } as const,
          })),
        };
      }
    }
    return msg;
  });
}

export function supportsOpenAIExplicitPromptCache(model: string): boolean {
  const normalized = (model || "").toLowerCase();
  const match = normalized.match(/(?:^|\/)gpt-(\d+)(?:\.(\d+))?/);
  if (!match) return false;
  const major = Number(match[1]);
  const minor = Number(match[2] || 0);
  return major > 5 || (major === 5 && minor >= 6);
}

function isOpenAICacheableChatContent(content: any): boolean {
  return (
    content?.type === "text" ||
    content?.type === "image_url" ||
    content?.type === "input_audio" ||
    content?.type === "file" ||
    content?.type === "refusal"
  );
}

function lastCacheableContentIndex(content: any[]): number {
  for (let i = content.length - 1; i >= 0; i -= 1) {
    if (isOpenAICacheableChatContent(content[i])) return i;
  }
  return -1;
}

export function applyOpenAIChatCaching(
  request: UnifiedChatRequest,
  _provider?: any,
  context?: any
): UnifiedChatRequest {
  const next: UnifiedChatRequest = {
    ...request,
    messages: (request.messages || []).map((message) => ({
      ...message,
      content: Array.isArray(message.content)
        ? message.content.map((part: any) => ({ ...part }))
        : message.content,
    })),
    tools: Array.isArray(request.tools)
      ? request.tools.map((tool: any) =>
          tool?.function
            ? {
                ...tool,
                function: {
                  ...tool.function,
                  parameters: { ...tool.function.parameters },
                },
              }
            : { ...tool }
        )
      : request.tools,
  };

  const cacheKey = deriveCacheSessionKey(context, request);
  if (cacheKey && !(next as any).prompt_cache_key) {
    (next as any).prompt_cache_key = cacheKey;
  }

  if (supportsOpenAIExplicitPromptCache(next.model)) {
    const candidates: Array<{ messageIndex: number; contentIndex?: number }> = [];
    next.messages.forEach((message: UnifiedMessage, messageIndex: number) => {
      if ((message as any).cache_control) {
        if (typeof message.content === "string" && message.content) {
          candidates.push({ messageIndex });
        } else if (Array.isArray(message.content)) {
          const contentIndex = lastCacheableContentIndex(message.content);
          if (contentIndex >= 0) candidates.push({ messageIndex, contentIndex });
        }
      }
      if (Array.isArray(message.content)) {
        message.content.forEach((part: any, contentIndex: number) => {
          if (part?.cache_control && isOpenAICacheableChatContent(part)) {
            candidates.push({ messageIndex, contentIndex });
          }
        });
      }
    });

    for (const candidate of candidates.slice(-3)) {
      const message = next.messages[candidate.messageIndex];
      if (typeof message.content === "string") {
        message.content = [
          {
            type: "text",
            text: message.content,
            prompt_cache_breakpoint: { mode: "explicit" },
          } as any,
        ];
      } else if (
        Array.isArray(message.content) &&
        candidate.contentIndex !== undefined
      ) {
        (message.content[candidate.contentIndex] as any).prompt_cache_breakpoint =
          { mode: "explicit" };
      }
    }
  }

  next.messages = stripMessagesCacheControl(next.messages);
  next.tools = stripToolsCacheControl(next.tools);
  return next;
}

export function openAIContentCacheBreakpoint(content: any, model: string): any {
  if (!supportsOpenAIExplicitPromptCache(model)) {
    return {};
  }
  return content?.prompt_cache_breakpoint
    ? { prompt_cache_breakpoint: content.prompt_cache_breakpoint }
    : {};
}

function providerName(provider?: any): string {
  return String(provider?.name || "").trim().toLowerCase();
}

function providerHost(provider?: any): string {
  const baseUrl = provider?.baseUrl || provider?.api_base_url || "";
  try {
    return new URL(baseUrl).hostname.toLowerCase();
  } catch {
    return String(baseUrl).toLowerCase();
  }
}

export function applyRequestCacheKey(
  request: UnifiedChatRequest,
  context?: any
): UnifiedChatRequest {
  const next = {
    ...request,
    messages: stripMessagesCacheControl(request.messages || []),
    tools: stripToolsCacheControl(request.tools),
  };
  const cacheKey = deriveCacheSessionKey(context, request);
  if (cacheKey) (next as any).prompt_cache_key = cacheKey;
  return next;
}

export function applyProviderNativeChatCaching(
  request: UnifiedChatRequest,
  provider?: any,
  context?: any
): UnifiedChatRequest {
  const name = providerName(provider);
  const host = providerHost(provider);

  if (
    name === "openai" ||
    name === "azure-openai" ||
    host === "api.openai.com" ||
    host.endsWith(".openai.azure.com")
  ) {
    return applyOpenAIChatCaching(request, provider, context);
  }

  if (
    name === "qwen" ||
    name === "qwen-auth" ||
    host.includes("dashscope") ||
    host.endsWith(".aliyuncs.com") ||
    host === "qwen.aikit.club"
  ) {
    return applyQwenPromptCaching(request);
  }

  if (
    name === "mistral" ||
    name === "cerebras" ||
    host === "api.mistral.ai" ||
    host === "api.cerebras.ai"
  ) {
    return applyRequestCacheKey(request, context);
  }

  // Opencode Zen – mirrors native ProviderTransform.options() session affinity:
  // native sets promptCacheKey = sessionID for every opencode model (and gpt-5
  // via opencode). We inject prompt_cache_key in the Unified body so Zen's
  // downstream OpenAI/Moonshot cache + sticky routing stays hot across turns.
  if (
    name === "opencode" ||
    name.startsWith("opencode") ||
    host === "opencode.ai" ||
    host.endsWith(".opencode.ai")
  ) {
    return applyRequestCacheKey(request, context);
  }

  return {
    ...request,
    messages: stripMessagesCacheControl(request.messages || []),
    tools: stripToolsCacheControl(request.tools),
  };
}

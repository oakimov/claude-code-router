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

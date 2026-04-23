import { UnifiedChatRequest, UnifiedMessage, MessageContent, TextContent } from "../types/llm";

/**
 * Helper to flatten array content to strings and remove cache_control
 */
function transformMessage(msg: UnifiedMessage): UnifiedMessage {
  const clonedMsg = { ...msg };
  if (Array.isArray(clonedMsg.content)) {
    const contentArray = clonedMsg.content as MessageContent[];

    const hasImages = contentArray.some(
      (part) => part.type === "image_url"
    );

    if (hasImages) {
      clonedMsg.content = contentArray.map((part) => {
        if (part.type === "text") {
          const { cache_control, ...rest } = part as TextContent;
          return rest;
        }
        return part;
      });
    } else {
      const textParts = contentArray
        .filter((part): part is TextContent => part.type === "text")
        .map((part) => part.text)
        .filter((text) => text && text.length > 0);

      clonedMsg.content = textParts.join("\n");
    }
  }

  if ((clonedMsg as any).cache_control) {
    delete (clonedMsg as any).cache_control;
  }

  return clonedMsg;
}

/**
 * Helper to transform tool_choice to Mistral-compatible format
 */
function transformToolChoice(toolChoice: UnifiedChatRequest["tool_choice"]): any {
  if (toolChoice === "auto" || toolChoice === "none") {
    return toolChoice;
  }

  if (toolChoice === "required") {
    return "any";
  }

  if (typeof toolChoice === "object" && toolChoice.function?.name) {
    return {
      type: "function",
      function: { name: toolChoice.function.name },
    };
  }

  return toolChoice;
}

/**
 * Helper to transform reasoning parameter to Mistral's reasoning_effort format
 */
function transformReasoning(reasoning: any): string | undefined {
  if (reasoning.effort) {
    const effort = reasoning.effort.toLowerCase();
    if (effort === "low" || effort === "medium" || effort === "high") {
      return effort;
    }
  }

  if (reasoning.max_tokens) {
    const tokens = reasoning.max_tokens;
    if (tokens < 1000) return "low";
    if (tokens < 5000) return "medium";
    return "high";
  }

  return "medium";
}

/**
 * Transform incoming request to Mistral-compatible format
 */
export function buildRequestBody(request: UnifiedChatRequest): Record<string, any> {
  // Create a shallow copy to avoid mutating the original request
  const req = { ...request };

  // 1. Process messages
  if (Array.isArray(req.messages)) {
    req.messages = req.messages.map((msg) => transformMessage(msg));
  }

  // 2. Defaults
  if (req.stream === undefined) {
    req.stream = true;
  }

  // 3. Tool Choice
  if (req.tool_choice) {
    req.tool_choice = transformToolChoice(req.tool_choice);
  }

  // 4. Tool Cleanup
  if (Array.isArray(req.tools)) {
    req.tools = req.tools.map((tool) => {
      if (tool?.function?.parameters?.$schema) {
        const params = { ...tool.function.parameters };
        delete params.$schema;
        return {
          ...tool,
          function: {
            ...tool.function,
            parameters: params,
          },
        };
      }
      return tool;
    });
  }

  // 5. Reasoning conversion
  if (req.reasoning && req.model) {
    const modelId = req.model;
    const supportsReasoning =
      modelId.startsWith("mistral-small-") ||
      modelId.startsWith("magistral-") ||
      modelId.startsWith("mistral-medium-") ||
      modelId === "mistral-vibe-cli-fast" ||
      modelId.startsWith("labs-leanstral-");

    if (supportsReasoning) {
      req.reasoning_effort = transformReasoning(req.reasoning);
    }
    delete req.reasoning;
  }

  return req;
}

/**
 * Transform a Mistral provider request back into a UnifiedChatRequest
 */
export async function transformRequestOut(request: any): Promise<UnifiedChatRequest> {
  // Mistral is OpenAI-compatible, so the request is already in a unified-like format.
  // We return it as a UnifiedChatRequest.
  return request as UnifiedChatRequest;
}

/**
 * Transform response back — convert Mistral's content-array thinking format
 * to the delta.thinking / delta.content shape expected by @musistudio/llms.
 */
export async function transformResponseOut(
  response: Response,
  providerName: string,
  logger?: any
): Promise<Response> {
  const contentType = response.headers.get("Content-Type") ?? "";

  if (contentType.includes("application/json")) {
    const jsonResponse = await response.json();
    logger?.debug({ response: jsonResponse }, `${providerName} response:`);

    const choice = jsonResponse.choices?.[0];
    if (choice?.message) {
      const message = choice.message;
      let thinkingText = "";
      let plainText = "";

      if (message.reasoning_content) {
        const rc = message.reasoning_content;
        thinkingText += typeof rc === "string" ? rc : (typeof rc?.text === "string" ? rc.text : JSON.stringify(rc));
        delete message.reasoning_content;
      }

      if (Array.isArray(message.content)) {
        for (const block of message.content) {
          if (block.type === "thinking") {
            const parts = Array.isArray(block.thinking) ? block.thinking : [block.thinking];
            thinkingText += parts.map((p: any) => {
              if (typeof p === "string") return p;
              if (p && typeof p.text === "string") return p.text;
              return JSON.stringify(p);
            }).join("");
          } else if (block.type === "text") {
            plainText += typeof block.text === "string" ? block.text : JSON.stringify(block.text ?? "");
          }
        }
        message.content = plainText;
      }

      if (thinkingText) {
        jsonResponse.thinking = { content: thinkingText };
      }
    }

    return new Response(JSON.stringify(jsonResponse), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  } else if (contentType.includes("stream")) {
    if (!response.body) return response;

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let buffer = "";

    return new Response(new ReadableStream({
      async start(controller) {
        const reader = response.body!.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              if (buffer.trim()) controller.enqueue(encoder.encode(buffer));
              break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() ?? "";

            for (const line of lines) {
              if (!line.trim()) {
                controller.enqueue(encoder.encode(line + "\n"));
                continue;
              }
              if (!line.startsWith("data:") || line.trim() === "data: [DONE]") {
                controller.enqueue(encoder.encode(line + "\n"));
                continue;
              }

              try {
                const rawDataStr = line.slice(5).trim();
                const data = JSON.parse(rawDataStr);

                const choice = data.choices?.[0];
                if (choice?.delta) {
                  const delta = choice.delta;
                  let thinkingText = "";
                  const deltaContent = delta.content;

                  if (delta.reasoning_content) {
                    const rc = delta.reasoning_content;
                    thinkingText += typeof rc === "string" ? rc : (typeof rc?.text === "string" ? rc.text : JSON.stringify(rc));
                    delete delta.reasoning_content;
                  }

                  if (Array.isArray(deltaContent)) {
                    for (const block of deltaContent) {
                      if (block.type === "thinking") {
                        const parts = Array.isArray(block.thinking) ? block.thinking : [block.thinking];
                        thinkingText += parts.map((p: any) => {
                          if (typeof p === "string") return p;
                          if (p && typeof p.text === "string") return p.text;
                          return JSON.stringify(p);
                        }).join("");
                      }
                    }
                    const plainText = deltaContent
                      .filter((b: any) => b.type === "text")
                      .map((b: any) => typeof b.text === "string" ? b.text : JSON.stringify(b.text ?? ""))
                      .join("");

                    delete delta.content;
                    if (thinkingText) delta.thinking = { content: thinkingText };
                    if (plainText) delta.content = plainText;
                  }

                  controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n`));
                } else {
                  controller.enqueue(encoder.encode(line + "\n"));
                }
              } catch {
                controller.enqueue(encoder.encode(line + "\n"));
              }
            }
          }
        } catch (error) {
          controller.error(error);
        } finally {
          try { reader.releaseLock(); } catch { }
          controller.close();
        }
      }
    }), {
      status: response.status,
      statusText: response.statusText,
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" },
    });
  }

  return response;
}

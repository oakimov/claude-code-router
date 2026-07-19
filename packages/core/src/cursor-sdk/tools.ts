import { randomUUID } from "crypto";
import type { SDKCustomTool, SDKCustomToolResult, SDKJsonValue } from "@cursor/sdk";
import type { UnifiedChatRequest } from "@/types/llm";
import type { CursorSdkSession } from "./session";

function normalizeInputSchema(parameters: unknown): Record<string, SDKJsonValue> {
  if (!parameters) {
    return { type: "object", properties: {} };
  }
  if (typeof parameters === "string") {
    try {
      const parsed = JSON.parse(parameters);
      if (parsed && typeof parsed === "object") {
        return parsed as Record<string, SDKJsonValue>;
      }
    } catch {
      return { type: "object", properties: {} };
    }
  }
  if (typeof parameters === "object") {
    return parameters as Record<string, SDKJsonValue>;
  }
  return { type: "object", properties: {} };
}

export function toCustomTools(
  request: UnifiedChatRequest,
  session: CursorSdkSession
): Record<string, SDKCustomTool> {
  const tools: Record<string, SDKCustomTool> = {};
  if (!Array.isArray(request.tools)) return tools;

  for (const tool of request.tools) {
    const name = tool.function?.name;
    if (!name || typeof name !== "string") continue;

    tools[name] = {
      description: tool.function.description || `Claude Code host tool: ${name}`,
      inputSchema: normalizeInputSchema(tool.function.parameters),
      async execute(args, context) {
        const toolCallId = context.toolCallId || randomUUID();
        session.metrics.customToolCalls += 1;
        return session.parkHostTool({
          id: toolCallId,
          name,
          args: (args || {}) as Record<string, unknown>,
        }) as Promise<SDKCustomToolResult>;
      },
    };
  }

  return tools;
}

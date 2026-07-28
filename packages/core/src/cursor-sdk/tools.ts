import { randomUUID } from "crypto";
import type { SDKCustomTool, SDKCustomToolResult, SDKJsonValue } from "@cursor/sdk";
import type { UnifiedChatRequest } from "@/types/llm";
import { EMPTY_HOST_ENVIRONMENT } from "./host-env";
import { sanitizeToolCallId } from "@/utils/toolCallId";
import type { CursorSdkSession } from "./session";
import {
  buildScratchPathCorrection,
  findScratchPaths,
  scratchDetectionApplies,
} from "./tool-paths";

/**
 * Cap on corrective results per session. Beyond this the call is forwarded
 * unchanged: a model that keeps insisting may be acting on an explicit user
 * request about that path, and the host is the right place to fail.
 */
const MAX_SCRATCH_CORRECTIONS = 3;

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

/**
 * Per-request tally. Session metrics are cumulative and the SDK may invoke
 * `execute` as soon as `agent.send` resolves — before the response stream is
 * even constructed — so a start/end delta on the session counter silently
 * misses the very first violation of a turn.
 */
export type TurnToolMetrics = {
  scratchViolations: number;
  scratchCorrections: number;
};

export function createTurnToolMetrics(): TurnToolMetrics {
  return { scratchViolations: 0, scratchCorrections: 0 };
}

export function toCustomTools(
  request: UnifiedChatRequest,
  session: CursorSdkSession,
  logger?: any,
  turn: TurnToolMetrics = createTurnToolMetrics()
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
        // The SDK hands back two ids joined by a newline (`call-…-3\nfc_…_0`).
        // Anthropic rejects the whole request if that reaches a tool_use.id,
        // and the bad id would persist in the transcript for every later turn.
        const toolCallId =
          sanitizeToolCallId(context.toolCallId) || randomUUID();
        const toolArgs = (args || {}) as Record<string, unknown>;
        const hostEnv = session.hostEnv || EMPTY_HOST_ENVIRONMENT;

        // The model aimed a host tool at its own sandbox — the observable
        // symptom of it treating the scratch workspace as the user's project.
        const hits = scratchDetectionApplies(hostEnv)
          ? findScratchPaths(toolArgs, session.workspaceDir)
          : [];
        if (hits.length) {
          session.metrics.scratchPathViolations += 1;
          turn.scratchViolations += 1;
          const correct =
            session.metrics.scratchPathCorrections < MAX_SCRATCH_CORRECTIONS;
          logger?.warn?.(
            {
              sessionKey: session.key,
              tool: name,
              hits,
              corrected: correct,
              violations: session.metrics.scratchPathViolations,
            },
            "cursor-sdk host tool call referenced the scratch workspace"
          );
          if (correct) {
            session.metrics.scratchPathCorrections += 1;
            turn.scratchCorrections += 1;
            return buildScratchPathCorrection(
              hits,
              name,
              session.workspaceDir,
              hostEnv
            ) as SDKCustomToolResult;
          }
        }

        session.metrics.customToolCalls += 1;
        return session.parkHostTool({
          id: toolCallId,
          name,
          args: toolArgs,
        }) as Promise<SDKCustomToolResult>;
      },
    };
  }

  return tools;
}

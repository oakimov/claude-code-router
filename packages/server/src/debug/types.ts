export type DebugTarget = "ccr" | "direct";

export type InboundProtocol = "chat_completions" | "messages" | "responses";

/** Canonical CCR reasoning effort tokens (`ThinkLevel` in `@caeliq/llms`). */
export const REASONING_EFFORTS = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "ultra",
] as const;

export type ReasoningEffort = (typeof REASONING_EFFORTS)[number];

export interface TokenUsage {
  input?: number;
  output?: number;
  total?: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
}

export interface CapturedLlmExchange {
  url: string;
  method: string;
  requestHeaders: Record<string, string>;
  requestBody: unknown;
  status: number;
  responseHeaders: Record<string, string>;
  responseBody: string;
  streaming: boolean;
  usage?: TokenUsage;
}

export function errorExchangeFromMessage(
  message: string,
  last?: CapturedLlmExchange
): CapturedLlmExchange {
  if (last && (last.status > 0 || last.responseBody)) return last;
  return {
    url: last?.url || "",
    method: last?.method || "POST",
    requestHeaders: last?.requestHeaders || {},
    requestBody: last?.requestBody,
    status: last?.status || 0,
    responseHeaders: last?.responseHeaders || {},
    responseBody:
      last?.responseBody || JSON.stringify({ error: message }, null, 2),
    streaming: last?.streaming ?? false,
  };
}

export interface OpenAiFunctionToolSpec {
  id: string;
  description: string;
  parameters: Record<string, unknown>;
}

export interface DebugChatInput {
  messages: unknown[];
  target: DebugTarget;
  protocol: InboundProtocol;
  provider: string;
  model: string;
  system: string;
  tools: unknown;
  stream: boolean;
  reasoningEffort?: ReasoningEffort;
  headers?: Record<string, string>;
}

export interface DebugModelConfig {
  url: string;
  id: string;
  apiKey: string;
  headers: Record<string, string>;
  authKind?:
    | "claude-auth"
    | "codex"
    | "qwen-auth"
    | "antigravity-auth"
    | "xai-auth";
  hint?: string;
}

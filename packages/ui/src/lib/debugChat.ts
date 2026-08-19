import type { UIMessage } from "ai";
import type { Provider } from "@/types";
import { getAuthHintKey } from "@/lib/providerMeta";

export type DebugTarget = "ccr" | "direct";
export type InboundProtocol = "chat_completions" | "messages" | "responses";

/** Matches `DEFAULT_DEBUG_INSTRUCTIONS` in `packages/server/src/debug/model.ts`. */
export const DEFAULT_DEBUG_SYSTEM_PROMPT =
  "You are the CCR Debug agent. Answer briefly so the request and response exchange is easy to inspect.";

const SYSTEM_SESSION_KEY = "ccr.debug.system";
const CHAT_SESSION_KEY = "ccr.debug.chat";

export type DebugChatSession = {
  messages: UIMessage[];
  usageByMessage: Record<string, unknown>;
};

export function loadDebugSystem(): string {
  try {
    const raw = sessionStorage.getItem(SYSTEM_SESSION_KEY);
    if (raw != null) return raw;
  } catch {
    // Ignore quota / private-mode failures.
  }
  return DEFAULT_DEBUG_SYSTEM_PROMPT;
}

export function saveDebugSystem(value: string): void {
  try {
    sessionStorage.setItem(SYSTEM_SESSION_KEY, value);
  } catch {
    // Ignore quota / private-mode failures.
  }
}

function isRestoredMessage(value: unknown): value is UIMessage {
  if (!value || typeof value !== "object") return false;
  const rec = value as Record<string, unknown>;
  return typeof rec.id === "string" && typeof rec.role === "string" && Array.isArray(rec.parts);
}

export function loadDebugChat(): DebugChatSession {
  try {
    const raw = sessionStorage.getItem(CHAT_SESSION_KEY);
    if (!raw) return { messages: [], usageByMessage: {} };
    const parsed = JSON.parse(raw);
    const messages = Array.isArray(parsed?.messages)
      ? parsed.messages.filter(isRestoredMessage)
      : [];
    const usageByMessage =
      parsed?.usageByMessage && typeof parsed.usageByMessage === "object"
        ? parsed.usageByMessage
        : {};
    return { messages, usageByMessage };
  } catch {
    return { messages: [], usageByMessage: {} };
  }
}

export function saveDebugChat(session: DebugChatSession): void {
  try {
    sessionStorage.setItem(CHAT_SESSION_KEY, JSON.stringify(session));
  } catch {
    // Ignore quota / private-mode failures.
  }
}

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

export function parseReasoningEffort(value: unknown): ReasoningEffort | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.trim().toLowerCase();
  return (REASONING_EFFORTS as readonly string[]).includes(normalized)
    ? (normalized as ReasoningEffort)
    : undefined;
}

export function applyReasoningEffortToBody(
  body: Record<string, unknown>,
  protocol: InboundProtocol,
  effort: ReasoningEffort | undefined
): Record<string, unknown> {
  if (!effort) return body;
  if (protocol === "messages") {
    if (effort === "none") {
      body.thinking = { type: "disabled" };
      if (body.output_config && typeof body.output_config === "object") {
        const next = { ...(body.output_config as Record<string, unknown>) };
        delete next.effort;
        if (Object.keys(next).length) body.output_config = next;
        else delete body.output_config;
      }
    } else {
      body.thinking = { type: "adaptive" };
      body.output_config = {
        ...((body.output_config && typeof body.output_config === "object"
          ? body.output_config
          : {}) as Record<string, unknown>),
        effort,
      };
    }
    return body;
  }
  if (protocol === "responses") {
    body.reasoning = {
      ...((body.reasoning && typeof body.reasoning === "object"
        ? body.reasoning
        : {}) as Record<string, unknown>),
      effort,
    };
    return body;
  }
  body.reasoning_effort = effort;
  return body;
}

export type HeaderRow = {
  id: string;
  enabled: boolean;
  key: string;
  value: string;
};

export type CapturedExchange = {
  url: string;
  method: string;
  requestHeaders: Record<string, string>;
  requestBody: unknown;
  status: number;
  responseHeaders: Record<string, string>;
  responseBody: string;
  streaming: boolean;
  usage?: {
    input?: number;
    output?: number;
    total?: number;
    cacheRead?: number;
    cacheWrite?: number;
    reasoning?: number;
  };
};

export function prettyJsonOrText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return value;
    try {
      return JSON.stringify(JSON.parse(trimmed), null, 2);
    } catch {
      return value;
    }
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export type ParsedDebugResponseError = {
  status: number;
  code?: string;
  message: string;
};

function parseNestedProviderDetail(message: string): {
  status?: number;
  detail?: string;
} {
  const idx = message.indexOf("): ");
  if (idx < 0) return {};
  let tail = message.slice(idx + 3).trim();
  if (!tail.startsWith("{")) return {};
  // Some upstream errors embed JSON with backslash-escaped quotes in the message string.
  if (tail.includes('\\"')) {
    tail = tail.replace(/\\"/g, '"');
  }
  try {
    const nested = JSON.parse(tail) as Record<string, unknown>;
    const detail =
      typeof nested.detail === "string"
        ? nested.detail
        : typeof nested.title === "string"
          ? nested.title
          : typeof nested.message === "string"
            ? nested.message
            : undefined;
    return {
      status: typeof nested.status === "number" ? nested.status : undefined,
      detail,
    };
  } catch {
    return {};
  }
}

function enrichFromProviderMessage(
  message: string,
  httpStatus: number,
  code?: string,
  explicitStatus?: number
): ParsedDebugResponseError {
  let status = explicitStatus ?? httpStatus;
  const providerStatusMatch = message.match(/:\s*(\d{3})\)\s*:/);
  if (providerStatusMatch) {
    status = Number(providerStatusMatch[1]);
  }

  const nested = parseNestedProviderDetail(message);
  if (nested.status != null && !providerStatusMatch) status = nested.status;

  const summary =
    nested.detail ||
    (providerStatusMatch && message.includes("): ")
      ? message.slice(message.indexOf("): ") + 3).trim()
      : "") ||
    message ||
    "Request failed";

  return {
    status: status || httpStatus || 0,
    code,
    message: summary,
  };
}

/** Extract HTTP status, error code, and human-readable message from any debug response body. */
export function parseDebugResponseError(
  body: string,
  httpStatus = 0
): ParsedDebugResponseError | null {
  const trimmed = body.trim();
  if (!trimmed) {
    return httpStatus >= 400 ? { status: httpStatus, message: `HTTP ${httpStatus}` } : null;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return httpStatus >= 400 || httpStatus === 0
      ? { status: httpStatus, message: trimmed }
      : null;
  }

  if (!parsed || typeof parsed !== "object") {
    return httpStatus >= 400 ? { status: httpStatus, message: trimmed } : null;
  }

  const rec = parsed as Record<string, unknown>;

  if (typeof rec.error === "string" && rec.error.trim()) {
    return enrichFromProviderMessage(rec.error.trim(), httpStatus);
  }

  const errObj =
    rec.error && typeof rec.error === "object"
      ? (rec.error as Record<string, unknown>)
      : null;

  if (errObj) {
    const message = typeof errObj.message === "string" ? errObj.message : "";
    const code =
      typeof errObj.code === "string"
        ? errObj.code
        : typeof errObj.type === "string"
          ? errObj.type
          : undefined;
    if (!message && !code && httpStatus < 400) return null;
    const explicitStatus = typeof errObj.status === "number" ? errObj.status : undefined;
    return enrichFromProviderMessage(message || trimmed, httpStatus, code, explicitStatus);
  }

  if (rec.type === "error" && typeof rec.message === "string" && rec.message) {
    return enrichFromProviderMessage(
      rec.message,
      httpStatus,
      typeof rec.code === "string" ? rec.code : undefined
    );
  }

  if (httpStatus >= 400 && typeof rec.message === "string") {
    return {
      status: httpStatus,
      code: typeof rec.code === "string" ? rec.code : undefined,
      message: rec.message,
    };
  }

  return httpStatus >= 400 ? { status: httpStatus, message: trimmed } : null;
}

export function isDebugResponseError(body: string, httpStatus = 0): boolean {
  if (httpStatus >= 400) return true;
  return parseDebugResponseError(body, httpStatus) != null;
}

/** Fallback exchange when useChat errors and no captured upstream body arrived. */
export function exchangeFromChatError(message: string): CapturedExchange {
  return {
    url: "",
    method: "POST",
    requestHeaders: {},
    requestBody: undefined,
    status: 0,
    responseHeaders: {},
    responseBody: prettyJsonOrText({ error: message }),
    streaming: false,
  };
}

export function formatDebugResponseNotice(error: ParsedDebugResponseError): string {
  const code = error.code ? ` · ${error.code}` : "";
  return `${error.status}${code}: ${error.message}`;
}

export function readDebugExchangeFromSseLine(line: string): CapturedExchange | null {
  const trimmed = line.trim();
  if (!trimmed.startsWith("data:")) return null;
  const payload = trimmed.slice(5).trim();
  if (!payload || payload === "[DONE]") return null;
  try {
    const json = JSON.parse(payload) as { type?: string; data?: unknown };
    if (json.type === "data-llm-exchange" && json.data && typeof json.data === "object") {
      return json.data as CapturedExchange;
    }
  } catch {
    // Ignore non-JSON SSE lines.
  }
  return null;
}

/** Read AI SDK UI message SSE for `data-llm-exchange` parts. Resolves when the stream ends. */
export async function tapDebugExchangeStream(
  stream: ReadableStream<Uint8Array>,
  onExchange: (exchange: CapturedExchange) => void
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const exchange = readDebugExchangeFromSseLine(line);
        if (exchange) onExchange(exchange);
      }
    }
    const tail = readDebugExchangeFromSseLine(buffer + decoder.decode());
    if (tail) onExchange(tail);
  } catch {
    // Stream closed or cancelled — exchange may already have been applied.
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // Already released.
    }
  }
}

export const EXAMPLE_TOOLS_JSON = `[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get the current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": { "type": "string", "description": "City name" }
        },
        "required": ["city"]
      }
    }
  }
]`;

export function newHeaderRow(partial?: Partial<HeaderRow>): HeaderRow {
  return {
    id: crypto.randomUUID(),
    enabled: true,
    key: "",
    value: "",
    ...partial,
  };
}

export function rowsToHeaders(rows: HeaderRow[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const row of rows) {
    if (!row.enabled || !row.key.trim()) continue;
    out[row.key.trim()] = row.value;
  }
  return out;
}

export function headersToRows(headers: Record<string, string>): HeaderRow[] {
  const rows = Object.entries(headers).map(([key, value]) =>
    newHeaderRow({ key, value, enabled: true })
  );
  return rows.length > 0 ? rows : [newHeaderRow()];
}

export function parseHeadersJson(raw: string): Record<string, string> {
  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    const out: Record<string, string> = {};
    for (const [key, value] of Object.entries(parsed)) {
      out[key] = String(value ?? "");
    }
    return out;
  } catch {
    const out: Record<string, string> = {};
    for (const line of raw.split(/\r?\n/)) {
      const separator = line.indexOf(":");
      if (separator <= 0) continue;
      const key = line.slice(0, separator).trim();
      if (!key) continue;
      out[key] = line.slice(separator + 1).trim();
    }
    return out;
  }
}

export function transformerNames(provider?: Provider | null): string[] {
  const use = provider?.transformer?.use;
  if (!Array.isArray(use)) return [];
  return use.map((item) => (Array.isArray(item) ? String(item[0]) : String(item))).filter(Boolean);
}

export function guessInboundProtocol(provider?: Provider | null): InboundProtocol {
  const url = provider?.api_base_url || "";
  if (/\/messages\/?$/i.test(url)) return "messages";
  if (/\/responses\/?$/i.test(url)) return "responses";
  if (/\/chat\/completions\/?$/i.test(url)) return "chat_completions";
  const names = transformerNames(provider).map((n) => n.toLowerCase());
  if (names.includes("anthropic") || names.includes("claude-auth")) return "messages";
  if (names.includes("openai-responses")) return "responses";
  return "chat_completions";
}

export function ccrPathForProtocol(protocol: InboundProtocol): string {
  if (protocol === "messages") return "/v1/messages";
  if (protocol === "responses") return "/v1/responses";
  return "/v1/chat/completions";
}

export function requestUrlForTarget(
  target: DebugTarget,
  protocol: InboundProtocol,
  provider?: Provider | null
): string {
  if (target === "direct") {
    return provider?.api_base_url || "";
  }
  return `${window.location.origin}${ccrPathForProtocol(protocol)}`;
}

export function isLiteralApiKey(apiKey?: string): boolean {
  const value = (apiKey || "").trim();
  if (!value) return false;
  if (value.toLowerCase() === "oauth") return false;
  if (value.startsWith("$")) return false;
  return true;
}

export function oauthRenewKind(
  provider?: Provider | null
): "claude-auth" | "codex" | "qwen-auth" | "antigravity-auth" | "xai-auth" | null {
  const hint = provider ? getAuthHintKey(provider) : null;
  if (hint === "claude_oauth") return "claude-auth";
  if (hint === "qwen_jwt") return "qwen-auth";
  if (hint === "codex_oauth") return "codex";
  if (hint === "antigravity_oauth") return "antigravity-auth";
  if (hint === "xai_oauth") return "xai-auth";
  return null;
}

export function directAuthHeader(
  provider: Provider,
  protocol: InboundProtocol
): { key: string; value: string } | null {
  if (!isLiteralApiKey(provider.api_key)) return null;
  if (
    protocol === "messages" ||
    /\/messages\/?$/i.test(provider.api_base_url || "") ||
    /anthropic/i.test(provider.api_base_url || "")
  ) {
    return { key: "x-api-key", value: provider.api_key };
  }
  return { key: "Authorization", value: `Bearer ${provider.api_key}` };
}

export function mergeAuthIntoRows(
  rows: HeaderRow[],
  auth: { key: string; value: string } | null,
  customWins: boolean
): HeaderRow[] {
  const next = rows.filter(
    (row) =>
      row.key.trim().toLowerCase() !== "authorization" &&
      row.key.trim().toLowerCase() !== "x-api-key"
  );
  if (!auth) {
    return next.length > 0 ? next : [newHeaderRow()];
  }
  const existing = rows.find(
    (row) => row.key.trim().toLowerCase() === auth.key.toLowerCase()
  );
  if (customWins && existing && existing.value && existing.value !== auth.value) {
    return rows.length > 0 ? rows : [newHeaderRow()];
  }
  return [newHeaderRow({ key: auth.key, value: auth.value, enabled: true }), ...next.filter((r) => r.key.trim() || r.value.trim())];
}

export type WireMessage = { role: string; content: string };

type TextPart = { type: string; text?: string };

export function uiMessagesToWire(
  messages: Array<{ role: string; parts?: TextPart[]; content?: unknown }>,
  pendingUserText?: string
): WireMessage[] {
  const out: WireMessage[] = [];
  for (const message of messages) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    let text = "";
    if (Array.isArray(message.parts)) {
      text = message.parts
        .filter((part) => part.type === "text")
        .map((part) => part.text || "")
        .join("\n");
    } else if (typeof message.content === "string") {
      text = message.content;
    }
    if (text.trim()) out.push({ role: message.role, content: text });
  }
  const pending = pendingUserText?.trim();
  if (pending) {
    const last = out[out.length - 1];
    if (!(last?.role === "user" && last.content === pending)) {
      out.push({ role: "user", content: pending });
    }
  }
  return out;
}

export function parseToolsJson(raw: string): unknown[] {
  try {
    const parsed = JSON.parse(raw || "[]");
    if (Array.isArray(parsed)) return parsed;
    if (parsed && typeof parsed === "object" && Array.isArray((parsed as { tools?: unknown }).tools)) {
      return (parsed as { tools: unknown[] }).tools;
    }
    if (parsed && typeof parsed === "object") return [parsed];
  } catch {
    // Invalid JSON is treated as no tools so the body still prerenders.
  }
  return [];
}

function asOpenAiFunctionTools(tools: unknown[]): Array<{
  type: "function";
  function: { name: string; description?: string; parameters?: unknown };
}> {
  const out: Array<{
    type: "function";
    function: { name: string; description?: string; parameters?: unknown };
  }> = [];
  for (const item of tools) {
    if (!item || typeof item !== "object") continue;
    const rec = item as Record<string, any>;
    const fn = rec.function && typeof rec.function === "object" ? rec.function : rec;
    const name = String(fn.name || rec.name || "").trim();
    if (!name) continue;
    out.push({
      type: "function",
      function: {
        name,
        description: fn.description || rec.description || undefined,
        parameters: fn.parameters || rec.parameters || rec.input_schema || { type: "object", properties: {} },
      },
    });
  }
  return out;
}

/** Endpoint-shaped JSON that Body tab Send posts (Chat Completions / Messages / Responses). */
export function buildEndpointBody(args: {
  protocol: InboundProtocol;
  model: string;
  system: string;
  messages: WireMessage[];
  toolsJson: string;
  stream: boolean;
  reasoningEffort?: string;
}): Record<string, unknown> {
  const tools = asOpenAiFunctionTools(parseToolsJson(args.toolsJson));
  const turns = args.messages.filter((m) => m.role === "user" || m.role === "assistant");
  const effort = parseReasoningEffort(args.reasoningEffort);

  if (args.protocol === "messages") {
    const body: Record<string, unknown> = {
      model: args.model,
      max_tokens: 1024,
      messages: turns.map((m) => ({ role: m.role, content: m.content })),
      stream: args.stream,
    };
    if (args.system.trim()) body.system = args.system.trim();
    if (tools.length) {
      body.tools = tools.map((tool) => ({
        name: tool.function.name,
        description: tool.function.description || "",
        input_schema: tool.function.parameters || { type: "object", properties: {} },
      }));
    }
    return applyReasoningEffortToBody(body, args.protocol, effort);
  }

  if (args.protocol === "responses") {
    const body: Record<string, unknown> = {
      model: args.model,
      input: turns.map((m) => ({ role: m.role, content: m.content })),
      stream: args.stream,
    };
    if (args.system.trim()) body.instructions = args.system.trim();
    if (tools.length) {
      body.tools = tools.map((tool) => ({
        type: "function",
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
      }));
    }
    return applyReasoningEffortToBody(body, args.protocol, effort);
  }

  const messages: WireMessage[] = [...turns];
  if (args.system.trim()) {
    messages.unshift({ role: "system", content: args.system.trim() });
  }
  const body: Record<string, unknown> = {
    model: args.model,
    messages,
    stream: args.stream,
  };
  if (args.stream) {
    body.stream_options = { include_usage: true };
  }
  if (tools.length) body.tools = tools;
  return applyReasoningEffortToBody(body, args.protocol, effort);
}

export function buildChatPayload(
  protocol: InboundProtocol,
  model: string,
  messages: Array<{ role: string; content: string }>,
  stream: boolean
): Record<string, unknown> {
  return buildEndpointBody({
    protocol,
    model,
    system: "",
    messages,
    toolsJson: "[]",
    stream,
  });
}

const AUTH_HEADER_RE =
  /^(authorization|x-api-key|api-key|x-goog-api-key|anthropic-api-key)$/i;

function placeholderAuthValue(key: string, value: string): string {
  if (key.toLowerCase() === "authorization" && /^bearer\s+/i.test(value)) {
    return "Bearer PLACEHOLDER";
  }
  return "PLACEHOLDER";
}

function shellQuote(value: string): string {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

export function copyCurlCommand(args: {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}): string {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  for (const [key, value] of Object.entries(args.headers || {})) {
    const lower = key.toLowerCase();
    if (AUTH_HEADER_RE.test(key)) {
      headers[key] = placeholderAuthValue(key, value);
    } else if (lower !== "content-type") {
      headers[key] = value;
    } else if (value) {
      headers["Content-Type"] = value;
    }
  }
  const hasAuth = Object.keys(headers).some((key) => AUTH_HEADER_RE.test(key));
  if (!hasAuth) {
    headers.Authorization = "Bearer PLACEHOLDER";
  }

  let curl = `curl -X ${shellQuote(args.method)} ${shellQuote(args.url || "")}`;
  for (const [key, value] of Object.entries(headers)) {
    curl += ` \\\n  -H ${shellQuote(`${key}: ${value}`)}`;
  }
  if (args.method !== "GET" && args.body != null) {
    const payload =
      typeof args.body === "string" ? args.body : JSON.stringify(args.body, null, 2);
    curl += ` \\\n  --data-raw ${shellQuote(payload)}`;
  }
  return curl;
}

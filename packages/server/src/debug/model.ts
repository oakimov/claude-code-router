import {
  getAntigravityAccessToken,
  getClaudeAccessToken,
  getCodexAccessToken,
  getQwenAccessToken,
  getXaiAccessToken,
} from "@caeliq/llms";
import { resolveCodexPat } from "@caeliq/ccr-shared";
import {
  REASONING_EFFORTS,
  type DebugChatInput,
  type DebugModelConfig,
  type InboundProtocol,
  type ReasoningEffort,
} from "./types";

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

export const DEFAULT_DEBUG_INSTRUCTIONS =
  "You are the CCR Debug agent. Answer briefly so the request and response exchange is easy to inspect.";

export function isLiteralApiKey(apiKey: unknown): boolean {
  const value = String(apiKey ?? "").trim();
  if (!value) return false;
  if (value.toLowerCase() === "oauth") return false;
  if (value.startsWith("$")) return false;
  return true;
}

export function transformerNames(provider: any): string[] {
  const use = provider?.transformer?.use;
  if (!Array.isArray(use)) return [];
  return use.map((item: unknown) =>
    Array.isArray(item) ? String(item[0] ?? "") : String(item ?? "")
  ).filter(Boolean);
}

export type OAuthKind =
  | "claude-auth"
  | "codex"
  | "qwen-auth"
  | "antigravity-auth"
  | "xai-auth";

export function oauthKindForProvider(provider: any): OAuthKind | null {
  const names = transformerNames(provider).map((n) => n.toLowerCase());
  if (names.includes("claude-auth")) return "claude-auth";
  if (names.includes("qwen-auth")) return "qwen-auth";
  if (names.includes("antigravity-auth")) return "antigravity-auth";
  if (names.includes("xai-auth")) return "xai-auth";
  if (names.includes("codex")) {
    if (codexPatForProvider(provider)) return null;
    return "codex";
  }
  return null;
}

export function codexPatForProvider(provider: any): string | undefined {
  return resolveCodexPat(provider?.api_key ?? provider?.apiKey, {
    allowBareEnvName: true,
  });
}

export function interpolateConfigString(value: string): string {
  return String(value || "").replace(
    /\$\{([^}]+)\}|\$([A-Z_][A-Z0-9_]*)/g,
    (match, braced, unbraced) => {
      const varName = braced || unbraced;
      return process.env[varName] || match;
    }
  );
}

export async function resolveProviderApiKey(provider: any): Promise<string> {
  const name = String(provider?.name || "provider");
  const kind = oauthKindForProvider(provider);
  if (kind) {
    const token = await oauthAccessToken(kind);
    if (!token) {
      throw new Error(`Provider "${name}" OAuth token from CCR config is empty.`);
    }
    return token;
  }
  const resolved = interpolateConfigString(String(provider?.api_key || "").trim());
  if (!resolved) {
    throw new Error(`Provider "${name}" has no api_key in CCR config.`);
  }
  if (resolved.toLowerCase() === "oauth") {
    throw new Error(
      `Provider "${name}" is set to oauth in CCR config but has no OAuth transformer.`
    );
  }
  if (resolved.startsWith("$")) {
    throw new Error(
      `Provider "${name}" api_key refers to ${resolved}, which is not set in the environment.`
    );
  }
  return resolved;
}

async function oauthAccessToken(kind: OAuthKind): Promise<string> {
  if (kind === "claude-auth") {
    const tokens = await getClaudeAccessToken();
    return String(tokens.access_token || "");
  }
  if (kind === "codex") {
    const tokens = await getCodexAccessToken();
    return String(tokens.access_token || "");
  }
  if (kind === "qwen-auth") {
    const tokens = await getQwenAccessToken();
    return String((tokens as { token?: string }).token || "");
  }
  if (kind === "antigravity-auth") {
    const tokens = await getAntigravityAccessToken();
    return String(tokens.access_token || "");
  }
  const tokens = await getXaiAccessToken();
  return String(tokens.access_token || "");
}

export function authHeadersForProvider(
  protocol: InboundProtocol,
  apiKey: string,
  oauthKind: OAuthKind | null
): Record<string, string> {
  if (oauthKind === "claude-auth") {
    return {
      Authorization: `Bearer ${apiKey}`,
      "anthropic-beta": "oauth-2025-04-20",
      "anthropic-version": "2023-06-01",
    };
  }
  if (protocol === "messages") {
    return {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    };
  }
  return { Authorization: `Bearer ${apiKey}` };
}

function codexTokenMetadata(tokens: {
  access_token: string;
  id_token?: string;
  account_id?: string;
}): { accountId?: string; isFedramp: boolean } {
  try {
    const encoded = tokens.id_token?.split(".")[1];
    if (!encoded) {
      return { accountId: tokens.account_id, isFedramp: false };
    }
    const payload = JSON.parse(Buffer.from(encoded, "base64url").toString());
    const claims = payload?.["https://api.openai.com/auth"];
    const accountId =
      tokens.account_id ||
      (typeof claims?.chatgpt_account_id === "string"
        ? claims.chatgpt_account_id
        : undefined);
    return {
      accountId,
      isFedramp: claims?.chatgpt_account_is_fedramp === true,
    };
  } catch {
    return { accountId: tokens.account_id, isFedramp: false };
  }
}

async function resolveDirectAuth(
  provider: any,
  protocol: InboundProtocol
): Promise<{
  apiKey: string;
  authKind: OAuthKind | null;
  headers: Record<string, string>;
}> {
  const authKind = oauthKindForProvider(provider);
  if (authKind === "codex") {
    const tokens = await getCodexAccessToken();
    const metadata = codexTokenMetadata(tokens);
    const headers: Record<string, string> = {
      Authorization: `Bearer ${tokens.access_token}`,
      originator: "codex_cli_rs",
      "User-Agent": "codex_cli_rs/0.145.0",
    };
    if (metadata.accountId) headers["ChatGPT-Account-ID"] = metadata.accountId;
    if (metadata.isFedramp) headers["X-OpenAI-Fedramp"] = "true";
    return { apiKey: tokens.access_token, authKind, headers };
  }
  const apiKey = await resolveProviderApiKey(provider);
  return {
    apiKey,
    authKind,
    headers: authHeadersForProvider(protocol, apiKey, authKind),
  };
}

function withCodexClientVersion(url: string, authKind: OAuthKind | null): string {
  if (authKind !== "codex") return url;
  const parsed = new URL(url);
  if (!parsed.searchParams.has("client_version")) {
    parsed.searchParams.set("client_version", "0.145.0");
  }
  return parsed.toString();
}

export function ccrPathForProtocol(protocol: InboundProtocol): string {
  if (protocol === "messages") return "/v1/messages";
  if (protocol === "responses") return "/v1/responses";
  return "/v1/chat/completions";
}

export function providerEndpointUrl(provider: any): string {
  return String(provider?.api_base_url || "").trim().replace(/\/+$/, "");
}

export function guessInboundProtocol(provider: any): InboundProtocol {
  const url = String(provider?.api_base_url || "");
  if (/\/messages\/?$/i.test(url)) return "messages";
  if (/\/responses\/?$/i.test(url)) return "responses";
  if (/\/chat\/completions\/?$/i.test(url)) return "chat_completions";
  const names = transformerNames(provider).map((n) => n.toLowerCase());
  if (names.includes("anthropic") || names.includes("claude-auth")) return "messages";
  if (names.includes("openai-responses")) return "responses";
  return "chat_completions";
}

export function findProvider(config: any, name: string): any | undefined {
  const providers = config?.Providers || config?.providers || [];
  if (!Array.isArray(providers)) return undefined;
  return providers.find((p: any) => p && p.name === name);
}

export function deriveApiBase(apiBaseUrl: string): string {
  return String(apiBaseUrl || "")
    .trim()
    .replace(/\/+$/, "")
    .replace(/\/chat\/completions$/i, "")
    .replace(/\/responses$/i, "")
    .replace(/\/messages$/i, "");
}

export function ensureV1Base(url: string): string {
  const trimmed = String(url || "").replace(/\/+$/, "");
  if (!trimmed) return trimmed;
  if (/\/v\d+(?:beta)?$/i.test(trimmed) || /\/openai$/i.test(trimmed)) return trimmed;
  try {
    const parsed = new URL(trimmed);
    return parsed.pathname === "/" || parsed.pathname === ""
      ? `${trimmed}/v1`
      : trimmed;
  } catch {
    return `${trimmed}/v1`;
  }
}

export function mastraModelRef(
  protocol: InboundProtocol,
  modelId: string
): { providerId: string; modelId: string } {
  if (protocol === "messages") {
    return { providerId: "anthropic", modelId };
  }
  return { providerId: "openai", modelId };
}

export function modelUrlForProtocol(
  protocol: InboundProtocol,
  baseOrOrigin: string
): string {
  return ensureV1Base(baseOrOrigin);
}

function sanitizeUserHeaders(
  headers: Record<string, string> | undefined
): Record<string, string> {
  if (!headers || typeof headers !== "object") return {};
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    const name = String(key || "").trim();
    if (!name) continue;
    const lower = name.toLowerCase();
    if (lower === "authorization" || lower === "x-api-key") continue;
    out[name] = String(value ?? "");
  }
  return out;
}

export async function resolveDebugModel(
  input: Pick<DebugChatInput, "target" | "protocol" | "provider" | "model" | "headers">,
  config: any
): Promise<DebugModelConfig> {
  const provider = findProvider(config, input.provider);
  if (!provider) {
    throw new Error(`Unknown provider "${input.provider}"`);
  }
  const modelName = String(input.model || "").trim();
  if (!modelName) {
    throw new Error("Model is required");
  }

  const extraHeaders = sanitizeUserHeaders(input.headers);
  const port = Number(config?.PORT) || 3456;

  if (input.target === "direct") {
    const derived = deriveApiBase(provider.api_base_url || "");
    if (!derived) {
      throw new Error(`Provider "${input.provider}" has no api_base_url in CCR config.`);
    }
    if (codexPatForProvider(provider)) {
      throw new Error(
        "Codex PAT direct mode is not supported because the backend requires account metadata. Use CCR mode."
      );
    }
    const auth = await resolveDirectAuth(provider, input.protocol);
    const ref = mastraModelRef(input.protocol, modelName);
    return {
      url: modelUrlForProtocol(input.protocol, derived),
      id: `${ref.providerId}/${ref.modelId}`,
      apiKey: auth.apiKey,
      authKind: auth.authKind ?? undefined,
      headers: {
        ...extraHeaders,
        ...auth.headers,
      },
    };
  }

  const ref = mastraModelRef(input.protocol, `${provider.name},${modelName}`);
  return {
    url: modelUrlForProtocol(input.protocol, `http://127.0.0.1:${port}`),
    id: `${ref.providerId}/${ref.modelId}`,
    apiKey: String(config?.APIKEY || "ccr-debug"),
    headers: extraHeaders,
  };
}

export async function executeDebugRequest(
  args: {
    target: "ccr" | "direct";
    protocol: InboundProtocol;
    provider: string;
    headers?: Record<string, string>;
    body: unknown;
    signal?: AbortSignal;
  },
  config: any
): Promise<{ status: number; headers: Record<string, string>; body: string }> {
  const provider = findProvider(config, args.provider);
  if (!provider) {
    throw new Error(`Unknown provider "${args.provider}"`);
  }
  const extraHeaders = sanitizeUserHeaders(args.headers);
  let url: string;
  let auth: Record<string, string>;
  if (args.target === "direct") {
    url = providerEndpointUrl(provider);
    if (!url) {
      throw new Error(`Provider "${args.provider}" has no api_base_url in CCR config.`);
    }
    if (codexPatForProvider(provider)) {
      throw new Error(
        "Codex PAT direct mode is not supported because the backend requires account metadata. Use CCR mode."
      );
    }
    const resolvedAuth = await resolveDirectAuth(provider, args.protocol);
    url = withCodexClientVersion(url, resolvedAuth.authKind);
    auth = resolvedAuth.headers;
  } else {
    const port = Number(config?.PORT) || 3456;
    url = `http://127.0.0.1:${port}${ccrPathForProtocol(args.protocol)}`;
    const apiKey = String(config?.APIKEY || "ccr-debug");
    auth =
      args.protocol === "messages"
        ? { "x-api-key": apiKey }
        : { Authorization: `Bearer ${apiKey}` };
  }
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...extraHeaders,
      ...auth,
    },
    body: JSON.stringify(args.body ?? {}),
    signal: args.signal,
  });
  const responseHeaders: Record<string, string> = {};
  response.headers.forEach((value, key) => {
    responseHeaders[key] = value;
  });
  return {
    status: response.status,
    headers: responseHeaders,
    body: await response.text(),
  };
}

export function parseDebugChatBody(body: any): DebugChatInput {
  const target = body?.target === "direct" ? "direct" : "ccr";
  const protocolRaw = String(body?.protocol || "chat_completions");
  const protocol: DebugChatInput["protocol"] =
    protocolRaw === "messages" || protocolRaw === "responses"
      ? protocolRaw
      : "chat_completions";
  const messages = Array.isArray(body?.messages) ? body.messages : [];
  return {
    messages,
    target,
    protocol,
    provider: String(body?.provider || "").trim(),
    model: String(body?.model || "").trim(),
    system: typeof body?.system === "string" ? body.system : "",
    tools: body?.tools,
    // The agent playground always uses the streaming UI transport. The raw
    // Body tab is the place to exercise non-streaming provider requests.
    stream: true,
    reasoningEffort: parseReasoningEffort(body?.reasoningEffort ?? body?.reasoning_effort),
    headers:
      body?.headers && typeof body.headers === "object" && !Array.isArray(body.headers)
        ? body.headers
        : undefined,
  };
}

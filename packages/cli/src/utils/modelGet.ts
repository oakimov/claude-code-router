import { confirm } from "@inquirer/prompts";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { backupConfigFile, readConfigFile, readConfigFileRaw, writeConfigFile } from "./index";
import { CONFIG_FILE } from "@caeliq/ccr-shared";
import type { ProviderConfig } from "@caeliq/ccr-shared";
import {
  buildCliCodexHeaders,
  recoverCliCodexOAuth,
  resolveCliCodexAuth,
} from "./codex-auth";

const CODEX_AUTH_FILE = join(homedir(), ".claude-code-router", "codex_auth.json");
const CLAUDE_AUTH_FILE = join(homedir(), ".claude-code-router", "claude_auth.json");
const CONFIG_PATH_DISPLAY = "~/.claude-code-router/config.json";
const READABLE_CONFIG_FILE = process.env.CCR_CONFIG_FILE || CONFIG_FILE;
const DEFAULT_CODEX_CLIENT_VERSION = "0.145.0";
const CLAUDE_OAUTH_REQUIRED_BETA = "oauth-2025-04-20";

const RESET = "\x1B[0m";
const DIM = "\x1B[2m";
const CYAN = "\x1B[36m";
const BOLDCYAN = "\x1B[1m\x1B[36m";
const GREEN = "\x1B[32m";
const YELLOW = "\x1B[33m";
const BOLDYELLOW = "\x1B[1m\x1B[33m";

interface ClaudeAuthTokens {
  access_token?: string;
  refresh_token?: string;
  expires_at?: number;
  token_type?: string;
}

interface ConfigWithProviders {
  Providers?: ProviderConfig[];
  [key: string]: any;
}

interface ResolvedEndpoint {
  kind: "openai" | "gemini" | "codex" | "cursor" | "anthropic";
  url: string;
}

function transformerUseIncludes(provider: ProviderConfig, transformerName: string): boolean {
  const use = provider.transformer?.use;
  if (!Array.isArray(use)) {
    return false;
  }

  const normalizedTransformerName = transformerName.toLowerCase();
  return use.some((entry: unknown) => {
    if (typeof entry === "string") {
      return entry.toLowerCase() === normalizedTransformerName;
    }
    if (Array.isArray(entry) && typeof entry[0] === "string") {
      return entry[0].toLowerCase() === normalizedTransformerName;
    }
    if (entry && typeof entry === "object" && typeof (entry as any).name === "string") {
      return (entry as any).name.toLowerCase() === normalizedTransformerName;
    }
    return false;
  });
}

function isClaudeAuthProvider(provider: ProviderConfig): boolean {
  return transformerUseIncludes(provider, "claude-auth");
}

function isAnthropicProvider(provider: ProviderConfig): boolean {
  const normalizedName = normalizeProviderName(provider.name);
  const baseUrl = String(provider.api_base_url || "").toLowerCase();
  return (
    normalizedName === "claude" ||
    normalizedName === "anthropic" ||
    baseUrl.includes("api.anthropic.com") ||
    transformerUseIncludes(provider, "anthropic")
  );
}

function isCursorProvider(provider: ProviderConfig): boolean {
  if (normalizeProviderName(provider.name) === "cursor") {
    return true;
  }

  return transformerUseIncludes(provider, "cursor-sdk");
}

if (!READABLE_CONFIG_FILE) {
  throw new Error("Config file path is not available.");
}

function assertConfigExists(): void {
  if (!existsSync(READABLE_CONFIG_FILE)) {
    throw new Error(`Configuration file not found at ${CONFIG_PATH_DISPLAY}.`);
  }
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.trim().length > 0;
}

function normalizeProviderName(name: string): string {
  return name.trim().toLowerCase();
}

function findProvider(config: ConfigWithProviders, providerName: string): ProviderConfig | undefined {
  const normalizedName = normalizeProviderName(providerName);
  return config.Providers?.find((provider) => normalizeProviderName(provider.name) === normalizedName);
}

function getValueByPath(obj: any, path: string): any {
  if (!path) return obj;
  return path.split(".").reduce((acc, part) => acc && acc[part], obj);
}

function getProviderApiKey(provider: ProviderConfig): string | undefined {
  if (!isNonEmptyString(provider.api_key)) {
    return undefined;
  }

  const unresolvedEnvPattern = /^\$\{?[A-Z_][A-Z0-9_]*\}?$/;
  if (unresolvedEnvPattern.test(provider.api_key.trim())) {
    return undefined;
  }

  return provider.api_key;
}

function readClaudeAuthTokens(): ClaudeAuthTokens {
  if (!existsSync(CLAUDE_AUTH_FILE)) {
    throw new Error("Claude OAuth tokens not found. Run `ccr claude-auth` first.");
  }

  try {
    return JSON.parse(readFileSync(CLAUDE_AUTH_FILE, "utf-8")) as ClaudeAuthTokens;
  } catch {
    throw new Error("Failed to read Claude OAuth tokens from ~/.claude-code-router/claude_auth.json.");
  }
}

function getClaudeAccessToken(): string {
  const tokens = readClaudeAuthTokens();

  if (!isNonEmptyString(tokens.access_token)) {
    throw new Error("Claude OAuth access token is missing. Run `ccr claude-auth` again.");
  }

  if (typeof tokens.expires_at === "number" && tokens.expires_at <= Date.now() / 1000) {
    throw new Error("Claude OAuth access token is expired. Run `ccr claude-auth` again.");
  }

  return tokens.access_token;
}

function getRequestApiKey(provider: ProviderConfig): string | undefined {
  if (isCursorProvider(provider)) {
    const apiKey = getProviderApiKey(provider)?.trim();
    // Mirror server resolveCursorApiKey: only concrete crsr_ keys from config;
    // unresolved ${ENV} placeholders are already stripped by getProviderApiKey.
    if (
      apiKey &&
      apiKey.startsWith("crsr_") &&
      !apiKey.includes("${") &&
      !apiKey.includes("$CURSOR_API_KEY")
    ) {
      return apiKey;
    }
    const fromEnv = process.env.CURSOR_API_KEY?.trim();
    if (fromEnv) {
      return fromEnv;
    }
    return undefined;
  }

  if (isClaudeAuthProvider(provider)) {
    return getClaudeAccessToken();
  }

  return getProviderApiKey(provider);
}

function getMissingApiKeyMessage(provider: ProviderConfig): string {
  if (normalizeProviderName(provider.name) === "codex") {
    return "Codex authentication unavailable. No valid api_key (PAT) and no OAuth tokens found. " +
           "Set api_key to a PAT, or run `ccr codex-auth` for OAuth.";
  }

  if (isCursorProvider(provider)) {
    return "Cursor authentication unavailable. Set Providers[].api_key to a crsr_ key, " +
           "or export CURSOR_API_KEY.";
  }

  if (isClaudeAuthProvider(provider)) {
    return "Claude authentication unavailable. Run `ccr claude-auth` for OAuth.";
  }

  return `Provider \"${provider.name}\" does not have a usable API key configured.`;
}

function appendApiKeyToUrl(url: string, apiKey: string): string {
  const parsedUrl = new URL(url);
  parsedUrl.searchParams.set("key", apiKey);
  return parsedUrl.toString();
}

function maskApiKeyInUrl(url: string): string {
  const parsedUrl = new URL(url);
  if (parsedUrl.searchParams.has("key")) {
    parsedUrl.searchParams.set("key", "***");
  }
  return parsedUrl.toString();
}

function deriveOpenAIModelsUrl(apiBaseUrl: string): string {
  const parsedUrl = new URL(apiBaseUrl);
  const pathname = parsedUrl.pathname;
  const suffixes = [
    "/chat/completions",
    "/responses",
    "/messages",
    "/completions",
    "/embeddings",
    "/v1beta/chat/completions",
    "/v1beta/responses",
    "/v1beta/messages",
  ];

  const matchedSuffix = suffixes.find((suffix) => pathname.endsWith(suffix));
  if (matchedSuffix) {
    parsedUrl.pathname = pathname.slice(0, -matchedSuffix.length) + "/models";
    parsedUrl.search = "";
    return parsedUrl.toString();
  }

  if (pathname.endsWith("/models")) {
    parsedUrl.search = "";
    return parsedUrl.toString();
  }

  const trimmedPath = pathname.endsWith("/") ? pathname.slice(0, -1) : pathname;
  if (trimmedPath === "" || trimmedPath === "/") {
    parsedUrl.pathname = "/v1/models";
    parsedUrl.search = "";
    return parsedUrl.toString();
  }

  const pathSegments = trimmedPath.split("/").filter(Boolean);
  if (pathSegments.length === 1 && /^v\d+(beta\d+)?$/i.test(pathSegments[0])) {
    parsedUrl.pathname = `/${pathSegments[0]}/models`;
    parsedUrl.search = "";
    return parsedUrl.toString();
  }

  parsedUrl.pathname = "/v1/models";
  parsedUrl.search = "";
  return parsedUrl.toString();
}

function deriveAnthropicModelsUrl(apiBaseUrl: string): string {
  const parsedUrl = new URL(apiBaseUrl);
  const pathname = parsedUrl.pathname;
  const suffixes = ["/v1/messages", "/messages", "/v1/complete", "/complete"];

  const matchedSuffix = suffixes.find((suffix) => pathname.endsWith(suffix));
  if (matchedSuffix) {
    parsedUrl.pathname = pathname.slice(0, -matchedSuffix.length) + "/v1/models";
  } else {
    parsedUrl.pathname = "/v1/models";
  }

  parsedUrl.search = "";
  return parsedUrl.toString();
}

function resolveModelsEndpoint(provider: ProviderConfig): ResolvedEndpoint {
  const normalizedName = normalizeProviderName(provider.name);

  if (isCursorProvider(provider)) {
    return { kind: "cursor", url: "Cursor.models.list (@cursor/sdk)" };
  }

  if (normalizedName === "gemini") {
    const url = provider.models_api_url || "https://generativelanguage.googleapis.com/v1beta/models";
    return { kind: "gemini", url };
  }

  if (normalizedName === "openai") {
    const url = provider.models_api_url || deriveOpenAIModelsUrl(provider.api_base_url);
    return { kind: "openai", url };
  }

  if (normalizedName === "codex") {
    if (!provider.models_api_url) {
      throw new Error("Provider \"codex\" requires \"models_api_url\" to be configured for model discovery.");
    }
    return { kind: "codex", url: provider.models_api_url };
  }

  if (isAnthropicProvider(provider)) {
    const url = provider.models_api_url || deriveAnthropicModelsUrl(provider.api_base_url);
    return { kind: "anthropic", url };
  }

  if (provider.models_api_url) {
    return { kind: "openai", url: provider.models_api_url };
  }

  return { kind: "openai", url: deriveOpenAIModelsUrl(provider.api_base_url) };
}

async function parseErrorMessage(response: Response): Promise<string> {
  try {
    const payload = await response.json();
    if (typeof payload?.error === "string") {
      return payload.error;
    }
    if (typeof payload?.error?.message === "string") {
      return payload.error.message;
    }
    if (typeof payload?.message === "string") {
      return payload.message;
    }
  } catch {
    const text = await response.text().catch(() => "");
    if (text.trim()) {
      return text.trim();
    }
  }
  return response.statusText || "Unknown error";
}

function describeHttpError(status: number): string {
  if (status === 401 || status === 403) {
    return "Invalid API key or insufficient permissions";
  }
  if (status === 404) {
    return "Model list endpoint was not found";
  }
  return "Request failed";
}

async function fetchOpenAIModels(apiKey: string, url: string): Promise<string[]> {
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  });

  if (!response.ok) {
    const errorMessage = await parseErrorMessage(response);
    throw new Error(`${describeHttpError(response.status)} (${response.status}): ${errorMessage}`);
  }

  const payload = await response.json();
  if (!Array.isArray(payload?.data)) {
    throw new Error("Unsupported response format: expected a data array");
  }

  return payload.data
    .map((model: any) => (typeof model?.id === "string" ? model.id.trim() : ""))
    .filter(Boolean);
}

async function fetchGeminiModels(apiKey: string, url: string): Promise<string[]> {
  const response = await fetch(appendApiKeyToUrl(url, apiKey), {
    method: "GET",
  });

  if (!response.ok) {
    const errorMessage = await parseErrorMessage(response);
    throw new Error(`${describeHttpError(response.status)} (${response.status}): ${errorMessage}`);
  }

  const payload = await response.json();
  if (!Array.isArray(payload?.models)) {
    throw new Error("Unsupported response format: expected a models array");
  }

  return payload.models
    .map((model: any) => {
      if (typeof model?.name !== "string") {
        return "";
      }
      return model.name.startsWith("models/") ? model.name.slice("models/".length) : model.name;
    })
    .map((model: string) => model.trim())
    .filter(Boolean);
}

function appendClaudeAuthQueryParams(url: string): string {
  const parsedUrl = new URL(url);
  parsedUrl.searchParams.set("beta", "true");
  return parsedUrl.toString();
}

function buildAnthropicModelHeaders(apiKey: string, provider: ProviderConfig): Record<string, string> {
  const headers: Record<string, string> = {
    "anthropic-version": "2023-06-01",
  };

  if (isClaudeAuthProvider(provider)) {
    headers["Authorization"] = `Bearer ${apiKey}`;
    headers["anthropic-beta"] = CLAUDE_OAUTH_REQUIRED_BETA;
  } else {
    headers["x-api-key"] = apiKey;
  }

  return headers;
}

async function fetchAnthropicModels(apiKey: string, url: string, provider: ProviderConfig): Promise<string[]> {
  const finalUrl = isClaudeAuthProvider(provider) ? appendClaudeAuthQueryParams(url) : url;
  const response = await fetch(finalUrl, {
    method: "GET",
    headers: buildAnthropicModelHeaders(apiKey, provider),
  });

  if (!response.ok) {
    const errorMessage = await parseErrorMessage(response);
    throw new Error(`${describeHttpError(response.status)} (${response.status}): ${errorMessage}`);
  }

  const payload = await response.json();
  if (!Array.isArray(payload?.data)) {
    throw new Error("Unsupported Anthropic response format: expected a data array");
  }

  return payload.data
    .map((model: any) => (typeof model?.id === "string" ? model.id.trim() : ""))
    .filter(Boolean);
}

function getCodexClientVersion(provider?: ProviderConfig): string {
  const configuredVersion =
    typeof provider?.codex_client_version === "string"
      ? provider.codex_client_version.trim()
      : "";
  const envVersion = process.env.CCR_CODEX_CLIENT_VERSION?.trim() || "";
  return configuredVersion || envVersion || DEFAULT_CODEX_CLIENT_VERSION;
}

function appendCodexQueryParams(url: string, provider?: ProviderConfig): string {
  const parsedUrl = new URL(url);
  if (!parsedUrl.searchParams.has("client_version")) {
    parsedUrl.searchParams.set("client_version", getCodexClientVersion(provider));
  }
  return parsedUrl.toString();
}

export async function fetchWithCodexAuth(
  url: string,
  provider: ProviderConfig
): Promise<Response> {
  const clientVersion = getCodexClientVersion(provider);
  let auth = await resolveCliCodexAuth(getProviderApiKey(provider));
  let response = await fetch(appendCodexQueryParams(url, provider), {
    method: "GET",
    headers: buildCliCodexHeaders(auth, clientVersion),
  });

  if (response.status === 401 && auth.mode === "oauth") {
    const recovered = await recoverCliCodexOAuth(auth);
    if (recovered) {
      auth = recovered;
      try {
        await response.body?.cancel();
      } catch {
        // The response may already be fully buffered.
      }
      response = await fetch(appendCodexQueryParams(url, provider), {
        method: "GET",
        headers: buildCliCodexHeaders(auth, clientVersion),
      });
    }
  }
  return response;
}

async function fetchCodexModels(url: string, provider: ProviderConfig): Promise<string[]> {
  const response = await fetchWithCodexAuth(url, provider);
  if (!response.ok) {
    const errorMessage = await parseErrorMessage(response);
    throw new Error(`${describeHttpError(response.status)} (${response.status}): ${errorMessage}`);
  }

  const payload = await response.json();
  if (!Array.isArray(payload?.models)) {
    throw new Error("Unsupported Codex response format: expected a models array");
  }

  return payload.models
    .map((model: any) => (typeof model?.slug === "string" ? model.slug.trim() : ""))
    .filter(Boolean);
}

async function fetchCursorModels(apiKey: string): Promise<string[]> {
  let Cursor: { models: { list: (opts?: { apiKey?: string }) => Promise<any[]> } };
  try {
    ({ Cursor } = await import("@cursor/sdk"));
  } catch (error: any) {
    throw new Error(
      `Failed to load @cursor/sdk for Cursor model discovery: ${error?.message || error}`
    );
  }

  let models: any[];
  try {
    models = await Cursor.models.list({ apiKey });
  } catch (error: any) {
    const message = error?.message || String(error);
    if (/auth|unauthorized|api.?key|401|403/i.test(message)) {
      throw new Error(`Invalid API key or insufficient permissions: ${message}`);
    }
    throw new Error(`Cursor.models.list failed: ${message}`);
  }

  if (!Array.isArray(models)) {
    throw new Error("Unsupported Cursor SDK response: expected a models array");
  }

  return models
    .map((model: any) => (typeof model?.id === "string" ? model.id.trim() : ""))
    .filter(Boolean);
}

async function fetchCustomModels(
  apiKey: string | undefined,
  url: string,
  format: any,
  kind: string,
  provider?: ProviderConfig
): Promise<string[]> {
  const headers: Record<string, string> = {};
  let finalUrl = url;

  if (kind === "gemini") {
    if (!apiKey) throw new Error("A Gemini API key is required.");
    finalUrl = appendApiKeyToUrl(url, apiKey);
  } else if (kind === "codex") {
    if (!provider) throw new Error("Codex provider configuration is required.");
  } else if (kind === "anthropic" && provider) {
    if (!apiKey) throw new Error("An Anthropic API key is required.");
    if (isClaudeAuthProvider(provider)) {
      finalUrl = appendClaudeAuthQueryParams(url);
    }
    Object.assign(headers, buildAnthropicModelHeaders(apiKey, provider));
  } else {
    if (!apiKey) throw new Error("An API key is required.");
    headers["Authorization"] = `Bearer ${apiKey}`;
  }

  const response =
    kind === "codex" && provider
      ? await fetchWithCodexAuth(finalUrl, provider)
      : await fetch(finalUrl, {
          method: "GET",
          headers,
        });

  if (!response.ok) {
    const errorMessage = await parseErrorMessage(response);
    throw new Error(`${describeHttpError(response.status)} (${response.status}): ${errorMessage}`);
  }

  const payload = await response.json();
  const list = getValueByPath(payload, format.listPath || "");

  if (!Array.isArray(list)) {
    throw new Error(`Unsupported response format: expected an array at path "${format.listPath || "root"}"`);
  }

  return list
    .map((item: any) => {
      let id = typeof item === "string" ? item : getValueByPath(item, format.idPath || "");
      if (typeof id !== "string") {
        return "";
      }

      if (format.stripPrefix && id.startsWith(format.stripPrefix)) {
        id = id.slice(format.stripPrefix.length);
      }
      return id.trim();
    })
    .filter(Boolean);
}

async function fetchRemoteModels(apiKey: string | undefined, provider: ProviderConfig, endpoint: ResolvedEndpoint): Promise<string[]> {
  if (endpoint.kind === "cursor") {
    if (!apiKey) throw new Error(getMissingApiKeyMessage(provider));
    return fetchCursorModels(apiKey);
  }

  if (provider.models_response_format) {
    return fetchCustomModels(apiKey, endpoint.url, provider.models_response_format, endpoint.kind, provider);
  }

  if (endpoint.kind === "gemini") {
    if (!apiKey) throw new Error(getMissingApiKeyMessage(provider));
    return fetchGeminiModels(apiKey, endpoint.url);
  }

  if (endpoint.kind === "codex") {
    return fetchCodexModels(endpoint.url, provider);
  }

  if (endpoint.kind === "anthropic") {
    if (!apiKey) throw new Error(getMissingApiKeyMessage(provider));
    return fetchAnthropicModels(apiKey, endpoint.url, provider);
  }

  if (!apiKey) throw new Error(getMissingApiKeyMessage(provider));
  return fetchOpenAIModels(apiKey, endpoint.url);
}

function getEndpointHelpText(provider: ProviderConfig, endpoint: ResolvedEndpoint): string | undefined {
  if (endpoint.kind === "codex") {
    return `Codex parsing uses models[].slug for provider \"${provider.name}\".`;
  }

  if (endpoint.kind === "cursor") {
    return `Cursor discovery uses Cursor.models.list and writes ids into Providers[].models (no cursor-models.json side cache).`;
  }

  if (endpoint.kind === "anthropic") {
    return `Anthropic parsing uses data[].id for provider \"${provider.name}\".`;
  }

  return undefined;
}

function printEndpointHelp(provider: ProviderConfig, endpoint: ResolvedEndpoint): void {
  const helpText = getEndpointHelpText(provider, endpoint);
  if (helpText) {
    console.log(`${DIM}${helpText}${RESET}`);
  }
}

function printAuthSource(provider: ProviderConfig): void {
  if (normalizeProviderName(provider.name) === "codex") {
    const apiKey = getProviderApiKey(provider);
    if (apiKey && apiKey.startsWith("at-")) {
      console.log(`${BOLDCYAN}Auth source:${RESET} api_key (PAT)`);
    } else {
      console.log(`${BOLDCYAN}Auth source:${RESET} ${CODEX_AUTH_FILE}`);
    }
    return;
  }

  if (isCursorProvider(provider)) {
    const apiKey = getProviderApiKey(provider);
    if (apiKey && apiKey.startsWith("crsr_")) {
      console.log(`${BOLDCYAN}Auth source:${RESET} api_key (crsr_...)`);
    } else {
      console.log(`${BOLDCYAN}Auth source:${RESET} CURSOR_API_KEY`);
    }
  }

  if (isClaudeAuthProvider(provider)) {
    console.log(`${BOLDCYAN}Auth source:${RESET} ${CLAUDE_AUTH_FILE}`);
  }
}

function printRequestContext(provider: ProviderConfig, endpoint: ResolvedEndpoint): void {
  printAuthSource(provider);
  printEndpointHelp(provider, endpoint);
}

function printNoModelsMessage(provider: ProviderConfig, endpoint: ResolvedEndpoint): void {
  console.log(`\n${DIM}No remote models were returned for ${provider.name}.${RESET}`);
  printRequestContext(provider, endpoint);
}

function printModelList(models: string[]): void {
  for (const model of models) {
    console.log(`  ${CYAN}- ${model}${RESET}`);
  }
}

function printRemoteModels(provider: ProviderConfig, endpoint: ResolvedEndpoint, models: string[]): void {
  if (models.length === 0) {
    printNoModelsMessage(provider, endpoint);
    return;
  }

  printModelList(models);
  printRequestContext(provider, endpoint);
}

function hasRemoteModels(models: string[]): boolean {
  return models.length > 0;
}

function printSuccess(provider: ProviderConfig, endpoint: ResolvedEndpoint, models: string[]): void {
  console.log(`\n${BOLDCYAN}Provider:${RESET} ${provider.name}`);
  console.log(
    `${BOLDCYAN}Endpoint:${RESET} ${
      endpoint.kind === "cursor" ? endpoint.url : maskApiKeyInUrl(endpoint.url)
    }`
  );
  console.log(`${BOLDCYAN}Remote models:${RESET} ${models.length}`);
  console.log(`${GREEN}✓ API key validated successfully${RESET}\n`);
  printRemoteModels(provider, endpoint, models);
}

function shouldSyncRemoteModels(models: string[]): boolean {
  return hasRemoteModels(models);
}

async function syncMissingModels(provider: ProviderConfig, remoteModels: string[]): Promise<void> {
  if (!shouldSyncRemoteModels(remoteModels)) {
    return;
  }

  const configuredModels = Array.isArray(provider.models) ? provider.models : [];
  const configuredSet = new Set(configuredModels);
  const missingModels = remoteModels.filter((model) => !configuredSet.has(model));

  if (missingModels.length === 0) {
    console.log(`\n${DIM}No missing models to sync.${RESET}`);
    return;
  }

  console.log(`\n${BOLDCYAN}Missing models:${RESET} ${missingModels.length}`);
  printModelList(missingModels);

  const shouldSync = await confirm({
    message: `${BOLDYELLOW}Add missing models to provider \"${provider.name}\"?${RESET}`,
    default: false,
  });

  if (!shouldSync) {
    console.log(`\n${DIM}Skipped syncing models.${RESET}`);
    return;
  }

  const latestConfig = await readConfigFileRaw();
  const latestProvider = findProvider(latestConfig, provider.name);

  if (!latestProvider) {
    throw new Error(`Provider \"${provider.name}\" was removed before sync could be applied.`);
  }

  const latestModels = Array.isArray(latestProvider.models) ? latestProvider.models : [];
  const latestModelSet = new Set(latestModels);
  const modelsToAdd = missingModels.filter((model) => !latestModelSet.has(model));

  if (modelsToAdd.length === 0) {
    console.log(`\n${DIM}No missing models remain after reloading config.${RESET}`);
    return;
  }

  latestProvider.models = [...latestModels, ...modelsToAdd];
  await backupConfigFile();
  await writeConfigFile(latestConfig);

  console.log(`\n${GREEN}✓ Added models to ${provider.name}:${RESET}`);
  printModelList(modelsToAdd);
}

async function syncUnavailableModels(provider: ProviderConfig, remoteModels: string[]): Promise<void> {
  if (!shouldSyncRemoteModels(remoteModels)) {
    return;
  }

  const configuredModels = Array.isArray(provider.models) ? provider.models : [];
  const remoteSet = new Set(remoteModels);
  const unavailableModels = configuredModels.filter((model) => !remoteSet.has(model));

  if (unavailableModels.length === 0) {
    console.log(`\n${DIM}No unavailable configured models to remove.${RESET}`);
    return;
  }

  console.log(`\n${BOLDCYAN}Configured models not returned by API:${RESET} ${unavailableModels.length}`);
  printModelList(unavailableModels);

  const shouldRemove = await confirm({
    message: `${BOLDYELLOW}Remove unavailable models from provider \"${provider.name}\"?${RESET}`,
    default: false,
  });

  if (!shouldRemove) {
    console.log(`\n${DIM}Skipped removing unavailable models.${RESET}`);
    return;
  }

  const latestConfig = await readConfigFileRaw();
  const latestProvider = findProvider(latestConfig, provider.name);

  if (!latestProvider) {
    throw new Error(`Provider \"${provider.name}\" was removed before sync could be applied.`);
  }

  const latestModels = Array.isArray(latestProvider.models) ? latestProvider.models : [];
  const unavailableSet = new Set(unavailableModels);
  const remoteModelSet = new Set(remoteModels);
  const modelsToRemove = latestModels.filter(
    (model) => unavailableSet.has(model) && !remoteModelSet.has(model)
  );

  if (modelsToRemove.length === 0) {
    console.log(`\n${DIM}No unavailable models remain after reloading config.${RESET}`);
    return;
  }

  latestProvider.models = latestModels.filter((model) => !modelsToRemove.includes(model));
  await backupConfigFile();
  await writeConfigFile(latestConfig);

  console.log(`\n${GREEN}✓ Removed models from ${provider.name}:${RESET}`);
  printModelList(modelsToRemove);
}

async function syncModelDifferences(provider: ProviderConfig, remoteModels: string[]): Promise<void> {
  await syncMissingModels(provider, remoteModels);

  const latestConfig = (await readConfigFile()) as ConfigWithProviders;
  const latestProvider = findProvider(latestConfig, provider.name) || provider;
  await syncUnavailableModels(latestProvider, remoteModels);
}

function validateParsedModels(provider: ProviderConfig, endpoint: ResolvedEndpoint, models: string[]): void {
  if (!hasRemoteModels(models)) {
    return;
  }

  if (endpoint.kind === "codex") {
    const invalidModel = models.find((model) => model.includes("/"));
    if (invalidModel) {
      throw new Error(`Unexpected Codex model slug format: ${invalidModel}`);
    }
  }
}

function uniqueSortedModels(models: string[]): string[] {
  return Array.from(new Set(models)).sort((a, b) => a.localeCompare(b));
}

export async function runModelGet(providerName: string, options: { listPath?: string; idPath?: string; stripPrefix?: string } = {}): Promise<void> {
  try {
    assertConfigExists();
    const config = (await readConfigFile()) as ConfigWithProviders;

    if (!Array.isArray(config.Providers) || config.Providers.length === 0) {
      throw new Error("No providers are configured.");
    }

    const provider = findProvider(config, providerName);
    if (!provider) {
      throw new Error(`Provider \"${providerName}\" not found in configuration.`);
    }

    const endpoint = resolveModelsEndpoint(provider);
    const apiKey =
      endpoint.kind === "codex" ? undefined : getRequestApiKey(provider);
    if (endpoint.kind !== "codex" && !apiKey) {
      throw new Error(getMissingApiKeyMessage(provider));
    }

    // Merge CLI options with provider config
    if (options.listPath || options.idPath || options.stripPrefix) {
      provider.models_response_format = {
        ...provider.models_response_format,
        ...(options.listPath ? { listPath: options.listPath } : {}),
        ...(options.idPath ? { idPath: options.idPath } : {}),
        ...(options.stripPrefix ? { stripPrefix: options.stripPrefix } : {}),
      };
    }

    const remoteModels = uniqueSortedModels(await fetchRemoteModels(apiKey, provider, endpoint));

    validateParsedModels(provider, endpoint, remoteModels);
    printSuccess(provider, endpoint, remoteModels);
    await syncModelDifferences(provider, remoteModels);
  } catch (error: any) {
    console.error(`\n${YELLOW}Error:${RESET} ${error.message}`);
    process.exit(1);
  }
}

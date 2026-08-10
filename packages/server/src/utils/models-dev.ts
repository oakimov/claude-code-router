/**
 * Server-owned models.dev lookup for gateway model discovery.
 *
 * The CLI has a separate models.dev lifecycle for `ccr codex-config`. The
 * server intentionally owns this client and cache because it can be deployed
 * and run without the CLI process.
 */

const DEFAULT_URL = "https://models.dev/api.json";
const DEFAULT_TIMEOUT_MS = 10_000;
const DEFAULT_CACHE_TTL_MS = 60 * 60 * 1_000;
const FAILURE_RETRY_MS = 60_000;

const GATEWAY_EFFORT_LEVELS = new Set([
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
]);

interface ModelsDevRow {
  key: string;
  provider: string;
  raw: Record<string, unknown>;
}

export interface ModelsDevModelInfo {
  key: string;
  nativeProvider: string;
  id: string;
  name: string;
  description: string;
  family: string;
  context: number;
  input: number;
  output: number;
  reasoning: boolean;
  effortLevels: string[];
}

export interface ModelsDevIndex {
  byModelName: Map<string, ModelsDevRow[]>;
  size: number;
}

export interface ModelsDevCatalogSource {
  get(): Promise<ModelsDevIndex | null>;
}

interface ModelsDevCacheOptions {
  load?: () => Promise<ModelsDevIndex | null>;
  ttlMs?: number;
  onError?: (error: unknown) => void;
  now?: () => number;
}

/**
 * models.dev flattens `base_model` inheritance in api.json, so wrapper and
 * reseller providers can expose the same id. These family rules identify the
 * native/base-model provider instead of relying on API object order.
 */
const NATIVE_PROVIDER_RULES: Array<{
  pattern: RegExp;
  provider: string;
}> = [
  { pattern: /^claude(?:-|$)/, provider: "anthropic" },
  { pattern: /^(?:gpt|chatgpt|codex|o\d)(?:-|$)/, provider: "openai" },
  {
    pattern: /^(?:codestral|devstral|magistral|ministral|mistral|mixtral|pixtral)(?:-|$)/,
    provider: "mistral",
  },
  { pattern: /^glm(?:-|$)/, provider: "zhipuai" },
  { pattern: /^gemini(?:-|$)/, provider: "google" },
  { pattern: /^deepseek(?:-|$)/, provider: "deepseek" },
  { pattern: /^grok(?:-|$)/, provider: "xai" },
  { pattern: /^command(?:-|$)/, provider: "cohere" },
  { pattern: /^qwen(?:-|$)/, provider: "alibaba" },
];

function isDisabled(): boolean {
  const value = process.env.CCR_MODELSDEV_DISABLE?.toLowerCase();
  return value === "1" || value === "true" || value === "yes";
}

/**
 * Normalize only the model-name side of CCR's `provider,model_name` id.
 * Provider prefixes (CCR or vendor-style `vendor/model`) never participate in
 * metadata matching.
 */
export function normalizeModelsDevModelName(value: string): string {
  let name = value.trim();
  const comma = name.indexOf(",");
  if (comma >= 0) name = name.slice(comma + 1);
  const slash = name.lastIndexOf("/");
  if (slash >= 0) name = name.slice(slash + 1);
  return name.replace(/\[1m\]$/i, "").trim().toLowerCase();
}

function positiveSafeInteger(value: unknown): number {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
}

function parseEffortLevels(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const levels: string[] = [];
  for (const option of value) {
    if (!option || typeof option !== "object") continue;
    const candidate = option as { type?: unknown; values?: unknown };
    if (candidate.type !== "effort" || !Array.isArray(candidate.values)) continue;
    for (const rawLevel of candidate.values) {
      const level = String(rawLevel || "").toLowerCase();
      if (GATEWAY_EFFORT_LEVELS.has(level) && !levels.includes(level)) {
        levels.push(level);
      }
    }
  }
  return levels;
}

function nativeProviderFor(rows: ModelsDevRow[], modelName: string): string | null {
  const identities = [
    modelName,
    ...rows.map((row) =>
      typeof row.raw.family === "string" ? row.raw.family.toLowerCase() : ""
    ),
  ].filter(Boolean);
  for (const identity of identities) {
    const rule = NATIVE_PROVIDER_RULES.find(({ pattern }) => pattern.test(identity));
    if (rule && rows.some((row) => row.provider === rule.provider)) {
      return rule.provider;
    }
  }

  // A unique row cannot be confused with a reseller duplicate. For duplicate
  // ids without a known native family, omit enrichment rather than selecting
  // an arbitrary provider and advertising provider-specific capabilities.
  return rows.length === 1 ? rows[0].provider : null;
}

function toInfo(row: ModelsDevRow): ModelsDevModelInfo {
  const raw = row.raw;
  const limit =
    raw.limit && typeof raw.limit === "object"
      ? (raw.limit as Record<string, unknown>)
      : {};
  return {
    key: row.key,
    nativeProvider: row.provider,
    id: typeof raw.id === "string" ? raw.id : normalizeModelsDevModelName(row.key),
    name: typeof raw.name === "string" ? raw.name : "",
    description: typeof raw.description === "string" ? raw.description : "",
    family: typeof raw.family === "string" ? raw.family.toLowerCase() : "",
    context: positiveSafeInteger(limit.context),
    input: positiveSafeInteger(limit.input),
    output: positiveSafeInteger(limit.output),
    reasoning: raw.reasoning === true,
    effortLevels: parseEffortLevels(raw.reasoning_options),
  };
}

export function buildModelsDevIndex(catalog: Record<string, unknown>): ModelsDevIndex {
  const byModelName = new Map<string, ModelsDevRow[]>();
  let size = 0;

  for (const [providerId, providerValue] of Object.entries(catalog || {})) {
    if (!providerValue || typeof providerValue !== "object") continue;
    const provider = providerValue as Record<string, unknown>;
    if (!provider.models || typeof provider.models !== "object") continue;

    for (const [modelKey, modelValue] of Object.entries(
      provider.models as Record<string, unknown>
    )) {
      if (!modelValue || typeof modelValue !== "object") continue;
      const raw = modelValue as Record<string, unknown>;
      const rawId = typeof raw.id === "string" ? raw.id : modelKey;
      const modelName = normalizeModelsDevModelName(rawId);
      if (!modelName) continue;

      const row: ModelsDevRow = {
        key: `${providerId}/${modelKey}`,
        provider: providerId.toLowerCase(),
        raw,
      };
      const rows = byModelName.get(modelName) || [];
      rows.push(row);
      byModelName.set(modelName, rows);
      size += 1;
    }
  }

  return { byModelName, size };
}

/** Match by model name only and return metadata from the native provider. */
export function lookupNativeModelsDevModel(
  index: ModelsDevIndex | null,
  modelName: string
): ModelsDevModelInfo | null {
  if (!index) return null;
  const normalized = normalizeModelsDevModelName(modelName);
  if (!normalized) return null;
  const rows = index.byModelName.get(normalized);
  if (!rows?.length) return null;
  const nativeProvider = nativeProviderFor(rows, normalized);
  if (!nativeProvider) return null;
  const nativeRow = rows.find((row) => row.provider === nativeProvider);
  return nativeRow ? toInfo(nativeRow) : null;
}

export async function fetchModelsDevCatalog(): Promise<ModelsDevIndex | null> {
  if (isDisabled()) return null;

  const url = process.env.CCR_MODELSDEV_URL || DEFAULT_URL;
  const timeout = Number(process.env.CCR_MODELSDEV_TIMEOUT) || DEFAULT_TIMEOUT_MS;
  const response = await fetch(url, {
    headers: {
      accept: "application/json",
      "user-agent": "claude-code-router-server (+https://models.dev)",
    },
    signal: AbortSignal.timeout(timeout),
  });
  if (!response.ok) throw new Error(`models.dev returned HTTP ${response.status}`);
  return buildModelsDevIndex((await response.json()) as Record<string, unknown>);
}

/**
 * Process-local server cache. A refresh failure keeps the last good catalog,
 * while a cold failure degrades `/v1/models` to its base routing entries.
 */
export class ModelsDevCatalogCache implements ModelsDevCatalogSource {
  private readonly load: () => Promise<ModelsDevIndex | null>;
  private readonly ttlMs: number;
  private readonly onError?: (error: unknown) => void;
  private readonly now: () => number;
  private current: ModelsDevIndex | null = null;
  private expiresAt = 0;
  private inFlight: Promise<ModelsDevIndex | null> | null = null;

  constructor(options: ModelsDevCacheOptions = {}) {
    this.load = options.load || fetchModelsDevCatalog;
    this.ttlMs = options.ttlMs || DEFAULT_CACHE_TTL_MS;
    this.onError = options.onError;
    this.now = options.now || Date.now;
  }

  get(): Promise<ModelsDevIndex | null> {
    if (this.now() < this.expiresAt) return Promise.resolve(this.current);
    if (this.inFlight) return this.inFlight;
    this.inFlight = this.refresh().finally(() => {
      this.inFlight = null;
    });
    return this.inFlight;
  }

  private async refresh(): Promise<ModelsDevIndex | null> {
    try {
      const next = await this.load();
      if (next) this.current = next;
      this.expiresAt = this.now() + this.ttlMs;
    } catch (error) {
      this.expiresAt = this.now() + Math.min(this.ttlMs, FAILURE_RETRY_MS);
      this.onError?.(error);
    }
    return this.current;
  }
}

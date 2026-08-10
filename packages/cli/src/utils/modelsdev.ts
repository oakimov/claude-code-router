/**
 * Model metadata lookup from the models.dev community catalog.
 *
 * https://models.dev/api.json carries per-provider model rows with friendly
 * names, context limits, reasoning effort levels, and modalities — everything
 * the Codex catalog needs, in one document. It is fetched live on every
 * invocation; a failure is non-fatal and callers fall back to defaults.
 */

const DEFAULT_URL = "https://models.dev/api.json";
const DEFAULT_TIMEOUT_MS = 10_000;

/** Effort tokens Codex understands. Anything else in the catalog is dropped. */
const KNOWN_EFFORT = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "ultra",
];

export interface ModelDevInfo {
  /** Original "<provider>/<id>" key. */
  key: string;
  provider: string;
  name: string;
  /** Max context tokens (limit.context), 0 when unknown. */
  context: number;
  /** Max output tokens (limit.output), 0 when unknown. */
  output: number;
  reasoning: boolean;
  /** Ordered effort levels from reasoning_options, empty when not effort-based. */
  effortLevels: string[];
  toolCall: boolean;
  attachment: boolean;
  modalitiesIn: string[];
}

interface IndexRow {
  key: string;
  provider: string;
  raw: any;
}

/**
 * models.dev flattens base-model inheritance, so several providers can carry
 * the same model id. Keep this in sync with the server-side discovery client:
 * duplicate ids resolve to the native model-family provider, never to CCR's
 * configured provider name or whichever row happens to advertise efforts.
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

export interface ModelsDevIndex {
  /** Lower-cased bare model id → rows that share it. */
  byBareId: Map<string, IndexRow[]>;
  size: number;
}

function isDisabled(): boolean {
  const value = process.env.CCR_MODELSDEV_DISABLE;
  return value === "1" || value === "true" || value === "yes";
}

function normalizeModelName(value: string): string {
  const slash = value.lastIndexOf("/");
  return (slash === -1 ? value : value.slice(slash + 1)).trim().toLowerCase();
}

function parseEffortLevels(raw: any): string[] {
  if (!Array.isArray(raw)) return [];
  const out: string[] = [];
  for (const option of raw) {
    if (!option || typeof option !== "object") continue;
    if (option.type !== "effort" || !Array.isArray(option.values)) continue;
    for (const value of option.values) {
      const level = String(value || "").toLowerCase();
      if (KNOWN_EFFORT.includes(level) && !out.includes(level)) {
        out.push(level);
      }
    }
  }
  return out;
}

function toInfo(row: IndexRow): ModelDevInfo {
  const raw = row.raw || {};
  const limit = raw.limit || {};
  const modalities = raw.modalities || {};
  return {
    key: row.key,
    provider: row.provider,
    name: typeof raw.name === "string" ? raw.name : "",
    context: positiveSafeInteger(limit.context),
    output: positiveSafeInteger(limit.output),
    reasoning: Boolean(raw.reasoning),
    effortLevels: parseEffortLevels(raw.reasoning_options),
    toolCall: Boolean(raw.tool_call),
    attachment: Boolean(raw.attachment),
    modalitiesIn: Array.isArray(modalities.input)
      ? modalities.input.map((m: unknown) => String(m))
      : [],
  };
}

function positiveSafeInteger(value: unknown): number {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
}

export function buildModelsDevIndex(
  catalog: Record<string, any>
): ModelsDevIndex {
  const byBareId = new Map<string, IndexRow[]>();
  let size = 0;

  for (const [providerId, provider] of Object.entries(catalog || {})) {
    const models = (provider as any)?.models;
    if (!models || typeof models !== "object") continue;

    for (const [modelId, raw] of Object.entries(models)) {
      const key = `${providerId}/${modelId}`;
      const rawId =
        typeof (raw as any)?.id === "string" ? (raw as any).id : modelId;
      const bare = normalizeModelName(rawId);
      if (!bare) continue;
      const rows = byBareId.get(bare) || [];
      rows.push({ key, provider: providerId.toLowerCase(), raw });
      byBareId.set(bare, rows);
      size += 1;
    }
  }

  return { byBareId, size };
}

/**
 * Fetch and index the catalog. Returns null when disabled or unreachable —
 * a models.dev outage must never block catalog generation.
 */
export async function fetchModelsDevCatalog(): Promise<ModelsDevIndex | null> {
  if (isDisabled()) return null;

  const url = process.env.CCR_MODELSDEV_URL || DEFAULT_URL;
  const timeout = Number(process.env.CCR_MODELSDEV_TIMEOUT) || DEFAULT_TIMEOUT_MS;

  try {
    const res = await fetch(url, {
      headers: {
        accept: "application/json",
        // models.dev answers 403 to a bare Node user-agent.
        "user-agent": "claude-code-router (+https://models.dev)",
      },
      signal: AbortSignal.timeout(timeout),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`);
    }
    return buildModelsDevIndex((await res.json()) as Record<string, any>);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(
      `Warning: models.dev lookup failed (${message}). Using default model metadata.`
    );
    return null;
  }
}

function nativeProviderFor(rows: IndexRow[], modelName: string): string | null {
  if (rows.length === 1) return rows[0].provider;

  const identities = [
    modelName,
    ...rows.map((row) =>
      typeof row.raw?.family === "string" ? row.raw.family.toLowerCase() : ""
    ),
  ].filter(Boolean);
  for (const identity of identities) {
    const rule = NATIVE_PROVIDER_RULES.find(({ pattern }) =>
      pattern.test(identity)
    );
    if (rule && rows.some((row) => row.provider === rule.provider)) {
      return rule.provider;
    }
  }
  return null;
}

/**
 * Look up one model. `modelId` may be a bare id or already namespaced.
 * Returns null on a miss so the caller can apply its own defaults.
 */
export function lookupModel(
  index: ModelsDevIndex | null,
  modelId: string
): ModelDevInfo | null {
  if (!index || !modelId) return null;

  const modelName = normalizeModelName(modelId);
  const rows = index.byBareId.get(modelName);
  if (!rows?.length) return null;

  const nativeProvider = nativeProviderFor(rows, modelName);
  if (!nativeProvider) return null;
  const nativeRow = rows.find((row) => row.provider === nativeProvider);
  return nativeRow ? toInfo(nativeRow) : null;
}

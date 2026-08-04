/**
 * Claude Code's bundled model capability catalog (extracted from v2.1.220).
 *
 * Every model-dependent decision in the claude-auth impersonation path (beta
 * flags, effort support, thinking shape, max_tokens ceiling) is driven by
 * `capabilities` membership here, never by matching on the model name.
 */
export interface ClaudeModelCatalogEntry {
  /** Context window in tokens, or undefined when the model predates 1M-context routing. */
  window?: number;
  /** Router carries a native (unconditional) 1M-token context window. */
  nativeOneMillion: boolean;
  /** Model can opt into a 1M-token window via the context-1m-2025-08-07 beta. */
  supportsOneMillionBeta: boolean;
  /** Model id accepts an explicit "[1m]" wire suffix. */
  supportsOneMillionSuffix: boolean;
  maxOutputTokens: { default: number; upper: number };
  defaultEffort?: string;
  capabilities: string[];
}

export const CLAUDE_MODEL_CATALOG: Record<string, ClaudeModelCatalogEntry> = {
  "claude-3-5-haiku": {
    nativeOneMillion: false,
    supportsOneMillionBeta: false,
    supportsOneMillionSuffix: false,
    maxOutputTokens: { default: 8192, upper: 8192 },
    capabilities: [],
  },
  "claude-haiku-4-5": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: false,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 32000, upper: 64000 },
    capabilities: ["context_management"],
  },
  "claude-3-5-sonnet": {
    nativeOneMillion: false,
    supportsOneMillionBeta: false,
    supportsOneMillionSuffix: false,
    maxOutputTokens: { default: 8192, upper: 8192 },
    capabilities: [],
  },
  "claude-3-7-sonnet": {
    nativeOneMillion: false,
    supportsOneMillionBeta: false,
    supportsOneMillionSuffix: false,
    maxOutputTokens: { default: 32000, upper: 64000 },
    capabilities: [],
  },
  "claude-sonnet-4-0": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 32000, upper: 64000 },
    capabilities: ["context_management"],
  },
  "claude-sonnet-4-5": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 32000, upper: 64000 },
    capabilities: ["context_management"],
  },
  "claude-sonnet-4-6": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 32000, upper: 128000 },
    capabilities: ["effort", "max_effort", "adaptive_thinking", "context_management"],
  },
  "claude-sonnet-5": {
    window: 1e6,
    nativeOneMillion: true,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: false,
    maxOutputTokens: { default: 64000, upper: 128000 },
    defaultEffort: "high",
    capabilities: [
      "effort",
      "max_effort",
      "xhigh_effort",
      "adaptive_thinking",
      "mid_conv_system",
      "context_management",
    ],
  },
  "claude-opus-4-0": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: false,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 32000, upper: 32000 },
    capabilities: ["context_management"],
  },
  "claude-opus-4-1": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: false,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 32000, upper: 32000 },
    capabilities: ["context_management"],
  },
  "claude-opus-4-5": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: false,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 32000, upper: 64000 },
    capabilities: ["context_management"],
  },
  "claude-opus-4-6": {
    window: 200000,
    nativeOneMillion: false,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 64000, upper: 128000 },
    capabilities: ["effort", "max_effort", "adaptive_thinking", "context_management"],
  },
  "claude-opus-4-7": {
    window: 1e6,
    nativeOneMillion: true,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 64000, upper: 128000 },
    defaultEffort: "xhigh",
    capabilities: [
      "effort",
      "max_effort",
      "xhigh_effort",
      "adaptive_thinking",
      "context_management",
      "fast_mode",
    ],
  },
  "claude-opus-4-8": {
    window: 1e6,
    nativeOneMillion: true,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 64000, upper: 128000 },
    defaultEffort: "high",
    capabilities: [
      "effort",
      "max_effort",
      "xhigh_effort",
      "adaptive_thinking",
      "mid_conv_system",
      "context_management",
      "fast_mode",
      "lean_prompt",
    ],
  },
  "claude-opus-5": {
    window: 1e6,
    nativeOneMillion: true,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: true,
    maxOutputTokens: { default: 64000, upper: 128000 },
    defaultEffort: "high",
    capabilities: [
      "effort",
      "max_effort",
      "xhigh_effort",
      "adaptive_thinking",
      "mid_conv_system",
      "context_management",
      "fast_mode",
      "lean_prompt",
      "refusal_fallback",
      "opus_5_prompt_bundle",
    ],
  },
  "claude-fable-5": {
    window: 1e6,
    nativeOneMillion: true,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: false,
    maxOutputTokens: { default: 64000, upper: 128000 },
    defaultEffort: "high",
    capabilities: [
      "effort",
      "max_effort",
      "xhigh_effort",
      "adaptive_thinking",
      "rejects_disabled_thinking",
      "mid_conv_system",
      "context_management",
      "lean_prompt",
      "fable_5_mitigations",
      "refusal_fallback",
    ],
  },
  "claude-mythos-5": {
    window: 1e6,
    nativeOneMillion: true,
    supportsOneMillionBeta: true,
    supportsOneMillionSuffix: false,
    maxOutputTokens: { default: 64000, upper: 128000 },
    capabilities: [],
  },
};

/** The literal wire marker CCR/Claude Code use to request a model's 1M-token context window. */
const ONE_MILLION_CONTEXT_MARKER = /\[1m\]/gi;

/** Strip the "[1m]" wire marker, reporting whether it was present. */
export function stripOneMillionContextMarker(modelId: string | undefined): {
  modelId: string;
  requestedOneMillion: boolean;
} {
  const raw = modelId || "";
  const requestedOneMillion = ONE_MILLION_CONTEXT_MARKER.test(raw);
  ONE_MILLION_CONTEXT_MARKER.lastIndex = 0;
  return { modelId: raw.replace(ONE_MILLION_CONTEXT_MARKER, ""), requestedOneMillion };
}

/**
 * Normalize a routed model id (CCR "provider,model" selector, an optional
 * "[1m]" marker, and Anthropic's dated model-id suffix) down to a catalog
 * key, e.g. "anthropic,claude-opus-5-20260101[1m]" -> "claude-opus-5".
 */
export function normalizeModelIdForCatalog(modelId: string | undefined): string {
  if (!modelId) return "";
  let id = modelId.trim();
  const commaIndex = id.indexOf(",");
  if (commaIndex >= 0) id = id.slice(commaIndex + 1);
  id = stripOneMillionContextMarker(id).modelId.trim();
  // Anthropic model ids carry a dated or "@YYYYMMDD" revision suffix that is
  // not part of the catalog's family key.
  id = id.replace(/[-@]\d{8}$/, "");
  return id.toLowerCase();
}

/** Look up a model's catalog entry. Unknown models resolve to undefined (minimal capability set). */
export function lookupClaudeModelCatalogEntry(
  modelId: string | undefined
): ClaudeModelCatalogEntry | undefined {
  const key = normalizeModelIdForCatalog(modelId);
  if (!key) return undefined;
  return CLAUDE_MODEL_CATALOG[key];
}

export function catalogEntryHasCapability(
  entry: ClaudeModelCatalogEntry | undefined,
  capability: string
): boolean {
  return entry?.capabilities.includes(capability) ?? false;
}

/**
 * Whether extended-thinking betas apply to this model. The catalog has no
 * dedicated "supports_thinking" column, but every entry that predates the
 * Claude 4 thinking generation (claude-3-5-haiku, claude-3-5-sonnet,
 * claude-3-7-sonnet) — plus the not-yet-classified claude-mythos-5 — carries
 * an empty capability set; every thinking-capable model carries at least
 * "context_management". A non-empty capability set is therefore a reliable,
 * self-documenting proxy that also degrades unknown models to "no thinking"
 * rather than guessing.
 */
export function catalogEntrySupportsThinking(
  entry: ClaudeModelCatalogEntry | undefined
): boolean {
  return (entry?.capabilities.length ?? 0) > 0;
}

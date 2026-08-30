import type { CachePrefixDiff } from "./cache-prefix-debug";

/**
 * Provider-family-specific cache contracts. OpenAI-style message prefix +
 * prompt_cache_key is only one of several; Cursor uses conversation/lifecycle,
 * Anthropic uses ephemeral breakpoints, Gemini uses cachedContent resources.
 */
export type CacheFamily =
  | "openai_prefix"
  | "anthropic_ephemeral"
  | "gemini_cached_content"
  | "cursor_conversation"
  | "deepseek_prefix"
  | "unknown";

export type CacheVerdict =
  | "cold"
  | "warm-start"
  | "hit"
  | "partial"
  | "expected-miss"
  | "unexpected-miss"
  | "unknown";

export type CachePrediction = {
  family: CacheFamily;
  firstTurn: boolean;
  /** Provider-specific: was a hit expected this turn? */
  predictedHit: boolean;
  /** Why we predicted miss/hit (for logs). */
  reason: string;
  /** OpenAI-style prefix intactness when available (diagnostic for Cursor). */
  prefixIntact?: boolean;
  firstDivergencePath?: string;
  approxPrefixTokensLost?: number;
  conversationId?: string;
  conversationIdSource?: string;
  /** Cursor lifecycle action when family is cursor_conversation. */
  lifecycleAction?: string;
  hostPrefixIntact?: boolean;
};

export type CursorCacheLifecycle = {
  sessionKey?: string;
  action: string;
  reason?: string;
};

export type ResolveCacheFamilyInput = {
  provider?: string;
  model?: string;
  body?: Record<string, any> | null;
  cursorLifecycle?: CursorCacheLifecycle | null;
};

const CURSOR_PROVIDER_RE = /\bcursor\b/i;
const DEEPSEEK_PROVIDER_RE = /deepseek/i;
const ANTHROPIC_PROVIDER_RE = /anthropic|claude/i;
const GEMINI_PROVIDER_RE = /gemini|google|vertex/i;

function hasEphemeralBreakpoint(value: unknown, depth = 0): boolean {
  if (!value || typeof value !== "object" || depth > 8) return false;
  const cc = (value as any).cache_control;
  if (cc && typeof cc === "object" && cc.type === "ephemeral") return true;
  if (Array.isArray(value)) {
    return value.some((item) => hasEphemeralBreakpoint(item, depth + 1));
  }
  for (const child of Object.values(value as Record<string, unknown>)) {
    if (hasEphemeralBreakpoint(child, depth + 1)) return true;
  }
  return false;
}

function geminiCachedContentName(
  body: Record<string, any> | null | undefined
): string | undefined {
  if (!body || typeof body !== "object") return undefined;
  const name = body.cachedContent ?? body.cached_content;
  return typeof name === "string" && name ? name : undefined;
}

/**
 * Prefer explicit provider/transformer id; fall back to body sniffing.
 */
export function resolveCacheFamily(input: ResolveCacheFamilyInput): CacheFamily {
  const provider = String(input.provider || "");
  if (input.cursorLifecycle || CURSOR_PROVIDER_RE.test(provider)) {
    return "cursor_conversation";
  }
  if (DEEPSEEK_PROVIDER_RE.test(provider)) return "deepseek_prefix";
  if (ANTHROPIC_PROVIDER_RE.test(provider)) return "anthropic_ephemeral";
  if (GEMINI_PROVIDER_RE.test(provider)) return "gemini_cached_content";

  const body = input.body;
  if (body && typeof body === "object") {
    if (geminiCachedContentName(body)) return "gemini_cached_content";
    if (hasEphemeralBreakpoint(body)) return "anthropic_ephemeral";
    if (
      Array.isArray(body.messages) ||
      Array.isArray(body.input) ||
      typeof body.prompt_cache_key === "string"
    ) {
      return "openai_prefix";
    }
  }
  return "unknown";
}

/**
 * Join what we predicted against what upstream reported.
 * Labels intentionally match the historical sse-debug-tap strings.
 */
export function classifyCacheOutcome(
  prediction: CachePrediction | null | undefined,
  hitRatio: number | undefined
): CacheVerdict {
  if (hitRatio === undefined) return "unknown";
  if (!prediction || prediction.firstTurn) {
    return hitRatio > 0 ? "warm-start" : "cold";
  }
  if (prediction.predictedHit) {
    return hitRatio > 0 ? "hit" : "unexpected-miss";
  }
  return hitRatio > 0 ? "partial" : "expected-miss";
}

function fromPrefixDiff(
  family: CacheFamily,
  diff: CachePrefixDiff | null | undefined,
  extras?: Partial<CachePrediction>
): CachePrediction {
  if (!diff) {
    return {
      family,
      firstTurn: true,
      predictedHit: false,
      reason: extras?.reason || "no-prefix-diff",
      ...extras,
    };
  }
  const predictedHit = !diff.firstTurn && diff.prefixIntact;
  let reason: string;
  if (diff.firstTurn) reason = "first-turn";
  else if (diff.prefixIntact) reason = "prefix-intact";
  else reason = diff.firstDivergencePath || diff.change || "prefix-broken";
  return {
    family,
    firstTurn: diff.firstTurn,
    predictedHit,
    reason: extras?.reason || reason,
    prefixIntact: diff.prefixIntact,
    firstDivergencePath: diff.firstDivergencePath,
    approxPrefixTokensLost: diff.approxPrefixTokensLost,
    conversationId: diff.conversationId,
    conversationIdSource: diff.conversationIdSource,
    ...extras,
  };
}

/** OpenAI Chat/Responses, Codex, Zen, Cerebras, OpenRouter, Mistral, xAI. */
export function predictOpenAiPrefix(
  diff: CachePrefixDiff | null | undefined
): CachePrediction {
  return fromPrefixDiff("openai_prefix", diff);
}

/** DeepSeek uses the same outbound prefix contract; hit tokens are separate. */
export function predictDeepSeekPrefix(
  diff: CachePrefixDiff | null | undefined
): CachePrediction {
  return fromPrefixDiff("deepseek_prefix", diff);
}

/**
 * Anthropic caches only when ephemeral breakpoints exist and the covered
 * prefix stayed intact. No breakpoints → predicted miss.
 */
export function predictAnthropicEphemeral(
  diff: CachePrefixDiff | null | undefined,
  body?: Record<string, any> | null
): CachePrediction {
  // When body is supplied, require at least one ephemeral breakpoint.
  if (body !== undefined && body !== null && !hasEphemeralBreakpoint(body)) {
    return {
      family: "anthropic_ephemeral",
      firstTurn: diff?.firstTurn ?? true,
      predictedHit: false,
      reason: "no-ephemeral-breakpoints",
      prefixIntact: diff?.prefixIntact,
      firstDivergencePath: diff?.firstDivergencePath,
      approxPrefixTokensLost: diff?.approxPrefixTokensLost,
      conversationId: diff?.conversationId,
      conversationIdSource: diff?.conversationIdSource,
    };
  }

  return fromPrefixDiff("anthropic_ephemeral", diff, {
    reason: diff?.firstTurn
      ? "first-turn"
      : diff?.prefixIntact
        ? "ephemeral-prefix-intact"
        : diff?.firstDivergencePath || "ephemeral-prefix-broken",
  });
}

/** Track last seen Gemini cachedContent name per conversation for rotation. */
const geminiCachedNames = new Map<string, string>();
const MAX_GEMINI_NAMES = 256;

export function __resetGeminiCachedContentNamesForTests(): void {
  geminiCachedNames.clear();
}

function rememberGeminiName(conversationId: string, name: string): string | undefined {
  const previous = geminiCachedNames.get(conversationId);
  geminiCachedNames.delete(conversationId);
  geminiCachedNames.set(conversationId, name);
  while (geminiCachedNames.size > MAX_GEMINI_NAMES) {
    const oldest = geminiCachedNames.keys().next().value;
    if (!oldest) break;
    geminiCachedNames.delete(oldest);
  }
  return previous;
}

export function predictGeminiCachedContent(opts: {
  diff?: CachePrefixDiff | null;
  body?: Record<string, any> | null;
  conversationId?: string;
}): CachePrediction {
  const { diff, body } = opts;
  const name = geminiCachedContentName(body);
  const conversationId =
    opts.conversationId || diff?.conversationId || "gemini-anon";

  if (!name) {
    return {
      family: "gemini_cached_content",
      firstTurn: diff?.firstTurn ?? true,
      predictedHit: false,
      reason: "no-cached-content",
      prefixIntact: diff?.prefixIntact,
      conversationId,
      conversationIdSource: diff?.conversationIdSource,
    };
  }

  const previous = rememberGeminiName(conversationId, name);
  if (!previous) {
    return {
      family: "gemini_cached_content",
      firstTurn: true,
      predictedHit: false,
      reason: "first-cached-content",
      prefixIntact: diff?.prefixIntact,
      conversationId,
      conversationIdSource: diff?.conversationIdSource,
    };
  }
  if (previous !== name) {
    return {
      family: "gemini_cached_content",
      firstTurn: false,
      predictedHit: false,
      reason: "cached-content-rotated",
      prefixIntact: false,
      firstDivergencePath: "cachedContent",
      conversationId,
      conversationIdSource: diff?.conversationIdSource,
    };
  }
  // Same resource: still require host prefix intact when we have a diff.
  if (diff && !diff.firstTurn && !diff.prefixIntact) {
    return fromPrefixDiff("gemini_cached_content", diff, {
      reason: "cached-content-prefix-broken",
    });
  }
  return {
    family: "gemini_cached_content",
    firstTurn: false,
    predictedHit: true,
    reason: "cached-content-reused",
    prefixIntact: diff?.prefixIntact ?? true,
    conversationId,
    conversationIdSource: diff?.conversationIdSource,
  };
}

const CURSOR_HIT_ACTIONS = new Set(["resume-parked", "send-incremental"]);
const CURSOR_FIRST_ACTIONS = new Set(["send-full"]);

/**
 * Cursor has no prompt_cache_key. Prediction follows lifecycle:
 * resume/incremental → hit expected; retire/replay or fresh send → miss expected.
 */
export function predictCursorConversation(opts: {
  lifecycle?: CursorCacheLifecycle | null;
  diff?: CachePrefixDiff | null;
}): CachePrediction {
  const lifecycle = opts.lifecycle;
  const diff = opts.diff;
  const action = lifecycle?.action || "";
  const sessionKey = lifecycle?.sessionKey;

  if (!action) {
    // Fall back to OpenAI-style prefix if lifecycle was not stashed.
    return fromPrefixDiff("cursor_conversation", diff, {
      reason: "cursor-lifecycle-missing",
      hostPrefixIntact: diff?.prefixIntact,
    });
  }

  if (CURSOR_FIRST_ACTIONS.has(action) || action === "send-full") {
    return {
      family: "cursor_conversation",
      firstTurn: true,
      predictedHit: false,
      reason: lifecycle?.reason || action,
      lifecycleAction: action,
      conversationId: sessionKey || diff?.conversationId,
      conversationIdSource: sessionKey ? "session" : diff?.conversationIdSource,
      hostPrefixIntact: diff?.prefixIntact,
      prefixIntact: diff?.prefixIntact,
    };
  }

  if (CURSOR_HIT_ACTIONS.has(action)) {
    return {
      family: "cursor_conversation",
      firstTurn: false,
      predictedHit: true,
      reason: lifecycle?.reason || action,
      lifecycleAction: action,
      conversationId: sessionKey || diff?.conversationId,
      conversationIdSource: sessionKey ? "session" : diff?.conversationIdSource,
      hostPrefixIntact: diff?.prefixIntact,
      // Do not use host message prefix as Cursor prediction; keep as diagnostic.
      prefixIntact: true,
    };
  }

  // retire-and-replay-full and any unknown retirement-like action
  return {
    family: "cursor_conversation",
    firstTurn: false,
    predictedHit: false,
    reason: lifecycle?.reason || action || "cursor-retire",
    lifecycleAction: action,
    conversationId: sessionKey || diff?.conversationId,
    conversationIdSource: sessionKey ? "session" : diff?.conversationIdSource,
    hostPrefixIntact: diff?.prefixIntact,
    prefixIntact: false,
    firstDivergencePath: diff?.firstDivergencePath,
    approxPrefixTokensLost: diff?.approxPrefixTokensLost,
  };
}

/**
 * Build the right prediction for this outbound leg.
 */
export function buildCachePrediction(opts: {
  provider?: string;
  model?: string;
  body?: Record<string, any> | null;
  diff?: CachePrefixDiff | null;
  cursorLifecycle?: CursorCacheLifecycle | null;
  conversationId?: string;
}): CachePrediction {
  const family = resolveCacheFamily({
    provider: opts.provider,
    model: opts.model,
    body: opts.body,
    cursorLifecycle: opts.cursorLifecycle,
  });

  switch (family) {
    case "cursor_conversation":
      return predictCursorConversation({
        lifecycle: opts.cursorLifecycle,
        diff: opts.diff,
      });
    case "anthropic_ephemeral":
      return predictAnthropicEphemeral(opts.diff, opts.body);
    case "gemini_cached_content":
      return predictGeminiCachedContent({
        diff: opts.diff,
        body: opts.body,
        conversationId: opts.conversationId || opts.diff?.conversationId,
      });
    case "deepseek_prefix":
      return predictDeepSeekPrefix(opts.diff);
    case "openai_prefix":
      return predictOpenAiPrefix(opts.diff);
    default:
      return fromPrefixDiff("unknown", opts.diff, { reason: "unknown-family" });
  }
}

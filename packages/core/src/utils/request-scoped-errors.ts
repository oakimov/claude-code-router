/**
 * Request-scoped error classification (CLIProxyAPI `request-scoped-errors` port).
 *
 * The default fallback classifier (`isFallbackEligibleError`) only looks at
 * HTTP status codes. That misclassifies two cases:
 * - a retryable status carrying a terminal body (e.g. 429 with
 *   `context_length_exceeded` — retrying another model with the same prompt
 *   fails the same way, so stop immediately);
 * - a terminal status carrying a retryable body (e.g. 400 with an overload /
 *   quota message — worth trying the next fallback model).
 *
 * Rules are evaluated in order; provider rules take precedence over global
 * rules. A rule matches when its `status` (when set) equals the error status
 * AND its body patterns (when set) hit. Within the body groups, any `match`
 * substring (case-insensitive) OR any `match_regex` hit matches.
 *
 * Config shapes (all accepted everywhere):
 * - provider entry: `request_scoped_errors` / `requestScopedErrors` /
 *   `request-scoped-errors` (the provider service stores the first spelling)
 * - top level: the same three spellings
 *
 * ```json
 * {
 *   "request_scoped_errors": [
 *     { "status": 400, "match": ["maximum_context_length"], "action": "stop" },
 *     { "status": 400, "match": ["server_is_overloaded"], "action": "continue" }
 *   ]
 * }
 * ```
 */
import { stripOneMillionContextMarker } from "./claude-model-catalog";

export type RequestScopedErrorAction =
  | "stop"
  | "stop-and-cooldown"
  | "continue"
  | "continue-and-cooldown";

export interface RequestScopedErrorRule {
  status?: number;
  match?: string[];
  match_regex?: string[];
  matchRegex?: string[];
  "match-regex"?: string[];
  action: RequestScopedErrorAction;
  cooldown_seconds?: number;
  cooldownSeconds?: number;
}

export interface NormalizedScopedErrorRule {
  status?: number;
  /** Lowercased substring patterns. */
  match: string[];
  matchRegex: RegExp[];
  action: RequestScopedErrorAction;
  cooldownSeconds: number;
}

/** A config rule rejected by validation; `index` is -1 for a non-array list. */
export interface ScopedErrorRuleIssue {
  index: number;
  reason: string;
}

/** Default cooldown for `*-and-cooldown` actions (matches legacy 60s transient). */
export const DEFAULT_SCOPED_ERROR_COOLDOWN_SECONDS = 60;

/** Accepted spellings of the rule-list key, canonical first. */
export const SCOPED_ERROR_RULE_KEYS = [
  "request_scoped_errors",
  "requestScopedErrors",
  "request-scoped-errors",
] as const;

const ACTIONS: ReadonlySet<string> = new Set<RequestScopedErrorAction>([
  "stop",
  "stop-and-cooldown",
  "continue",
  "continue-and-cooldown",
]);

const REGEX_KEYS = ["match_regex", "matchRegex", "match-regex"] as const;
const COOLDOWN_KEYS = ["cooldown_seconds", "cooldownSeconds"] as const;

/**
 * Every key a rule may carry. An unknown key is almost always a misspelled
 * pattern key (`matches`, `regex`); ignoring it would widen the rule to every
 * body, so such rules are rejected instead.
 */
const RULE_KEYS: ReadonlySet<string> = new Set<string>([
  "status",
  "match",
  ...REGEX_KEYS,
  "action",
  ...COOLDOWN_KEYS,
]);

/** A pattern list, or the rejection reason when the value is not one. */
function readPatternList(
  value: unknown,
  key: string
): string[] | { error: string } {
  if (value === undefined) return [];
  if (
    !Array.isArray(value) ||
    value.some((entry) => typeof entry !== "string" || entry.length === 0)
  ) {
    return { error: `"${key}" must be an array of non-empty strings` };
  }
  return value as string[];
}

/** Validate and compile one raw rule; a string is the rejection reason. */
function compileRule(entry: unknown): NormalizedScopedErrorRule | string {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
    return "rule must be an object";
  }
  const rule = entry as Record<string, unknown>;
  const unknownKeys = Object.keys(rule).filter((key) => !RULE_KEYS.has(key));
  if (unknownKeys.length) {
    return `unknown key(s): ${unknownKeys.map((key) => `"${key}"`).join(", ")}`;
  }
  const action = rule.action;
  if (typeof action !== "string" || !ACTIONS.has(action)) {
    return `"action" must be one of ${Array.from(ACTIONS).join(", ")}`;
  }
  const status = rule.status;
  if (
    status !== undefined &&
    (typeof status !== "number" ||
      !Number.isInteger(status) ||
      status < 100 ||
      status > 599)
  ) {
    return `"status" must be an integer HTTP status (100-599)`;
  }
  const match = readPatternList(rule.match, "match");
  if (!Array.isArray(match)) return match.error;
  const matchRegex: RegExp[] = [];
  for (const key of REGEX_KEYS) {
    const sources = readPatternList(rule[key], key);
    if (!Array.isArray(sources)) return sources.error;
    for (const source of sources) {
      try {
        matchRegex.push(new RegExp(source));
      } catch (error: unknown) {
        const detail = error instanceof Error ? error.message : String(error);
        return `invalid "${key}" pattern ${JSON.stringify(source)}: ${detail}`;
      }
    }
  }
  let cooldownSeconds = DEFAULT_SCOPED_ERROR_COOLDOWN_SECONDS;
  for (const key of COOLDOWN_KEYS) {
    const raw = rule[key];
    if (raw === undefined) continue;
    if (typeof raw !== "number" || !Number.isFinite(raw) || raw < 1) {
      return `"${key}" must be a number >= 1`;
    }
    cooldownSeconds = Math.floor(raw);
    break;
  }
  return {
    ...(status !== undefined ? { status } : {}),
    match: match.map((pattern) => pattern.toLowerCase()),
    matchRegex,
    action: action as RequestScopedErrorAction,
    cooldownSeconds,
  };
}

/**
 * Compile raw config rules. An invalid rule (unknown key, bad action, status,
 * pattern list, regex or cooldown) is dropped and reported via `onInvalid`;
 * it is never widened into a broader match.
 */
export function normalizeRequestScopedErrorRules(
  input: unknown,
  onInvalid?: (issue: ScopedErrorRuleIssue) => void
): NormalizedScopedErrorRule[] {
  if (input === undefined || input === null) return [];
  if (!Array.isArray(input)) {
    onInvalid?.({ index: -1, reason: "rule list must be an array" });
    return [];
  }
  const rules: NormalizedScopedErrorRule[] = [];
  input.forEach((entry, index) => {
    const compiled = compileRule(entry);
    if (typeof compiled === "string") {
      onInvalid?.({ index, reason: compiled });
      return;
    }
    rules.push(compiled);
  });
  return rules;
}

const compiledRuleCache = new WeakMap<object, NormalizedScopedErrorRule[]>();

/**
 * Compiled rules for a raw config list, cached per list object so validation
 * (and its `onInvalid` reports) runs once per config value, not per request.
 */
export function compiledScopedErrorRules(
  input: unknown,
  onInvalid?: (issue: ScopedErrorRuleIssue) => void
): NormalizedScopedErrorRule[] {
  if (!input || typeof input !== "object") {
    return normalizeRequestScopedErrorRules(input, onInvalid);
  }
  const cached = compiledRuleCache.get(input);
  if (cached) return cached;
  const compiled = normalizeRequestScopedErrorRules(input, onInvalid);
  compiledRuleCache.set(input, compiled);
  return compiled;
}

/** Read a rule list from a config carrier under any accepted key spelling. */
export function readScopedErrorRulesFromCarrier(carrier: unknown): unknown {
  if (!carrier || typeof carrier !== "object") return undefined;
  const record = carrier as Record<string, unknown>;
  for (const key of SCOPED_ERROR_RULE_KEYS) {
    if (record[key] !== undefined) return record[key];
  }
  return undefined;
}

/** Top-level (global) rule list from a config service, any key spelling. */
export function readGlobalScopedErrorRules(configService: {
  get(key: string): unknown;
}): unknown {
  for (const key of SCOPED_ERROR_RULE_KEYS) {
    const value = configService.get(key);
    if (value !== undefined) return value;
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Raw classification text.
//
// Client envelopes and logs only ever see sanitized error text cut to 240
// chars, which hides markers that sit later in a provider body. A producer
// that reads the raw upstream body attaches a bounded copy for matching only.
// It lives under a non-enumerable symbol key, so JSON serialization, object
// spread, pino's error serializer and sanitizeErrorForLog never see it.
// ---------------------------------------------------------------------------

/** Upper bound (chars) on the raw text kept for rule matching. */
export const MAX_SCOPED_ERROR_CLASSIFICATION_CHARS = 16_384;

const CLASSIFICATION_TEXT = Symbol.for(
  "ccr.requestScopedErrors.classificationText"
);

/** Attach the raw upstream error text an error was built from. */
export function attachScopedErrorClassificationText(
  error: unknown,
  rawText: unknown
): void {
  if (!error || typeof error !== "object") return;
  if (rawText === undefined || rawText === null) return;
  const text = String(rawText).slice(0, MAX_SCOPED_ERROR_CLASSIFICATION_CHARS);
  if (!text) return;
  Object.defineProperty(error, CLASSIFICATION_TEXT, {
    value: text,
    enumerable: false,
    writable: false,
    configurable: true,
  });
}

/** Raw classification text previously attached to an error, if any. */
export function readScopedErrorClassificationText(
  error: unknown
): string | undefined {
  if (!error || typeof error !== "object") return undefined;
  const value = (error as Record<symbol, unknown>)[CLASSIFICATION_TEXT];
  return typeof value === "string" ? value : undefined;
}

/** HTTP status for classification: explicit field first, then upstream snapshot. */
export function errorStatusForClassification(error: any): number | undefined {
  if (typeof error?.statusCode === "number") return error.statusCode;
  if (typeof error?.status === "number") return error.status;
  if (typeof error?.upstream?.status === "number") return error.upstream.status;
  return undefined;
}

/**
 * CCR's own catch-all codes. They say nothing about the upstream body, so a
 * pattern hitting them would match every provider failure.
 */
const GENERIC_ERROR_CODES: ReadonlySet<string> = new Set([
  "provider_response_error",
  "internal_error",
]);

/**
 * Searchable body text. Prefers the raw upstream text a producer attached;
 * without it, falls back to the message plus any captured upstream body. A
 * specific error code is appended either way. The error `type` (always
 * `api_error` for provider failures) is never included.
 */
export function errorTextForClassification(error: any): string {
  const parts: string[] = [];
  const raw = readScopedErrorClassificationText(error);
  if (raw !== undefined) {
    parts.push(raw);
  } else {
    if (typeof error?.message === "string" && error.message) {
      parts.push(error.message);
    }
    const upstreamBody = error?.upstream?.body;
    if (upstreamBody !== undefined) {
      try {
        parts.push(
          typeof upstreamBody === "string"
            ? upstreamBody
            : JSON.stringify(upstreamBody)
        );
      } catch {
        parts.push(String(upstreamBody));
      }
    }
  }
  if (
    typeof error?.code === "string" &&
    error.code &&
    !GENERIC_ERROR_CODES.has(error.code)
  ) {
    parts.push(error.code);
  }
  return parts.join("\n");
}

function ruleMatchesBody(
  rule: NormalizedScopedErrorRule,
  text: string
): boolean {
  if (rule.match.length === 0 && rule.matchRegex.length === 0) return true;
  const lower = text.toLowerCase();
  if (rule.match.some((pattern) => lower.includes(pattern))) return true;
  if (rule.matchRegex.some((re) => re.test(text))) return true;
  return false;
}

/** First matching rule, or undefined when nothing matches. */
export function matchScopedErrorRule(
  error: any,
  rules: NormalizedScopedErrorRule[]
): NormalizedScopedErrorRule | undefined {
  if (!rules.length) return undefined;
  const status = errorStatusForClassification(error);
  const text = errorTextForClassification(error);
  for (const rule of rules) {
    if (rule.status !== undefined && rule.status !== status) continue;
    if (!ruleMatchesBody(rule, text)) continue;
    return rule;
  }
  return undefined;
}

export type ScopedErrorDecision =
  | { kind: "default" }
  | {
      kind: "stop" | "continue";
      rule: NormalizedScopedErrorRule;
      cooldown: boolean;
    };

/**
 * Classify an upstream error against provider rules first, then global rules.
 * `stop` forces a terminal failure (no fallback); `continue` forces fallback
 * eligibility even when the status alone would not qualify.
 */
export function decideScopedError(
  error: any,
  providerRules: NormalizedScopedErrorRule[],
  globalRules: NormalizedScopedErrorRule[]
): ScopedErrorDecision {
  const rule =
    matchScopedErrorRule(error, providerRules) ??
    matchScopedErrorRule(error, globalRules);
  if (!rule) return { kind: "default" };
  const stop = rule.action === "stop" || rule.action === "stop-and-cooldown";
  return {
    kind: stop ? "stop" : "continue",
    rule,
    cooldown:
      rule.action === "stop-and-cooldown" ||
      rule.action === "continue-and-cooldown",
  };
}

// ---------------------------------------------------------------------------
// Cooldown registry for `*-and-cooldown` actions.
//
// `provider,model` → expiry timestamp. Only the fallback loop consults it: a
// cooled-down model is skipped as a fallback candidate (and never waited
// for) until the window lapses. The routed primary model is still attempted.
//
// One registry per scope (a server namespace's provider service): preset
// namespaces register their own providers, possibly under the same names.
// ---------------------------------------------------------------------------

/**
 * Cooldown key. The model drops its `[1m]` context marker: the primary route
 * strips it before dispatch while fallback entries keep it as configured,
 * yet both name the same upstream model.
 */
export function scopedErrorCooldownKey(
  providerName: string,
  model: string
): string {
  return `${providerName},${stripOneMillionContextMarker(model).modelId}`;
}

export class ScopedErrorCooldownRegistry {
  private readonly expiries = new Map<string, number>();

  put(
    providerName: string,
    model: string,
    cooldownSeconds: number = DEFAULT_SCOPED_ERROR_COOLDOWN_SECONDS,
    now: number = Date.now()
  ): void {
    if (!providerName || !model) return;
    this.prune(now);
    this.expiries.set(
      scopedErrorCooldownKey(providerName, model),
      now + Math.max(1, cooldownSeconds) * 1000
    );
  }

  isCooledDown(
    providerName: string,
    model: string,
    now: number = Date.now()
  ): boolean {
    const key = scopedErrorCooldownKey(providerName, model);
    const expiresAt = this.expiries.get(key);
    if (expiresAt === undefined) return false;
    if (expiresAt <= now) {
      this.expiries.delete(key);
      return false;
    }
    return true;
  }

  /** Drop every expired entry so keys that are never re-read cannot pile up. */
  prune(now: number = Date.now()): void {
    for (const [key, expiresAt] of this.expiries) {
      if (expiresAt <= now) this.expiries.delete(key);
    }
  }

  /** Number of tracked entries, expired or not. */
  get size(): number {
    return this.expiries.size;
  }
}

const cooldownRegistries = new WeakMap<object, ScopedErrorCooldownRegistry>();

/** The cooldown registry owned by `scope`, created on first use. */
export function scopedErrorCooldownsFor(
  scope: object
): ScopedErrorCooldownRegistry {
  let registry = cooldownRegistries.get(scope);
  if (!registry) {
    registry = new ScopedErrorCooldownRegistry();
    cooldownRegistries.set(scope, registry);
  }
  return registry;
}

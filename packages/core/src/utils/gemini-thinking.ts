import { ThinkLevel, UnifiedChatRequest } from "../types/llm";
import { resolveOutboundReasoningSummary } from "./reasoning-effort";

/**
 * Translate Claude Code's effort setting into the thinking dialect each Gemini
 * family expects.
 *
 * Claude Code is authoritative: the effort it sends (`output_config.effort`,
 * surfaced as `reasoning.effort`) decides how much the model thinks. The model
 * id is never rewritten — a user who configured `gemini-3-pro-low` keeps talking
 * to `gemini-3-pro-low`; only the request's thinkingConfig changes. Antigravity
 * tier-suffixed ids accept the whole level set for their family, so the suffix
 * caps nothing here (it does still select the upstream quota bucket).
 *
 * Two upstream dialects, never mixed — sending thinkingLevel and thinkingBudget
 * in one request is a documented 400:
 *
 *  - level  (Gemini 3+): thinkingLevel, per-family value set
 *  - budget (Gemini 2.5, Claude served through Antigravity): thinkingBudget in tokens
 */

/** Upstream `thinkingLevel` values, ascending. */
type GeminiThinkingLevel = "minimal" | "low" | "medium" | "high";

const LEVEL_ORDER: GeminiThinkingLevel[] = ["minimal", "low", "medium", "high"];

type ThinkingDialect =
  | { kind: "level"; levels: GeminiThinkingLevel[] }
  | { kind: "budget"; min: number; max: number; requireMin?: boolean }
  /** Unknown family: only ask for thought parts, never guess a level or budget. */
  | { kind: "includeOnly" }
  /** Image models must not carry thinking config at all. */
  | { kind: "none" };

export type GeminiThinkingConfig = {
  includeThoughts?: boolean;
  thinkingLevel?: GeminiThinkingLevel;
  thinkingBudget?: number;
};

function normalizeModelId(model: string): string {
  return String(model || "")
    .toLowerCase()
    .replace(/^models\//, "");
}

/**
 * Which thinking dialect a model speaks. Matching is by family prefix so tier
 * and preview suffixes (`-low`, `-high`, `-tiered`, `-preview`, `-agent`) all
 * resolve to their family.
 */
export function resolveThinkingDialect(model: string): ThinkingDialect {
  const id = normalizeModelId(model);
  if (!id) return { kind: "includeOnly" };

  // Image generation: thoughts are not returned and thinkingConfig is rejected
  // or ignored depending on the surface.
  if (id.includes("-image") || id.includes("imagen")) return { kind: "none" };

  // Claude models served through Antigravity: budget dialect, and a budget below
  // the Anthropic minimum is rejected — drop thinking rather than send it.
  if (id.startsWith("claude")) {
    return { kind: "budget", min: 1024, max: 64000, requireMin: true };
  }

  // Gemini 3 and later, including 3.1 / 3.5 / 3.6 minors and `gemini-pro-agent`.
  if (/^gemini-3(\.\d+)?\b/.test(id) || id.startsWith("gemini-pro-agent")) {
    // The original Gemini 3 Pro ships low|high only; later Pro minors add medium.
    if (/^gemini-3-pro\b/.test(id)) {
      return { kind: "level", levels: ["low", "high"] };
    }
    // Flash and Flash-Lite additionally accept minimal.
    if (id.includes("flash")) {
      return { kind: "level", levels: LEVEL_ORDER };
    }
    return { kind: "level", levels: ["low", "medium", "high"] };
  }

  // Gemini 2.5: token budgets. Flash/Lite may disable thinking with 0.
  if (/^gemini-2\.5/.test(id)) {
    return id.includes("pro")
      ? { kind: "budget", min: 128, max: 32768 }
      : { kind: "budget", min: 0, max: 24576 };
  }

  return { kind: "includeOnly" };
}

/**
 * Clamp a requested effort onto the levels a family accepts.
 *
 * Rounds *up* when the exact level is missing so a request never silently loses
 * reasoning depth: `medium` on Gemini 3 Pro (low|high) becomes `high`, and
 * Claude/CCR's `xhigh`/`max`/`ultra` — which are not Gemini enum values —
 * become `high`.
 */
export function translateThinkingLevel(
  effort: string,
  levels: GeminiThinkingLevel[]
): GeminiThinkingLevel {
  const requested = effort.toLowerCase();
  const supported = LEVEL_ORDER.filter((level) => levels.includes(level));
  if (!supported.length) return "high";

  // Efforts above Gemini's range collapse onto the family's ceiling.
  if (requested === "xhigh" || requested === "max" || requested === "ultra") {
    return supported[supported.length - 1];
  }

  const requestedIndex = LEVEL_ORDER.indexOf(requested as GeminiThinkingLevel);
  if (requestedIndex < 0) return supported[supported.length - 1];

  return (
    supported.find((level) => LEVEL_ORDER.indexOf(level) >= requestedIndex) ??
    supported[supported.length - 1]
  );
}

/** Effort → token budget, as a share of the family's ceiling. */
function translateThinkingBudget(effort: string, max: number): number {
  const share =
    effort === "minimal" ? 0.1 : effort === "low" ? 0.25 : effort === "medium" ? 0.5 : 1;
  return Math.round(max * share);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Build the `generationConfig.thinkingConfig` for a request.
 *
 * Returns undefined when the request asks for no thinking at all, or when the
 * model cannot carry the config.
 */
export function buildGeminiThinkingConfig(
  request: Pick<UnifiedChatRequest, "model" | "reasoning" | "max_tokens"> & {
    thinking?: UnifiedChatRequest["thinking"];
  }
): GeminiThinkingConfig | undefined {
  const reasoning = request.reasoning;
  const thinkingDisabled = request.thinking?.type === "disabled";
  // No reasoning field at all means the client never asked about thinking —
  // leave the model on its own default rather than inventing a config.
  if (!reasoning || thinkingDisabled) return undefined;

  const dialect = resolveThinkingDialect(request.model);
  if (dialect.kind === "none") return undefined;

  const effort = (reasoning.effort as ThinkLevel | undefined)?.toLowerCase();
  // Claude Code's `none` means the user turned thinking off. Gemini 3 cannot
  // disable thinking outright, so ask for the family's floor and stop
  // requesting thought parts.
  const off = effort === "none" || reasoning.enabled === false;
  // Shared opt-in: reasoning.summary / REASONING_AUTO_SUMMARY / provider
  // reasoningSummary request visible thoughts. summary:"none" hides them.
  const summary = resolveOutboundReasoningSummary(request);
  const wantThoughts = reasoning.summary === "none" ? false : summary ? true : !off;

  if (dialect.kind === "includeOnly") {
    return off ? undefined : { includeThoughts: wantThoughts };
  }

  if (dialect.kind === "level") {
    const supported = LEVEL_ORDER.filter((level) =>
      dialect.levels.includes(level)
    );
    if (off) {
      return { includeThoughts: false, thinkingLevel: supported[0] };
    }
    // Effort absent (some clients send `thinking` without an effort): keep the
    // model default level but still ask for thought parts, since Antigravity
    // only streams them when includeThoughts is set.
    if (!effort) return { includeThoughts: wantThoughts };
    return {
      includeThoughts: wantThoughts,
      thinkingLevel: translateThinkingLevel(effort, dialect.levels),
    };
  }

  // Budget dialect.
  if (off) {
    // A 0 budget disables thinking where the family allows it; otherwise
    // omitting the config is the only way off, since Claude's floor is 1024.
    return dialect.min === 0
      ? { includeThoughts: false, thinkingBudget: 0 }
      : undefined;
  }

  // An explicit client budget wins over the effort-derived one.
  const explicit = reasoning.max_tokens;
  const requested =
    typeof explicit === "number"
      ? explicit
      : effort
        ? translateThinkingBudget(effort, dialect.max)
        : undefined;

  // Effort absent and no budget: ask for thought parts at the model default.
  if (typeof requested !== "number") {
    return { includeThoughts: wantThoughts };
  }

  let budget = clamp(requested, dialect.min, dialect.max);

  // Thinking has to leave room for the answer.
  if (request.max_tokens && budget >= request.max_tokens) {
    budget = request.max_tokens - 1;
  }
  if (budget < dialect.min) {
    // The floor and the answer budget cannot both be satisfied. Anthropic
    // rejects sub-floor budgets outright, so drop thinking there; elsewhere the
    // API floor wins.
    if (dialect.requireMin) return undefined;
    budget = dialect.min;
  }

  return { includeThoughts: wantThoughts, thinkingBudget: budget };
}

/**
 * OpenAI model metadata imported from Codex CLI's bundled catalog
 * (codex-rs `models-manager/models.json`), so CCR sends what Codex sends:
 *
 * - reasoning effort: only the model's supported levels. `ultra` is a picker
 *   alias and never reaches the wire (`ModelInfo::resolve_reasoning_effort`).
 * - Responses Lite: the request shape Codex uses for `use_responses_lite`
 *   models (`core/src/client.rs::build_responses_request`).
 * - verbosity: only models with `support_verbosity`.
 *
 * Refresh the table when Codex adds or retires catalog models.
 */
import {
  isGpt6FamilyModel,
  normalizeReasoningEffort,
  type ReasoningSummaryLevel,
} from "./reasoning-effort";
import type { ThinkLevel } from "@/types/llm";

export type CodexTextVerbosity = "low" | "medium" | "high";

export interface CodexModelSpec {
  /** `supported_reasoning_levels`, lowest first. */
  efforts: readonly ThinkLevel[];
  /** `multi_agent_reasoning_effort`: what `ultra` resolves to, when set. */
  ultraEffort?: ThinkLevel;
  /** `use_responses_lite`. */
  responsesLite: boolean;
  /** `support_verbosity`. */
  verbosity: boolean;
}

const LOW_TO_XHIGH: readonly ThinkLevel[] = ["low", "medium", "high", "xhigh"];
const LOW_TO_MAX: readonly ThinkLevel[] = [...LOW_TO_XHIGH, "max"];
const LOW_TO_ULTRA: readonly ThinkLevel[] = [...LOW_TO_MAX, "ultra"];

const lite = (
  efforts: readonly ThinkLevel[],
  ultraEffort?: ThinkLevel
): CodexModelSpec => ({
  efforts,
  ...(ultraEffort ? { ultraEffort } : {}),
  responsesLite: true,
  verbosity: true,
});

/** Codex `models-manager/models.json` at codex commit 822e58cc3d. */
const CODEX_MODEL_CATALOG: Readonly<Record<string, CodexModelSpec>> = {
  "gpt-6-astra": lite(LOW_TO_ULTRA, "xhigh"),
  "gpt-6.1-sol": lite(LOW_TO_ULTRA, "xhigh"),
  "gpt-6-sol": lite(LOW_TO_ULTRA),
  "gpt-6-luna": lite(LOW_TO_MAX),
  "gpt-5.6-sol": lite(LOW_TO_ULTRA),
  "gpt-5.6-terra": lite(LOW_TO_ULTRA),
  "gpt-5.6-luna": lite(LOW_TO_MAX),
  "gpt-daybreak-blue-latest": lite(LOW_TO_ULTRA),
  "gpt-daybreak-red-latest": lite(LOW_TO_ULTRA),
  "codex-auto-review": lite(LOW_TO_MAX),
  "gpt-5.5": { efforts: LOW_TO_XHIGH, responsesLite: false, verbosity: true },
};

/**
 * GPT-6 Luna slugs (`gpt-6-luna`, `gpt-6.1-luna`, `openai/gpt-6-luna`,
 * `codex,gpt-6.1-luna`). Anchored so `gpt-6-sol` / `gpt-6-astra` /
 * `gpt-5.6-luna` do not match.
 */
export function isGpt6LunaModel(model: unknown): boolean {
  if (typeof model !== "string" || !model) return false;
  return /(?:^|[/,:])gpt-6(?:\.\d+)?-luna(?:$|[.-])/i.test(model);
}

/** Catalog slug: drop a `provider,` / `org/` prefix, lowercase. */
function catalogSlug(model: string): string {
  return model.slice(Math.max(model.lastIndexOf(","), model.lastIndexOf("/")) + 1)
    .trim()
    .toLowerCase();
}

/**
 * Catalog metadata for a model id, or undefined for models Codex does not
 * list. GPT-6 slugs newer than the table get the family's shape (Luna tops
 * out at `max`), so a new minor release keeps working before a refresh.
 */
export function codexModelSpec(model: unknown): CodexModelSpec | undefined {
  if (typeof model !== "string" || !model) return undefined;
  const listed = CODEX_MODEL_CATALOG[catalogSlug(model)];
  if (listed) return listed;
  if (!isGpt6FamilyModel(model)) return undefined;
  return lite(isGpt6LunaModel(model) ? LOW_TO_MAX : LOW_TO_ULTRA);
}

/** Every effort level CCR knows, lowest first. */
const EFFORT_ORDER: readonly ThinkLevel[] = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "ultra",
];

/**
 * The effort a catalog model is sent. `ultra` follows Codex: the model's
 * multi-agent effort, else `max`, else the highest level below `ultra`. Any
 * other unsupported level moves to the nearest end of the supported range
 * (`none` / `minimal` → the lowest). Unknown tokens and unlisted models pass
 * through unchanged.
 */
export function resolveCodexReasoningEffort(
  model: unknown,
  effort: ThinkLevel | undefined
): ThinkLevel | undefined {
  const spec = codexModelSpec(model);
  if (!effort || !spec || spec.efforts.length === 0) return effort;
  const levels = spec.efforts;
  if (effort === "ultra") {
    if (spec.ultraEffort && levels.includes(spec.ultraEffort)) {
      return spec.ultraEffort;
    }
    if (levels.includes("max")) return "max";
    return [...levels].reverse().find((level) => level !== "ultra");
  }
  if (levels.includes(effort)) return effort;
  const rank = EFFORT_ORDER.indexOf(effort);
  if (rank < 0) return effort;
  const sent = levels.filter((level) => level !== "ultra");
  if (rank < EFFORT_ORDER.indexOf(sent[0])) return sent[0];
  return sent[sent.length - 1];
}

/**
 * Apply `resolveCodexReasoningEffort` to a Responses/Unified request in
 * place. Covers convert (`openai-responses`) and same-protocol wire-keep
 * (`codex`). A disabled-reasoning request raised to a supported level is
 * re-enabled, since the model cannot run without reasoning.
 */
export function applyCodexReasoningEffort(request: {
  model?: unknown;
  reasoning?: { effort?: unknown; enabled?: boolean } | null;
}): void {
  if (!request.reasoning) return;
  const effort = normalizeReasoningEffort(request.reasoning.effort);
  const resolved = resolveCodexReasoningEffort(request.model, effort);
  if (!resolved || resolved === effort) return;
  request.reasoning.effort = resolved;
  if (request.reasoning.enabled === false) {
    request.reasoning.enabled = true;
  }
}

/**
 * `text.verbosity` implied by `REASONING_AUTO_SUMMARY`: detailed thinking
 * pairs with verbose answers, concise with terse ones.
 */
export function verbosityForReasoningSummary(
  summary: ReasoningSummaryLevel | undefined
): CodexTextVerbosity | undefined {
  if (summary === "detailed") return "high";
  if (summary === "concise") return "low";
  if (summary === "auto") return "medium";
  return undefined;
}

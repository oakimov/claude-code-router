import type { ModelListItem, ModelSelection, ModelVariant } from "@cursor/sdk";

type ParameterDefinition = NonNullable<ModelListItem["parameters"]>[number];
type ParameterValue = NonNullable<ModelVariant["params"]>[number];

/**
 * Reasoning the client asked for. `undefined`: none (absent, `none`, or
 * disabled). `{}`: on at the model's default effort. `{ effort }`: on at
 * that level.
 */
export type CursorReasoning = { effort?: string } | undefined;

/** Cursor's `fast` parameter. CCR never selects fast mode. */
const FAST_PARAM = "fast";
const THINKING_PARAM = "thinking";
const CONTEXT_PARAM = "context";

/** Effort levels low to high, across Cursor's and the clients' spellings. */
const EFFORT_RANK = [
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "ultra",
];

function isEffortParam(id: string): boolean {
  return id !== THINKING_PARAM && /effort|reasoning/i.test(id);
}

/** Rank of an effort value; GPT's `extra-high` is the clients' `xhigh`. */
function effortRank(value: string): number {
  const normalized = value.trim().toLowerCase().replace(/^extra[-_ ]?/, "x");
  return EFFORT_RANK.indexOf(normalized);
}

function contextTokens(value: string): number {
  const match = /^(\d+(?:\.\d+)?)\s*([km]?)$/i.exec(value.trim());
  if (!match) return Number.POSITIVE_INFINITY;
  const scale = { "": 1, k: 1e3, m: 1e6 }[match[2].toLowerCase() as "" | "k" | "m"];
  return Number.parseFloat(match[1]) * scale;
}

/**
 * Parameter definitions for a model. Variant-only catalog entries get
 * definitions derived from their variants, in order of appearance.
 */
function definitionsOf(model: ModelListItem): ParameterDefinition[] {
  if (model.parameters?.length) return model.parameters;
  const values = new Map<string, string[]>();
  for (const variant of model.variants || []) {
    for (const param of variant.params || []) {
      const seen = values.get(param.id) || [];
      if (!seen.includes(param.value)) seen.push(param.value);
      values.set(param.id, seen);
    }
  }
  return [...values].map(([id, list]) => ({
    id,
    values: list.map((value) => ({ value })),
  }));
}

/** The allowed value closest to the requested effort, or undefined. */
function closestEffort(
  definition: ParameterDefinition,
  effort: string
): string | undefined {
  const want = effortRank(effort);
  if (want < 0) {
    return definition.values.find(
      (v) => v.value.toLowerCase() === effort.trim().toLowerCase()
    )?.value;
  }
  let best: { value: string; rank: number; distance: number } | undefined;
  for (const { value } of definition.values) {
    const rank = effortRank(value);
    if (rank < 0) continue;
    const distance = Math.abs(rank - want);
    // Ties go to the lower level: never more reasoning than asked for.
    if (
      !best ||
      distance < best.distance ||
      (distance === best.distance && rank < best.rank)
    ) {
      best = { value, rank, distance };
    }
  }
  return best?.value;
}

/**
 * The value CCR wants for each parameter.
 *
 * Without reasoning, every parameter takes its first allowed value. That is
 * what Cursor itself runs when a parameter is omitted, and the catalog lists
 * the least reasoning first (`thinking=false`, `reasoning=none`, else the
 * lowest effort). With reasoning, `thinking=true` and the requested effort
 * (nearest allowed level), or the default variant's effort when none was
 * given. Context is the smallest window (Grok 4.7's 500k is rejected on the
 * run stream) and fast mode is always off.
 */
function wantedParams(
  model: ModelListItem,
  reasoning: CursorReasoning
): Map<string, string> {
  const defaults = new Map(
    (model.variants?.find((v) => v.isDefault)?.params || []).map((p) => [
      p.id,
      p.value,
    ])
  );
  const wanted = new Map<string, string>();
  for (const definition of definitionsOf(model)) {
    const values = definition.values.map((v) => v.value);
    if (!values.length) continue;
    let value = values[0];
    if (definition.id === FAST_PARAM) {
      value = values.includes("false") ? "false" : value;
    } else if (definition.id === CONTEXT_PARAM) {
      value = values.reduce((a, b) =>
        contextTokens(b) < contextTokens(a) ? b : a
      );
    } else if (reasoning && definition.id === THINKING_PARAM) {
      value = values.includes("true") ? "true" : value;
    } else if (reasoning && isEffortParam(definition.id)) {
      const requested = reasoning.effort
        ? closestEffort(definition, reasoning.effort)
        : undefined;
      const fallback = defaults.get(definition.id);
      value =
        requested ||
        (fallback && values.includes(fallback) ? fallback : value);
    }
    wanted.set(definition.id, value);
  }
  return wanted;
}

/** Narrowing order: what changes behavior most decides first. */
function priorityOf(id: string): number {
  if (id === FAST_PARAM) return 0;
  if (id === THINKING_PARAM) return 1;
  if (isEffortParam(id)) return 2;
  if (id === CONTEXT_PARAM) return 3;
  return 4;
}

/**
 * The preset variant closest to `wanted`. Presets carry params the
 * definitions omit (Claude's `cyber`), so CCR copies a preset instead of
 * composing one. Each parameter, in priority order, narrows the candidates
 * to the rows that carry the wanted value, when any do.
 */
function pickVariant(
  variants: ModelVariant[],
  wanted: Map<string, string>
): ModelVariant | undefined {
  let candidates = variants.filter((v) => v.params?.length);
  const order = [...wanted.keys()].sort(
    (a, b) => priorityOf(a) - priorityOf(b)
  );
  for (const id of order) {
    const value = wanted.get(id);
    const matching = candidates.filter((v) =>
      v.params.some((p: ParameterValue) => p.id === id && p.value === value)
    );
    if (matching.length) candidates = matching;
  }
  return candidates.find((v) => v.isDefault) || candidates[0];
}

function findListedModel(
  models: ModelListItem[],
  modelId: string
): ModelListItem | undefined {
  const want = modelId.toLowerCase();
  return models.find(
    (m) =>
      m.id === modelId ||
      m.displayName === modelId ||
      (Array.isArray(m.aliases) && m.aliases.includes(modelId)) ||
      m.id.toLowerCase() === want ||
      m.displayName.toLowerCase() === want
  );
}

/**
 * Map a CCR model id and the client's reasoning request onto an SDK
 * ModelSelection from a Cursor.models.list catalog. Parameterized models
 * (Grok 4.7 context/effort/fast) always get explicit params, as the Cursor
 * SDK docs recommend; see `wantedParams` for the values.
 *
 * Grok 4.7 has no off switch, so no reasoning selects 256k+low+non-fast;
 * Claude models select `thinking=false`, GPT 5.4+ `reasoning=none`.
 */
export function selectCursorModelSelection(
  models: ModelListItem[],
  modelId: string,
  reasoning?: CursorReasoning
): ModelSelection {
  const found = findListedModel(models, modelId);
  if (!found) return { id: modelId };

  const wanted = wantedParams(found, reasoning);
  const variant = pickVariant(found.variants || [], wanted);
  if (variant?.params?.length) {
    return { id: found.id, params: variant.params };
  }
  if (wanted.size) {
    return {
      id: found.id,
      params: [...wanted].map(([id, value]) => ({ id, value })),
    };
  }
  return { id: found.id };
}

/** Stable id+params key for persist/compare. Order of params does not matter. */
export function cursorModelFingerprint(selection: ModelSelection): string {
  const params = [...(selection.params || [])]
    .map((param) => `${param.id}=${param.value}`)
    .sort();
  return `${selection.id}|${params.join(",")}`;
}

export function cursorModelSelectionsEqual(
  left: ModelSelection | undefined,
  right: ModelSelection
): boolean {
  if (!left?.id) return false;
  return cursorModelFingerprint(left) === cursorModelFingerprint(right);
}

export function isCursorInvalidRegistryModelError(err: unknown): boolean {
  const message = (
    typeof (err as { message?: unknown })?.message === "string"
      ? (err as { message: string }).message
      : String(err ?? "")
  ).toLowerCase();
  return message.includes("invalid parameters for registry model");
}

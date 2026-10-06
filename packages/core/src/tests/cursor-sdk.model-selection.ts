import assert from "node:assert/strict";
import type { ModelListItem } from "@cursor/sdk";
import {
  cursorModelFingerprint,
  cursorModelSelectionsEqual,
  isCursorInvalidRegistryModelError,
  selectCursorModelSelection,
  type CursorReasoning,
} from "../cursor-sdk/model-selection";
import { cursorReasoningKey, extractCursorReasoning } from "../cursor-sdk/shared";

type Param = { id: string; value: string };

/** Every combination of the given parameter values, in catalog order. */
function variantsOf(
  axes: Array<[string, string[]]>,
  isDefault: (params: Param[]) => boolean = () => false
): NonNullable<ModelListItem["variants"]> {
  let rows: Param[][] = [[]];
  for (const [id, values] of axes) {
    rows = rows.flatMap((row) => values.map((value) => [...row, { id, value }]));
  }
  return rows.map((params) => ({
    displayName: params.map((p) => p.value).join(" "),
    ...(isDefault(params) ? { isDefault: true } : {}),
    params,
  }));
}

const valueOf = (params: Param[], id: string) =>
  params.find((p) => p.id === id)?.value;

/** Live Cursor.models.list shapes (ids/params only), catalog order kept. */
const grok47: ModelListItem = {
  id: "grok-4.7",
  displayName: "Grok 4.7",
  parameters: [
    { id: "context", values: [{ value: "256k" }, { value: "500k" }] },
    {
      id: "reasoning_effort",
      values: ["low", "medium", "high", "xhigh"].map((value) => ({ value })),
    },
    { id: "fast", values: [{ value: "false" }, { value: "true" }] },
  ],
  variants: variantsOf(
    [
      ["context", ["256k", "500k"]],
      ["reasoning_effort", ["low", "medium", "high", "xhigh"]],
      ["fast", ["false", "true"]],
    ],
    (p) =>
      valueOf(p, "context") === "500k" &&
      valueOf(p, "reasoning_effort") === "high" &&
      valueOf(p, "fast") === "true"
  ),
};

/** Variants carry `cyber`, which the parameter definitions do not list. */
const claudeOpus5: ModelListItem = {
  id: "claude-opus-5",
  displayName: "Claude Opus 5",
  parameters: [
    { id: "thinking", values: [{ value: "false" }, { value: "true" }] },
    { id: "context", values: [{ value: "300k" }, { value: "1m" }] },
    {
      id: "effort",
      values: ["low", "medium", "high", "xhigh", "max"].map((value) => ({ value })),
    },
    { id: "fast", values: [{ value: "false" }, { value: "true" }] },
  ],
  variants: [
    ...variantsOf([
      ["cyber", ["false"]],
      ["thinking", ["false"]],
      ["context", ["300k", "1m"]],
      ["effort", ["low", "medium", "high"]],
      ["fast", ["false", "true"]],
    ]),
    ...variantsOf(
      [
        ["cyber", ["false"]],
        ["thinking", ["true"]],
        ["context", ["300k", "1m"]],
        ["effort", ["low", "medium", "high", "xhigh", "max"]],
        ["fast", ["false", "true"]],
      ],
      (p) =>
        valueOf(p, "context") === "1m" &&
        valueOf(p, "effort") === "high" &&
        valueOf(p, "fast") === "false"
    ),
  ],
};

const gpt55: ModelListItem = {
  id: "gpt-5.5",
  displayName: "GPT-5.5",
  parameters: [
    { id: "context", values: [{ value: "272k" }, { value: "1m" }] },
    {
      id: "reasoning",
      values: ["none", "low", "medium", "high", "extra-high"].map((value) => ({
        value,
      })),
    },
    { id: "fast", values: [{ value: "false" }, { value: "true" }] },
  ],
  variants: variantsOf(
    [
      ["context", ["272k", "1m"]],
      ["reasoning", ["none", "low", "medium", "high", "extra-high"]],
      ["fast", ["false", "true"]],
    ],
    (p) =>
      valueOf(p, "context") === "1m" &&
      valueOf(p, "reasoning") === "medium" &&
      valueOf(p, "fast") === "false"
  ),
};

const catalog = [grok47, claudeOpus5, gpt55];

function paramsFor(modelId: string, reasoning: CursorReasoning) {
  return Object.fromEntries(
    (selectCursorModelSelection(catalog, modelId, reasoning).params || []).map(
      (p) => [p.id, p.value]
    )
  );
}

// No reasoning: Grok 4.7 has no off switch, so its least reasoning, never fast.
assert.deepEqual(paramsFor("grok-4.7", undefined), {
  context: "256k",
  reasoning_effort: "low",
  fast: "false",
});
// Reasoning without an effort: the catalog default's effort.
assert.deepEqual(paramsFor("grok-4.7", {}), {
  context: "256k",
  reasoning_effort: "high",
  fast: "false",
});
for (const effort of ["low", "medium", "high", "xhigh"]) {
  assert.deepEqual(
    paramsFor("grok-4.7", { effort }),
    { context: "256k", reasoning_effort: effort, fast: "false" },
    `grok effort ${effort}`
  );
}
// Unlisted levels go to the nearest one, never above what was asked when tied.
assert.equal(paramsFor("grok-4.7", { effort: "minimal" }).reasoning_effort, "low");
assert.equal(paramsFor("grok-4.7", { effort: "max" }).reasoning_effort, "xhigh");

// Claude: a real off switch. The preset keeps `cyber`.
assert.deepEqual(paramsFor("claude-opus-5", undefined), {
  cyber: "false",
  thinking: "false",
  context: "300k",
  effort: "low",
  fast: "false",
});
assert.deepEqual(paramsFor("claude-opus-5", {}), {
  cyber: "false",
  thinking: "true",
  context: "300k",
  effort: "high",
  fast: "false",
});
assert.deepEqual(paramsFor("claude-opus-5", { effort: "max" }), {
  cyber: "false",
  thinking: "true",
  context: "300k",
  effort: "max",
  fast: "false",
});

// GPT: `reasoning=none` is the off switch; `xhigh` is spelled `extra-high`.
assert.deepEqual(paramsFor("gpt-5.5", undefined), {
  context: "272k",
  reasoning: "none",
  fast: "false",
});
assert.deepEqual(paramsFor("gpt-5.5", {}), {
  context: "272k",
  reasoning: "medium",
  fast: "false",
});
assert.equal(paramsFor("gpt-5.5", { effort: "xhigh" }).reasoning, "extra-high");

// A fast default still loses to its non-fast twin.
const composer: ModelListItem = {
  id: "composer-2.5",
  displayName: "Composer 2.5",
  parameters: [{ id: "fast", values: [{ value: "false" }, { value: "true" }] }],
  variants: [
    { displayName: "Fast", isDefault: true, params: [{ id: "fast", value: "true" }] },
    { displayName: "Standard", params: [{ id: "fast", value: "false" }] },
  ],
};
assert.deepEqual(selectCursorModelSelection([composer], "composer-2.5", {}).params, [
  { id: "fast", value: "false" },
]);

// Parameter-only catalog entries: explicit params from the same rules.
const paramOnly: ModelListItem = {
  id: "param-only",
  displayName: "Param Only",
  parameters: [
    { id: "fast", values: [{ value: "true" }, { value: "false" }] },
    { id: "reasoning_effort", values: [{ value: "low" }, { value: "high" }] },
  ],
};
assert.deepEqual(
  selectCursorModelSelection([paramOnly], "param-only", { effort: "high" }).params,
  [
    { id: "fast", value: "false" },
    { id: "reasoning_effort", value: "high" },
  ]
);
assert.deepEqual(
  selectCursorModelSelection([paramOnly], "param-only", undefined).params,
  [
    { id: "fast", value: "false" },
    { id: "reasoning_effort", value: "low" },
  ]
);

assert.deepEqual(selectCursorModelSelection(catalog, "missing", {}), {
  id: "missing",
});

// Unified request → reasoning request, for each inbound protocol's mapping.
assert.equal(extractCursorReasoning({ messages: [] }), undefined);
// Anthropic: Claude Code sends adaptive thinking plus output_config.effort.
assert.deepEqual(
  extractCursorReasoning({
    thinking: { type: "adaptive" },
    reasoning: { effort: "high", enabled: true },
  }),
  { effort: "high" }
);
// Anthropic thinking without effort, or Responses `reasoning: {}`.
assert.deepEqual(extractCursorReasoning({ reasoning: { enabled: true } }), {});
assert.deepEqual(extractCursorReasoning({ thinking: { type: "adaptive" } }), {});
// Explicit off.
assert.equal(
  extractCursorReasoning({
    thinking: { type: "disabled" },
    reasoning: { enabled: false },
  }),
  undefined
);
assert.equal(
  extractCursorReasoning({ reasoning: { effort: "none", enabled: false } }),
  undefined
);
assert.equal(cursorReasoningKey(undefined), "off");
assert.equal(cursorReasoningKey({}), "on");
assert.equal(cursorReasoningKey({ effort: "high" }), "on:high");

const selected = selectCursorModelSelection(catalog, "grok-4.7", { effort: "high" });
assert.equal(
  cursorModelFingerprint(selected),
  "grok-4.7|context=256k,fast=false,reasoning_effort=high"
);
assert.equal(cursorModelSelectionsEqual(selected, selected), true);
assert.equal(cursorModelSelectionsEqual({ id: "grok-4.7" }, selected), false);
assert.equal(
  isCursorInvalidRegistryModelError(
    new Error('AI Model Not Found Invalid parameters for registry model: "grok-4.7"')
  ),
  true
);

console.log("cursor-sdk.model-selection: ok");

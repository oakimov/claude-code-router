/**
 * Codex catalog import: per-model effort levels and `ultra` resolution,
 * Responses Lite request shape, and `text.verbosity` from
 * REASONING_AUTO_SUMMARY — matching what Codex CLI sends.
 */
import assert from "node:assert/strict";
import { CodexTransformer } from "../transformer/codex.transformer";
import {
  codexModelSpec,
  resolveCodexReasoningEffort,
  verbosityForReasoningSummary,
} from "../utils/codex-model-catalog";

function codex(): CodexTransformer {
  const transformer = new CodexTransformer();
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });
  return transformer;
}

function context(id: string, config: Record<string, unknown> = {}) {
  return {
    req: {
      id,
      server: { configService: { get: (key: string) => config[key] } },
    },
  };
}

async function send(
  body: Record<string, any>,
  options: { provider?: Record<string, any>; config?: Record<string, unknown> } = {}
) {
  const result = await codex().transformRequestIn(
    body,
    { baseUrl: "https://chatgpt.com/backend-api/codex", ...options.provider },
    context(`catalog-${Math.random()}`, options.config)
  );
  return {
    body: result.body as any,
    headers: (result as any).config.headers as Record<string, string>,
  };
}

function testCatalogLookup() {
  for (const model of [
    "gpt-6.1-sol",
    "codex,gpt-6.1-sol",
    "openai/GPT-6.1-SOL",
  ]) {
    assert.equal(codexModelSpec(model)?.ultraEffort, "xhigh", model);
  }
  assert.equal(codexModelSpec("gpt-5.5")?.responsesLite, false);
  assert.equal(codexModelSpec("gpt-5.6-terra")?.responsesLite, true);
  // A GPT-6 minor newer than the table keeps the family shape.
  assert.equal(codexModelSpec("gpt-6.2-sol")?.responsesLite, true);
  assert.equal(resolveCodexReasoningEffort("gpt-6.2-luna", "ultra"), "max");
  assert.equal(codexModelSpec("o9-pro"), undefined);
}

function testEffortRanges() {
  // gpt-5.5 tops out at xhigh: max and ultra come down to it.
  assert.equal(resolveCodexReasoningEffort("gpt-5.5", "max"), "xhigh");
  assert.equal(resolveCodexReasoningEffort("gpt-5.5", "ultra"), "xhigh");
  assert.equal(resolveCodexReasoningEffort("gpt-5.5", "minimal"), "low");
  assert.equal(resolveCodexReasoningEffort("gpt-5.5", "high"), "high");
  // Luna models list max but not ultra.
  assert.equal(resolveCodexReasoningEffort("gpt-5.6-luna", "ultra"), "max");
  assert.equal(resolveCodexReasoningEffort("gpt-5.6-sol", "max"), "max");
  // Codex-only tokens and models outside the catalog pass through.
  assert.equal(resolveCodexReasoningEffort("gpt-6-sol", "disabled" as any), "disabled");
  assert.equal(resolveCodexReasoningEffort("o9-pro", "ultra"), "ultra");
}

/** Codex `build_responses_request` for a `use_responses_lite` model. */
async function testResponsesLiteShape() {
  const { body, headers } = await send({
    model: "gpt-6.1-sol",
    instructions: "Be precise.",
    input: [
      { role: "developer", content: "Repo rules." },
      {
        role: "user",
        content: [
          { type: "input_text", text: "look" },
          { type: "input_image", image_url: "data:image/png;base64,AA", detail: "high" },
        ],
      },
      { type: "function_call", call_id: "c1", name: "read", arguments: "{}" },
      {
        type: "function_call_output",
        call_id: "c1",
        output: [{ type: "input_image", image_url: "data:image/png;base64,BB", detail: "original" }],
      },
    ],
    tools: [
      { type: "web_search" },
      { type: "function", name: "read", description: "Read", parameters: { type: "object" } },
      { type: "custom", name: "apply_patch", description: "Patch", format: { type: "text" } },
    ],
    parallel_tool_calls: true,
    reasoning: { effort: "ultra", summary: "auto" },
  }, { provider: { parallelToolCalls: true } });

  assert.equal(headers["x-openai-internal-codex-responses-lite"], "true");
  assert.equal(body.tools, undefined);
  assert.deepEqual(body.input[0], {
    type: "additional_tools",
    role: "developer",
    tools: [
      { type: "web_search" },
      {
        type: "namespace",
        name: "functions",
        description: "",
        tools: [
          { type: "function", name: "read", description: "Read", parameters: { type: "object" } },
          { type: "custom", name: "apply_patch", description: "Patch", format: { type: "text" } },
        ],
      },
    ],
  });
  // Developer messages still fold into instructions; the tools item stays.
  assert.equal(body.instructions, "Be precise.\n\nRepo rules.");
  assert.equal(body.input.filter((item: any) => item.role === "developer").length, 1);
  assert.equal(body.parallel_tool_calls, false);
  assert.deepEqual(body.reasoning, { effort: "xhigh", summary: "auto", context: "all_turns" });
  assert.equal("detail" in body.input[1].content[1], false);
  assert.equal("detail" in body.input[3].output[0], false);
}

async function testClassicModelKeepsTools() {
  const { body, headers } = await send({
    model: "gpt-5.5",
    input: [{ role: "user", content: "hi" }],
    tools: [{ type: "function", name: "read", parameters: { type: "object" } }],
    reasoning: { effort: "max" },
  });
  assert.equal(headers["x-openai-internal-codex-responses-lite"], undefined);
  assert.equal(body.tools.length, 1);
  assert.equal(body.input[0].role, "user");
  assert.equal(body.reasoning.effort, "xhigh");
  assert.equal(body.reasoning.context, undefined);
}

async function testLiteWithoutToolsAddsNoItem() {
  const { body } = await send({
    model: "gpt-6-luna",
    input: [{ role: "user", content: "hi" }],
  });
  assert.equal(body.input.length, 1);
  assert.deepEqual(body.reasoning, { context: "all_turns" });
}

async function testVerbosity() {
  assert.equal(verbosityForReasoningSummary("detailed"), "high");
  assert.equal(verbosityForReasoningSummary("auto"), "medium");
  assert.equal(verbosityForReasoningSummary("concise"), "low");
  assert.equal(verbosityForReasoningSummary(undefined), undefined);

  const base = { model: "gpt-6.1-sol", input: [{ role: "user", content: "hi" }] };
  for (const [setting, expected] of [
    [true, "high"],
    ["detailed", "high"],
    ["auto", "medium"],
    ["concise", "low"],
    [undefined, undefined],
  ] as const) {
    const { body } = await send(structuredClone(base), {
      config: { REASONING_AUTO_SUMMARY: setting },
    });
    assert.equal(body.text?.verbosity, expected, String(setting));
    assert.equal(body.verbosity, undefined);
  }

  // The client's own value wins, then the provider's.
  let { body } = await send(
    { ...structuredClone(base), text: { verbosity: "low", format: { type: "text" } } },
    { provider: { verbosity: "high" }, config: { REASONING_AUTO_SUMMARY: "detailed" } }
  );
  assert.deepEqual(body.text, { verbosity: "low", format: { type: "text" } });
  ({ body } = await send(structuredClone(base), {
    provider: { verbosity: "medium" },
    config: { REASONING_AUTO_SUMMARY: "detailed" },
  }));
  assert.equal(body.text.verbosity, "medium");

  // Models outside the catalog: only an explicit provider setting applies.
  ({ body } = await send(
    { model: "o9-pro", input: [{ role: "user", content: "hi" }] },
    { config: { REASONING_AUTO_SUMMARY: "detailed" } }
  ));
  assert.equal(body.text, undefined);
  ({ body } = await send(
    { model: "o9-pro", input: [{ role: "user", content: "hi" }] },
    { provider: { verbosity: "low" } }
  ));
  assert.equal(body.text.verbosity, "low");
}

async function main() {
  testCatalogLookup();
  testEffortRanges();
  await testResponsesLiteShape();
  await testClassicModelKeepsTools();
  await testLiteWithoutToolsAddsNoItem();
  await testVerbosity();
  console.log("codex.model-catalog: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

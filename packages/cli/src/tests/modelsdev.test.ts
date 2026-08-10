/**
 * models.dev model-id lookup and native-provider disambiguation for
 * `ccr codex-config`. Hermetic: the fixture is indexed in memory.
 */
import assert from "node:assert/strict";
import { buildCatalogEntry, type CatalogModel } from "../utils/codexConfig";
import { buildModelsDevIndex, lookupModel } from "../utils/modelsdev";

function main(): void {
  const index = buildModelsDevIndex({
    opencode: {
      models: {
        "deepseek-v4-flash-free": {
          id: "deepseek-v4-flash-free",
          name: "DeepSeek V4 Flash Free",
          family: "deepseek-v4-flash",
          reasoning: true,
          reasoning_options: [
            { type: "effort", values: ["low", "high", "max"] },
          ],
          limit: { context: 200_000, output: 128_000 },
        },
      },
    },
    mistral: {
      models: {
        "codestral-latest": {
          id: "codestral-latest",
          name: "Codestral Latest",
          reasoning: false,
          limit: { context: 256_000, output: 4_096 },
        },
      },
    },
    openai: {
      models: {
        "gpt-5.6-sol": {
          id: "gpt-5.6-sol",
          name: "GPT-5.6 Sol",
          family: "gpt-sol",
          reasoning: true,
          reasoning_options: [
            { type: "effort", values: ["low", "medium", "high", "max"] },
          ],
          limit: { context: 1_050_000, output: 128_000 },
        },
      },
    },
    reseller: {
      models: {
        "gpt-5.6-sol": {
          id: "gpt-5.6-sol",
          name: "Reseller GPT",
          family: "gpt-sol",
          reasoning: true,
          reasoning_options: [{ type: "effort", values: ["low"] }],
          limit: { context: 32_000, output: 4_096 },
        },
      },
    },
    one: {
      models: {
        ambiguous: {
          id: "ambiguous",
          reasoning_options: [{ type: "effort", values: ["low"] }],
        },
      },
    },
    two: {
      models: {
        ambiguous: {
          id: "ambiguous",
          reasoning_options: [{ type: "effort", values: ["max"] }],
        },
      },
    },
  });

  // A 1:1 id match is authoritative and preserves every declared effort.
  const deepseek = lookupModel(index, "deepseek-v4-flash-free");
  assert.equal(deepseek?.provider, "opencode");
  assert.equal(deepseek?.context, 200_000);
  assert.deepEqual(deepseek?.effortLevels, ["low", "high", "max"]);

  // Duplicate ids resolve through the native provider, never CCR's provider
  // name, catalog order, or the first row that happens to carry efforts.
  const gpt = lookupModel(index, "gpt-5.6-sol");
  assert.equal(gpt?.provider, "openai");
  assert.equal(gpt?.name, "GPT-5.6 Sol");
  assert.deepEqual(gpt?.effortLevels, ["low", "medium", "high", "max"]);

  // An unresolved duplicate is a miss instead of an arbitrary selection.
  assert.equal(lookupModel(index, "ambiguous"), null);
  // Similar-looking ids are not substituted for an exact miss.
  assert.equal(lookupModel(index, "deepseek-v4-flash"), null);

  const codestral = lookupModel(index, "codestral-latest");
  assert.deepEqual(codestral?.effortLevels, []);

  const matched: CatalogModel = {
    providerName: "opencode-openai",
    modelName: "deepseek-v4-flash-free",
    slug: "opencode-openai,deepseek-v4-flash-free",
    info: deepseek,
  };
  const entry = buildCatalogEntry(matched, 0, null);
  assert.deepEqual(
    entry.supported_reasoning_levels.map((level: any) => level.effort),
    ["low", "high", "max"]
  );
  assert.equal(entry.default_reasoning_level, "low");

  const nonReasoning = buildCatalogEntry(
    {
      providerName: "codestral",
      modelName: "codestral-latest",
      slug: "codestral,codestral-latest",
      info: codestral,
    },
    1,
    {
      default_reasoning_level: "medium",
      supported_reasoning_levels: [{ effort: "medium", description: "" }],
    }
  );
  assert.ok(!("default_reasoning_level" in nonReasoning));
  assert.deepEqual(nonReasoning.supported_reasoning_levels, []);
}

main();

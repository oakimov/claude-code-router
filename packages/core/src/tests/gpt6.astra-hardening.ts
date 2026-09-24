/**
 * GPT-6 Astra rejects reasoning.effort none/minimal and sampling knobs
 * (temperature/top_p/logprobs). Harden convert + Codex wire-keep paths.
 */
import assert from "node:assert/strict";
import { CodexTransformer } from "../transformer/codex.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import {
  applyGpt6ReasoningEffortCoercion,
  coerceGpt6ReasoningEffort,
  isGpt6FamilyModel,
  isGpt6LunaModel,
  stripGpt6UnsupportedSampling,
} from "../utils/reasoning-effort";

function mockCodexAuth(transformer: CodexTransformer) {
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });
}

function testModelDetection() {
  assert.equal(isGpt6FamilyModel("gpt-6-astra"), true);
  assert.equal(isGpt6FamilyModel("gpt-6-sol"), true);
  assert.equal(isGpt6FamilyModel("gpt-6-luna"), true);
  assert.equal(isGpt6FamilyModel("gpt-6"), true);
  assert.equal(isGpt6FamilyModel("openai/gpt-6-astra"), true);
  assert.equal(isGpt6FamilyModel("codex,gpt-6-astra"), true);
  assert.equal(isGpt6FamilyModel("gpt-6.0-astra"), true);
  assert.equal(isGpt6FamilyModel("gpt-5.6-sol"), false);
  assert.equal(isGpt6FamilyModel("gpt-5.6-astra"), false);
  assert.equal(isGpt6FamilyModel("gpt-60"), false);
  assert.equal(isGpt6FamilyModel("not-gpt-6"), false);
}

function testEffortHelpers() {
  assert.equal(coerceGpt6ReasoningEffort("gpt-6-astra", "none"), "low");
  assert.equal(coerceGpt6ReasoningEffort("gpt-6-astra", "minimal"), "low");
  assert.equal(coerceGpt6ReasoningEffort("gpt-6-astra", "medium"), "medium");
  assert.equal(coerceGpt6ReasoningEffort("gpt-5.6-sol", "none"), "none");

  const req = {
    model: "gpt-6-astra",
    reasoning: { effort: "none" as const, enabled: false },
  };
  applyGpt6ReasoningEffortCoercion(req);
  assert.equal(req.reasoning.effort, "low");
  assert.equal(req.reasoning.enabled, true);

  const other = {
    model: "gpt-5.6-luna",
    reasoning: { effort: "none" as const, enabled: false },
  };
  applyGpt6ReasoningEffortCoercion(other);
  assert.equal(other.reasoning.effort, "none");
  assert.equal(other.reasoning.enabled, false);
}

function testLunaUltraClamp() {
  assert.equal(isGpt6LunaModel("gpt-6-luna"), true);
  assert.equal(isGpt6LunaModel("openai/gpt-6-luna"), true);
  assert.equal(isGpt6LunaModel("codex,gpt-6-luna"), true);
  assert.equal(isGpt6LunaModel("gpt-6-sol"), false);
  assert.equal(isGpt6LunaModel("gpt-6-astra"), false);
  assert.equal(isGpt6LunaModel("gpt-5.6-luna"), false);
  assert.equal(isGpt6LunaModel(undefined), false);

  // Luna tops out at max (no ultra); Astra/Sol keep ultra.
  assert.equal(coerceGpt6ReasoningEffort("gpt-6-luna", "ultra"), "max");
  assert.equal(coerceGpt6ReasoningEffort("gpt-6-astra", "ultra"), "ultra");
  assert.equal(coerceGpt6ReasoningEffort("gpt-6-sol", "ultra"), "ultra");
  assert.equal(coerceGpt6ReasoningEffort("gpt-6-luna", "high"), "high");

  const req = {
    model: "codex,gpt-6-luna",
    reasoning: { effort: "ultra" as const },
  };
  applyGpt6ReasoningEffortCoercion(req);
  assert.equal(req.reasoning.effort, "max");
}

function testSamplingStrip() {
  const astra: Record<string, any> = {
    model: "gpt-6-astra",
    temperature: 0.2,
    top_p: 0.9,
    top_logprobs: 5,
    logprobs: true,
    include: ["reasoning.encrypted_content", "message.output_text.logprobs"],
  };
  stripGpt6UnsupportedSampling(astra);
  assert.equal(astra.temperature, undefined);
  assert.equal(astra.top_p, undefined);
  assert.equal(astra.top_logprobs, undefined);
  assert.equal(astra.logprobs, undefined);
  assert.deepEqual(astra.include, ["reasoning.encrypted_content"]);

  const luna: Record<string, any> = {
    model: "gpt-5.6-luna",
    temperature: 0.2,
    top_p: 0.9,
  };
  stripGpt6UnsupportedSampling(luna);
  assert.equal(luna.temperature, 0.2);
  assert.equal(luna.top_p, 0.9);
}

async function responsesConvertCoercesNoneAndStripsSampling() {
  const responses = new OpenAIResponsesTransformer();
  const wire = await responses.transformRequestIn(
    {
      model: "gpt-6-astra",
      messages: [{ role: "user", content: "hi" }],
      reasoning: { effort: "none", enabled: false },
      temperature: 0.7,
      top_p: 0.5,
    } as any,
    {},
    {}
  );
  assert.equal((wire as any).reasoning?.effort, "low");
  assert.equal((wire as any).temperature, undefined);
  assert.equal((wire as any).top_p, undefined);
}

async function responsesConvertCoercesMinimal() {
  const responses = new OpenAIResponsesTransformer();
  const wire = await responses.transformRequestIn(
    {
      model: "openai/gpt-6-astra",
      messages: [{ role: "user", content: "hi" }],
      reasoning: { effort: "minimal", enabled: true },
    } as any,
    {},
    {}
  );
  assert.equal((wire as any).reasoning?.effort, "low");
}

async function responsesLeavesNonGpt6NoneAlone() {
  const responses = new OpenAIResponsesTransformer();
  const wire = await responses.transformRequestIn(
    {
      model: "gpt-5.6-sol",
      messages: [{ role: "user", content: "hi" }],
      reasoning: { effort: "none", enabled: false },
      temperature: 0.1,
    } as any,
    {},
    {}
  );
  assert.equal((wire as any).reasoning?.effort, "none");
  // Non-gpt-6 Responses destinations may still carry temperature; Codex strips
  // it separately. stripGpt6 must not touch this model.
  assert.equal((wire as any).temperature, 0.1);
}

async function codexWireKeepCoercesAndStripsTopP() {
  const transformer = new CodexTransformer();
  mockCodexAuth(transformer);
  const result = await transformer.transformRequestIn(
    {
      model: "gpt-6-astra",
      input: [{ role: "user", content: "hi" }],
      reasoning: { effort: "none" },
      temperature: 0.3,
      top_p: 0.8,
      stream: false,
    },
    { baseUrl: "https://chatgpt.com/backend-api/codex" },
    { req: { id: "gpt6-codex-keep" } }
  );
  const body = result.body as any;
  assert.equal(body.reasoning?.effort, "low");
  assert.equal(body.temperature, undefined);
  assert.equal(body.top_p, undefined);
  assert.equal(body.store, false);
  assert.equal(body.stream, true);
}

async function codexAlwaysStripsTopP() {
  const transformer = new CodexTransformer();
  mockCodexAuth(transformer);
  const result = await transformer.transformRequestIn(
    {
      model: "gpt-5.6-luna",
      input: [{ role: "user", content: "hi" }],
      reasoning: { effort: "none" },
      temperature: 0.3,
      top_p: 0.8,
    },
    { baseUrl: "https://chatgpt.com/backend-api/codex" },
    { req: { id: "codex-top-p" } }
  );
  const body = result.body as any;
  assert.equal(body.reasoning?.effort, "none");
  assert.equal(body.temperature, undefined);
  assert.equal(body.top_p, undefined);
}

async function main() {
  testModelDetection();
  testEffortHelpers();
  testLunaUltraClamp();
  testSamplingStrip();
  await responsesConvertCoercesNoneAndStripsSampling();
  await responsesConvertCoercesMinimal();
  await responsesLeavesNonGpt6NoneAlone();
  await codexWireKeepCoercesAndStripsTopP();
  await codexAlwaysStripsTopP();
  console.log("gpt6.astra-hardening: all tests passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

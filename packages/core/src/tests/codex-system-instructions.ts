/**
 * Codex rejects role:"system"/"developer" in Responses input. All system-like
 * content must fold into top-level `instructions` (no drops, no residual
 * system input items) — including multi-block prompts and mid-session model
 * switches that carry more than one system message.
 */
import assert from "node:assert/strict";
import { CodexTransformer } from "../transformer/codex.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";

function mockCodexAuth(transformer: CodexTransformer) {
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });
}

async function transform(messages: any[]) {
  const unified = await new OpenAIResponsesTransformer().transformRequestIn(
    { model: "gpt-5.6-luna", messages } as any,
    {},
    {}
  );
  const transformer = new CodexTransformer();
  mockCodexAuth(transformer);
  return transformer.transformRequestIn(
    unified,
    { baseUrl: "https://chatgpt.com/backend-api/codex" },
    { req: { id: "codex-system-test" } }
  );
}

function assertNoSystemInInput(input: any[]) {
  for (const item of input) {
    assert.notEqual(
      item.role,
      "system",
      `Codex input must not contain role:system (got ${JSON.stringify(item)})`
    );
    assert.notEqual(
      item.role,
      "developer",
      `Codex input must not contain role:developer (got ${JSON.stringify(item)})`
    );
  }
}

async function foldsMultipleSystemMessagesIntoInstructions() {
  const result = await transform([
    { role: "system", content: "You are Codex from model A." },
    {
      role: "system",
      content: [{ type: "text", text: "Session context from prior model." }],
    },
    { role: "developer", content: "Be concise." },
    { role: "user", content: "continue" },
  ]);

  const body = result.body as any;
  assertNoSystemInInput(body.input || []);
  assert.equal(
    body.instructions,
    "You are Codex from model A.\n\nSession context from prior model.\n\nBe concise."
  );
  assert.equal(body.input.length, 1);
  assert.equal(body.input[0].role, "user");
}

async function foldsListOnlySystemWithoutDroppingText() {
  const result = await transform([
    {
      role: "system",
      content: [
        { type: "text", text: "Block one." },
        { type: "text", text: "Block two." },
      ],
    },
    { role: "user", content: "hi" },
  ]);

  const body = result.body as any;
  assertNoSystemInInput(body.input || []);
  assert.equal(body.instructions, "Block one.\n\nBlock two.");
}

async function singleStringSystemStillWorks() {
  const result = await transform([
    { role: "system", content: "Only one." },
    { role: "user", content: "ping" },
  ]);

  const body = result.body as any;
  assertNoSystemInInput(body.input || []);
  assert.equal(body.instructions, "Only one.");
}

async function noSystemYieldsEmptyInstructions() {
  const result = await transform([{ role: "user", content: "ping" }]);
  const body = result.body as any;
  assert.equal(body.instructions, "");
  assertNoSystemInInput(body.input || []);
}

async function responsesInstructionsAreFoldedOnce() {
  const unified = await new OpenAIResponsesTransformer().transformRequestOut({
    model: "gpt-5.6-luna",
    instructions: "Top-level instructions.",
    input: [
      { role: "developer", content: "Input developer message." },
      { role: "user", content: "ping" },
    ],
  } as any);
  const converted = await new OpenAIResponsesTransformer().transformRequestIn(
    unified,
    {},
    {}
  );
  const transformer = new CodexTransformer();
  mockCodexAuth(transformer);
  const result = await transformer.transformRequestIn(
    converted,
    { baseUrl: "https://chatgpt.com/backend-api/codex" },
    { req: { id: "codex-responses-system-test" } }
  );
  const body = result.body as any;

  assert.equal(
    body.instructions,
    "Top-level instructions.\n\nInput developer message."
  );
  assert.equal(
    body.instructions.match(/Top-level instructions\./g)?.length,
    1
  );
  assertNoSystemInInput(body.input || []);
}

async function responsesKeepFoldsSystemWithoutRebuildingInput() {
  const transformer = new CodexTransformer();
  mockCodexAuth(transformer);
  const ciphertext = "gAAAAABlcodex-keep-encrypted-reasoning";
  const result = await transformer.transformRequestIn(
    {
      model: "gpt-5.6-luna",
      store: false,
      include: ["reasoning.encrypted_content"],
      instructions: "Keep me.",
      input: [
        { role: "system", content: "Do not leak." },
        {
          type: "reasoning",
          id: "rs_keep",
          summary: [],
          encrypted_content: ciphertext,
        },
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "continue" }],
        },
      ],
    },
    { baseUrl: "https://chatgpt.com/backend-api/codex" },
    { req: { id: "codex-keep-system" } }
  );
  const body = result.body as any;
  assert.equal(body.instructions, "Keep me.\n\nDo not leak.");
  assertNoSystemInInput(body.input || []);
  assert.equal(body.store, false);
  assert.equal(body.stream, true);
  assert.deepEqual(body.include, ["reasoning.encrypted_content"]);
  assert.equal(body.input[0].encrypted_content, ciphertext);
  assert.equal(body.input[0].id, "rs_keep");
}

async function main() {
  await foldsMultipleSystemMessagesIntoInstructions();
  await foldsListOnlySystemWithoutDroppingText();
  await singleStringSystemStillWorks();
  await noSystemYieldsEmptyInstructions();
  await responsesInstructionsAreFoldedOnce();
  await responsesKeepFoldsSystemWithoutRebuildingInput();
  console.log("codex-system-instructions: all tests passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

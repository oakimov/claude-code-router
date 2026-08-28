/**
 * Codex rejects role:"system"/"developer" in Responses input. All system-like
 * content must fold into top-level `instructions` (no drops, no residual
 * system input items) — including multi-block prompts and mid-session model
 * switches that carry more than one system message.
 */
import assert from "node:assert/strict";
import { CodexTransformer } from "../transformer/codex.transformer";

function mockCodexAuth(transformer: CodexTransformer) {
  (transformer as any).resolveAuth = async () => ({
    mode: "oauth",
    token: "test-token",
    accountId: "test-account",
    isFedramp: false,
  });
}

async function transform(messages: any[]) {
  const transformer = new CodexTransformer();
  mockCodexAuth(transformer);
  return transformer.transformRequestIn(
    { model: "gpt-5.6-luna", messages } as any,
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
  assert.equal(body.instructions, "Block one.\nBlock two.");
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

async function main() {
  await foldsMultipleSystemMessagesIntoInstructions();
  await foldsListOnlySystemWithoutDroppingText();
  await singleStringSystemStillWorks();
  await noSystemYieldsEmptyInstructions();
  console.log("codex-system-instructions: all tests passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

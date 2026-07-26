/**
 * Tool schemas for Claude models served through Antigravity.
 *
 * Antigravity parses our Gemini `generateContent` body into the Gemini Schema
 * proto and re-emits it as an Anthropic tool schema. Probing the live endpoint
 * pinned the failure exactly:
 *
 *   flat object schema                → 200
 *   `anyOf` with no sibling `type`    → 400 tools.N.custom.input_schema invalid
 *   `anyOf` plus a sibling `type`     → 500 "type and anyOf cannot be both populated"
 *   nested `anyOf`                    → 400, same as above
 *
 * Claude Code's own SendMessage tool (index 18 of its tool set) has exactly the
 * failing shape, so every session using it broke. Gemini models accept typeless
 * unions, so the collapse must not touch them.
 */
import assert from "node:assert/strict";
import { buildRequestBody } from "../utils/gemini.util";
import { collapseTypelessUnions } from "../utils/schema";

/** Claude Code's SendMessage input_schema, as captured from the request log. */
const SEND_MESSAGE_SCHEMA = {
  $schema: "https://json-schema.org/draft/2020-12/schema",
  type: "object",
  properties: {
    to: { description: "Recipient: teammate name", type: "string" },
    summary: {
      description: "A 5-10 word summary shown as a preview in the UI",
      type: "string",
      maxLength: 200,
    },
    message: {
      anyOf: [
        { description: "Plain text message content", type: "string" },
        {
          anyOf: [
            {
              type: "object",
              properties: {
                type: { type: "string", const: "shutdown_request" },
                reason: { type: "string" },
              },
              required: ["type"],
              additionalProperties: false,
            },
          ],
        },
      ],
    },
  },
  required: ["to", "message"],
  additionalProperties: false,
};

function toolsFor(model: string) {
  const body = buildRequestBody({
    model,
    messages: [{ role: "user", content: "hi" }],
    tools: [
      {
        type: "function",
        function: {
          name: "SendMessage",
          description: "Send a message to another agent",
          parameters: JSON.parse(JSON.stringify(SEND_MESSAGE_SCHEMA)),
        },
      },
    ],
  } as any);
  return body.tools[0].functionDeclarations[0].parameters;
}

/** Every subschema must carry a type, and none may keep a union. */
function assertNoTypelessUnion(schema: any, path = "$") {
  if (!schema || typeof schema !== "object") return;
  if (Array.isArray(schema)) {
    schema.forEach((item, i) => assertNoTypelessUnion(item, `${path}[${i}]`));
    return;
  }
  for (const key of ["anyOf", "oneOf"]) {
    assert.equal(schema[key], undefined, `${path}.${key} must be collapsed`);
  }
  if (schema.properties) {
    for (const [name, value] of Object.entries(schema.properties)) {
      assert.ok(
        (value as any)?.type,
        `${path}.properties.${name} has no type — Antigravity cannot map it`
      );
      assertNoTypelessUnion(value, `${path}.properties.${name}`);
    }
  }
  if (schema.items) assertNoTypelessUnion(schema.items, `${path}.items`);
}

function testClaudeModelsGetCollapsedUnions() {
  for (const model of ["claude-opus-4-6-thinking", "claude-sonnet-4-6", "models/claude-fable-5"]) {
    const params = toolsFor(model);
    assertNoTypelessUnion(params, model);
    // The union collapses onto its first typed branch.
    assert.equal(params.properties.message.type, "STRING");
    assert.match(params.properties.message.description, /Plain text message content/);
    // Dropped branches are still mentioned, once.
    assert.equal(
      (params.properties.message.description.match(/alternatives:/g) || []).length,
      1
    );
    // Untouched siblings stay intact.
    assert.equal(params.properties.summary.maxLength, 200);
    assert.deepEqual(params.required, ["to", "message"]);
  }
}

function testGeminiModelsKeepUnions() {
  for (const model of ["gemini-3.6-flash-tiered", "gemini-3.1-pro-low", "gemini-2.5-flash"]) {
    const params = toolsFor(model);
    assert.ok(
      Array.isArray(params.properties.message.anyOf),
      `${model} must keep anyOf — Gemini accepts typeless unions`
    );
  }
}

/** Unit behaviour of the collapse itself. */
function testCollapseHelper() {
  // Nested unions collapse from the inside out.
  const collapsed = collapseTypelessUnions({
    type: "OBJECT",
    properties: {
      x: { anyOf: [{ anyOf: [{ type: "INTEGER" }] }, { type: "STRING" }] },
    },
  });
  assert.equal(collapsed.properties.x.type, "INTEGER");
  assert.equal(collapsed.properties.x.anyOf, undefined);

  // A union that already has a type is left alone — Gemini rejects both being
  // populated, so we must never create that combination either.
  const typed = collapseTypelessUnions({ type: "STRING", anyOf: [{ type: "STRING" }] });
  assert.equal(typed.type, "STRING");
  assert.ok(typed.anyOf, "existing type+anyOf is not ours to rewrite");

  // Arrays keep their item schema.
  const arr = collapseTypelessUnions({
    type: "ARRAY",
    items: { anyOf: [{ type: "STRING" }, { type: "NUMBER" }] },
  });
  assert.equal(arr.items.type, "STRING");
  assert.equal(arr.items.anyOf, undefined);

  // Nothing to do: schema returned unchanged.
  const plain = { type: "OBJECT", properties: { a: { type: "STRING" } } };
  assert.deepEqual(collapseTypelessUnions(plain), plain);
}

function main() {
  testClaudeModelsGetCollapsedUnions();
  testGeminiModelsKeepUnions();
  testCollapseHelper();
  console.log("antigravity.claude-tool-schema: PASS");
}

main();

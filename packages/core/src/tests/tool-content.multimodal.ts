/**
 * Multimodal tool-result round-trips across destination protocols.
 */
import assert from "node:assert/strict";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { responsesRequestToUnified } from "../utils/openai.responses.util";
import { buildRequestBody as buildGeminiBody } from "../utils/gemini.util";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";
import {
  anthropicToolResultToUnified,
  extractToolMediaForStringToolApis,
  unifiedToolContentToAnthropic,
} from "../utils/tool-content";

const IMAGE_URL = "data:image/png;base64,iVBORw0KGgo=";

function multimodalToolMessages() {
  return [
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "webfetch", arguments: "{}" },
        },
      ],
    },
    {
      role: "tool",
      tool_call_id: "call_1",
      content: [
        { type: "text", text: "Image fetched successfully" },
        {
          type: "image_url",
          image_url: { url: IMAGE_URL, detail: "auto" },
        },
      ],
    },
  ];
}

async function testResponsesRoundTrip() {
  const unified = responsesRequestToUnified({
    model: "muse",
    input: [
      {
        type: "function_call",
        call_id: "call_1",
        name: "webfetch",
        arguments: "{}",
      },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: [
          { type: "input_text", text: "Image fetched successfully" },
          { type: "input_image", image_url: IMAGE_URL, detail: "auto" },
          {
            type: "input_file",
            filename: "doc.pdf",
            file_data: "data:application/pdf;base64,AAAA",
          },
        ],
      },
    ],
  });
  const tool = unified.messages.find((m: any) => m.role === "tool");
  assert.ok(Array.isArray(tool.content));
  assert.equal(tool.content[1].type, "image_url");
  assert.equal(tool.content[2].type, "file");

  const out = await new OpenAIResponsesTransformer().transformRequestIn(
    structuredClone(unified) as any,
    {},
    {}
  );
  const fco = (out as any).input.find(
    (i: any) => i.type === "function_call_output"
  );
  assert.deepEqual(fco.output, [
    { type: "input_text", text: "Image fetched successfully" },
    { type: "input_image", image_url: IMAGE_URL, detail: "auto" },
    {
      type: "input_file",
      filename: "doc.pdf",
      file_data: "data:application/pdf;base64,AAAA",
    },
  ]);
}

async function testAnthropicToolResultImages() {
  const tf = new AnthropicTransformer();
  const result = await tf.transformRequestIn(
    {
      model: "claude-sonnet-4",
      messages: multimodalToolMessages() as any,
    } as any,
    { name: "anthropic" } as any
  );
  const body = (result as any).body;
  const user = body.messages.find(
    (m: any) =>
      m.role === "user" &&
      Array.isArray(m.content) &&
      m.content.some((c: any) => c.type === "tool_result")
  );
  assert.ok(user);
  const tr = user.content.find((c: any) => c.type === "tool_result");
  assert.ok(Array.isArray(tr.content));
  assert.equal(tr.content[0].type, "text");
  assert.equal(tr.content[1].type, "image");
  assert.equal(tr.content[1].source.type, "base64");
  assert.equal(tr.content[1].source.media_type, "image/png");

  // Inbound: Anthropic tool_result with image → Unified image_url
  const back = anthropicToolResultToUnified(tr.content);
  assert.ok(Array.isArray(back));
  assert.equal((back as any)[1].type, "image_url");
  assert.ok((back as any)[1].image_url.url.startsWith("data:image/png"));

  // Helper symmetry for document
  const withDoc = unifiedToolContentToAnthropic([
    { type: "text", text: "pdf" },
    {
      type: "file",
      filename: "a.pdf",
      file_data: "data:application/pdf;base64,BBBB",
      media_type: "application/pdf",
    },
  ]);
  assert.ok(Array.isArray(withDoc));
  assert.equal((withDoc as any)[1].type, "document");
}

async function testGeminiToolMediaSiblings() {
  const body = buildGeminiBody(
    {
      model: "gemini-2.5-flash",
      messages: multimodalToolMessages() as any,
    } as any,
    {}
  );
  const userTurn = body.contents.find(
    (c: any) =>
      c.role === "user" &&
      c.parts?.some((p: any) => p.functionResponse) &&
      c.parts?.some((p: any) => p.inlineData)
  );
  assert.ok(userTurn, "expected functionResponse + inlineData siblings");
  const fr = userTurn.parts.find((p: any) => p.functionResponse).functionResponse;
  assert.equal(fr.response.result, "Image fetched successfully");
  const img = userTurn.parts.find((p: any) => p.inlineData).inlineData;
  assert.equal(img.mimeType, "image/png");
  assert.equal(img.data, "iVBORw0KGgo=");
}

async function testChatExtractsToolMedia() {
  const tf = new OpenAITransformer();
  const out = await tf.transformRequestIn(
    {
      model: "gpt-4o",
      messages: multimodalToolMessages() as any,
    } as any,
    { name: "openai" },
    {}
  );
  const tool = out.messages.find((m: any) => m.role === "tool");
  assert.equal(tool.content, "Image fetched successfully");
  const followUp = out.messages.find(
    (m: any) =>
      m.role === "user" &&
      Array.isArray(m.content) &&
      m.content.some((p: any) => p.type === "image_url")
  );
  assert.ok(followUp);
  assert.equal(followUp.content[1].image_url.url, IMAGE_URL);
}

async function testMistralExtractsToolMedia() {
  const body = buildMistralBody(
    {
      model: "mistral-large-latest",
      messages: multimodalToolMessages() as any,
    } as any,
    {}
  );
  const tool = body.messages.find((m: any) => m.role === "tool");
  assert.equal(tool.content, "Image fetched successfully");
  const followUp = body.messages.find(
    (m: any) =>
      m.role === "user" &&
      Array.isArray(m.content) &&
      m.content.some((p: any) => p.type === "image_url")
  );
  assert.ok(followUp);
}

async function testVertexClaudeToolImages() {
  const { buildRequestBody } = await import("../utils/vertex-claude.util");
  const body = buildRequestBody({
    model: "claude-sonnet-4@20250514",
    messages: multimodalToolMessages() as any,
  } as any);
  const user = body.messages.find(
    (m: any) =>
      m.role === "user" &&
      Array.isArray(m.content) &&
      m.content.some((c: any) => c.type === "tool_result")
  );
  assert.ok(user);
  const tr = user.content.find((c: any) => c.type === "tool_result");
  assert.ok(Array.isArray(tr.content));
  assert.equal(tr.content[1].type, "image");
}

async function testExtractHelperIsIdempotentOnStrings() {
  const msgs = [
    { role: "tool", tool_call_id: "c1", content: "plain" },
    { role: "user", content: "hi" },
  ];
  const out = extractToolMediaForStringToolApis(msgs);
  assert.deepEqual(out, msgs);
}

async function main() {
  await testResponsesRoundTrip();
  await testAnthropicToolResultImages();
  await testGeminiToolMediaSiblings();
  await testChatExtractsToolMedia();
  await testMistralExtractsToolMedia();
  await testVertexClaudeToolImages();
  await testExtractHelperIsIdempotentOnStrings();
  console.log("tool-content.multimodal: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

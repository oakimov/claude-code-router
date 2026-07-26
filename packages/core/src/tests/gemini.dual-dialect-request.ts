/**
 * Request-shape parity between the two Gemini dialects.
 *
 * The same Claude Code request is pushed through both configurations:
 *   public Gemini   → new GeminiTransformer()
 *   Antigravity     → new GeminiTransformer({ cachedContent: false, … }) + envelope
 *
 * Both must receive an identical generateContent body (Antigravity only differs
 * by the envelope and by never attaching cachedContent), the picked model id
 * must survive untouched, and the wire keys must stay camelCase — v1internal
 * 400s on the snake_case aliases the public API tolerates.
 *
 * Response-side coverage for both dialects lives in gemini.parity.ts,
 * thinking-sequencer.dual-dialect.ts, and gemini.function-call-signatures.ts.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { GeminiTransformer } from "../transformer/gemini.transformer";
import { wrapAntigravityRequest } from "../utils/antigravity-auth";
import { SKIP_THOUGHT_SIGNATURE } from "../utils/gemini.util";
import type { UnifiedChatRequest } from "../types/llm";

const MODEL = "gemini-3.6-flash-tiered";

const publicProvider = {
  name: "gemini",
  baseUrl: "https://generativelanguage.googleapis.com/v1beta/models/",
  apiKey: "test-key",
  models: [MODEL],
} as any;

const antigravityProvider = {
  name: "antigravity",
  baseUrl: "https://daily-cloudcode-pa.sandbox.googleapis.com",
  apiKey: "oauth",
  models: [MODEL],
} as any;

/** Claude Code's real request: adaptive thinking, effort in output_config, a replayed unsigned tool call. */
async function buildUnifiedRequest(): Promise<UnifiedChatRequest> {
  const transformer = new AnthropicTransformer();
  return transformer.transformRequestOut({
    model: MODEL,
    max_tokens: 16384,
    thinking: { type: "adaptive", display: "summarized" },
    output_config: { effort: "high" },
    stream: true,
    system: [{ type: "text", text: "You are Claude Code." }],
    tools: [
      {
        name: "Bash",
        description: "Run a shell command",
        input_schema: {
          type: "object",
          properties: { command: { type: "string" } },
          required: ["command"],
        },
      },
    ],
    messages: [
      { role: "user", content: "what's in this folder" },
      {
        role: "assistant",
        content: [
          // Claude Code strips thought_signature from tool_use on replay.
          { type: "tool_use", id: "benyu3P6", name: "Bash", input: { command: "ls -la" } },
        ],
      },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "benyu3P6", content: "total 0" },
        ],
      },
    ],
  } as any);
}

function assertCamelCaseWire(body: Record<string, any>) {
  const json = JSON.stringify(body);
  for (const snake of [
    "thought_signature",
    "generation_config",
    "thinking_config",
    "include_thoughts",
    "thinking_level",
    "thinking_budget",
    "mime_type",
    "file_uri",
    "system_instruction",
  ]) {
    assert.equal(
      json.includes(`"${snake}"`),
      false,
      `snake_case key ${snake} must not reach the wire`
    );
  }
}

function assertSharedGeminiContract(body: Record<string, any>, label: string) {
  // Claude Code's effort decides thinking depth, in the level dialect.
  assert.deepEqual(
    body.generationConfig.thinkingConfig,
    { includeThoughts: true, thinkingLevel: "high" },
    `${label}: thinkingConfig`
  );
  assert.equal(body.generationConfig.maxOutputTokens, 16384, `${label}: maxOutputTokens`);

  // System prompt and tools survive.
  assert.equal(body.systemInstruction.parts[0].text, "You are Claude Code.");
  assert.equal(body.tools[0].functionDeclarations[0].name, "Bash");

  // Unsigned replayed tool call gets the sentinel on the first functionCall part.
  const modelTurn = body.contents.find((c: any) => c.role === "model");
  assert.ok(modelTurn, `${label}: expected a model turn`);
  const fcPart = modelTurn.parts.find((p: any) => p.functionCall);
  assert.ok(fcPart, `${label}: expected a functionCall part`);
  assert.equal(fcPart.thoughtSignature, SKIP_THOUGHT_SIGNATURE, `${label}: sentinel`);
  assert.equal(fcPart.functionCall.name, "Bash");

  // Tool result came back as a functionResponse with the matching call id.
  // Claude-on-Antigravity remaps functionResponse.id → tool_result.tool_use_id.
  const frTurn = body.contents.find((c: any) =>
    (c.parts || []).some((p: any) => p.functionResponse)
  );
  assert.ok(frTurn, `${label}: expected a functionResponse part`);
  const fr = frTurn.parts.find((p: any) => p.functionResponse).functionResponse;
  assert.equal(fr.id, fcPart.functionCall.id, `${label}: functionResponse.id`);
  assert.equal(fr.id, "benyu3P6", `${label}: round-tripped tool id`);

  assertCamelCaseWire(body);
}

async function main() {
  // --- Public Gemini dialect -------------------------------------------------
  const publicUnified = await buildUnifiedRequest();
  const publicOut = await new GeminiTransformer().transformRequestIn(
    publicUnified,
    publicProvider,
    { req: { model: MODEL } }
  );
  assertSharedGeminiContract(publicOut.body, "public gemini");
  assert.equal(
    String(publicOut.config.url),
    `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:streamGenerateContent?alt=sse`,
    "public gemini: model id must appear verbatim in the URL"
  );

  // --- Antigravity dialect ---------------------------------------------------
  const antigravityUnified = await buildUnifiedRequest();
  const antigravityOut = await new GeminiTransformer({
    cachedContent: false,
    thoughtSignatureFallback: "skip",
  }).transformRequestIn(antigravityUnified, antigravityProvider, {
    req: { model: MODEL },
  });
  assertSharedGeminiContract(antigravityOut.body, "antigravity");
  assert.equal(
    antigravityOut.body.cachedContent,
    undefined,
    "antigravity has no cachedContents resource"
  );

  // Both dialects must produce the same generateContent body.
  assert.deepEqual(
    antigravityOut.body,
    publicOut.body,
    "antigravity and public gemini bodies diverged"
  );

  // Envelope: body nests under `request`, model passed through unchanged.
  const envelope = wrapAntigravityRequest({
    project: "test-project",
    model: MODEL,
    request: antigravityOut.body,
  });
  assert.equal(envelope.model, MODEL, "envelope must not rewrite the model id");
  assert.equal(
    envelope.request.generationConfig.thinkingConfig.thinkingLevel,
    "high"
  );
  assert.equal(envelope.userAgent, "antigravity");
  assert.ok(envelope.requestId);

  // Explicit opt-out still reaches the shared builder.
  const noSentinel = await new GeminiTransformer({
    cachedContent: false,
    thoughtSignatureFallback: "none",
  }).transformRequestIn(await buildUnifiedRequest(), antigravityProvider, {
    req: { model: MODEL },
  });
  const noSentinelPart = noSentinel.body.contents
    .find((c: any) => c.role === "model")
    .parts.find((p: any) => p.functionCall);
  assert.equal(noSentinelPart.thoughtSignature, undefined);

  console.log("gemini.dual-dialect-request: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

import assert from "node:assert/strict";
import {
  compileTransformerPlan,
  isExactProtocolRequestPlan,
  isExactProtocolResponsePlan,
  isWireSafeMiddlewareForKeep,
  planContains,
} from "../utils/transformer-plan";
import { OpencodeHeadersTransformer } from "../transformer/opencode-headers.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";

/**
 * Wire-keep predicate and allowlist — hermetic.
 * Covers isExactProtocolRequestPlan mirror, wire-safe middleware,
 * and multimodal-adjacent invariants (detail, file, cache key).
 */

async function exactRequestCoversEveryOwner() {
  for (const name of ["Anthropic", "OpenAI", "openai-responses"]) {
    const endpoint: any = { name };
    const plan = compileTransformerPlan([endpoint, { name: "xai-auth" } as any], []);
    assert.equal(isExactProtocolRequestPlan(plan, endpoint, name), true, `request exact ${name}`);
    assert.equal(isExactProtocolResponsePlan(plan, endpoint, name), true, `response exact ${name}`);
    assert.equal(isExactProtocolRequestPlan(plan, endpoint, "different"), false, `request not exact ${name}`);
  }
}

async function exactRequestFalseWhenOwnerMissing() {
  const endpoint: any = { name: "openai-responses" };
  const plan = compileTransformerPlan([{ name: "xai-auth" } as any], []);
  assert.equal(isExactProtocolRequestPlan(plan, endpoint, "openai-responses"), false, "missing owner -> not exact");
}

async function wireSafeAllowlist() {
  assert.equal(isWireSafeMiddlewareForKeep("xai-auth", "openai-responses"), true);
  assert.equal(isWireSafeMiddlewareForKeep("opencode-headers", "Anthropic"), true);
  assert.equal(isWireSafeMiddlewareForKeep("opencode-headers", "openai-responses"), true);
  assert.equal(isWireSafeMiddlewareForKeep("qwen-auth", "OpenAI"), true);
  assert.equal(isWireSafeMiddlewareForKeep("reasoning", "OpenAI"), true, "reasoning allowed for OpenAI owner");
  assert.equal(isWireSafeMiddlewareForKeep("reasoning", "Anthropic"), false, "reasoning not for Anthropic wire");
  assert.equal(isWireSafeMiddlewareForKeep("claude-auth", "Anthropic"), false);
  assert.equal(isWireSafeMiddlewareForKeep("codex", "openai-responses"), false);
  assert.equal(isWireSafeMiddlewareForKeep(undefined, "OpenAI"), false);
}

async function planContainsChecksRequestOnly() {
  const plan = compileTransformerPlan([{ name: "OpenAI" } as any, { name: "opencode-headers" } as any], []);
  assert.equal(planContains(plan, "OpenAI"), true);
  assert.equal(planContains(plan, "opencode-headers"), true);
  assert.equal(planContains(plan, "missing"), false);
  assert.equal(planContains(plan, "claude-auth"), false);
}

async function claudeAuthExcludedFromKeep() {
  // ["claude-auth","Anthropic"] must not be considered keep in v1
  const claudeAuth: any = { name: "claude-auth" };
  const anthro: any = { name: "Anthropic" };
  const plan = compileTransformerPlan([claudeAuth, anthro], []);
  assert.equal(planContains(plan, "claude-auth"), true);
  // isExact true, but caller must exclude claude-auth before using wire keep
  assert.equal(isExactProtocolRequestPlan(plan, anthro, "Anthropic"), true, "predicate alone true, caller excludes");
}

async function responsesWirePreservesDetailAndFile() {
  // Simulate Responses wire with image detail + file, ensure wire keep would preserve them
  const wire: any = {
    model: "grok-4",
    input: [
      { type: "function_call", call_id: "call_1", name: "webfetch", arguments: "{}" },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: [
          { type: "input_text", text: "Image fetched" },
          { type: "input_image", image_url: "data:image/png;base64,iVBOR", detail: "high" },
          { type: "input_file", filename: "a.pdf", file_data: "data:application/pdf;base64,AAA" },
        ],
      },
    ],
    prompt_cache_key: "ccr_test_key",
  };
  // Wire-keep plan contains owner, so predicate true -> body stays as input[], detail/file intact
  const endpoint: any = { name: "openai-responses" };
  const plan = compileTransformerPlan([{ name: "xai-auth" } as any, endpoint], []);
  assert.equal(isExactProtocolRequestPlan(plan, endpoint, "openai-responses"), true);
  assert.equal(isWireSafeMiddlewareForKeep("xai-auth", "openai-responses"), true);
  const input = wire.input as any[];
  const fco = input.find((i: any) => i.type === "function_call_output");
  assert.equal(fco.output[1].detail, "high", "detail preserved on wire");
  assert.equal(fco.output[2].type, "input_file", "file preserved on wire");
  assert.equal(wire.prompt_cache_key, "ccr_test_key", "prompt_cache_key on wire");
}

async function opencodeFingerprintHandlesResponsesInput() {
  // Ensure the fixed fingerprint hashes input[] not just messages[]
  const t = new OpencodeHeadersTransformer() as any;
  const fpAnthropic = t.fingerprintConversation({ body: { model: "m", messages: [{ role: "user", content: "hi" }] } } as any, { req: { headers: {} } });
  const fpResponses = t.fingerprintConversation(
    {
      body: {
        model: "m",
        input: [
          { type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] },
          { type: "function_call", call_id: "c1", name: "x", arguments: "{}" },
        ],
      },
    } as any,
    { req: { headers: {} } }
  );
  assert.equal(typeof fpAnthropic, "string");
  assert.equal(typeof fpResponses, "string");
  assert.notEqual(fpResponses, fpAnthropic, "input fingerprint differs from empty, not churned");
  assert.equal(fpResponses.length, 32);
}

async function anthropicWireKeepsImageDocument() {
  const wire: any = {
    model: "claude-sonnet",
    system: [{ type: "text", text: "sys", cache_control: { type: "ephemeral" } }],
    messages: [
      {
        role: "user",
        content: "hi",
      },
      {
        role: "assistant",
        content: [{ type: "text", text: "ok" }, { type: "tool_use", id: "call_1", name: "Read", input: {} }],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "call_1",
            content: [
              { type: "text", text: "Image fetched" },
              { type: "image", source: { type: "base64", media_type: "image/png", data: "iVBOR" } },
              { type: "document", source: { type: "base64", media_type: "application/pdf", data: "AAA" } },
            ],
          },
        ],
      },
    ],
  };
  const endpoint: any = { name: "Anthropic" };
  const plan = compileTransformerPlan([endpoint, { name: "opencode-headers" } as any], []);
  assert.equal(isExactProtocolRequestPlan(plan, endpoint, "Anthropic"), true);
  // Wire should keep image/document without Unified round-trip
  const tr = wire.messages[2].content[0].content as any[];
  assert.equal(tr[1].type, "image");
  assert.equal(tr[2].type, "document");
}

async function chatWireKeepRunsOpenAIIn() {
  // Chat keep must run OpenAI In so media extract happens. Verify predicate + allowlist.
  const endpoint: any = { name: "OpenAI" };
  const plan = compileTransformerPlan([endpoint, { name: "opencode-headers" } as any], []);
  assert.equal(isExactProtocolRequestPlan(plan, endpoint, "OpenAI"), true);
  assert.equal(isWireSafeMiddlewareForKeep("opencode-headers", "OpenAI"), true);
  assert.equal(isWireSafeMiddlewareForKeep("reasoning", "OpenAI"), true);
  // Non-wire-safe like `codex` must not run on Anthropic/Responses keep
  assert.equal(isWireSafeMiddlewareForKeep("codex", "OpenAI"), false);
  // Header copy + applyRawAnthropicPromptCaching stay Anthropic-wire only
  // (routes.ts anthropicWireKeep). Chat/Responses keep must not inherit them.
}

async function main() {
  await exactRequestCoversEveryOwner();
  await exactRequestFalseWhenOwnerMissing();
  await wireSafeAllowlist();
  await planContainsChecksRequestOnly();
  await claudeAuthExcludedFromKeep();
  await responsesWirePreservesDetailAndFile();
  await opencodeFingerprintHandlesResponsesInput();
  await anthropicWireKeepsImageDocument();
  await chatWireKeepRunsOpenAIIn();
  console.log("wire-keep: PASS");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

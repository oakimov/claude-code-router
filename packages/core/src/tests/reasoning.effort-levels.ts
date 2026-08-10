import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { ReasoningTransformer } from "../transformer/reasoning.transformer";
import { THINK_LEVELS, ThinkLevel, UnifiedChatRequest } from "../types/llm";
import { isDeepSeekThinkingRequest } from "../utils/deepseek.util";

const ANTHROPIC_EFFORT: Record<ThinkLevel, string | undefined> = {
  none: undefined,
  minimal: "low",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "max",
  ultra: "max",
};

function unifiedRequest(effort: ThinkLevel): UnifiedChatRequest {
  return {
    model: "model",
    messages: [{ role: "user", content: "hello" }],
    reasoning: { effort, enabled: effort !== "none" },
  };
}

async function testInboundProtocols(): Promise<void> {
  const chat = new OpenAITransformer();
  const responses = new OpenAIResponsesTransformer();
  const anthropic = new AnthropicTransformer();

  for (const effort of THINK_LEVELS) {
    const chatUnified = await chat.transformRequestOut({
      model: "provider,model",
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: effort,
    });
    assert.equal(chatUnified.reasoning?.effort, effort, `Chat inbound ${effort}`);
    assert.equal(
      chatUnified.reasoning?.enabled,
      effort !== "none",
      `Chat inbound enabled ${effort}`
    );

    const responsesUnified = await responses.transformRequestOut({
      model: "provider,model",
      input: "hello",
      reasoning: { effort },
    });
    assert.equal(
      responsesUnified.reasoning?.effort,
      effort,
      `Responses inbound ${effort}`
    );
    assert.equal(
      responsesUnified.reasoning?.enabled,
      effort !== "none",
      `Responses inbound enabled ${effort}`
    );

    const anthropicUnified = await anthropic.transformRequestOut({
      model: "provider,model",
      max_tokens: 128,
      messages: [{ role: "user", content: "hello" }],
      thinking: { type: "adaptive" },
      output_config: { effort },
    });
    assert.equal(
      anthropicUnified.reasoning?.effort,
      effort,
      `Anthropic inbound ${effort}`
    );
    assert.equal(
      anthropicUnified.reasoning?.enabled,
      effort !== "none",
      `Anthropic inbound enabled ${effort}`
    );
  }

  const disabledAnthropic = await anthropic.transformRequestOut({
    model: "provider,model",
    max_tokens: 128,
    messages: [{ role: "user", content: "hello" }],
    thinking: { type: "disabled" },
  });
  assert.deepEqual(disabledAnthropic.reasoning, { enabled: false });
}

async function testOutboundProtocols(): Promise<void> {
  const chat = new OpenAITransformer();
  const responses = new OpenAIResponsesTransformer();
  const anthropic = new AnthropicTransformer();
  const anthropicProvider = {
    apiKey: "test-key",
    baseUrl: "https://api.anthropic.test",
    transformer: { use: [] },
  } as any;

  for (const effort of THINK_LEVELS) {
    const chatWire = await chat.transformRequestIn(
      unifiedRequest(effort),
      {},
      {}
    );
    assert.equal(
      chatWire.reasoning_effort,
      effort,
      `Chat outbound ${effort}`
    );
    assert.equal(chatWire.reasoning, undefined, `Chat canonical cleanup ${effort}`);

    const responsesRequest = unifiedRequest(effort);
    if (effort === "none") {
      (responsesRequest.reasoning as any).summary = "detailed";
    }
    const responsesWire = await responses.transformRequestIn(
      responsesRequest,
      {},
      {}
    );
    assert.equal(
      responsesWire.reasoning?.effort,
      effort,
      `Responses outbound ${effort}`
    );
    if (effort === "none") {
      assert.equal((responsesWire.reasoning as any).summary, undefined);
    }

    const anthropicResult = await anthropic.transformRequestIn(
      unifiedRequest(effort),
      anthropicProvider,
      {}
    );
    const anthropicWire = anthropicResult.body;
    if (effort === "none") {
      assert.deepEqual(anthropicWire.thinking, { type: "disabled" });
      assert.equal(anthropicWire.output_config, undefined);
    } else {
      assert.deepEqual(anthropicWire.thinking, { type: "adaptive" });
      assert.equal(
        anthropicWire.output_config?.effort,
        ANTHROPIC_EFFORT[effort],
        `Anthropic outbound ${effort}`
      );
    }
  }
}

async function testNoneDisablesReasoningTransformers(): Promise<void> {
  const request = unifiedRequest("none");
  assert.equal(
    isDeepSeekThinkingRequest(request, {
      name: "deepseek",
      baseUrl: "https://api.deepseek.test",
    }),
    false
  );

  const transformed = await new ReasoningTransformer().transformRequestIn(
    request
  );
  assert.equal(transformed.enable_thinking, false);
  assert.deepEqual(transformed.thinking, {
    type: "disabled",
    budget_tokens: -1,
  });
}

async function main(): Promise<void> {
  await testInboundProtocols();
  await testOutboundProtocols();
  await testNoneDisablesReasoningTransformers();
  console.log("reasoning.effort-levels: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

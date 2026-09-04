/**
 * Hermetic FIM util + outbound transformer tests (no live network).
 */
import assert from "node:assert/strict";
import {
  V1_FIM_INBOUND_KIND,
  inboundToUnifiedFim,
  shouldFimPassthrough,
  outboundFamilyFromTransformerName,
  isFimProviderTransformerName,
  buildQwenFimPrompt,
  encodeDeepseekFimBody,
  DEEPSEEK_FIM_MAX_TOKENS,
  resolveFimMistralUrl,
  resolveFimDeepseekUrl,
  resolveFimQwenCompletionsUrl,
  normalizeToFimClientJson,
  encodeFimResponseForInbound,
  normalizeFimSseDataPayload,
} from "../utils/fim";
import { FimMistralTransformer } from "../transformer/fim/fim.mistral.transformer";
import { FimDeepseekTransformer } from "../transformer/fim/fim.deepseek.transformer";
import { FimQwenTransformer } from "../transformer/fim/fim.qwen.transformer";
import { matchClientProtocol } from "../routing/protocol-endpoints";

async function main() {
  // Protocol registration
  {
    const m = matchClientProtocol("POST", "/v1/fim/completions");
    assert.ok(m);
    assert.equal(m!.protocol, "openai_fim_completions");
    assert.equal(m!.ownerTransformerName, "Fim");
    assert.equal(m!.canonicalPath, "/v1/fim/completions");

    const alias = matchClientProtocol("POST", "/fim/completions");
    assert.ok(alias);
    assert.equal(alias!.isAlias, true);
    assert.equal(matchClientProtocol("POST", "/v1/completions"), null);
  }

  // Inbound validation (Codestral only)
  {
    const unified = inboundToUnifiedFim(
      {
        model: "codestral-latest",
        prompt: "def add(a, b):\n",
        suffix: "\n    return a + b",
        max_tokens: 64,
      },
      V1_FIM_INBOUND_KIND
    );
    assert.equal(unified.prompt, "def add(a, b):\n");
    assert.equal(unified.suffix, "\n    return a + b");
    assert.equal(unified.max_tokens, 64);

    let threw = false;
    try {
      inboundToUnifiedFim({ model: "x" }, V1_FIM_INBOUND_KIND);
    } catch (err: any) {
      threw = true;
      assert.equal(err.statusCode, 400);
    }
    assert.equal(threw, true);
  }

  // Same-kind passthrough helper
  {
    assert.equal(shouldFimPassthrough("mistral", "mistral"), true);
    assert.equal(shouldFimPassthrough("deepseek", "deepseek"), true);
    assert.equal(shouldFimPassthrough("qwen", "qwen"), true);
    assert.equal(shouldFimPassthrough("mistral", "qwen"), false);
    assert.equal(outboundFamilyFromTransformerName("fim.mistral"), "mistral");
    assert.equal(isFimProviderTransformerName("fim.mistral"), true);
    assert.equal(isFimProviderTransformerName("mistral"), false);
  }

  const unified = {
    model: "codestral-latest",
    prompt: "prefix",
    suffix: "suffix",
    max_tokens: 128,
    temperature: 0.2,
    stream: false,
  };

  // fim.mistral same-kind: body passthrough
  {
    const tf = new FimMistralTransformer();
    const clientBody = {
      model: "bare",
      prompt: "prefix",
      suffix: "suffix",
      max_tokens: 128,
      custom_field: "keep-me",
    };
    const out = (await tf.transformRequestIn(
      unified,
      {
        name: "codestral-fim",
        baseUrl: "https://codestral.mistral.ai/v1/fim/completions",
        apiKey: "sk-test",
        models: ["codestral-latest"],
      },
      {
        fimInboundKind: "mistral",
        fimClientBody: clientBody,
      } as any
    )) as any;
    assert.equal(out.config.__fimPassthrough, true);
    assert.equal(out.body.custom_field, "keep-me");
    assert.equal(out.body.model, "codestral-latest");
    assert.equal(
      String(out.config.url),
      "https://codestral.mistral.ai/v1/fim/completions"
    );
  }

  // fim.deepseek cross-family from Codestral
  {
    const tf = new FimDeepseekTransformer();
    const out = (await tf.transformRequestIn(
      { ...unified, max_tokens: 8000 },
      {
        name: "deepseek-fim",
        baseUrl: "https://api.deepseek.com/beta/completions",
        apiKey: "sk-ds",
        models: ["deepseek-chat"],
      },
      { fimInboundKind: "mistral" } as any
    )) as any;
    assert.equal(out.config.__fimPassthrough, false);
    assert.equal(out.body.prompt, "prefix");
    assert.equal(out.body.suffix, "suffix");
    assert.equal(out.body.max_tokens, DEEPSEEK_FIM_MAX_TOKENS);
    assert.deepEqual(out.body.thinking, { type: "disabled" });
  }

  // fim.qwen — LM Studio + DashScope
  {
    const tf = new FimQwenTransformer();
    const expectedPrompt = buildQwenFimPrompt("prefix", "suffix");
    assert.equal(
      expectedPrompt,
      "<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>"
    );

    const lm = (await tf.transformRequestIn(
      unified,
      {
        name: "lmstudio-qwen-fim",
        baseUrl: "http://127.0.0.1:1234/v1/completions",
        apiKey: "lm-studio",
        models: ["qwen2.5-coder-7b-instruct"],
      },
      { fimInboundKind: "mistral" } as any
    )) as any;
    assert.equal(lm.body.prompt, expectedPrompt);
    assert.equal(lm.body.suffix, undefined);
    assert.equal(String(lm.config.url), "http://127.0.0.1:1234/v1/completions");

    const ds = (await tf.transformRequestIn(
      unified,
      {
        name: "dashscope-qwen-fim",
        baseUrl:
          "https://dashscope.aliyuncs.com/compatible-mode/v1/completions",
        apiKey: "sk-dash",
        models: ["qwen-coder-turbo"],
      },
      { fimInboundKind: "mistral" } as any
    )) as any;
    assert.equal(ds.body.prompt, expectedPrompt);
  }

  // Future same-kind deepseek passthrough
  {
    const tf = new FimDeepseekTransformer();
    const clientBody = {
      model: "deepseek-chat",
      prompt: "p",
      suffix: "s",
      max_tokens: 100,
    };
    const out = (await tf.transformRequestIn(
      { model: "deepseek-chat", prompt: "p", suffix: "s", max_tokens: 100 },
      {
        name: "deepseek-fim",
        baseUrl: "https://api.deepseek.com/beta",
        apiKey: "sk",
        models: ["deepseek-chat"],
      },
      { fimInboundKind: "deepseek", fimClientBody: clientBody } as any
    )) as any;
    assert.equal(out.config.__fimPassthrough, true);
    assert.equal(out.body.max_tokens, 100);
    assert.equal(out.body.thinking, undefined);
    assert.equal(
      String(out.config.url),
      "https://api.deepseek.com/beta/completions"
    );
  }

  assert.equal(
    resolveFimMistralUrl("https://api.mistral.ai").pathname,
    "/v1/fim/completions"
  );
  assert.equal(
    resolveFimDeepseekUrl("https://api.deepseek.com").pathname,
    "/beta/completions"
  );
  assert.equal(
    resolveFimQwenCompletionsUrl(
      "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).pathname,
    "/compatible-mode/v1/completions"
  );

  {
    // Codestral upstream → mistral inbound wire
    const normalized = encodeFimResponseForInbound(
      {
        id: "x",
        object: "chat.completion",
        model: "codestral-latest",
        created: 1,
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "filled" },
            finish_reason: "stop",
          },
        ],
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
      },
      "mistral"
    );
    assert.equal(normalized.object, "chat.completion");
    assert.equal(normalized.choices[0].message.content, "filled");
    assert.equal((normalized.choices[0] as any).text, undefined);
    assert.equal(normalized.model, "codestral-latest");
  }

  {
    // LM Studio text_completion → mistral inbound (v1 client) wire
    const normalized = encodeFimResponseForInbound(
      {
        id: "cmpl-qwen",
        object: "text_completion",
        model: "qwen/qwen2.5-coder-14b",
        created: 42,
        choices: [
          { index: 0, text: "    result = a + b", finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
      },
      "mistral"
    );
    assert.equal(normalized.object, "chat.completion");
    assert.equal(normalized.choices[0].message.role, "assistant");
    assert.equal(normalized.choices[0].message.content, "    result = a + b");
    assert.equal(normalized.choices[0].finish_reason, "stop");
    assert.equal(normalized.usage.completion_tokens, 5);
  }

  {
    // Future qwen inbound: encode to text_completion (symmetric with inbound)
    const normalized = encodeFimResponseForInbound(
      {
        id: "chat",
        object: "chat.completion",
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "mid" },
            finish_reason: "stop",
          },
        ],
      },
      "qwen"
    );
    assert.equal(normalized.object, "text_completion");
    assert.equal(normalized.choices[0].text, "mid");
  }

  {
    const chunk = normalizeFimSseDataPayload(
      JSON.stringify({
        id: "c",
        object: "text_completion",
        choices: [{ index: 0, text: "x", finish_reason: null }],
      }),
      "mistral"
    );
    const parsed = JSON.parse(chunk);
    assert.equal(parsed.object, "chat.completion.chunk");
    assert.equal(parsed.choices[0].delta.content, "x");
    assert.equal(parsed.choices[0].text, undefined);
  }

  {
    const body = encodeDeepseekFimBody({
      model: "m",
      prompt: "p",
      max_tokens: 99999,
    });
    assert.equal(body.max_tokens, DEEPSEEK_FIM_MAX_TOKENS);
  }

  console.log("fim.transformers: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

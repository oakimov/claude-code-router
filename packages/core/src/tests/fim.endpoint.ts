/**
 * FIM endpoint integration: separate pipeline, Router.fim, mock upstream.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { errorHandler } from "../api/middleware";
import { registerApiRoutes } from "../api/routes";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TokenizerService } from "../services/tokenizer";
import { TransformerService } from "../services/transformer";

const logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
};

interface Captured {
  url: string;
  body: any;
}

async function buildApp(config: Record<string, unknown>) {
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      HOST: "127.0.0.1",
      ...config,
      providers: config.Providers || config.providers,
    },
    logger: logger as any,
  });
  const transformerService = new TransformerService(
    configService,
    logger as any
  );
  await transformerService.initialize();
  const providerService = new ProviderService(
    configService,
    transformerService,
    logger as any
  );
  const tokenizerService = new TokenizerService(configService, logger as any);

  const app = Fastify({ logger: false });
  app.setErrorHandler(errorHandler);
  app.decorate("configService", configService);
  app.decorate("providerService", providerService);
  app.decorate("transformerService", transformerService);
  app.decorate("tokenizerService", tokenizerService);
  await registerApiRoutes(app);

  return {
    app,
    cleanup: () => {},
  };
}

async function main() {
  const captured: Captured[] = [];
  const originalFetch = globalThis.fetch;

  globalThis.fetch = (async (input: any, init?: any) => {
    const url = typeof input === "string" ? input : input?.url || String(input);
    const body = init?.body ? JSON.parse(init.body) : undefined;
    captured.push({ url, body });
    return new Response(
      JSON.stringify({
        id: "fim_test",
        object: "chat.completion",
        model: "codestral-latest",
        created: 1,
        usage: {
          prompt_tokens: 1,
          completion_tokens: 1,
          total_tokens: 2,
        },
        choices: [
          {
            index: 0,
            message: { role: "assistant", content: "mid" },
            finish_reason: "stop",
          },
        ],
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }) as any;

  try {
    // Codestral passthrough via Router.fim
    {
      captured.length = 0;
      const { app, cleanup } = await buildApp({
        HOST: "127.0.0.1",
        Providers: [
          {
            name: "codestral-fim",
            api_base_url:
              "https://codestral.mistral.ai/v1/fim/completions",
            api_key: "sk-test",
            models: ["codestral-latest"],
            transformer: { use: ["fim.mistral"] },
          },
        ],
        Router: {
          default: "codestral-fim,codestral-latest",
          fim: "codestral-fim,codestral-latest",
        },
      });

      try {
        const res = await app.inject({
          method: "POST",
          url: "/v1/fim/completions",
          payload: {
            model: "claude-sonnet-4-5",
            prompt: "def f():\n",
            suffix: "\n    pass",
            max_tokens: 32,
            custom_keep: true,
          },
        });
        assert.equal(res.statusCode, 200, res.body);
        const json = res.json();
        assert.equal(json.object, "chat.completion");
        assert.equal(json.choices[0].message.content, "mid");
        assert.equal(json.choices[0].message.role, "assistant");
        assert.equal(json.choices[0].text, undefined);
        assert.equal(captured.length, 1);
        assert.equal(
          captured[0].url,
          "https://codestral.mistral.ai/v1/fim/completions"
        );
        assert.equal(captured[0].body.prompt, "def f():\n");
        assert.equal(captured[0].body.suffix, "\n    pass");
        assert.equal(captured[0].body.custom_keep, true);
        assert.equal(captured[0].body.model, "codestral-latest");
      } finally {
        await app.close();
        cleanup();
      }
    }

    // Cross-family qwen from Codestral inbound
    {
      captured.length = 0;
      globalThis.fetch = (async (input: any, init?: any) => {
        const url =
          typeof input === "string" ? input : input?.url || String(input);
        const body = init?.body ? JSON.parse(init.body) : undefined;
        captured.push({ url, body });
        // LM Studio upstream shape
        return new Response(
          JSON.stringify({
            id: "cmpl_lm",
            object: "text_completion",
            model: "qwen2.5-coder-7b-instruct",
            choices: [{ index: 0, text: "mid", finish_reason: "stop" }],
          }),
          { status: 200, headers: { "content-type": "application/json" } }
        );
      }) as any;

      const { app, cleanup } = await buildApp({
        HOST: "127.0.0.1",
        Providers: [
          {
            name: "lmstudio-qwen-fim",
            api_base_url: "http://127.0.0.1:1234/v1/completions",
            api_key: "lm-studio",
            models: ["qwen2.5-coder-7b-instruct"],
            transformer: { use: ["fim.qwen"] },
          },
        ],
        Router: {
          fim: "lmstudio-qwen-fim,qwen2.5-coder-7b-instruct",
        },
      });

      try {
        const res = await app.inject({
          method: "POST",
          url: "/v1/fim/completions",
          payload: {
            model: "anything",
            prompt: "pre",
            suffix: "suf",
          },
        });
        assert.equal(res.statusCode, 200, res.body);
        const json = res.json();
        // Cross-family: encode to mistral inbound wire
        assert.equal(json.object, "chat.completion");
        assert.equal(json.choices[0].message.content, "mid");
        assert.equal(json.choices[0].text, undefined);
        assert.equal(captured.length, 1);
        assert.equal(
          captured[0].url,
          "http://127.0.0.1:1234/v1/completions"
        );
        assert.equal(
          captured[0].body.prompt,
          "<|fim_prefix|>pre<|fim_suffix|>suf<|fim_middle|>"
        );
        assert.equal(captured[0].body.suffix, undefined);
      } finally {
        await app.close();
        cleanup();
      }
    }

    // Codestral chat.completion upstream stays Codestral client wire (passthrough)
    {
      captured.length = 0;
      globalThis.fetch = (async (input: any, init?: any) => {
        const url =
          typeof input === "string" ? input : input?.url || String(input);
        const body = init?.body ? JSON.parse(init.body) : undefined;
        captured.push({ url, body });
        return new Response(
          JSON.stringify({
            id: "chat_shaped",
            object: "chat.completion",
            model: "codestral-latest",
            created: 1,
            usage: {
              prompt_tokens: 1,
              completion_tokens: 1,
              total_tokens: 2,
            },
            choices: [
              {
                index: 0,
                finish_reason: "stop",
                message: {
                  role: "assistant",
                  content: "    else:\n        pass",
                },
              },
            ],
          }),
          { status: 200, headers: { "content-type": "application/json" } }
        );
      }) as any;

      const { app, cleanup } = await buildApp({
        HOST: "127.0.0.1",
        Providers: [
          {
            name: "codestral-fim",
            api_base_url:
              "https://codestral.mistral.ai/v1/fim/completions",
            api_key: "sk-test",
            models: ["codestral-latest"],
            transformer: { use: ["fim.mistral"] },
          },
        ],
        Router: { fim: "codestral-fim,codestral-latest" },
      });

      try {
        const res = await app.inject({
          method: "POST",
          url: "/v1/fim/completions",
          payload: {
            model: "codestral-fim,codestral-latest",
            prompt: "def f():\n",
            suffix: "\n    return 1",
          },
        });
        assert.equal(res.statusCode, 200, res.body);
        const json = res.json();
        assert.equal(json.object, "chat.completion");
        assert.equal(
          json.choices[0].message.content,
          "    else:\n        pass"
        );
        assert.equal(json.choices[0].message.role, "assistant");
        assert.equal(json.choices[0].text, undefined);
      } finally {
        await app.close();
        cleanup();
      }
    }

    // Missing fim.* transformer → 400
    {
      const { app, cleanup } = await buildApp({
        HOST: "127.0.0.1",
        Providers: [
          {
            name: "chat-only",
            api_base_url: "https://api.mistral.ai/v1/chat/completions",
            api_key: "sk",
            models: ["mistral-small"],
            transformer: { use: ["mistral"] },
          },
        ],
        Router: { fim: "chat-only,mistral-small" },
      });

      try {
        const res = await app.inject({
          method: "POST",
          url: "/v1/fim/completions",
          payload: { model: "x", prompt: "hi" },
        });
        assert.equal(res.statusCode, 400);
        const body = res.json();
        assert.ok(
          String(body?.error?.code || "").includes("fim_transformer") ||
            String(body?.error?.message || "").includes("fim.")
        );
      } finally {
        await app.close();
        cleanup();
      }
    }

    // Missing prompt → 400
    {
      const { app, cleanup } = await buildApp({
        HOST: "127.0.0.1",
        Providers: [
          {
            name: "codestral-fim",
            api_base_url:
              "https://codestral.mistral.ai/v1/fim/completions",
            api_key: "sk",
            models: ["codestral-latest"],
            transformer: { use: ["fim.mistral"] },
          },
        ],
        Router: { fim: "codestral-fim,codestral-latest" },
      });
      try {
        const res = await app.inject({
          method: "POST",
          url: "/v1/fim/completions",
          payload: { model: "x" },
        });
        assert.equal(res.statusCode, 400);
      } finally {
        await app.close();
        cleanup();
      }
    }

    console.log("fim.endpoint: ok");
  } finally {
    globalThis.fetch = originalFetch;
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

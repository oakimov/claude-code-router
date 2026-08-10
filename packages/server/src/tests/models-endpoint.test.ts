/**
 * GET /v1/models: OpenAI list shape, provider,model ids, alias path,
 * single-model lookup, de-duplication, and API key enforcement.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { encodeClaudeModelAlias } from "@caeliq/ccr-shared";
import { registerModelsRoutes, listModels } from "../routes/models";
import { apiKeyAuth, detectClientProtocol } from "../middleware/auth";
import {
  buildModelsDevIndex,
  lookupNativeModelsDevModel,
  ModelsDevCatalogCache,
  type ModelsDevIndex,
} from "../utils/models-dev";

const SERVER_CONFIG = {
  initialConfig: {
    MODEL_ID_OUTPUT: "masked",
    providers: [
      { name: "deepseek", models: ["deepseek-chat", "deepseek-reasoner"] },
      { name: "openrouter", models: ["anthropic/claude-3.5-sonnet"] },
      // Duplicate entry and junk that must not reach the output.
      { name: "deepseek", models: ["deepseek-chat"] },
      { name: "empty", models: [] },
      { name: "", models: ["orphan"] },
      { name: "bad-models", models: "not-an-array" },
    ],
  },
};

const AUTH_CONFIG = {
  APIKEY: "test-router-secret",
  PORT: 3456,
  Providers: [{ name: "deepseek" }],
};

const MODELS_DEV_INDEX = buildModelsDevIndex({
  anthropic: {
    models: {
      "claude-sonnet-5": {
        id: "claude-sonnet-5",
        name: "Claude Sonnet 5",
        description: "Everyday Claude agent model",
        family: "claude-sonnet",
        reasoning: true,
        reasoning_options: [
          { type: "effort", values: ["low", "medium", "high", "xhigh", "max"] },
        ],
        limit: { context: 1_000_000, output: 128_000 },
      },
    },
  },
  openai: {
    models: {
      "gpt-5.6-sol": {
        id: "gpt-5.6-sol",
        name: "GPT-5.6 Sol",
        description: "Native OpenAI description",
        family: "gpt-sol",
        reasoning: true,
        reasoning_options: [
          {
            type: "effort",
            values: ["none", "low", "medium", "high", "xhigh", "max", "ultra"],
          },
        ],
        limit: { context: 1_050_000, input: 922_000, output: 128_000 },
      },
    },
  },
  vivgrid: {
    models: {
      "gpt-5.6-sol": {
        id: "gpt-5.6-sol",
        name: "Wrong reseller row",
        description: "Must not be selected",
        family: "gpt-sol",
        reasoning: false,
        limit: { context: 8_192, output: 1_024 },
      },
    },
  },
  mistral: {
    models: {
      "codestral-latest": {
        id: "codestral-latest",
        name: "Codestral (latest)",
        description: "Mistral code model",
        family: "codestral",
        reasoning: false,
        limit: { context: 256_000, output: 4_096 },
      },
    },
  },
  zhipuai: {
    models: {
      "glm-5.2": {
        id: "glm-5.2",
        name: "GLM-5.2",
        description: "Native GLM description",
        family: "glm",
        reasoning: true,
        reasoning_options: [{ type: "effort", values: ["high", "max"] }],
        limit: { context: 1_000_000, output: 131_072 },
      },
    },
  },
  zai: {
    models: {
      "glm-5.2": {
        id: "glm-5.2",
        name: "Inherited GLM row",
        description: "Must not be selected",
        family: "glm",
        reasoning: true,
        reasoning_options: [{ type: "effort", values: ["low"] }],
        limit: { context: 200_000, output: 32_000 },
      },
    },
  },
  native: {
    models: {
      "reasoner-small": {
        id: "reasoner-small",
        name: "Reasoner Small",
        description: "Small reasoning model",
        family: "reasoner",
        reasoning: true,
        reasoning_options: [{ type: "effort", values: ["low", "high"] }],
        limit: { context: 300_000, output: 16_000 },
      },
    },
  },
});

const NULL_CATALOG = { get: async () => null };

async function buildApp(
  config: any,
  withAuth: boolean,
  modelsDev: ModelsDevIndex | null = null
) {
  const app = Fastify({ logger: false });
  if (withAuth) {
    app.addHook("onRequest", async (req: any) => {
      const url = new URL(`http://127.0.0.1${req.url}`);
      req.pathname = url.pathname;
      detectClientProtocol(req);
    });
    app.addHook("preHandler", async (req, reply) => {
      await new Promise<void>((resolve) => {
        apiKeyAuth(AUTH_CONFIG)(req, reply, resolve);
      });
    });
  }
  await registerModelsRoutes(app, config, {
    catalog: modelsDev ? { get: async () => modelsDev } : NULL_CATALOG,
  });
  return app;
}

async function main(): Promise<void> {
  // Matching ignores CCR's provider prefix and selects the native models.dev
  // provider rather than a reseller duplicate.
  {
    const fromCodex = lookupNativeModelsDevModel(
      MODELS_DEV_INDEX,
      "codex,gpt-5.6-sol"
    );
    const fromCursor = lookupNativeModelsDevModel(
      MODELS_DEV_INDEX,
      "cursor,gpt-5.6-sol"
    );
    assert.equal(fromCodex?.nativeProvider, "openai");
    assert.equal(fromCodex?.name, "GPT-5.6 Sol");
    assert.deepEqual(fromCursor, fromCodex);

    const glm = lookupNativeModelsDevModel(MODELS_DEV_INDEX, "cursor,glm-5.2");
    assert.equal(glm?.nativeProvider, "zhipuai");
    assert.equal(glm?.description, "Native GLM description");
  }

  // The approved mock shapes: non-reasoning => haiku, >300k reasoning =>
  // opus, GPT-5.6 Sol => fable, while direct Anthropic family wins.
  {
    const models = listModels(
      {
        initialConfig: {
          MODEL_ID_OUTPUT: "masked",
          providers: [
            { name: "codestral", models: ["codestral-latest"] },
            { name: "codex", models: ["gpt-5.6-sol"] },
            { name: "cursor", models: ["glm-5.2"] },
            { name: "claude", models: ["claude-sonnet-5"] },
            { name: "custom", models: ["reasoner-small"] },
          ],
        },
      },
      MODELS_DEV_INDEX
    );
    assert.deepEqual(models[0], {
      id: encodeClaudeModelAlias("codestral,codestral-latest"),
      object: "model",
      created: 0,
      owned_by: "codestral",
      display_name: "Codestral (latest)",
      description: "Mistral code model",
      anthropic_family_tier: "haiku",
      context_window: 256_000,
      max_input_tokens: 256_000,
      supports_1m: false,
      max_output_tokens: 4_096,
    });
    assert.equal(models[1].anthropic_family_tier, "fable");
    assert.equal(models[1].context_window, 1_050_000);
    assert.equal(models[1].max_input_tokens, 922_000);
    assert.deepEqual(models[1].effort_levels, [
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
    ]);
    assert.equal(models[2].anthropic_family_tier, "opus");
    assert.equal(models[2].max_output_tokens, 131_072);
    assert.deepEqual(models[2].effort_levels, ["high", "max"]);
    assert.equal(models[3].anthropic_family_tier, "sonnet");
    assert.equal(models[3].id, "claude,claude-sonnet-5");
    assert.equal(models[4].anthropic_family_tier, "sonnet");
  }

  // Server cache is independent of the CLI lifecycle and keeps its last good
  // catalog when a refresh fails.
  {
    let now = 0;
    let loads = 0;
    const errors: unknown[] = [];
    const cache = new ModelsDevCatalogCache({
      ttlMs: 10,
      now: () => now,
      onError: (error) => errors.push(error),
      load: async () => {
        loads += 1;
        if (loads === 1) return MODELS_DEV_INDEX;
        throw new Error("offline");
      },
    });
    assert.equal(await cache.get(), MODELS_DEV_INDEX);
    now = 5;
    assert.equal(await cache.get(), MODELS_DEV_INDEX);
    assert.equal(loads, 1);
    now = 11;
    assert.equal(await cache.get(), MODELS_DEV_INDEX);
    assert.equal(loads, 2);
    assert.equal(errors.length, 1);
  }

  // listModels: ids, de-duplication, and skipping malformed providers
  {
    const models = listModels(SERVER_CONFIG);
    assert.deepEqual(
      models.map((m) => m.id),
      [
        encodeClaudeModelAlias("deepseek,deepseek-chat"),
        encodeClaudeModelAlias("deepseek,deepseek-reasoner"),
        encodeClaudeModelAlias("openrouter,anthropic/claude-3.5-sonnet"),
      ]
    );
    assert.equal(models[0].object, "model");
    assert.equal(models[0].owned_by, "deepseek");
    // A fixed created keeps identical configs byte-identical across restarts.
    assert.equal(models[0].created, 0);
  }

  // Literal is the default. Invalid values also fail closed to literal.
  {
    const providers = [{ name: "codex", models: ["gpt-5.6-sol"] }];
    assert.equal(
      listModels({ initialConfig: { providers } })[0].id,
      "codex,gpt-5.6-sol"
    );
    assert.equal(
      listModels({
        initialConfig: { MODEL_ID_OUTPUT: "unknown", providers },
      })[0].id,
      "codex,gpt-5.6-sol"
    );
    assert.equal(
      listModels({
        initialConfig: { MODEL_ID_OUTPUT: "masked", providers },
      })[0].id,
      encodeClaudeModelAlias("codex,gpt-5.6-sol")
    );
  }

  // GET /v1/models returns an OpenAI list envelope
  {
    const app = await buildApp(SERVER_CONFIG, false);
    const res = await app.inject({ method: "GET", url: "/v1/models" });
    assert.equal(res.statusCode, 200);
    const body = res.json();
    assert.equal(body.object, "list");
    assert.equal(body.data.length, 3);
    assert.equal(
      body.data[0].id,
      encodeClaudeModelAlias("deepseek,deepseek-chat")
    );
    await app.close();
  }

  // The HTTP surface defaults to literal ids when the option is absent.
  {
    const app = await buildApp(
      {
        initialConfig: {
          providers: [{ name: "codex", models: ["gpt-5.6-sol"] }],
        },
      },
      false
    );
    const res = await app.inject({ method: "GET", url: "/v1/models" });
    assert.equal(res.statusCode, 200);
    assert.equal(res.json().data[0].id, "codex,gpt-5.6-sol");
    await app.close();
  }

  // The HTTP route emits the same enriched metadata as listModels.
  {
    const config = {
      initialConfig: {
        MODEL_ID_OUTPUT: "masked",
        providers: [{ name: "codex", models: ["gpt-5.6-sol"] }],
      },
    };
    const app = await buildApp(config, false, MODELS_DEV_INDEX);
    const res = await app.inject({ method: "GET", url: "/v1/models" });
    assert.equal(res.statusCode, 200);
    const model = res.json().data[0];
    assert.equal(model.id, encodeClaudeModelAlias("codex,gpt-5.6-sol"));
    assert.equal(model.display_name, "GPT-5.6 Sol");
    assert.equal(model.anthropic_family_tier, "fable");
    assert.equal(model.max_input_tokens, 922_000);
    await app.close();
  }

  // /models is an alias for /v1/models
  {
    const app = await buildApp(SERVER_CONFIG, false);
    const alias = await app.inject({ method: "GET", url: "/models" });
    const canonical = await app.inject({ method: "GET", url: "/v1/models" });
    assert.equal(alias.statusCode, 200);
    assert.deepEqual(alias.json(), canonical.json());
    await app.close();
  }

  // Single-model lookup accepts both the canonical id and advertised alias.
  {
    const app = await buildApp(SERVER_CONFIG, false);
    const canonicalId = "openrouter,anthropic/claude-3.5-sonnet";
    const aliasId = encodeClaudeModelAlias(canonicalId);
    const canonical = await app.inject({
      method: "GET",
      url: `/v1/models/${encodeURIComponent(canonicalId)}`,
    });
    const aliased = await app.inject({
      method: "GET",
      url: `/v1/models/${aliasId}`,
    });
    assert.equal(canonical.statusCode, 200);
    assert.equal(aliased.statusCode, 200);
    assert.deepEqual(aliased.json(), canonical.json());
    const body = canonical.json();
    assert.equal(body.id, aliasId);
    assert.equal(body.owned_by, "openrouter");
    await app.close();
  }

  // Unknown model → OpenAI-shaped 404
  {
    const app = await buildApp(SERVER_CONFIG, false);
    const res = await app.inject({
      method: "GET",
      url: "/v1/models/nope,nothing",
    });
    assert.equal(res.statusCode, 404);
    assert.equal(res.json().error.code, "model_not_found");
    await app.close();
  }

  // No providers configured → empty list, not an error
  {
    const app = await buildApp({ initialConfig: { providers: [] } }, false);
    const res = await app.inject({ method: "GET", url: "/v1/models" });
    assert.equal(res.statusCode, 200);
    assert.deepEqual(res.json(), { object: "list", data: [] });
    await app.close();
  }

  // Production routes prefer the live ProviderService over stale boot config.
  {
    const app = await buildApp(SERVER_CONFIG, false);
    (app as any)._server = {
      providerService: {
        getProviders: () => [{ name: "live", models: ["current-model"] }],
      },
    };
    const res = await app.inject({ method: "GET", url: "/v1/models" });
    assert.deepEqual(
      res.json().data.map((model: any) => model.id),
      [encodeClaudeModelAlias("live,current-model")]
    );
    await app.close();
  }

  // Protected by the same API key check as the rest of the surface
  {
    const app = await buildApp(SERVER_CONFIG, true);

    const missing = await app.inject({ method: "GET", url: "/v1/models" });
    assert.equal(missing.statusCode, 401);
    assert.deepEqual(missing.json(), {
      error: {
        message: "APIKEY is missing",
        type: "authentication_error",
        param: null,
        code: "missing_api_key",
      },
    });

    const queryKey = await app.inject({
      method: "GET",
      url: `/v1/models?api_key=${AUTH_CONFIG.APIKEY}`,
    });
    assert.equal(queryKey.statusCode, 400);
    assert.equal(queryKey.json().error.code, "query_api_key_rejected");

    const authorized = await app.inject({
      method: "GET",
      url: "/v1/models",
      headers: { authorization: `Bearer ${AUTH_CONFIG.APIKEY}` },
    });
    assert.equal(authorized.statusCode, 200);
    assert.equal(authorized.json().data.length, 3);

    await app.close();
  }

  console.log("models-endpoint: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

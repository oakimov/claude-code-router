/**
 * GET /v1/models: OpenAI list shape, provider,model ids, alias path,
 * single-model lookup, de-duplication, and API key enforcement.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { registerModelsRoutes, listModels } from "../routes/models";
import { apiKeyAuth, detectClientProtocol } from "../middleware/auth";

const SERVER_CONFIG = {
  initialConfig: {
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

async function buildApp(config: any, withAuth: boolean) {
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
  await registerModelsRoutes(app, config);
  return app;
}

async function main(): Promise<void> {
  // listModels: ids, de-duplication, and skipping malformed providers
  {
    const models = listModels(SERVER_CONFIG);
    assert.deepEqual(
      models.map((m) => m.id),
      [
        "deepseek,deepseek-chat",
        "deepseek,deepseek-reasoner",
        "openrouter,anthropic/claude-3.5-sonnet",
      ]
    );
    assert.equal(models[0].object, "model");
    assert.equal(models[0].owned_by, "deepseek");
    // A fixed created keeps identical configs byte-identical across restarts.
    assert.equal(models[0].created, 0);
  }

  // GET /v1/models returns an OpenAI list envelope
  {
    const app = await buildApp(SERVER_CONFIG, false);
    const res = await app.inject({ method: "GET", url: "/v1/models" });
    assert.equal(res.statusCode, 200);
    const body = res.json();
    assert.equal(body.object, "list");
    assert.equal(body.data.length, 3);
    assert.equal(body.data[0].id, "deepseek,deepseek-chat");
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

  // Single-model lookup, including an id containing a comma and a slash
  {
    const app = await buildApp(SERVER_CONFIG, false);
    const res = await app.inject({
      method: "GET",
      url: `/v1/models/${encodeURIComponent("openrouter,anthropic/claude-3.5-sonnet")}`,
    });
    assert.equal(res.statusCode, 200);
    const body = res.json();
    assert.equal(body.id, "openrouter,anthropic/claude-3.5-sonnet");
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
      ["live,current-model"]
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

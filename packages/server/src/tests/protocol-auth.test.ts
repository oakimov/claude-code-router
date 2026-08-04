/**
 * Protocol-aware API key auth: Bearer, x-api-key, rejected query keys,
 * and OpenAI-shaped 401 envelopes.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { apiKeyAuth, detectClientProtocol } from "../middleware/auth";

const CONFIG = {
  APIKEY: "test-router-secret",
  HOST: "127.0.0.1",
  PORT: 3456,
  Providers: [{ name: "test" }],
};

async function buildApp() {
  const app = Fastify({ logger: false });

  app.addHook("onRequest", async (req: any) => {
    const url = new URL(`http://127.0.0.1${req.url}`);
    req.pathname = url.pathname;
    detectClientProtocol(req);
  });

  app.addHook("preHandler", async (req, reply) => {
    await new Promise<void>((resolve) => {
      apiKeyAuth(CONFIG)(req, reply, resolve);
    });
  });

  app.post("/v1/chat/completions", async () => ({ ok: true }));
  app.post("/v1/responses", async () => ({ ok: true }));
  app.post("/v1/messages", async () => ({ ok: true }));
  app.post("/preset/demo/v1/responses", async () => ({ ok: true }));
  app.post("/admin", async () => ({ ok: true }));
  return app;
}

async function main(): Promise<void> {
  const app = await buildApp();

  // Missing key → OpenAI-shaped 401 on Chat
  {
    const res = await app.inject({
      method: "POST",
      url: "/v1/chat/completions",
      payload: { model: "x", messages: [] },
    });
    assert.equal(res.statusCode, 401);
    const body = res.json();
    assert.equal(body.error.code, "missing_api_key");
    assert.ok(body.error.message);
  }

  // Bearer accepted
  {
    const res = await app.inject({
      method: "POST",
      url: "/v1/chat/completions",
      headers: { authorization: `Bearer ${CONFIG.APIKEY}` },
      payload: { model: "x", messages: [] },
    });
    assert.equal(res.statusCode, 200);
  }

  // x-api-key accepted
  {
    const res = await app.inject({
      method: "POST",
      url: "/v1/responses",
      headers: { "x-api-key": CONFIG.APIKEY },
      payload: { model: "x", input: "hi" },
    });
    assert.equal(res.statusCode, 200);
  }

  // Invalid key → protocol-shaped 401
  {
    const res = await app.inject({
      method: "POST",
      url: "/v1/responses",
      headers: { authorization: "Bearer wrong" },
      payload: { model: "x", input: "hi" },
    });
    assert.equal(res.statusCode, 401);
    const body = res.json();
    assert.equal(body.error.code, "invalid_api_key");
  }

  // Query API key rejected
  {
    const res = await app.inject({
      method: "POST",
      url: "/v1/chat/completions?key=secret",
      headers: { authorization: `Bearer ${CONFIG.APIKEY}` },
      payload: { model: "x", messages: [] },
    });
    assert.equal(res.statusCode, 400);
    const body = res.json();
    assert.equal(body.error.code, "query_api_key_rejected");
  }

  // Preset path works with Bearer
  {
    const res = await app.inject({
      method: "POST",
      url: "/preset/demo/v1/responses",
      headers: { authorization: `Bearer ${CONFIG.APIKEY}` },
      payload: { model: "x", input: "hi" },
    });
    assert.equal(res.statusCode, 200);
  }

  // A non-protocol endpoint may legitimately use a query parameter named key.
  {
    const res = await app.inject({
      method: "POST",
      url: "/admin?key=filter-value",
      headers: { authorization: `Bearer ${CONFIG.APIKEY}` },
    });
    assert.equal(res.statusCode, 200);
  }

  await app.close();
  console.log("protocol-auth: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

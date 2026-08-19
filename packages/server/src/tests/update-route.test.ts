import assert from "node:assert/strict";
import { createServer } from "../server";
import { apiKeyAuth } from "../middleware/auth";

const CONFIG = {
  APIKEY: "test-router-secret",
  HOST: "127.0.0.1",
  PORT: 3456,
  Providers: [{ name: "test" }],
};

async function testUpdateCheckRouteExists(): Promise<void> {
  const server = await createServer({
    initialConfig: CONFIG,
    logger: false,
    useJsonFile: false,
  });
  try {
    const response = await server.app.inject({
      method: "GET",
      url: "/api/update/check",
      headers: { "x-api-key": CONFIG.APIKEY },
    });
    assert.equal(response.statusCode, 200);
    const body = response.json();
    assert.equal(typeof body.hasUpdate, "boolean");
    assert.equal(typeof body.latestVersion, "string");
  } finally {
    await server.app.close();
  }
}

async function testFaviconBypassesAuth(): Promise<void> {
  const app = (await import("fastify")).default({ logger: false });
  app.addHook("preHandler", async (req, reply) => {
    await new Promise<void>((resolve) => {
      apiKeyAuth(CONFIG)(req, reply, resolve);
    });
  });
  app.get("/favicon.ico", async (_req, reply) => reply.status(204).send());
  try {
    const response = await app.inject({
      method: "GET",
      url: "/favicon.ico",
    });
    assert.equal(response.statusCode, 204);
  } finally {
    await app.close();
  }
}

async function main(): Promise<void> {
  await testUpdateCheckRouteExists();
  await testFaviconBypassesAuth();
  console.log("update-route tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

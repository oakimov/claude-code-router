import assert from "node:assert/strict";
import Fastify from "fastify";
import cookie from "@fastify/cookie";
import {
  apiKeysMatch,
  clearUiSessionCookie,
  clearUiSessionsForTests,
  createUiSession,
  revokeUiSession,
  setUiSessionCookie,
  UI_SESSION_COOKIE,
  UI_SESSION_LIMIT_FOR_TESTS,
  uiSessionCountForTests,
} from "../auth/ui-session";
import { apiKeyAuth } from "../middleware/auth";
import { createServer } from "../server";

const CONFIG = {
  APIKEY: "test-router-secret",
  HOST: "127.0.0.1",
  PORT: 3456,
  Providers: [{ name: "test" }],
};

async function buildApp() {
  const app = Fastify({ logger: false });
  await app.register(cookie);

  app.post("/api/auth/login", async (req: any, reply) => {
    if (!apiKeysMatch(req.body?.apiKey, CONFIG.APIKEY)) {
      return reply.status(401).send({ error: "Invalid API key" });
    }
    const id = createUiSession();
    setUiSessionCookie(reply, id);
    return { success: true };
  });

  app.addHook("preHandler", async (req, reply) => {
    if (req.url === "/api/auth/login") return;
    await new Promise<void>((resolve) => {
      apiKeyAuth(CONFIG)(req, reply, resolve);
    });
  });

  app.get("/api/protected", async () => ({ ok: true }));
  app.post("/api/protected", async () => ({ ok: true }));
  app.post("/api/auth/logout", async (req, reply) => {
    revokeUiSession(req);
    clearUiSessionCookie(reply);
    return { success: true };
  });
  return app;
}

function cookiePair(setCookie: string): string {
  return setCookie.split(";", 1)[0];
}

async function testRealServerLoginReadsConfigService(): Promise<void> {
  const server = await createServer({
    initialConfig: CONFIG,
    logger: false,
    useJsonFile: false,
  });
  try {
    const login = await server.app.inject({
      method: "POST",
      url: "/api/auth/login",
      payload: { apiKey: CONFIG.APIKEY },
    });
    assert.equal(login.statusCode, 200);
    assert.match(String(login.headers["set-cookie"]), new RegExp(`^${UI_SESSION_COOKIE}=`));
  } finally {
    await server.app.close();
  }
}

async function main(): Promise<void> {
  clearUiSessionsForTests();
  await testRealServerLoginReadsConfigService();
  clearUiSessionsForTests();
  const app = await buildApp();
  try {
    const invalid = await app.inject({
      method: "POST",
      url: "/api/auth/login",
      payload: { apiKey: "wrong" },
    });
    assert.equal(invalid.statusCode, 401);
    assert.equal(invalid.headers["set-cookie"], undefined);

    const login = await app.inject({
      method: "POST",
      url: "/api/auth/login",
      payload: { apiKey: CONFIG.APIKEY },
    });
    assert.equal(login.statusCode, 200);
    const setCookie = String(login.headers["set-cookie"]);
    assert.match(setCookie, new RegExp(`^${UI_SESSION_COOKIE}=`));
    assert.match(setCookie, /HttpOnly/i);
    assert.match(setCookie, /SameSite=Strict/i);
    assert.match(setCookie, /Path=\//i);
    assert.doesNotMatch(setCookie, new RegExp(CONFIG.APIKEY));
    const sessionCookie = cookiePair(setCookie);

    const httpsLogin = await app.inject({
      method: "POST",
      url: "/api/auth/login",
      headers: { "x-forwarded-proto": "https" },
      payload: { apiKey: CONFIG.APIKEY },
    });
    assert.match(String(httpsLogin.headers["set-cookie"]), /Secure/i);

    const protectedByCookie = await app.inject({
      method: "GET",
      url: "/api/protected",
      headers: { cookie: sessionCookie },
    });
    assert.equal(protectedByCookie.statusCode, 200);

    const crossOrigin = await app.inject({
      method: "POST",
      url: "/api/protected",
      headers: {
        cookie: sessionCookie,
        host: "127.0.0.1:3456",
        origin: "https://evil.example",
      },
    });
    assert.equal(crossOrigin.statusCode, 403);

    const missingOrigin = await app.inject({
      method: "POST",
      url: "/api/protected",
      headers: { cookie: sessionCookie },
    });
    assert.equal(missingOrigin.statusCode, 403);

    const sameOrigin = await app.inject({
      method: "POST",
      url: "/api/protected",
      headers: {
        cookie: sessionCookie,
        host: "127.0.0.1:3456",
        origin: "http://127.0.0.1:3456",
      },
    });
    assert.equal(sameOrigin.statusCode, 200);

    const proxiedHttps = await app.inject({
      method: "POST",
      url: "/api/protected",
      headers: {
        cookie: sessionCookie,
        host: "router.example.com",
        origin: "https://router.example.com",
        "x-forwarded-proto": "https",
      },
    });
    assert.equal(proxiedHttps.statusCode, 200);

    const headerAuth = await app.inject({
      method: "GET",
      url: "/api/protected",
      headers: { "x-api-key": CONFIG.APIKEY },
    });
    assert.equal(headerAuth.statusCode, 200);

    const bearerAuth = await app.inject({
      method: "GET",
      url: "/api/protected",
      headers: { authorization: `Bearer ${CONFIG.APIKEY}` },
    });
    assert.equal(bearerAuth.statusCode, 200);

    const logout = await app.inject({
      method: "POST",
      url: "/api/auth/logout",
      headers: {
        cookie: sessionCookie,
        host: "127.0.0.1:3456",
        origin: "http://127.0.0.1:3456",
      },
    });
    assert.equal(logout.statusCode, 200);
    assert.match(String(logout.headers["set-cookie"]), /Max-Age=0|Expires=/i);

    const revoked = await app.inject({
      method: "GET",
      url: "/api/protected",
      headers: { cookie: sessionCookie },
    });
    assert.equal(revoked.statusCode, 401);

    clearUiSessionsForTests();
    for (let i = 0; i < UI_SESSION_LIMIT_FOR_TESTS + 10; i += 1) {
      createUiSession();
    }
    assert.equal(uiSessionCountForTests(), UI_SESSION_LIMIT_FOR_TESTS);

    console.log("UI session authentication tests passed.");
  } finally {
    clearUiSessionsForTests();
    await app.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

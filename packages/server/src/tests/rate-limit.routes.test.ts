import assert from "node:assert/strict";
import Fastify from "fastify";
import rateLimit from "@fastify/rate-limit";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";

async function main(): Promise<void> {
  const app = Fastify({ logger: false });
  await app.register(rateLimit, {
    global: true,
    ...RATE_LIMIT_CONFIG,
  });

  const routeOptions = {
    config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
  };
  app.get("/first", routeOptions, async () => ({ ok: true }));
  app.get("/second", routeOptions, async () => ({ ok: true }));

  try {
    const first = await app.inject({ method: "GET", url: "/first" });
    assert.equal(first.statusCode, 200);
    assert.equal(first.headers["x-ratelimit-limit"], String(RATE_LIMIT_CONFIG.max));
    assert.equal(
      first.headers["x-ratelimit-remaining"],
      String(RATE_LIMIT_CONFIG.max - 1)
    );

    for (let i = 1; i < RATE_LIMIT_CONFIG.max; i += 1) {
      const response = await app.inject({ method: "GET", url: "/first" });
      assert.equal(response.statusCode, 200, `request ${i + 1} should pass`);
    }

    const limited = await app.inject({ method: "GET", url: "/first" });
    assert.equal(limited.statusCode, 429);
    assert.equal(limited.headers["x-ratelimit-remaining"], "0");
    assert.ok(limited.headers["retry-after"]);

    const independent = await app.inject({ method: "GET", url: "/second" });
    assert.equal(independent.statusCode, 200);
    assert.equal(
      independent.headers["x-ratelimit-remaining"],
      String(RATE_LIMIT_CONFIG.max - 1)
    );

    console.log("per-route rate-limit tests passed.");
  } finally {
    await app.close();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

/**
 * Endpoint route registration.
 *
 * `endPoint` is declared as a class field, so it only exists on an instance.
 * Transformers registered by `static TransformerName` (so that ["gemini", opts]
 * can pass options) therefore have to be instantiated before their native route
 * can be seen — and several transformers share one wire format, which Fastify
 * would reject as a duplicate route.
 */
import assert from "node:assert/strict";
import { TransformerService } from "../services/transformer";

const noop = () => {};
const logger = { info: noop, debug: noop, warn: noop, error: noop } as any;
const configService = { get: (_key: string, fallback: any = []) => fallback } as any;

async function main() {
  const service = new TransformerService(configService, logger);
  await service.initialize();

  const withEndpoint = service.getTransformersWithEndpoint();
  const byEndpoint = withEndpoint.map((entry) => [
    entry.name,
    entry.transformer.endPoint,
  ]);

  // Constructor-registered transformer still contributes its native route.
  assert.ok(
    byEndpoint.some(
      ([name, endPoint]) =>
        name === "gemini" && endPoint === "/v1beta/models/:modelAndAction"
    ),
    `gemini native endpoint missing from ${JSON.stringify(byEndpoint)}`
  );

  // Instance-registered transformers keep working.
  assert.ok(
    byEndpoint.some(
      ([name, endPoint]) => name === "Anthropic" && endPoint === "/v1/messages"
    ),
    `anthropic endpoint missing from ${JSON.stringify(byEndpoint)}`
  );

  // No endPoint may be claimed twice, or Fastify refuses to boot. `openai` is
  // registered before `vercel`, so it keeps /v1/chat/completions.
  const endPoints = withEndpoint.map((entry) => entry.transformer.endPoint);
  assert.equal(
    endPoints.length,
    new Set(endPoints).size,
    `duplicate endpoints: ${JSON.stringify(byEndpoint)}`
  );
  assert.equal(
    byEndpoint.find(([, endPoint]) => endPoint === "/v1/chat/completions")?.[0],
    "OpenAI"
  );

  // Every entry must actually have an endpoint to register.
  assert.ok(withEndpoint.every((entry) => Boolean(entry.transformer.endPoint)));

  // Boot check: the real router registers one POST route per entry. Fastify
  // throws FST_ERR_DUPLICATED_ROUTE on collisions, so this fails loudly instead
  // of taking the server down at startup.
  const { default: Fastify } = await import("fastify");
  const app = Fastify({ logger: false });
  for (const { transformer } of withEndpoint) {
    app.post(transformer.endPoint!, async () => ({ ok: true }));
  }
  await app.ready();
  await app.close();

  console.log("transformer-service.endpoints: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

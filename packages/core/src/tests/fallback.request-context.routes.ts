/**
 * Fallback attempts hand transformers the same request view as the primary
 * attempt. Fastify exposes `headers`, `server`, `signal`, … as prototype
 * getters, so cloning the request with an object spread dropped them: a
 * fallback transformer saw no client headers and no config service.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { errorHandler } from "../api/middleware";
import { registerApiRoutes } from "../api/routes";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TokenizerService } from "../services/tokenizer";
import { TransformerService } from "../services/transformer";

process.exitCode = 1;

const silent = { debug() {}, info() {}, warn() {}, error() {} };

type Seen = { provider: string; header?: unknown; config: boolean; signal: boolean };

async function main() {
  const seen: Seen[] = [];
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      Router: { default: "alpha,a" },
      fallback: { default: ["gamma,g"] },
      providers: ["alpha", "gamma"].map((name) => ({
        name,
        api_base_url: `https://${name}.invalid/v1/chat/completions`,
        api_key: `${name}-key`,
        models: [name[0]],
        transformer: { use: ["probe"] },
      })),
    },
  });
  const transformerService = new TransformerService(configService, silent);
  await transformerService.initialize();
  transformerService.registerTransformer("probe", {
    name: "probe",
    async transformRequestIn(request: any, provider: any, context: any) {
      seen.push({
        provider: provider?.name,
        header: context?.req?.headers?.["x-probe"],
        config: typeof context?.req?.server?.configService?.get === "function",
        signal: context?.req?.signal instanceof AbortSignal,
      });
      return request;
    },
  } as any);
  const providerService = new ProviderService(configService, transformerService, silent);
  const tokenizerService = new TokenizerService(configService, silent);
  await tokenizerService.initialize();

  const app = Fastify({ logger: false });
  app.decorate("configService", configService);
  app.decorate("transformerService", transformerService);
  app.decorate("providerService", providerService);
  app.decorate("tokenizerService", tokenizerService);
  app.setErrorHandler(errorHandler);
  await registerApiRoutes(app);

  const originalFetch = globalThis.fetch;
  globalThis.fetch = (async (input: any) => {
    const host = new URL(typeof input === "string" ? input : input.url).hostname;
    if (host.startsWith("alpha")) {
      return new Response("busy", { status: 503 });
    }
    return new Response(
      JSON.stringify({
        id: "chatcmpl-ok",
        object: "chat.completion",
        created: 1,
        model: "g",
        choices: [
          { index: 0, finish_reason: "stop", message: { role: "assistant", content: "ok" } },
        ],
      }),
      { status: 200, headers: { "content-type": "application/json" } }
    );
  }) as typeof fetch;
  try {
    const res = await app.inject({
      method: "POST",
      url: "/v1/chat/completions",
      headers: { "x-probe": "client-value" },
      payload: { model: "a", messages: [{ role: "user", content: "hi" }] },
    });
    assert.equal(res.statusCode, 200, res.body);
    assert.deepEqual(
      seen.map((entry) => entry.provider),
      ["alpha", "gamma"]
    );
    for (const entry of seen) {
      assert.equal(entry.header, "client-value", entry.provider);
      assert.equal(entry.config, true, entry.provider);
      assert.equal(entry.signal, true, entry.provider);
    }
  } finally {
    globalThis.fetch = originalFetch;
    await app.close();
  }
  console.log("fallback.request-context.routes: PASS");
  process.exitCode = 0;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

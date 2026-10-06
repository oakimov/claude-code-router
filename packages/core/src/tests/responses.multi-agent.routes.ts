/** Responses multi-agent repairs reach both primary and fallback wire keep. */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { registerApiRoutes } from "../api/routes";
import { errorHandler } from "../api/middleware";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TransformerService } from "../services/transformer";
import { TokenizerService } from "../services/tokenizer";

process.exitCode = 1;

const silent = { info() {}, debug() {}, warn() {}, error() {} };

async function main() {
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      Router: { default: "primary,m" },
      fallback: { default: ["secondary,m"] },
      orphanDelegationCompatibility: true,
      optimizeMultiAgentV2: true,
      providers: ["primary", "secondary"].map((name) => ({
        name,
        api_base_url: `https://${name}.invalid/v1/responses`,
        api_key: "dummy",
        models: ["m"],
        transformer: { use: ["openai-responses"] },
      })),
    },
  });
  const transformerService = new TransformerService(configService, silent);
  await transformerService.initialize();
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

  const reasoning = { type: "reasoning", id: "rs_1", encrypted_content: "opaque", summary: [] };
  const content = [
    { type: "input_text", text: "agent update" },
    { type: "input_image", image_url: "data:image/png;base64,aGVsbG8=", detail: "high" },
  ];
  const payload = {
    model: "primary,m",
    store: false,
    include: ["reasoning.encrypted_content"],
    input: [
      reasoning,
      { type: "agent_message", content },
      { type: "function_call", call_id: "call_1", name: "Read", arguments: "{}" },
      { type: "custom_tool_call_output", output: "delegated" },
      { type: "function_call_output", call_id: "call_1", output: "file" },
    ],
  };
  const originalPayload = structuredClone(payload);
  const expectedInput = [
    reasoning,
    { type: "message", role: "user", content },
    payload.input[2],
    payload.input[4],
    { type: "message", role: "user", content: "delegated" },
  ];
  const originalFetch = globalThis.fetch;
  try {
    for (const fallback of [false, true]) {
      const calls: Array<{ host: string; body: any }> = [];
      globalThis.fetch = (async (input: any, init: any) => {
        const host = new URL(String(input)).hostname;
        const body = JSON.parse(init.body);
        calls.push({ host, body });
        if (fallback && host === "primary.invalid") {
          return new Response("busy", { status: 503, headers: { "retry-after": "0.001" } });
        }
        return new Response(JSON.stringify({
          id: "resp_ok",
          object: "response",
          model: "m",
          status: "completed",
          output: [{ type: "message", role: "assistant", content: [{ type: "output_text", text: "done" }] }],
        }), { headers: { "Content-Type": "application/json" } });
      }) as typeof fetch;
      const result = await app.inject({
        method: "POST",
        url: "/v1/responses",
        headers: { "x-openai-subagent": "collab_spawn" },
        payload,
      });
      assert.equal(result.statusCode, 200, result.body);
      assert.deepEqual(calls.map((call) => call.host), fallback
        ? ["primary.invalid", "secondary.invalid"]
        : ["primary.invalid"]);
      for (const call of calls) {
        assert.deepEqual(call.body.input, expectedInput);
        assert.equal(call.body.model, "m");
        assert.equal(call.body.store, false);
        assert.deepEqual(call.body.include, ["reasoning.encrypted_content"]);
      }
      assert.deepEqual(payload, originalPayload);

      const count = calls.length;
      const rejected = await app.inject({ method: "POST", url: "/v1/responses", payload });
      assert.equal(rejected.statusCode, 400, rejected.body);
      assert.equal(calls.length, count, "missing header rejects before any upstream call");
    }
    console.log("responses.multi-agent.routes: PASS");
  } finally {
    globalThis.fetch = originalFetch;
    await app.close();
  }
}

main().then(() => { process.exitCode = 0; }).catch((error) => {
  console.error(error);
  process.exit(1);
});

/**
 * `POST /v1/responses/input_tokens` (and its `/responses/input_tokens` alias):
 * the Responses counterpart of `/v1/messages/count_tokens`.
 *
 * Covers the wire → counting-shape projection (instructions, text, tool calls,
 * tool results, tool schemas) and the full Fastify route lifecycle, including
 * the configured-tokenizer branch and the Responses error envelope.
 */
import assert from "node:assert/strict";
import Fastify from "fastify";
import { errorHandler } from "../api/middleware";
import { registerApiRoutes } from "../api/routes";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TokenizerService } from "../services/tokenizer";
import { TransformerService } from "../services/transformer";
import { responsesRequestToTokenizeRequest } from "../utils/openai.responses.util";

const logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
};

function testProjection() {
  const countRequest = responsesRequestToTokenizeRequest({
    model: "generic,gpt",
    instructions: "be brief",
    input: [
      { type: "message", role: "user", content: "hello" },
      {
        type: "function_call",
        call_id: "call_1",
        name: "Read",
        arguments: '{"path":"a.ts"}',
      },
      {
        type: "function_call_output",
        call_id: "call_1",
        output: "file contents",
      },
    ],
    tools: [
      {
        type: "function",
        name: "Read",
        description: "Read a file",
        parameters: { type: "object", properties: { path: { type: "string" } } },
      },
    ],
  });

  assert.equal(countRequest.system, "be brief");

  // The assistant tool call becomes a `tool_use` block whose input is the
  // parsed arguments — this is what the counting helpers actually count.
  const assistant = countRequest.messages.find((m) => m.role === "assistant");
  assert.ok(Array.isArray(assistant?.content));
  const block = (assistant!.content as any[]).find(
    (part) => part.type === "tool_use"
  );
  assert.deepEqual(block?.input, { path: "a.ts" });

  // The tool result stays a plain (counted) message.
  const toolResult = countRequest.messages.find((m) => m.role === "tool");
  assert.equal(toolResult?.content, "file contents");

  // Responses function tools flatten to the `input_schema` shape.
  assert.deepEqual(countRequest.tools, [
    {
      name: "Read",
      description: "Read a file",
      input_schema: { type: "object", properties: { path: { type: "string" } } },
    },
  ]);

  // A body the routed `/v1/responses` path rejects is rejected here too.
  assert.throws(
    () => responsesRequestToTokenizeRequest({ input: "x" }),
    (err: any) => err.statusCode === 400 && err.code === "invalid_body"
  );
  assert.throws(
    () =>
      responsesRequestToTokenizeRequest({
        model: "generic,gpt",
        input: "x",
        store: true,
      }),
    (err: any) => err.code === "unsupported_store"
  );
}

async function buildApp() {
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      Router: { default: "generic,gpt" },
      providers: [
        {
          name: "generic",
          api_base_url: "https://generic.invalid/v1/chat/completions",
          api_key: "generic-provider-key",
          models: ["gpt"],
        },
        {
          name: "tok",
          api_base_url: "https://tok.invalid/v1/chat/completions",
          api_key: "tok-provider-key",
          models: ["gpt"],
          // Present so getTokenizerConfigForModel resolves a config.
          tokenizer: { default: { type: "tiktoken", encoding: "cl100k_base" } },
        },
      ],
    },
  });
  const transformerService = new TransformerService(configService, logger);
  await transformerService.initialize();
  const providerService = new ProviderService(
    configService,
    transformerService,
    logger
  );
  const tokenizerService = new TokenizerService(configService, logger);
  await tokenizerService.initialize();

  const app = Fastify({ logger: false });
  app.decorate("configService", configService);
  app.decorate("transformerService", transformerService);
  app.decorate("providerService", providerService);
  app.decorate("tokenizerService", tokenizerService);
  app.setErrorHandler(errorHandler);
  await registerApiRoutes(app);
  return { app, tokenizerService };
}

async function testRoutes() {
  const { app, tokenizerService } = await buildApp();

  // Configured tokenizer branch: a `provider,model` whose provider declares a
  // tokenizer must route through TokenizerService.countTokens. Stub it so the
  // branch is observable rather than inferred from a token count.
  let countTokensCalls = 0;
  (tokenizerService as any).countTokens = async () => {
    countTokensCalls += 1;
    return { tokenCount: 4242, tokenizerUsed: "sentinel", cached: false };
  };

  const configured = await app.inject({
    method: "POST",
    url: "/v1/responses/input_tokens",
    payload: { model: "tok,gpt", input: "hello world" },
  });
  assert.equal(configured.statusCode, 200);
  assert.deepEqual(configured.json(), {
    object: "response.input_tokens",
    input_tokens: 4242,
  });
  assert.equal(countTokensCalls, 1);

  // Default tiktoken branch (provider without a tokenizer config).
  const fallback = await app.inject({
    method: "POST",
    url: "/responses/input_tokens",
    payload: { model: "generic,gpt", input: "hello world" },
  });
  assert.equal(fallback.statusCode, 200);
  const fallbackBody = fallback.json();
  assert.equal(fallbackBody.object, "response.input_tokens");
  assert.ok(fallbackBody.input_tokens > 0);
  assert.equal(countTokensCalls, 1, "tiktoken branch must not call the service");

  // Tool schemas and tool-call arguments contribute to the count.
  const withTools = await app.inject({
    method: "POST",
    url: "/responses/input_tokens",
    payload: {
      model: "generic,gpt",
      input: "hello world",
      tools: [
        {
          type: "function",
          name: "Read",
          description: "Read a file from disk",
          parameters: {
            type: "object",
            properties: { path: { type: "string" } },
          },
        },
      ],
    },
  });
  assert.ok(
    withTools.json().input_tokens > fallbackBody.input_tokens,
    "tool definitions must be counted"
  );

  // Invalid bodies return the Responses error envelope, not Anthropic's.
  const invalid = await app.inject({
    method: "POST",
    url: "/responses/input_tokens",
    payload: { input: "no model" },
  });
  assert.equal(invalid.statusCode, 400);
  const errorBody = invalid.json();
  assert.equal(errorBody.error.code, "invalid_body");
  assert.equal(errorBody.type, undefined, "must not use the Anthropic envelope");

  await app.close();
}

async function main() {
  testProjection();
  await testRoutes();
  console.log("responses.input-tokens: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

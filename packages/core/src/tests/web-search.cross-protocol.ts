/**
 * Hosted web search across inbound protocols against an Anthropic upstream:
 * - request: Responses `web_search` tools and Chat `web_search_options` become
 *   Anthropic's `web_search_20250305` server tool; Anthropic-only source data
 *   never reaches other providers' bodies;
 * - response: Anthropic `server_tool_use` / `web_search_tool_result` / text
 *   citations become Responses `web_search_call` items + `url_citation`
 *   annotations and Chat `annotations`, never a nameless tool call;
 * - history: a replayed searched Anthropic turn keeps its server blocks.
 */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import Fastify from "fastify";
import { errorHandler } from "../api/middleware";
import { registerApiRoutes } from "../api/routes";
import { ConfigService } from "../services/config";
import { ProviderService } from "../services/provider";
import { TokenizerService } from "../services/tokenizer";
import { TransformerService } from "../services/transformer";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

const logger = { debug() {}, info() {}, warn() {}, error() {} };

const GATEWAY_HEADERS = {
  "user-agent": "litellm/1.80.0",
  "x-app": "cli",
  "x-claude-code-session-id": "gateway-session",
};

const QUERY = "node lts";
const PREAMBLE = "Let me search.";
const ANSWER = "Node 24 is current.";
const RESULT = {
  type: "web_search_result",
  url: "https://nodejs.org",
  title: "Node.js",
  encrypted_content: "enc-result",
  page_age: null,
};
const CITATION = {
  type: "web_search_result_location",
  url: "https://nodejs.org",
  title: "Node.js",
  cited_text: "Node 24",
  encrypted_index: "enc-index",
};
const SERVER_USE = {
  type: "server_tool_use",
  id: "srvtoolu_1",
  name: "web_search",
  input: { query: QUERY },
};
const SERVER_RESULT = {
  type: "web_search_tool_result",
  tool_use_id: "srvtoolu_1",
  content: [RESULT],
};
const USAGE = { input_tokens: 5, output_tokens: 9, server_tool_use: { web_search_requests: 1 } };

function searchMessage() {
  return {
    id: "msg_search",
    type: "message",
    role: "assistant",
    model: "claude",
    content: [
      { type: "text", text: PREAMBLE },
      SERVER_USE,
      SERVER_RESULT,
      { type: "text", text: ANSWER, citations: [CITATION] },
    ],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: USAGE,
  };
}

function sse(type: string, data: Record<string, unknown>): string {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...data })}\n\n`;
}

function searchStream(): string {
  return [
    sse("message_start", {
      message: { id: "msg_stream", type: "message", role: "assistant", model: "claude", content: [], usage: { input_tokens: 5, output_tokens: 1 } },
    }),
    sse("content_block_start", { index: 0, content_block: { type: "text", text: "" } }),
    sse("content_block_delta", { index: 0, delta: { type: "text_delta", text: PREAMBLE } }),
    sse("content_block_stop", { index: 0 }),
    sse("content_block_start", { index: 1, content_block: { ...SERVER_USE, input: {} } }),
    sse("content_block_delta", { index: 1, delta: { type: "input_json_delta", partial_json: '{"query":' } }),
    sse("content_block_delta", { index: 1, delta: { type: "input_json_delta", partial_json: `"${QUERY}"}` } }),
    sse("content_block_stop", { index: 1 }),
    sse("content_block_start", { index: 2, content_block: SERVER_RESULT }),
    sse("content_block_stop", { index: 2 }),
    sse("content_block_start", { index: 3, content_block: { type: "text", text: "", citations: [] } }),
    sse("content_block_delta", { index: 3, delta: { type: "citations_delta", citation: CITATION } }),
    sse("content_block_delta", { index: 3, delta: { type: "text_delta", text: ANSWER } }),
    sse("content_block_stop", { index: 3 }),
    sse("message_delta", { delta: { stop_reason: "end_turn", stop_sequence: null }, usage: { output_tokens: 9 } }),
    sse("message_stop", {}),
  ].join("");
}

function chatOk(): Response {
  return new Response(
    JSON.stringify({
      id: "chatcmpl-x",
      object: "chat.completion",
      created: 1,
      model: "gpt",
      choices: [{ index: 0, finish_reason: "stop", message: { role: "assistant", content: "ok" } }],
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    }),
    { headers: { "content-type": "application/json" } }
  );
}

async function buildApp() {
  const configService = new ConfigService({
    useJsonFile: false,
    useEnvironmentVariables: false,
    initialConfig: {
      Router: { default: "anthropic,claude" },
      WEB_SEARCH_MAX_USES: 2,
      providers: [
        { name: "anthropic", api_base_url: "https://anthropic.invalid", api_key: "k", models: ["claude"], transformer: { use: ["Anthropic"] } },
        { name: "subscription", api_base_url: "https://subscription.invalid", api_key: "placeholder", models: ["claude"], transformer: { use: ["claude-auth", "Anthropic"] } },
        { name: "chat", api_base_url: "https://chat.invalid/v1/chat/completions", api_key: "k", models: ["gpt"], transformer: { use: ["OpenAI"] } },
        { name: "generic", api_base_url: "https://generic.invalid/v1/chat/completions", api_key: "k", models: ["gpt"] },
      ],
    },
  });
  const transformerService = new TransformerService(configService, logger);
  await transformerService.initialize();
  const providerService = new ProviderService(configService, transformerService, logger);
  const tokenizerService = new TokenizerService(configService, logger);
  await tokenizerService.initialize();
  const app = Fastify({ logger: false });
  app.decorate("configService", configService);
  app.decorate("transformerService", transformerService);
  app.decorate("providerService", providerService);
  app.decorate("tokenizerService", tokenizerService);
  app.setErrorHandler(errorHandler);
  await registerApiRoutes(app);
  return app;
}

function sseDataEvents(body: string): any[] {
  return body
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .filter((data) => data && data !== "[DONE]")
    .map((data) => JSON.parse(data));
}

const expectedWebSearchTool = (options: Record<string, unknown> = {}) => ({
  type: "web_search_20250305",
  name: "web_search",
  ...options,
});

async function main() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-web-search-"));
  const originalAuth = process.env.CCR_CLAUDE_AUTH_FILE;
  const originalDevice = process.env.CCR_CLAUDE_DEVICE_FILE;
  process.env.CCR_CLAUDE_AUTH_FILE = join(dir, "claude_auth.json");
  process.env.CCR_CLAUDE_DEVICE_FILE = join(dir, "claude_device.json");
  writeFileSync(
    process.env.CCR_CLAUDE_AUTH_FILE,
    JSON.stringify({ access_token: "t", token_type: "Bearer", expires_at: Math.floor(Date.now() / 1000) + 3600 }),
    { mode: 0o600 }
  );

  const app = await buildApp();
  const upstream: Array<{ url: string; body: any; headers: Headers }> = [];
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const body = JSON.parse(String(init?.body || "{}"));
    upstream.push({ url: String(input), body, headers: new Headers(init?.headers) });
    if (String(input).includes("chat.invalid") || String(input).includes("generic.invalid")) {
      return chatOk();
    }
    return body.stream
      ? new Response(searchStream(), { headers: { "content-type": "text/event-stream" } })
      : new Response(JSON.stringify(searchMessage()), { headers: { "content-type": "application/json" } });
  };

  try {
    // Anthropic-only source data stays request-local: a typed tool from an
    // Anthropic client never appears in a Chat provider's body.
    for (const provider of ["chat", "generic"]) {
      const result = await app.inject({
        method: "POST",
        url: "/v1/messages",
        payload: {
          model: `${provider},gpt`,
          max_tokens: 32,
          messages: [{ role: "user", content: "hi" }],
          tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 2 }],
        },
      });
      assert.equal(result.statusCode, 200, result.body);
      const body = upstream.at(-1)!.body;
      assert.equal("anthropic_tools" in body, false, `${provider}: anthropic_tools leaked`);
      // A destination that cannot host the search never gets it as a client
      // function the model could call.
      assert.equal(body.tools, undefined, `${provider}: ${JSON.stringify(body.tools)}`);
    }

    // Hosted search requested over Chat or Responses reaches a plain Chat
    // provider without the synthesized `web_search` function; the client's
    // own tools stay. An OpenAI-owned chain keeps the client's exact wire.
    {
      const lookup = { type: "function", function: { name: "lookup", parameters: { type: "object", properties: {} } } };
      const chat = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        payload: { model: "generic,gpt", messages: [{ role: "user", content: "hi" }], tools: [lookup], web_search_options: {} },
      });
      assert.equal(chat.statusCode, 200, chat.body);
      assert.deepEqual(upstream.at(-1)!.body.tools, [lookup]);

      const responses = await app.inject({
        method: "POST",
        url: "/v1/responses",
        payload: { model: "generic,gpt", input: "hi", tools: [{ type: "web_search" }], tool_choice: "required" },
      });
      assert.equal(responses.statusCode, 200, responses.body);
      assert.equal(upstream.at(-1)!.body.tools, undefined);
      assert.equal(upstream.at(-1)!.body.tool_choice, undefined);

      const exact = await app.inject({
        method: "POST",
        url: "/v1/chat/completions",
        payload: { model: "chat,gpt", messages: [{ role: "user", content: "hi" }], web_search_options: {} },
      });
      assert.equal(exact.statusCode, 200, exact.body);
      assert.deepEqual(upstream.at(-1)!.body.web_search_options, {});
      assert.equal(upstream.at(-1)!.body.tools, undefined);
    }

    for (const provider of ["anthropic", "subscription"]) {
      const model = `${provider},claude`;

      // Responses hosted web search → Anthropic server tool with its options;
      // custom tools are still renamed, web_search keeps its fixed name.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/responses",
          payload: {
            model,
            input: "search",
            tools: [
              {
                type: "web_search",
                filters: { allowed_domains: ["nodejs.org"] },
                user_location: { type: "approximate", city: "Berlin", country: "DE" },
              },
              { type: "function", name: "lookup", parameters: { type: "object", properties: {} } },
            ],
          },
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        const tools = upstream.at(-1)!.body.tools;
        assert.deepEqual(
          tools.find((tool: any) => tool.name === "web_search"),
          expectedWebSearchTool({
            max_uses: 2,
            allowed_domains: ["nodejs.org"],
            user_location: { type: "approximate", city: "Berlin", country: "DE" },
          }),
          `${provider}: responses hosted tool`
        );
        assert.ok(tools.some((tool: any) => tool.name === "mcp_Lookup"), JSON.stringify(tools));

        // JSON: search call item, then the answer carrying its citation.
        const output = result.json().output;
        assert.deepEqual(output[0], {
          type: "web_search_call",
          id: "ws_srvtoolu_1",
          status: "completed",
          action: { type: "search", query: QUERY },
        });
        const message = output.find((item: any) => item.type === "message");
        assert.equal(message.content[0].text, PREAMBLE + ANSWER);
        assert.deepEqual(message.content[0].annotations, [
          {
            type: "url_citation",
            url: CITATION.url,
            title: CITATION.title,
            start_index: PREAMBLE.length,
            end_index: PREAMBLE.length + ANSWER.length,
          },
        ]);
        assert.equal(output.some((item: any) => item.type === "function_call"), false);
      }

      // Responses streaming: preamble item closes, search item, answer item
      // with its citation rebased onto that item.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/responses",
          payload: { model, input: "search", stream: true, tools: [{ type: "web_search" }] },
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        const events = sseDataEvents(result.body);
        const added = events
          .filter((event) => event.type === "response.output_item.added")
          .map((event) => `${event.output_index}:${event.item.type}`);
        assert.deepEqual(added, ["0:message", "1:web_search_call", "2:message"], `${provider}: items`);
        const annotation = events.find((event) => event.type === "response.output_text.annotation.added");
        assert.deepEqual(annotation?.annotation, {
          type: "url_citation",
          url: CITATION.url,
          title: CITATION.title,
          start_index: 0,
          end_index: ANSWER.length,
        });
        assert.equal(annotation.output_index, 2);
        assert.deepEqual(
          events
            .filter((event) => event.output_index === 1)
            .map((event) => event.type),
          [
            "response.output_item.added",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.output_item.done",
          ]
        );
        const completed = events.find((event) => event.type === "response.completed").response;
        assert.deepEqual(
          completed.output.map((item: any) => item.type),
          ["message", "web_search_call", "message"]
        );
        assert.equal(completed.output[1].action.query, QUERY);
        assert.equal(completed.output[2].content[0].annotations.length, 1);
        assert.equal(
          events.some((event) => event.type === "response.function_call_arguments.delta"),
          false,
          `${provider}: search query leaked as a function call`
        );
      }

      // Chat web_search_options → the same Anthropic server tool.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/chat/completions",
          payload: {
            model,
            messages: [{ role: "user", content: "search" }],
            web_search_options: {
              user_location: { type: "approximate", approximate: { country: "DE" } },
            },
          },
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        assert.deepEqual(
          upstream.at(-1)!.body.tools,
          [expectedWebSearchTool({ max_uses: 2, user_location: { type: "approximate", country: "DE" } })],
          `${provider}: chat hosted tool`
        );
        const message = result.json().choices[0].message;
        assert.equal(message.content, PREAMBLE + ANSWER);
        assert.equal(message.tool_calls, undefined);
        assert.equal(message.web_search_calls, undefined);
        assert.deepEqual(message.annotations, [
          {
            type: "url_citation",
            url_citation: {
              url: CITATION.url,
              title: CITATION.title,
              content: CITATION.cited_text,
              start_index: PREAMBLE.length,
              end_index: PREAMBLE.length + ANSWER.length,
            },
          },
        ]);
      }

      // Chat streaming: annotations delta, no tool_calls or internal fields.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/chat/completions",
          payload: {
            model,
            stream: true,
            messages: [{ role: "user", content: "search" }],
            web_search_options: {},
          },
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        const deltas = sseDataEvents(result.body).map((chunk) => chunk.choices?.[0]?.delta ?? {});
        assert.equal(deltas.some((delta) => delta.tool_calls), false, `${provider}: orphan tool call`);
        assert.equal(deltas.some((delta) => delta.web_search_calls), false);
        const annotations = deltas.flatMap((delta) => delta.annotations ?? []);
        assert.equal(annotations.length, 1);
        assert.equal(annotations[0].url_citation.start_index, PREAMBLE.length);
        assert.equal(
          deltas.map((delta) => delta.content ?? "").join(""),
          PREAMBLE + ANSWER
        );
      }

      // History: a replayed searched turn keeps its server blocks, after
      // signed thinking and before the turn's text.
      {
        const result = await app.inject({
          method: "POST",
          url: "/v1/messages",
          headers: GATEWAY_HEADERS,
          payload: {
            model,
            max_tokens: 64,
            tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 1 }],
            messages: [
              { role: "user", content: "search" },
              {
                role: "assistant",
                content: [
                  { type: "thinking", thinking: "plan", signature: "sig" },
                  { type: "text", text: PREAMBLE },
                  SERVER_USE,
                  SERVER_RESULT,
                  { type: "text", text: ANSWER, citations: [CITATION] },
                ],
              },
              { role: "user", content: "which version?" },
            ],
          },
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        const assistant = upstream.at(-1)!.body.messages[1];
        assert.deepEqual(
          assistant.content.map((block: any) => block.type),
          ["thinking", "server_tool_use", "web_search_tool_result", "text"],
          `${provider}: replayed turn`
        );
        assert.deepEqual(assistant.content[1], SERVER_USE);
        assert.deepEqual(assistant.content[2], SERVER_RESULT);
        // The client's own typed tool keeps its own cap, not the config one.
        assert.deepEqual(upstream.at(-1)!.body.tools, [
          { type: "web_search_20250305", name: "web_search", max_uses: 1 },
        ]);
      }

      // Computer use: the client's computer-use beta is carried over only
      // while the request declares a computer_* tool; other client betas
      // never enter the emulated profile.
      for (const withTool of [true, false]) {
        const result = await app.inject({
          method: "POST",
          url: "/v1/messages",
          headers: { ...GATEWAY_HEADERS, "anthropic-beta": "computer-use-2025-11-24,unrelated-beta-2026-01-01" },
          payload: {
            model,
            max_tokens: 64,
            messages: [{ role: "user", content: "look" }],
            tools: withTool
              ? [{ type: "computer_20251124", name: "computer", display_width_px: 1024, display_height_px: 768 }]
              : [{ name: "lookup", input_schema: { type: "object", properties: {} } }],
          },
        });
        assert.equal(result.statusCode, 200, `${provider}: ${result.body}`);
        const beta = upstream.at(-1)!.headers.get("anthropic-beta") || "";
        assert.equal(beta.includes("computer-use-2025-11-24"), withTool, `${provider}: ${beta}`);
        assert.equal(beta.includes("unrelated-beta"), false, `${provider}: ${beta}`);
      }
    }

    // Recorded blocks only re-attach to the same turn: a changed turn text
    // (history rewritten between normalization and build) gets none.
    {
      const body = AnthropicTransformer.buildAnthropicBody(
        {
          model: "claude",
          messages: [
            { role: "user", content: "search" },
            { role: "assistant", content: "rewritten turn" },
          ],
        } as any,
        undefined,
        {
          protocolContext: {
            anthropicSource: {
              assistantServerBlocks: [{ text: "original turn", blocks: [SERVER_USE, SERVER_RESULT] }],
            },
          },
        }
      );
      assert.deepEqual(
        body.messages[1].content.map((block: any) => block.type),
        ["text"]
      );
    }

    console.log("web-search.cross-protocol: ok");
  } finally {
    globalThis.fetch = originalFetch;
    await app.close();
    if (originalAuth === undefined) delete process.env.CCR_CLAUDE_AUTH_FILE;
    else process.env.CCR_CLAUDE_AUTH_FILE = originalAuth;
    if (originalDevice === undefined) delete process.env.CCR_CLAUDE_DEVICE_FILE;
    else process.env.CCR_CLAUDE_DEVICE_FILE = originalDevice;
    rmSync(dir, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

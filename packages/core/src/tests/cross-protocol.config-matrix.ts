/**
 * Inbound Anthropic | Chat | Responses → the outbound wires used in the
 * live CCR config, including transformer.use chains, the provider→client
 * return path, and one replay turn.
 *
 * Config providers covered (by wire, not live OAuth):
 *   xai-supergrok   xai-auth → openai-responses
 *   claude          claude-auth → Anthropic
 *   antigravity     gemini (cachedContent:false) → Antigravity envelope
 *   codex           codex
 *   codestral       mistral
 *   openrouter/nvidia/opencode  OpenAI Chat (+ reasoning on OpenCode)
 *   chrome-nano     chrome-on-device → tooluse
 *   cursor          cache strip + SDK prompt flatten (no live Agent.run)
 */
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { ChromeOnDeviceTransformer } from "../transformer/chrome-on-device.transformer";
import { ClaudeAuthTransformer } from "../transformer/claude-auth.transformer";
import { CodexTransformer } from "../transformer/codex.transformer";
import { GeminiTransformer } from "../transformer/gemini.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";
import { OpenAITransformer } from "../transformer/openai.transformer";
import { ReasoningTransformer } from "../transformer/reasoning.transformer";
import { TooluseTransformer } from "../transformer/tooluse.transformer";
import { XaiAuthTransformer } from "../transformer/xai-auth.transformer";
import { toSdkPrompt } from "../cursor-sdk/prompt";
import { wrapAntigravityRequest } from "../utils/antigravity-auth";
import { transformResponseOut as geminiResponseOut } from "../utils/gemini.util";
import { buildRequestBody as buildMistralBody } from "../utils/mistral.util";
import { transformResponseOut as mistralResponseOut } from "../utils/mistral.util";
import {
  stripMessagesCacheControl,
  stripToolsCacheControl,
} from "../utils/cacheControl";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

const PLAN = "plan first";
const ANSWER = "visible answer";
const ANTHROPIC_SIG = "anth-sig-not-an-id";
const CIPHER = "enc-blob-not-an-id";
const REASONING_ID = "rs_abc123";

function sse(payloads: string[]): Response {
  const body = payloads.map((line) => `data: ${line}`).join("\n\n") + "\n\n";
  return new Response(body, { headers: { "Content-Type": "text/event-stream" } });
}

function parseChatSse(text: string): any[] {
  const chunks: any[] = [];
  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    chunks.push(JSON.parse(data));
  }
  return chunks;
}

type AnthropicSSE = { event: string; data: any };

function parseAnthropicSse(text: string): AnthropicSSE[] {
  const events: AnthropicSSE[] = [];
  for (const block of text.split("\n\n")) {
    const eventLine = block.split("\n").find((line) => line.startsWith("event: "));
    const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
    if (!eventLine || !dataLine) continue;
    const raw = dataLine.slice(6);
    if (raw === "[DONE]") continue;
    events.push({ event: eventLine.slice(7), data: JSON.parse(raw) });
  }
  return events;
}

function sessionCtx(extra?: Record<string, unknown>) {
  return {
    req: { id: "cfg-matrix", sessionId: "cache-session-1", log: logger },
    protocolContext: {},
    ...extra,
  } as any;
}

/** Same merge as `sendRequestToProvider` provider/model `transformRequestIn` loops. */
async function applyChain(
  unified: any,
  transformers: Array<{
    transformRequestIn: (
      request: any,
      provider: any,
      context: any
    ) => Promise<any>;
    name?: string;
  }>,
  provider: any,
  context: any
): Promise<{ body: any; config: any }> {
  let requestBody = structuredClone(unified);
  let config: any = {};
  for (const transformer of transformers) {
    const transformIn = await transformer.transformRequestIn(
      requestBody,
      provider,
      context
    );
    if (transformIn && typeof transformIn === "object" && "body" in transformIn) {
      requestBody = transformIn.body;
      const nextConfig = transformIn.config || {};
      config = {
        ...config,
        ...nextConfig,
        headers: { ...(config.headers || {}), ...(nextConfig.headers || {}) },
      };
    } else {
      requestBody = transformIn;
    }
  }
  return { body: requestBody, config };
}

function ensureHermeticClaudeAuth(): void {
  const configured = process.env.CCR_CLAUDE_AUTH_FILE;
  const defaultAuthFile = join(
    homedir(),
    ".claude-code-router",
    "claude_auth.json"
  );
  if ((configured && existsSync(configured)) || existsSync(defaultAuthFile)) {
    return;
  }
  const dir = mkdtempSync(join(tmpdir(), "ccr-claude-auth-matrix-"));
  writeFileSync(
    join(dir, "claude_auth.json"),
    JSON.stringify({
      access_token: "hermetic-subscription-token",
      token_type: "Bearer",
      expires_at: Math.floor(Date.now() / 1000) + 3600,
    }),
    { mode: 0o600 }
  );
  process.env.CCR_CLAUDE_AUTH_FILE = join(dir, "claude_auth.json");
}

async function inboundAnthropic() {
  return new AnthropicTransformer().transformRequestOut({
    model: "claude-sonnet-4-20250514",
    max_tokens: 64,
    system: [
      {
        type: "text",
        text: "stable system",
        cache_control: { type: "ephemeral" },
      },
    ],
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: PLAN, signature: ANTHROPIC_SIG },
          { type: "text", text: ANSWER },
          {
            type: "tool_use",
            id: "call_1",
            name: "Read",
            input: { path: "a.ts" },
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: "call_1",
            content: "ok",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
    ],
    tools: [
      {
        name: "Read",
        description: "read",
        input_schema: { type: "object", properties: { path: { type: "string" } } },
        cache_control: { type: "ephemeral" },
      },
    ],
  });
}

async function inboundChat() {
  return new OpenAITransformer().transformRequestOut({
    model: "gpt-4o",
    messages: [
      {
        role: "system",
        content: [
          {
            type: "text",
            text: "stable system",
            cache_control: { type: "ephemeral" },
          },
        ],
      },
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: ANSWER,
        reasoning_content: PLAN,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "Read", arguments: '{"path":"a.ts"}' },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "ok" },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "Read",
          description: "read",
          parameters: { type: "object", properties: { path: { type: "string" } } },
        },
        cache_control: { type: "ephemeral" },
      },
    ],
  });
}

async function inboundResponses() {
  return new OpenAIResponsesTransformer().transformRequestOut({
    model: "grok-4.6",
    input: [
      { role: "system", content: "stable system" },
      { role: "user", content: "hi" },
      {
        type: "reasoning",
        id: REASONING_ID,
        summary: [{ type: "summary_text", text: PLAN }],
        encrypted_content: CIPHER,
      },
      {
        type: "function_call",
        call_id: "call_1",
        name: "Read",
        arguments: '{"path":"a.ts"}',
      },
      { type: "function_call_output", call_id: "call_1", output: "ok" },
    ],
    tools: [
      {
        type: "function",
        name: "Read",
        description: "read",
        parameters: { type: "object", properties: { path: { type: "string" } } },
      },
    ],
  });
}

function assistantHasThinkingAndTool(bodyMessages: any[]): void {
  const assistant = bodyMessages.find(
    (m: any) => m.role === "assistant" && (m.thinking || m.reasoning_content)
  );
  assert.ok(assistant, "assistant thinking turn");
  assert.ok(assistant.tool_calls?.length, "assistant tool_calls");
}

async function testRequestChains() {
  const sources = [
    ["anthropic", await inboundAnthropic()],
    ["chat", await inboundChat()],
    ["responses", await inboundResponses()],
  ] as const;

  ensureHermeticClaudeAuth();

  for (const [source, unified] of sources) {
    assistantHasThinkingAndTool(unified.messages);

    const xai = await applyChain(
      unified,
      [new XaiAuthTransformer(), new OpenAIResponsesTransformer()],
      {
        name: "xai-supergrok",
        apiKey: "xai-test-key",
        baseUrl: "https://api.x.ai/v1",
      },
      sessionCtx()
    );
    assert.equal(xai.config.url, "https://api.x.ai/v1/responses");
    assert.equal(xai.config.headers.Authorization, "Bearer xai-test-key");
    const xaiInput = xai.body.input || [];
    assert.ok(
      xaiInput.some((item: any) => item.type === "reasoning"),
      `${source}→xAI reasoning item`
    );
    assert.ok(
      xaiInput.some(
        (item: any) =>
          item.type === "function_call" || item.type === "custom_tool_call"
      ),
      `${source}→xAI function_call`
    );
    assert.ok(
      JSON.stringify(xai.body).includes("cache_control") === false,
      `${source}→xAI must not forward Anthropic cache_control`
    );
    assert.ok(
      String(xai.body.prompt_cache_key || "").startsWith("ccr_"),
      `${source}→xAI prompt_cache_key`
    );

    const claude = await applyChain(
      unified,
      [new ClaudeAuthTransformer(), new AnthropicTransformer()],
      {
        name: "claude",
        apiKey: "no-key",
        baseUrl: "https://api.anthropic.com",
        transformer: { use: [{ name: "claude-auth" }, { name: "Anthropic" }] },
      },
      sessionCtx({
        protocolContext: {
          anthropicClientKind: "claude_code",
          anthropicDestinationInScope: true,
        },
        req: {
          id: "cfg-matrix",
          sessionId: "cache-session-1",
          log: logger,
          headers: { "user-agent": "claude-cli/2.0" },
        },
      })
    );
    assert.ok(
      String(claude.config.headers.Authorization || "").startsWith("Bearer ")
    );
    const anthAssistant = claude.body.messages.find(
      (m: any) => m.role === "assistant"
    );
    const types = anthAssistant.content.map((b: any) => b.type);
    assert.equal(types[0], "thinking", `${source}→claude thinking leads`);
    assert.ok(types.includes("tool_use"), `${source}→claude tool_use`);
    assert.notEqual(
      anthAssistant.content.find((b: any) => b.type === "thinking")?.signature,
      CIPHER
    );

    const geminiTf = new GeminiTransformer({
      cachedContent: false,
      thoughtSignatureFallback: "skip",
    } as any);
    (geminiTf as any).logger = logger;
    const gemini = await applyChain(
      { ...unified, model: "gemini-3-flash" },
      [geminiTf],
      {
        name: "antigravity",
        apiKey: "oauth",
        baseUrl: "https://daily-cloudcode-pa.sandbox.googleapis.com",
      },
      sessionCtx()
    );
    const wrapped = wrapAntigravityRequest({
      project: "test-project",
      model: "gemini-3-flash",
      request: gemini.body,
    });
    assert.equal(wrapped.userAgent, "antigravity");
    assert.equal(wrapped.model, "gemini-3-flash");
    assert.deepEqual(wrapped.request, gemini.body);
    const model = gemini.body.contents.find((c: any) => c.role === "model");
    assert.ok(
      model.parts.some((p: any) => p.functionCall),
      `${source}→antigravity functionCall`
    );
    assert.ok(
      JSON.stringify(gemini.body).includes("cache_control") === false,
      `${source}→Gemini must not leak cache_control`
    );

    const codexTf = new CodexTransformer();
    (codexTf as any).logger = logger;
    (codexTf as any).resolveAuth = async () => ({
      mode: "oauth",
      token: "t",
      accountId: "a",
      isFedramp: false,
    });
    const codex = await applyChain(
      { ...unified, model: "gpt-5.6-sol" },
      [codexTf],
      { name: "codex", baseUrl: "https://chatgpt.com/backend-api/codex" },
      sessionCtx()
    );
    assert.ok(
      (codex.body.input || []).some((item: any) => item.type === "reasoning"),
      `${source}→codex reasoning`
    );
    assert.equal(
      JSON.stringify(codex.body).includes("prompt_cache_breakpoint"),
      false,
      `${source}→codex key-based cache only`
    );

    const mistral = buildMistralBody(
      { ...unified, model: "codestral-latest" },
      sessionCtx(),
      { name: "codestral", baseUrl: "https://codestral.mistral.ai/v1/chat/completions" }
    );
    const mistralAssistant = mistral.messages.find(
      (m: any) => m.role === "assistant"
    );
    assert.ok(Array.isArray(mistralAssistant.content));
    assert.equal(mistralAssistant.content[0].type, "thinking");
    assert.equal(mistralAssistant.reasoning_content, undefined);
    assert.ok(String(mistral.prompt_cache_key || "").startsWith("ccr_"));

    const chatGeneric = await applyChain(
      unified,
      [new OpenAITransformer()],
      {
        name: "nvidia",
        baseUrl: "https://integrate.api.nvidia.com/v1/chat/completions",
      },
      sessionCtx()
    );
    const chatAssistant = chatGeneric.body.messages.find(
      (m: any) => m.role === "assistant"
    );
    assert.equal(chatAssistant.reasoning_content, PLAN);
    assert.equal(chatAssistant.thinking, undefined);
    assert.ok(chatAssistant.tool_calls?.length);
    assert.ok(
      JSON.stringify(chatGeneric.body).includes("cache_control") === false,
      `${source}→OpenAI-compat must strip Anthropic cache_control`
    );

    const opencode = await applyChain(
      { ...unified, model: "deepseek-v4-flash-free" },
      [new OpenAITransformer(), new ReasoningTransformer()],
      {
        name: "opencode",
        baseUrl: "https://opencode.ai/zen/v1/chat/completions",
      },
      sessionCtx()
    );
    assert.equal(opencode.body.enable_thinking, true);
    assert.equal(
      opencode.body.messages.find((m: any) => m.role === "assistant")
        .reasoning_content,
      PLAN
    );

    const chrome = await applyChain(
      unified,
      [new ChromeOnDeviceTransformer(), new TooluseTransformer()],
      { name: "chrome-nano", baseUrl: "http://127.0.0.1:3457" },
      sessionCtx()
    );
    assert.equal(chrome.config.url, "http://127.0.0.1:3457/v1/chat/completions");
    assert.ok(
      String(chrome.config.headers["x-ccr-session-id"] || "").startsWith("ccr_")
    );
    assert.ok(
      chrome.body.messages.some(
        (m: any) =>
          m.role === "system" &&
          typeof m.content === "string" &&
          m.content.includes("Tool mode is active")
      ),
      `${source}→chrome+tooluse reminder`
    );
    assert.ok(
      Array.isArray(chrome.body.tools),
      `${source}→chrome tools present (keys=${Object.keys(chrome.body || {}).join(",")})`
    );
    assert.ok(
      chrome.body.tools.some((t: any) => t.function?.name === "ExitTool"),
      `${source}→chrome+tooluse ExitTool`
    );
    assert.ok(
      JSON.stringify(chrome.body).includes("cache_control") === false,
      `${source}→chrome must strip cache_control`
    );

    const cursorBody = {
      ...structuredClone(unified),
      messages: stripMessagesCacheControl(unified.messages),
      tools: stripToolsCacheControl(unified.tools),
    };
    assert.ok(
      JSON.stringify(cursorBody).includes("cache_control") === false,
      `${source}→cursor strips cache_control`
    );
    const prompt = toSdkPrompt(cursorBody, {
      mode: "bridge",
      workspaceDir: "/tmp/ws",
    });
    const promptText = String((prompt as any).text || (prompt as any).content || prompt);
    assert.ok(promptText.includes("[system]"), `${source}→cursor system first`);
    assert.ok(promptText.includes("[user]"));
    assert.match(
      promptText,
      /\[assistant/,
      `${source}→cursor keeps assistant/tool_call history`
    );
  }
}

async function testPathBackAndReplay() {
  const anthropicClient = new AnthropicTransformer();
  (anthropicClient as any).logger = logger;
  const chatClient = new OpenAITransformer();
  const responsesClient = new OpenAIResponsesTransformer();
  (responsesClient as any).logger = logger;

  const grokStream = sse([
    `{"type":"response.reasoning_summary_text.delta","item_id":"${REASONING_ID}","delta":"${PLAN}"}`,
    JSON.stringify({
      type: "response.output_item.done",
      item: {
        id: REASONING_ID,
        type: "reasoning",
        summary: [{ type: "summary_text", text: PLAN }],
        encrypted_content: CIPHER,
      },
    }),
    `{"type":"response.output_text.delta","item_id":"msg_1","delta":"${ANSWER}"}`,
    JSON.stringify({
      type: "response.completed",
      response: {
        id: "resp_1",
        model: "grok-4.6",
        output: [
          {
            type: "reasoning",
            id: REASONING_ID,
            summary: [{ type: "summary_text", text: PLAN }],
            encrypted_content: CIPHER,
          },
          {
            type: "message",
            id: "msg_1",
            content: [{ type: "output_text", text: ANSWER }],
          },
        ],
      },
    }),
  ]);
  const grokUnified = await new OpenAIResponsesTransformer().transformResponseOut(
    grokStream
  );

  const anthEvents = parseAnthropicSse(
    await (
      await anthropicClient.transformResponseIn(grokUnified.clone(), {
        req: { id: "cfg-back-anth" },
      } as any)
    ).text()
  );
  const thinking = anthEvents
    .filter(
      (e) =>
        e.event === "content_block_delta" && e.data.delta.type === "thinking_delta"
    )
    .map((e) => e.data.delta.thinking)
    .join("");
  assert.equal(thinking, PLAN, "Grok→Anthropic thinking once");
  assert.deepEqual(
    anthEvents
      .filter((e) => e.event === "content_block_start")
      .map((e) => e.data.content_block.type),
    ["thinking", "text"]
  );

  const chatChunks = parseChatSse(
    await (await chatClient.transformResponseIn(grokUnified.clone(), {} as any)).text()
  );
  const chatThinking = chatChunks
    .map((c) => c.choices?.[0]?.delta?.thinking?.content || "")
    .join("");
  const chatReasoning = chatChunks
    .map((c) => c.choices?.[0]?.delta?.reasoning_content || "")
    .join("");
  assert.equal(chatThinking, "", "Chat Completions clients must not see Unified thinking");
  assert.equal(chatReasoning, PLAN);

  const geminiUnified = await geminiResponseOut(
    new Response(
      JSON.stringify({
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                {
                  text: PLAN,
                  thought: true,
                  thoughtSignature: ANTHROPIC_SIG,
                },
                { text: ANSWER },
              ],
            },
            finishReason: "STOP",
          },
        ],
      }),
      { headers: { "Content-Type": "application/json" } }
    ),
    "antigravity"
  );
  const geminiAnth: any = await (
    await anthropicClient.transformResponseIn(geminiUnified.clone(), {
      req: { id: "cfg-back-gemini" },
    } as any)
  ).json();
  assert.equal(
    geminiAnth.content.find((b: any) => b.type === "thinking")?.thinking,
    PLAN
  );
  assert.equal(
    geminiAnth.content.find((b: any) => b.type === "thinking")?.signature,
    ANTHROPIC_SIG
  );

  const mistralUnified = await mistralResponseOut(
    new Response(
      JSON.stringify({
        id: "m1",
        object: "chat.completion",
        choices: [
          {
            message: {
              role: "assistant",
              content: [
                { type: "thinking", thinking: [{ type: "text", text: PLAN }] },
                { type: "text", text: ANSWER },
              ],
            },
            finish_reason: "stop",
          },
        ],
      }),
      { headers: { "Content-Type": "application/json" } }
    ),
    "mistral",
    logger
  );
  const mistralChat: any = await (
    await chatClient.transformResponseIn(mistralUnified.clone(), {} as any)
  ).json();
  assert.equal(mistralChat.choices[0].message.thinking, undefined);
  assert.equal(mistralChat.choices[0].message.reasoning_content, PLAN);

  const responsesReplay = await responsesClient.transformRequestOut({
    model: "grok-4.6",
    input: [
      { role: "user", content: "hi" },
      {
        type: "reasoning",
        id: REASONING_ID,
        summary: [{ type: "summary_text", text: PLAN }],
        encrypted_content: CIPHER,
      },
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: ANSWER }],
      },
    ],
  });
  const replayed = await new OpenAIResponsesTransformer().transformRequestIn(
    structuredClone(responsesReplay),
    {},
    sessionCtx()
  );
  const reasoning = (replayed as any).input.find(
    (item: any) => item.type === "reasoning"
  );
  assert.equal(reasoning.encrypted_content, CIPHER);
  assert.equal(reasoning.summary[0].text, PLAN);
  assert.equal(
    (replayed as any).input.filter((item: any) => item.type === "reasoning")
      .length,
    1,
    "replay must not duplicate reasoning items"
  );
}

async function main() {
  await testRequestChains();
  await testPathBackAndReplay();
  console.log("cross-protocol.config-matrix: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

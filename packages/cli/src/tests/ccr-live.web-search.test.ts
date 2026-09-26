/**
 * Live server-side web search: CCR → Anthropic for all three chat protocols,
 * as third-party clients. Anthropic runs the search itself in every case.
 *
 * /v1/messages (the path Claude Code takes behind a gateway that drops its
 * fingerprint): the Anthropic-defined `web_search_20250305` tool must reach
 * Anthropic verbatim next to a renamed custom tool:
 * - non-streaming, with `stream` omitted (Messages default): JSON carrying
 *   `server_tool_use` (name `web_search`), a successful
 *   `web_search_tool_result`, a text answer and a counted search request;
 * - streaming: the same blocks as SSE events;
 * - a follow-up turn replaying the searched assistant turn (server blocks and
 *   their encrypted_content included) must be accepted.
 * Custom tool names must come back exactly as the client defined them.
 *
 * /v1/responses (`tools: [{type: "web_search"}]`) and /v1/chat/completions
 * (`web_search_options`), JSON and streaming: the search must surface as a
 * completed Responses `web_search_call` with its query plus `url_citation`
 * annotations, and as Chat `url_citation` annotations respectively.
 *
 * Opt-in (never in CI): `pnpm test:live`. Environment:
 *   CCR_URL         default http://localhost:3456
 *   CCR_API_KEY     required: the CCR server's API key
 *   CCR_LIVE_MODEL  default "claude,claude-haiku-4-5-20251001"
 * Each run performs six or more billed web searches (/v1/messages caps
 * each request at one via max_uses; Responses and Chat have no such cap).
 */
import assert from "node:assert/strict";

const CCR_URL = process.env.CCR_URL || "http://localhost:3456";
const API_KEY = requireApiKey();

function requireApiKey(): string {
  const key = process.env.CCR_API_KEY;
  if (!key) {
    console.error("ccr-live.web-search: set CCR_API_KEY to the CCR server's API key");
    process.exit(1);
  }
  return key;
}
const MODEL = process.env.CCR_LIVE_MODEL || "claude,claude-haiku-4-5-20251001";
const PROMPT =
  "Use web search to find the latest stable Node.js release version. " +
  "Answer in one sentence and name the source.";
const CUSTOM_TOOL = "get_time";
const TOOLS = [
  { type: "web_search_20250305", name: "web_search", max_uses: 1 },
  {
    name: CUSTOM_TOOL,
    description: "Get the current time. Not needed for this task.",
    input_schema: { type: "object", properties: {} },
  },
];

async function post(body: unknown, path = "/v1/messages"): Promise<Response> {
  return fetch(CCR_URL + path, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": API_KEY,
      authorization: `Bearer ${API_KEY}`,
      // Gateway-style headers: x-* survive but the claude-cli user agent and
      // x-stainless-* do not, so CCR's third-party emulation applies.
      "user-agent": "ccr-live-test/1.0",
      "x-app": "cli",
    },
    body: JSON.stringify(body),
  });
}

function describe(status: number, body: unknown): string {
  return `${status} ${JSON.stringify(body).slice(0, 600)}`;
}

/** Custom tool names must be the client's own, never CCR's mcp_ spelling. */
function assertClientToolNames(blocks: any[], label: string): void {
  for (const block of blocks) {
    if (block?.type === "tool_use") {
      assert.equal(block.name, CUSTOM_TOOL, `${label}: tool_use name ${block.name}`);
    }
    if (block?.type === "server_tool_use") {
      assert.equal(block.name, "web_search", `${label}: server_tool_use name ${block.name}`);
    }
  }
}

function assertSearchResult(blocks: any[], label: string): void {
  const result = blocks.find((block) => block?.type === "web_search_tool_result");
  assert.ok(result, `${label}: web_search_tool_result missing (${JSON.stringify(blocks).slice(0, 400)})`);
  assert.ok(
    Array.isArray(result.content),
    `${label}: web search failed: ${JSON.stringify(result.content)}`
  );
  assert.ok(result.content.length > 0, `${label}: web search returned no results`);
}

async function nonStreaming(): Promise<any[]> {
  // `stream` deliberately omitted: the Messages API default is non-streaming.
  const response = await post({
    model: MODEL,
    max_tokens: 1024,
    tools: TOOLS,
    messages: [{ role: "user", content: PROMPT }],
  });
  const text = await response.text();
  assert.match(
    response.headers.get("content-type") || "",
    /application\/json/,
    `non-streaming: content-type ${response.headers.get("content-type")}: ${text.slice(0, 300)}`
  );
  const json = JSON.parse(text);
  assert.equal(response.status, 200, `non-streaming: ${describe(response.status, json)}`);

  const content: any[] = json.content || [];
  const search = content.find((block) => block.type === "server_tool_use");
  assert.ok(search, `non-streaming: no server_tool_use: ${describe(response.status, json)}`);
  assert.equal(search.name, "web_search");
  assertSearchResult(content, "non-streaming");
  assertClientToolNames(content, "non-streaming");
  const answer = content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("");
  assert.ok(answer.trim(), `non-streaming: no text answer: ${describe(response.status, json)}`);
  const requests = json.usage?.server_tool_use?.web_search_requests ?? 0;
  assert.ok(requests >= 1, `non-streaming: usage.server_tool_use ${JSON.stringify(json.usage)}`);

  console.log(
    `  non-streaming: ${requests} search, ${
      content.find((block) => block.type === "web_search_tool_result").content.length
    } results; answer: ${answer.trim().slice(0, 120)}`
  );
  return content;
}

async function streaming(): Promise<void> {
  const response = await post({
    model: MODEL,
    max_tokens: 1024,
    stream: true,
    tools: TOOLS,
    messages: [{ role: "user", content: PROMPT }],
  });
  const text = await response.text();
  assert.equal(response.status, 200, `streaming: ${response.status} ${text.slice(0, 600)}`);
  assert.match(response.headers.get("content-type") || "", /text\/event-stream/);

  const events = text
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .filter((data) => data && data !== "[DONE]")
    .map((data) => JSON.parse(data));
  const error = events.find((event) => event.type === "error");
  assert.ok(!error, `streaming: error event ${JSON.stringify(error)}`);
  const started = events
    .filter((event) => event.type === "content_block_start")
    .map((event) => event.content_block);
  const search = started.find((block) => block?.type === "server_tool_use");
  assert.ok(search, `streaming: no server_tool_use block: ${JSON.stringify(started).slice(0, 400)}`);
  assert.equal(search.name, "web_search");
  assertSearchResult(started, "streaming");
  assertClientToolNames(started, "streaming");
  assert.ok(
    events.some((event) => event.type === "message_stop"),
    "streaming: no message_stop"
  );
  console.log(
    `  streaming: ${events.length} events, blocks ${started.map((block) => block?.type).join(",")}`
  );
}

async function replay(assistantContent: any[]): Promise<void> {
  const response = await post({
    model: MODEL,
    max_tokens: 512,
    tools: TOOLS,
    messages: [
      { role: "user", content: PROMPT },
      { role: "assistant", content: assistantContent },
      { role: "user", content: "In one short sentence: which major version is that?" },
    ],
  });
  const json = await response.json();
  assert.equal(response.status, 200, `replay: ${describe(response.status, json)}`);
  assertClientToolNames(json.content || [], "replay");
  console.log(
    `  replay: 200 ${(json.content || []).map((block: any) => block.type).join(",")}`
  );
}

function sseEvents(text: string): any[] {
  return text
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .filter((data) => data && data !== "[DONE]")
    .map((data) => JSON.parse(data));
}

function assertUrlCitations(annotations: any[], label: string, nested: boolean): void {
  assert.ok(annotations.length > 0, `${label}: no url_citation annotations`);
  for (const annotation of annotations) {
    assert.equal(annotation.type, "url_citation", `${label}: ${JSON.stringify(annotation)}`);
    const url = nested ? annotation.url_citation?.url : annotation.url;
    assert.match(url || "", /^https?:\/\//, `${label}: citation url ${JSON.stringify(annotation)}`);
  }
}

async function responsesSearch(stream: boolean): Promise<void> {
  const label = `responses${stream ? " (stream)" : ""}`;
  const response = await post(
    {
      model: MODEL,
      stream,
      max_output_tokens: 1024,
      input: PROMPT,
      tools: [{ type: "web_search" }],
    },
    "/v1/responses"
  );
  const text = await response.text();
  assert.equal(response.status, 200, `${label}: ${response.status} ${text.slice(0, 600)}`);
  let output: any[];
  if (stream) {
    const events = sseEvents(text);
    const annotationEvents = events.filter(
      (event) => event.type === "response.output_text.annotation.added"
    );
    assertUrlCitations(annotationEvents.map((event) => event.annotation), label, false);
    const completed = events.find((event) => event.type === "response.completed");
    assert.ok(completed, `${label}: no response.completed`);
    output = completed.response.output;
  } else {
    output = JSON.parse(text).output;
  }
  const call = output.find((item) => item.type === "web_search_call");
  assert.ok(call, `${label}: no web_search_call: ${JSON.stringify(output).slice(0, 400)}`);
  assert.equal(call.status, "completed", `${label}: ${JSON.stringify(call)}`);
  assert.ok(call.action?.query, `${label}: web_search_call without query`);
  const message = output.find((item) => item.type === "message" && item.content?.[0]?.annotations?.length);
  assert.ok(message, `${label}: no cited message: ${JSON.stringify(output).slice(0, 400)}`);
  assertUrlCitations(message.content[0].annotations, label, false);
  assert.equal(output.some((item) => item.type === "function_call"), false, `${label}: search leaked as function_call`);
  console.log(`  ${label}: query "${call.action.query}", ${message.content[0].annotations.length} citations`);
}

async function chatSearch(stream: boolean): Promise<void> {
  const label = `chat${stream ? " (stream)" : ""}`;
  const response = await post(
    {
      model: MODEL,
      stream,
      max_tokens: 1024,
      messages: [{ role: "user", content: PROMPT }],
      web_search_options: {},
    },
    "/v1/chat/completions"
  );
  const text = await response.text();
  assert.equal(response.status, 200, `${label}: ${response.status} ${text.slice(0, 600)}`);
  let annotations: any[];
  let answer: string;
  if (stream) {
    const deltas = sseEvents(text).map((chunk) => chunk.choices?.[0]?.delta ?? {});
    assert.equal(deltas.some((delta) => delta.tool_calls), false, `${label}: search leaked as tool_calls`);
    annotations = deltas.flatMap((delta) => delta.annotations ?? []);
    answer = deltas.map((delta) => delta.content ?? "").join("");
  } else {
    const message = JSON.parse(text).choices?.[0]?.message ?? {};
    assert.equal(message.tool_calls, undefined, `${label}: search leaked as tool_calls`);
    annotations = message.annotations ?? [];
    answer = message.content ?? "";
  }
  assertUrlCitations(annotations, label, true);
  assert.ok(answer.trim(), `${label}: no answer`);
  console.log(`  ${label}: ${annotations.length} citations; answer: ${answer.trim().slice(0, 100)}`);
}

async function main(): Promise<void> {
  console.log(`ccr-live.web-search: ${MODEL} via ${CCR_URL}`);
  const content = await nonStreaming();
  await streaming();
  await replay(content);
  await responsesSearch(false);
  await responsesSearch(true);
  await chatSearch(false);
  await chatSearch(true);
  console.log("ccr-live.web-search: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

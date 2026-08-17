/**
 * Responses reasoning items can carry the same summary on three event types.
 * Late handlers exist to rescue ciphertext / item id and must not re-send
 * content, because Unified `delta.thinking.content` is additive.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";
import { CodexTransformer } from "../transformer/codex.transformer";
import { OpenAIResponsesTransformer } from "../transformer/openai.responses.transformer";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

const SUMMARY = "Think harder";
const CIPHER = "CIPHER";
const REASONING_ID = "rs_abc";

function duplicatedReasoningEvents(includeSummaryDeltas: boolean): string[] {
  const events: string[] = ['{"type":"response.created","response":{"id":"resp_1","model":"gpt"}}'];
  if (includeSummaryDeltas) {
    events.push(
      `{"type":"response.reasoning_summary_text.delta","item_id":"${REASONING_ID}","delta":"Think "}`,
      `{"type":"response.reasoning_summary_text.delta","item_id":"${REASONING_ID}","delta":"harder"}`,
      `{"type":"response.reasoning_summary_text.done","item_id":"${REASONING_ID}","text":"${SUMMARY}"}`
    );
  }
  events.push(
    JSON.stringify({
      type: "response.output_item.done",
      item_id: REASONING_ID,
      item: {
        id: REASONING_ID,
        type: "reasoning",
        summary: [{ type: "summary_text", text: SUMMARY }],
        encrypted_content: CIPHER,
      },
    }),
    '{"type":"response.output_text.delta","item_id":"msg_1","delta":"Hi"}',
    JSON.stringify({
      type: "response.completed",
      response: {
        id: "resp_1",
        model: "gpt",
        output: [
          {
            type: "reasoning",
            id: REASONING_ID,
            summary: [{ type: "summary_text", text: SUMMARY }],
            encrypted_content: CIPHER,
          },
          {
            type: "message",
            id: "msg_1",
            content: [{ type: "output_text", text: "Hi" }],
          },
        ],
      },
    })
  );
  return events;
}

function sseResponse(lines: string[]): Response {
  const body = lines.map((line) => `data: ${line}`).join("\n\n") + "\n\n";
  return new Response(body, {
    headers: { "Content-Type": "text/event-stream" },
  });
}

interface ParsedStream {
  chunks: any[];
  thinkingText: string;
  text: string;
}

async function convertStream(
  transformer: OpenAIResponsesTransformer | CodexTransformer,
  lines: string[]
): Promise<ParsedStream> {
  (transformer as any).logger = logger;
  const out = await transformer.transformResponseOut(sseResponse(lines));
  const chunks: any[] = [];
  for (const line of (await out.text()).split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const data = line.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    chunks.push(JSON.parse(data));
  }
  return {
    chunks,
    thinkingText: chunks
      .map((chunk) => chunk.choices?.[0]?.delta?.thinking?.content || "")
      .join(""),
    text: chunks
      .map((chunk) => chunk.choices?.[0]?.delta?.content || "")
      .join(""),
  };
}

function replayMetadata(chunks: any[]): { encrypted?: string; id?: string } {
  let encrypted: string | undefined;
  let id: string | undefined;
  for (const chunk of chunks) {
    const thinking = chunk.choices?.[0]?.delta?.thinking;
    if (thinking?.encrypted_content) encrypted = thinking.encrypted_content;
    if (thinking?.id) id = thinking.id;
  }
  return { encrypted, id };
}

async function assertProviderOnce(
  transformer: OpenAIResponsesTransformer | CodexTransformer,
  includeSummaryDeltas: boolean
): Promise<ParsedStream> {
  const parsed = await convertStream(
    transformer,
    duplicatedReasoningEvents(includeSummaryDeltas)
  );
  assert.equal(parsed.thinkingText, SUMMARY);
  assert.equal(parsed.text, "Hi");
  const meta = replayMetadata(parsed.chunks);
  assert.equal(meta.encrypted, CIPHER);
  assert.equal(meta.id, REASONING_ID);
  return parsed;
}

type SSEEvent = { event: string; data: any };

async function anthropicEventsFromUnified(unified: Response): Promise<SSEEvent[]> {
  const transformer = new AnthropicTransformer();
  (transformer as any).logger = logger;
  const out = await transformer.transformResponseIn(unified, {
    req: { id: "reasoning-dup" },
  } as any);
  const events: SSEEvent[] = [];
  for (const block of (await out.text()).split("\n\n")) {
    const eventLine = block.split("\n").find((line) => line.startsWith("event: "));
    const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
    if (!eventLine || !dataLine) continue;
    const raw = dataLine.slice(6);
    if (raw === "[DONE]") continue;
    events.push({ event: eventLine.slice(7), data: JSON.parse(raw) });
  }
  return events;
}

function clientVisible(events: SSEEvent[]): {
  blockTypes: string[];
  thinking: string;
  text: string;
} {
  const blockTypes: string[] = [];
  let thinking = "";
  let text = "";
  for (const event of events) {
    if (event.event === "content_block_start") {
      blockTypes.push(event.data.content_block.type);
    }
    if (event.event === "content_block_delta") {
      if (event.data.delta.type === "thinking_delta") {
        thinking += event.data.delta.thinking || "";
      }
      if (event.data.delta.type === "text_delta") {
        text += event.data.delta.text || "";
      }
    }
  }
  return { blockTypes, thinking, text };
}

async function testOpenAIResponsesSummaryPlusTerminalIsOnce() {
  await assertProviderOnce(new OpenAIResponsesTransformer(), true);
}

async function testCodexSummaryPlusTerminalIsOnce() {
  await assertProviderOnce(new CodexTransformer(), true);
}

async function testOpenAIResponsesTerminalOnlyIsOnce() {
  await assertProviderOnce(new OpenAIResponsesTransformer(), false);
}

async function testCodexTerminalOnlyIsOnce() {
  await assertProviderOnce(new CodexTransformer(), false);
}

async function testAnthropicClientSeesThinkingThenText() {
  const tf = new OpenAIResponsesTransformer();
  (tf as any).logger = logger;
  const unified = await tf.transformResponseOut(
    sseResponse(duplicatedReasoningEvents(true))
  );
  const visible = clientVisible(await anthropicEventsFromUnified(unified));
  assert.deepEqual(visible.blockTypes, ["thinking", "text"]);
  assert.equal(visible.thinking, SUMMARY);
  assert.equal(visible.text, "Hi");
}

async function testAnthropicClientSeesTerminalOnlyThinkingThenText() {
  const tf = new CodexTransformer();
  (tf as any).logger = logger;
  const unified = await tf.transformResponseOut(
    sseResponse(duplicatedReasoningEvents(false))
  );
  const visible = clientVisible(await anthropicEventsFromUnified(unified));
  assert.deepEqual(visible.blockTypes, ["thinking", "text"]);
  assert.equal(visible.thinking, SUMMARY);
  assert.equal(visible.text, "Hi");
}

async function main() {
  await testOpenAIResponsesSummaryPlusTerminalIsOnce();
  await testCodexSummaryPlusTerminalIsOnce();
  await testOpenAIResponsesTerminalOnlyIsOnce();
  await testCodexTerminalOnlyIsOnce();
  await testAnthropicClientSeesThinkingThenText();
  await testAnthropicClientSeesTerminalOnlyThinkingThenText();
  console.log("responses.reasoning-duplication: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

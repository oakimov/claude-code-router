/**
 * Live prompt-caching check: CCR → Anthropic for all three chat protocols.
 *
 * Each protocol runs a fresh multi-turn conversation over a large, stable
 * system prompt (a per-run nonce keeps runs and protocols from sharing cache
 * entries). Two sources are checked for every turn:
 *
 *   1. Client-visible usage — turn 1 writes the prefix; every later turn must
 *      read at least the previous turn's whole prompt (minus a small
 *      tolerance) and reach a high hit ratio.
 *   2. CCR's debug log — the `cache outcome` record for the same request
 *      (matched by model and exact prompt/cached token counts) must report
 *      the same numbers and verdict `cold`/`warm-start` on turn 1, `hit`
 *      afterwards (never `partial` or `unexpected-miss`).
 *
 * Opt-in (never in CI): `pnpm test:live`. Requires CCR with LOG_LEVEL=debug.
 * Environment:
 *   CCR_URL         default http://localhost:3456
 *   CCR_API_KEY     required: the CCR server's API key
 *   CCR_LIVE_MODEL  default "claude,claude-haiku-4-5-20251001"
 *   CCR_LOG_DIR     default packages/server/ccr-config/logs (the Docker mount)
 *   CCR_LIVE_TURNS  default 4
 */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const CCR_URL = process.env.CCR_URL || "http://localhost:3456";
const API_KEY = requireApiKey();

function requireApiKey(): string {
  const key = process.env.CCR_API_KEY;
  if (!key) {
    console.error("ccr-live.caching: set CCR_API_KEY to the CCR server's API key");
    process.exit(1);
  }
  return key;
}
const MODEL = process.env.CCR_LIVE_MODEL || "claude,claude-haiku-4-5-20251001";
const UPSTREAM_MODEL = MODEL.includes(",") ? MODEL.slice(MODEL.indexOf(",") + 1) : MODEL;
const LOG_DIR =
  process.env.CCR_LOG_DIR ||
  resolve(dirname(fileURLToPath(import.meta.url)), "../../../server/ccr-config/logs");
const TURNS = Math.max(2, Number(process.env.CCR_LIVE_TURNS) || 4);

/** Tokens a follow-up read may fall short of the previous prompt. */
const READ_TOLERANCE = 64;
/** Minimum cached share of the prompt on follow-up turns. */
const MIN_HIT_RATIO = 0.9;
/** Allowed skew between this host's clock and CCR's log timestamps. */
const CLOCK_SLACK_MS = 5_000;

type Usage = { prompt: number; read: number; write?: number };
type Turn = { protocol: string; turn: number; usage: Usage; startedAt: number };

const nonce = randomUUID();

/** ~12k tokens of deterministic text: above every model's cacheable minimum. */
function stableSystemPrompt(protocol: string): string {
  const lines = [`Run ${nonce} (${protocol}). You are a terse assistant; follow the rules below.`];
  for (let i = 1; i <= 600; i += 1) {
    lines.push(
      `Rule ${i}: for request class ${i % 17}, prefer option ${(i * 7) % 13} and record code R${i}.`
    );
  }
  return lines.join("\n");
}

function question(turn: number): string {
  return `Turn ${turn}: reply with exactly the word ok${turn} and nothing else.`;
}

async function post(path: string, body: unknown): Promise<any> {
  const response = await fetch(CCR_URL + path, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-api-key": API_KEY,
      authorization: `Bearer ${API_KEY}`,
      "user-agent": "ccr-live-test/1.0",
    },
    body: JSON.stringify(body),
  });
  const json = await response.json();
  assert.equal(
    response.status,
    200,
    `${path} → ${response.status} ${JSON.stringify(json).slice(0, 400)}`
  );
  return json;
}

async function anthropicConversation(): Promise<Turn[]> {
  const system = stableSystemPrompt("anthropic");
  const messages: any[] = [];
  const turns: Turn[] = [];
  for (let turn = 1; turn <= TURNS; turn += 1) {
    messages.push({ role: "user", content: question(turn) });
    const startedAt = Date.now();
    const json = await post("/v1/messages", { model: MODEL, max_tokens: 64, system, messages });
    const u = json.usage || {};
    const read = u.cache_read_input_tokens || 0;
    const write = u.cache_creation_input_tokens || 0;
    turns.push({
      protocol: "anthropic",
      turn,
      startedAt,
      usage: { prompt: (u.input_tokens || 0) + read + write, read, write },
    });
    messages.push({ role: "assistant", content: json.content });
  }
  return turns;
}

async function chatConversation(): Promise<Turn[]> {
  const messages: any[] = [{ role: "system", content: stableSystemPrompt("chat") }];
  const turns: Turn[] = [];
  for (let turn = 1; turn <= TURNS; turn += 1) {
    messages.push({ role: "user", content: question(turn) });
    const startedAt = Date.now();
    const json = await post("/v1/chat/completions", {
      model: MODEL,
      max_tokens: 64,
      stream: false,
      messages,
    });
    const u = json.usage || {};
    turns.push({
      protocol: "chat",
      turn,
      startedAt,
      usage: {
        prompt: u.prompt_tokens || 0,
        read: u.prompt_tokens_details?.cached_tokens || 0,
        write: u.prompt_tokens_details?.cache_write_tokens,
      },
    });
    messages.push({ role: "assistant", content: json.choices?.[0]?.message?.content ?? "" });
  }
  return turns;
}

async function responsesConversation(): Promise<Turn[]> {
  const instructions = stableSystemPrompt("responses");
  const input: any[] = [];
  const turns: Turn[] = [];
  for (let turn = 1; turn <= TURNS; turn += 1) {
    input.push({ role: "user", content: [{ type: "input_text", text: question(turn) }] });
    const startedAt = Date.now();
    const json = await post("/v1/responses", {
      model: MODEL,
      max_output_tokens: 64,
      stream: false,
      instructions,
      input,
    });
    const u = json.usage || {};
    turns.push({
      protocol: "responses",
      turn,
      startedAt,
      usage: {
        prompt: u.input_tokens || 0,
        read: u.input_tokens_details?.cached_tokens || 0,
      },
    });
    input.push(...(json.output || []).filter((item: any) => item.type === "message"));
  }
  return turns;
}

function checkClientUsage(turns: Turn[]): string[] {
  const problems: string[] = [];
  for (const current of turns) {
    const { usage } = current;
    const label = `${current.protocol} T${current.turn}`;
    if (current.turn === 1) {
      if (usage.write !== undefined && usage.write <= 0) {
        problems.push(`${label}: expected a cache write, got ${usage.write}`);
      }
      continue;
    }
    const previous = turns[current.turn - 2].usage;
    if (usage.read < previous.prompt - READ_TOLERANCE) {
      problems.push(
        `${label}: read ${usage.read} < previous prompt ${previous.prompt} - ${READ_TOLERANCE}`
      );
    }
    if (usage.prompt > 0 && usage.read / usage.prompt < MIN_HIT_RATIO) {
      problems.push(
        `${label}: hit ratio ${(usage.read / usage.prompt).toFixed(3)} < ${MIN_HIT_RATIO}`
      );
    }
  }
  return problems;
}

function readCacheOutcomes(since: number): any[] {
  assert.ok(existsSync(LOG_DIR), `CCR log dir not found: ${LOG_DIR} (set CCR_LOG_DIR)`);
  const records: any[] = [];
  for (const file of readdirSync(LOG_DIR)) {
    if (!/^ccr.*\.log$/.test(file)) continue;
    const path = join(LOG_DIR, file);
    if (statSync(path).mtimeMs < since) continue;
    for (const line of readFileSync(path, "utf8").split("\n")) {
      if (!line.includes('"type":"cache outcome"')) continue;
      try {
        const record = JSON.parse(line);
        if (record.time >= since) records.push(record);
      } catch {
        // A line still being written; the flush wait below makes this rare.
      }
    }
  }
  return records;
}

function checkLog(turns: Turn[], records: any[]): string[] {
  const problems: string[] = [];
  const used = new Set<any>();
  for (const current of turns) {
    const label = `${current.protocol} T${current.turn}`;
    const record = records.find(
      (candidate) =>
        !used.has(candidate) &&
        candidate.model === UPSTREAM_MODEL &&
        // Log times come from CCR's clock (e.g. a container); allow drift.
        candidate.time >= current.startedAt - CLOCK_SLACK_MS &&
        candidate.promptTokens === current.usage.prompt &&
        (candidate.cachedTokens || 0) === current.usage.read
    );
    if (!record) {
      problems.push(
        `${label}: no cache outcome record with prompt=${current.usage.prompt} cached=${current.usage.read}`
      );
      continue;
    }
    used.add(record);
    console.log(
      `  log ${label.padEnd(12)} ${record.reqId}  verdict=${record.verdict}  reason=${record.predictionReason}`
    );
    const expected = current.turn === 1 ? ["cold", "warm-start"] : ["hit"];
    if (!expected.includes(record.verdict)) {
      problems.push(
        `${label}: log verdict ${record.verdict} (${record.predictionReason}), expected ${expected.join("/")}`
      );
    }
    if (current.usage.write !== undefined && (record.cacheWriteTokens || 0) !== current.usage.write) {
      problems.push(
        `${label}: log cacheWriteTokens ${record.cacheWriteTokens} != client ${current.usage.write}`
      );
    }
  }
  return problems;
}

function report(turns: Turn[]): void {
  for (const current of turns) {
    const { prompt, read, write } = current.usage;
    console.log(
      `  ${current.protocol.padEnd(9)} T${current.turn}  prompt=${prompt}  read=${read}  write=${
        write ?? "-"
      }  hit=${prompt ? ((100 * read) / prompt).toFixed(1) : "-"}%`
    );
  }
  const followUps = turns.filter((current) => current.turn > 1);
  const prompt = followUps.reduce((sum, current) => sum + current.usage.prompt, 0);
  const read = followUps.reduce((sum, current) => sum + current.usage.read, 0);
  console.log(
    `  follow-up hit ratio: ${prompt ? ((100 * read) / prompt).toFixed(1) : "-"}% (${read}/${prompt})`
  );
}

async function main(): Promise<void> {
  console.log(`ccr-live.caching: ${MODEL} via ${CCR_URL}, ${TURNS} turns, logs ${LOG_DIR}`);
  const since = Date.now() - CLOCK_SLACK_MS;
  const turns = [
    ...(await anthropicConversation()),
    ...(await chatConversation()),
    ...(await responsesConversation()),
  ];
  report(turns);

  // pino writes asynchronously; give the last records time to land.
  await new Promise((done) => setTimeout(done, 2_000));
  const records = readCacheOutcomes(since);
  assert.ok(
    records.length > 0,
    `no "cache outcome" records in ${LOG_DIR} since the run started; is LOG_LEVEL=debug?`
  );

  const problems = [...checkClientUsage(turns), ...checkLog(turns, records)];
  if (problems.length) {
    throw new Error(`caching problems:\n  ${problems.join("\n  ")}`);
  }
  console.log("ccr-live.caching: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

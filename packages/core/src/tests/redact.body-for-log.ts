import assert from "node:assert/strict";
import {
  DEFAULT_LOG_BODY_MAX_BYTES,
  sanitizeBodyForLog,
} from "../utils/redact";

/**
 * Body capture exists so an operator can read back the prompts actually sent
 * upstream. sanitizeUpstreamErrorText is unusable here: it collapses newlines
 * and clips every string to 240 chars. These assertions pin the difference.
 */
function preservesContent() {
  const systemPrompt =
    "You are Claude Code, Anthropic's official CLI for Claude.\n\n" +
    "Line two keeps its newline.\n" +
    "x".repeat(500);
  const body = JSON.stringify({
    model: "claude-sonnet-5",
    system: [{ type: "text", text: systemPrompt }],
  });

  const out = sanitizeBodyForLog(body);

  // The whole prompt survives — including the part past the 240-char mark
  // that sanitizeUpstreamErrorText would have cut.
  assert.ok(out.includes("x".repeat(500)));
  assert.ok(out.includes("You are Claude Code"));
  // Newlines are JSON-escaped in the source string and must stay escaped,
  // not be collapsed into spaces.
  assert.ok(out.includes("\\n\\n"));
}

function redactsSecrets() {
  const body = JSON.stringify({
    authorization: "Bearer sk-ant-oat01-abcdefghijklmnopqrstuvwxyz012345",
    api_key: "super-secret-value",
    nested: { refresh_token: "sk-ant-ort01-abcdefghijklmnopqrstuvwxyz" },
    prompt: "keep me",
  });

  const out = sanitizeBodyForLog(body);

  assert.ok(!out.includes("super-secret-value"));
  assert.ok(!out.includes("oat01-abcdefghijklmnopqrstuvwxyz"));
  assert.ok(!out.includes("ort01-abcdefghijklmnopqrstuvwxyz"));
  assert.ok(out.includes("[redacted]"));
  // Non-secret payload is untouched.
  assert.ok(out.includes("keep me"));
}

function truncatesWithMarker() {
  const body = "y".repeat(100);

  const out = sanitizeBodyForLog(body, 40);
  assert.ok(out.startsWith("y".repeat(40)));
  assert.ok(out.includes("[truncated 60 bytes]"));

  // Exactly at the cap: no marker.
  assert.equal(sanitizeBodyForLog("z".repeat(40), 40), "z".repeat(40));

  // A zero/negative cap must not throw or emit a negative count.
  assert.ok(sanitizeBodyForLog("abc", 0).includes("[truncated 3 bytes]"));
  assert.ok(!sanitizeBodyForLog("abc", -5).includes("-"));
}

function defaultCapApplies() {
  const body = "q".repeat(DEFAULT_LOG_BODY_MAX_BYTES + 10);
  const out = sanitizeBodyForLog(body);
  assert.ok(out.includes("[truncated 10 bytes]"));
}

function redactsEncryptedContent() {
  const body = JSON.stringify({
    type: "response.output_item.done",
    item: {
      type: "reasoning",
      encrypted_content: "gAAAAABlongcipherblobthatmustnotblowuplogs",
      summary: [{ type: "summary_text", text: "Checking gitignore" }],
    },
  });
  const out = sanitizeBodyForLog(body);
  assert.ok(!out.includes("gAAAAABlongcipher"));
  assert.ok(out.includes("[redacted-encrypted]"));
  assert.ok(out.includes("Checking gitignore"));
}

function main() {
  preservesContent();
  redactsSecrets();
  redactsEncryptedContent();
  truncatesWithMarker();
  defaultCapApplies();

  console.log("redact.body-for-log: ok");
}

try {
  main();
} catch (error) {
  console.error(error);
  process.exitCode = 1;
}

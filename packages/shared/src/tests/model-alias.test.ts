import assert from "node:assert/strict";
import {
  canonicalClaudeModelId,
  decodeClaudeModelAlias,
  encodeClaudeModelAlias,
  modelIdNeedsClaudeAlias,
} from "../model-alias";

function main(): void {
  const canonical = "codex,gpt-5.6-sol";
  const alias = "claude-636f6465782c6770742d352e362d736f6c";

  assert.equal(encodeClaudeModelAlias(canonical), alias);
  assert.equal(decodeClaudeModelAlias(alias), canonical);
  assert.equal(decodeClaudeModelAlias(`${alias}[1m]`), `${canonical}[1m]`);
  assert.equal(canonicalClaudeModelId(alias), canonical);

  assert.equal(modelIdNeedsClaudeAlias(canonical), true);
  assert.equal(modelIdNeedsClaudeAlias("claude,claude-sonnet-5"), false);
  assert.equal(modelIdNeedsClaudeAlias("Anthropic,claude-opus-5"), false);

  for (const invalid of [
    "claude-sonnet-5",
    "claude-0",
    "claude-ABCDEF",
    "claude-ff",
    `claude-${Buffer.from("missing-comma", "utf8").toString("hex")}`,
    `claude-${Buffer.from(",missing-provider", "utf8").toString("hex")}`,
    `claude-${Buffer.from("missing-model,", "utf8").toString("hex")}`,
  ]) {
    assert.equal(decodeClaudeModelAlias(invalid), null, invalid);
  }
}

main();

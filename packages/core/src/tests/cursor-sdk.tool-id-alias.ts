import assert from "node:assert/strict";
import {
  aliasHostToolId,
  resolveHostToolId,
  type CursorSdkSession,
} from "../cursor-sdk/session";

function testSession(): CursorSdkSession {
  return {
    toolIdAliases: { byOriginal: new Map(), byAlias: new Map() },
  } as unknown as CursorSdkSession;
}

const ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

// Conforming ids pass through unchanged and round-trip.
{
  const session = testSession();
  const alias = aliasHostToolId(session, "call_abc123");
  assert.equal(alias, "call_abc123");
  assert.equal(resolveHostToolId(session, "call_abc123"), "call_abc123");
}

// Newline-joined SDK ids get a stable, wire-safe alias.
{
  const session = testSession();
  const original = "call-0192-3\nfc_7f3a_0";
  const first = aliasHostToolId(session, original)!;
  assert.ok(first.length > 0 && first.length <= 64, `length: ${first}`);
  assert.match(first, ID_PATTERN);
  assert.ok(!first.includes("\n"));
  assert.equal(aliasHostToolId(session, original), first);
  assert.equal(resolveHostToolId(session, first), original);
}

// Distinct originals that sanitize alike get distinct aliases.
{
  const session = testSession();
  const a = aliasHostToolId(session, "a\nb")!;
  const b = aliasHostToolId(session, "a_b")!;
  assert.notEqual(a, b);
  assert.equal(resolveHostToolId(session, a), "a\nb");
  assert.equal(resolveHostToolId(session, b), "a_b");
}

// Overlong ids stay within the Responses 64-char limit.
{
  const session = testSession();
  const alias = aliasHostToolId(session, `call_${"x".repeat(200)}`)!;
  assert.ok(alias.length <= 64, `length: ${alias.length}`);
  assert.match(alias, ID_PATTERN);
}

// Unknown echoes pass through; empty input resolves to "".
{
  const session = testSession();
  assert.equal(resolveHostToolId(session, "call_unknown"), "call_unknown");
  assert.equal(resolveHostToolId(session, ""), "");
  assert.equal(resolveHostToolId(session, undefined), "");
}

// Empty originals get no alias (caller falls back to a fresh id).
{
  const session = testSession();
  assert.equal(aliasHostToolId(session, ""), undefined);
  assert.equal(aliasHostToolId(session, undefined), undefined);
}

console.log("cursor-sdk.tool-id-alias: ok");

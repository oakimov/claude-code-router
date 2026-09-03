import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  deletePersistedSession,
  getPersistedSession,
  pruneSessionRegistry,
  putPersistedSession,
  resetSessionRegistryForTests,
  SESSION_REGISTRY_TTL_MS,
} from "../session-registry";

const dir = mkdtempSync(join(tmpdir(), "ccr-session-registry-"));
process.env.CCR_SESSION_REGISTRY_DIR = dir;
try {
  // Round trip across a simulated restart (drop memory, keep file).
  putPersistedSession("zen", "conv-1", { sessionId: "ses_abc" }, 1000);
  assert.equal(getPersistedSession("zen", "conv-1", 2000)?.sessionId, "ses_abc");
  resetSessionRegistryForTests();
  assert.equal(getPersistedSession("zen", "conv-1", 2000)?.sessionId, "ses_abc");

  // Families are isolated.
  assert.equal(getPersistedSession("cursor", "conv-1", 2000), undefined);
  putPersistedSession(
    "cursor",
    "key-1",
    { sessionId: "agent-1", workspaceDir: "/w", model: "m" },
    1000
  );
  assert.deepEqual(getPersistedSession("cursor", "key-1", 2000), {
    sessionId: "agent-1",
    workspaceDir: "/w",
    model: "m",
    updatedAt: 1000,
  });

  // Overwrite wins (Zen re-roll then re-mint path).
  putPersistedSession("zen", "conv-1", { sessionId: "ses_def" }, 3000);
  assert.equal(getPersistedSession("zen", "conv-1", 4000)?.sessionId, "ses_def");

  // Delete forgets (retire path).
  deletePersistedSession("zen", "conv-1");
  assert.equal(getPersistedSession("zen", "conv-1", 4000), undefined);
  resetSessionRegistryForTests();
  assert.equal(getPersistedSession("zen", "conv-1", 4000), undefined);

  // TTL expiry reads as missing.
  putPersistedSession("zen", "old", { sessionId: "ses_old" }, 1000);
  assert.equal(
    getPersistedSession("zen", "old", 1000 + SESSION_REGISTRY_TTL_MS + 1),
    undefined
  );

  // Prune drops expired, keeps fresh.
  putPersistedSession("zen", "fresh", { sessionId: "ses_f" }, 5000);
  putPersistedSession("zen", "stale", { sessionId: "ses_s" }, 1000);
  // "stale" plus the earlier cursor/key-1 (updatedAt 1000) both expire.
  assert.equal(pruneSessionRegistry(1000 + SESSION_REGISTRY_TTL_MS + 1), 2);
  assert.equal(
    getPersistedSession("zen", "fresh", 1000 + SESSION_REGISTRY_TTL_MS + 1)?.sessionId,
    "ses_f"
  );

  // Corrupt file degrades to empty instead of throwing.
  resetSessionRegistryForTests();
  writeFileSync(join(dir, "ccr-sessions.json"), "not json{{{");
  resetSessionRegistryForTests();
  assert.equal(getPersistedSession("zen", "fresh", 6000), undefined);
  // Next write replaces the corrupt file.
  putPersistedSession("zen", "fresh", { sessionId: "ses_f2" }, 6000);
  const raw = readFileSync(join(dir, "ccr-sessions.json"), "utf-8");
  assert.ok(JSON.parse(raw).families.zen.fresh);

  // First load in a process prunes expired entries from the file itself.
  resetSessionRegistryForTests();
  writeFileSync(
    join(dir, "ccr-sessions.json"),
    JSON.stringify({
      version: 1,
      families: {
        zen: {
          gone: { sessionId: "ses_g", updatedAt: 1000 },
          live: { sessionId: "ses_l", updatedAt: 6000 },
        },
      },
    })
  );
  resetSessionRegistryForTests();
  assert.equal(
    getPersistedSession("zen", "live", 1000 + SESSION_REGISTRY_TTL_MS + 1)?.sessionId,
    "ses_l"
  );
  const pruned = JSON.parse(readFileSync(join(dir, "ccr-sessions.json"), "utf-8"));
  assert.ok(!("gone" in pruned.families.zen));
  assert.ok("live" in pruned.families.zen);
} finally {
  delete process.env.CCR_SESSION_REGISTRY_DIR;
  resetSessionRegistryForTests();
  rmSync(dir, { recursive: true, force: true });
}

console.log("session-registry: ok");

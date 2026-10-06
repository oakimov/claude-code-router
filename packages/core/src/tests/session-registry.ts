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
  updatePersistedSessions,
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
    {
      sessionId: "agent-1",
      workspaceDir: "/w",
      model: "m",
      modelFingerprint: "grok-4.7|context=256k,fast=false,reasoning_effort=high",
    },
    1000
  );
  assert.deepEqual(getPersistedSession("cursor", "key-1", 2000), {
    sessionId: "agent-1",
    workspaceDir: "/w",
    model: "m",
    modelFingerprint: "grok-4.7|context=256k,fast=false,reasoning_effort=high",
    updatedAt: 1000,
  });
  resetSessionRegistryForTests();
  assert.equal(
    getPersistedSession("cursor", "key-1", 2000)?.modelFingerprint,
    "grok-4.7|context=256k,fast=false,reasoning_effort=high"
  );
  const afterStartup = JSON.parse(
    readFileSync(join(dir, "ccr-sessions.json"), "utf-8")
  );
  assert.equal(
    afterStartup.families.cursor["key-1"].modelFingerprint,
    "grok-4.7|context=256k,fast=false,reasoning_effort=high"
  );

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

  // Batched update: puts and removes land in one write.
  putPersistedSession("cursor-inbound", "short", { sessionId: "ccrs_1" }, 7000);
  updatePersistedSessions(
    "cursor-inbound",
    { put: [{ key: "long", value: { sessionId: "ccrs_1" } }], remove: ["short"] },
    8000
  );
  resetSessionRegistryForTests();
  assert.equal(getPersistedSession("cursor-inbound", "short", 9000), undefined);
  assert.deepEqual(getPersistedSession("cursor-inbound", "long", 9000), {
    sessionId: "ccrs_1",
    updatedAt: 8000,
  });

  // The size cap is per family: churn in one family keeps another's oldest.
  putPersistedSession("cursor", "agent-old", { sessionId: "agent-x" }, 7000);
  for (let i = 0; i < 300; i += 1) {
    putPersistedSession("cursor-inbound", `k${i}`, { sessionId: `ccrs_${i}` }, 10_000 + i);
  }
  resetSessionRegistryForTests();
  assert.equal(getPersistedSession("cursor", "agent-old", 20_000)?.sessionId, "agent-x");
  assert.equal(getPersistedSession("zen", "live", 20_000)?.sessionId, "ses_l");
  assert.equal(getPersistedSession("cursor-inbound", "k0", 20_000), undefined);
  assert.equal(getPersistedSession("cursor-inbound", "k299", 20_000)?.sessionId, "ccrs_299");
  const capped = JSON.parse(readFileSync(join(dir, "ccr-sessions.json"), "utf-8"));
  assert.equal(Object.keys(capped.families["cursor-inbound"]).length, 256);
} finally {
  delete process.env.CCR_SESSION_REGISTRY_DIR;
  resetSessionRegistryForTests();
  rmSync(dir, { recursive: true, force: true });
}

console.log("session-registry: ok");

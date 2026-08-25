import assert from "node:assert/strict";
import { existsSync, writeFileSync, utimesSync } from "node:fs";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  ACTIVE_SERVER_LOG_NAME,
  DEFAULT_LOG_MAX_FILES,
  LogRetentionScheduler,
  pruneServerLogs,
  SERVER_LOG_HISTORY_NAME,
} from "../utils/log-retention";

function touch(dir: string, name: string, size: number, mtimeMs: number): void {
  const path = join(dir, name);
  writeFileSync(path, Buffer.alloc(size, 0x61));
  const at = new Date(mtimeMs);
  utimesSync(path, at, at);
}

async function testPruneKeepsNewestWithinCountAndDeletesOrphans(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), "ccr-log-retention-"));
  try {
    writeFileSync(join(dir, ACTIVE_SERVER_LOG_NAME), "active");
    writeFileSync(join(dir, SERVER_LOG_HISTORY_NAME), "history");
    touch(dir, "ccr-20260820100000.log", 100, 1_000);
    touch(dir, "ccr-20260821100000.log", 100, 2_000);
    touch(dir, "ccr-20260822100000.log", 100, 3_000);
    touch(dir, "ccr-20260823100000.log", 100, 4_000);
    touch(dir, "ccr-20260821100000.log.txt", 10, 2_000);
    writeFileSync(join(dir, "other.log"), "keep-me");

    const result = await pruneServerLogs(dir, {
      maxFiles: 2,
      maxTotalBytes: 10_000,
    });

    assert.deepEqual(result.kept.sort(), [
      "ccr-20260822100000.log",
      "ccr-20260823100000.log",
    ]);
    assert.ok(result.deleted.includes("ccr-20260820100000.log"));
    assert.ok(result.deleted.includes("ccr-20260821100000.log"));
    assert.ok(result.deleted.includes("ccr-20260821100000.log.txt"));
    assert.equal(existsSync(join(dir, ACTIVE_SERVER_LOG_NAME)), true);
    assert.equal(existsSync(join(dir, SERVER_LOG_HISTORY_NAME)), true);
    assert.equal(existsSync(join(dir, "other.log")), true);
    assert.equal(existsSync(join(dir, "ccr-20260823100000.log")), true);
    assert.equal(existsSync(join(dir, "ccr-20260820100000.log")), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

async function testPruneEnforcesTotalByteBudget(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), "ccr-log-retention-bytes-"));
  try {
    touch(dir, "ccr-20260823100000.log", 80, 3_000);
    touch(dir, "ccr-20260822100000.log", 80, 2_000);
    touch(dir, "ccr-20260821100000.log", 80, 1_000);

    const result = await pruneServerLogs(dir, {
      maxFiles: 10,
      maxTotalBytes: 100,
    });

    assert.deepEqual(result.kept, ["ccr-20260823100000.log"]);
    assert.equal(result.deleted.length, 2);
    assert.equal(existsSync(join(dir, "ccr-20260823100000.log")), true);
    assert.equal(existsSync(join(dir, "ccr-20260822100000.log")), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

async function testSchedulerRunsStartupAndDaily(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), "ccr-log-retention-sched-"));
  const calls: string[] = [];
  const timers: Array<{ ms: number; fn: () => void }> = [];
  try {
    touch(dir, "ccr-20260820100000.log", 10, 1_000);
    touch(dir, "ccr-20260821100000.log", 10, 2_000);
    touch(dir, "ccr-20260822100000.log", 10, 3_000);
    touch(dir, "ccr-20260823100000.log", 10, 4_000);

    const scheduler = new LogRetentionScheduler({
      logDir: dir,
      maxFiles: DEFAULT_LOG_MAX_FILES,
      maxTotalBytes: 10_000,
      intervalMs: 86_400_000,
      prune: async (logDir, options) => {
        calls.push("prune");
        return pruneServerLogs(logDir, options);
      },
      setIntervalFn: ((fn: () => void, ms: number) => {
        timers.push({ ms, fn });
        return 1 as any;
      }) as typeof setInterval,
      clearIntervalFn: (() => undefined) as typeof clearInterval,
    });

    scheduler.start();
    await new Promise((r) => setTimeout(r, 30));
    assert.ok(calls.length >= 1);
    assert.equal(timers.length, 1);
    assert.equal(timers[0].ms, 86_400_000);

    const before = calls.length;
    timers[0].fn();
    await new Promise((r) => setTimeout(r, 30));
    assert.ok(calls.length > before);
    assert.equal(existsSync(join(dir, "ccr-20260823100000.log")), true);
    assert.equal(existsSync(join(dir, "ccr-20260820100000.log")), false);

    scheduler.stop();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  await testPruneKeepsNewestWithinCountAndDeletesOrphans();
  await testPruneEnforcesTotalByteBudget();
  await testSchedulerRunsStartupAndDaily();
  console.log("log-retention: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

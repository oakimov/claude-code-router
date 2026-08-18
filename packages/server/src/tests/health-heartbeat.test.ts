/**
 * Periodic health report: window accounting, in-flight tracking, the persisted
 * snapshot the Web UI reads, and the guarantee that the timer never keeps the
 * process alive.
 */
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { mkdtempSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  HealthHeartbeat,
  resolveHeartbeatIntervalMs,
} from "../utils/health-heartbeat";

type Clock = { value: number };

function makeHeartbeat(clock: Clock, lines: string[], intervalMs = 600_000) {
  return new HealthHeartbeat({
    intervalMs,
    write: (line) => lines.push(line),
    now: () => clock.value,
  });
}

/** Minimal stand-ins for a Fastify request/reply pair. */
function makeExchange(id: string, sessionId?: string, provider?: string) {
  const raw = new EventEmitter() as any;
  raw.once = raw.once.bind(raw);
  return {
    request: { id, sessionId, provider },
    reply: { raw, statusCode: 200 },
    close(statusCode = 200) {
      this.reply.statusCode = statusCode;
      raw.emit("close");
    },
  };
}

function testReportCoversTheRequestedFields() {
  const clock: Clock = { value: 1_000 };
  const lines: string[] = [];
  const heartbeat = makeHeartbeat(clock, lines);

  clock.value += 5 * 60_000;
  const snapshot = heartbeat.report();

  const text = lines.join("\n");
  assert.match(text, /uptime 5m 00s/);
  assert.match(text, /memory rss /);
  assert.match(text, /heap /);
  assert.match(text, /(load [\d.]+|load n\/a)/);
  assert.match(text, /event loop mean /);
  assert.match(text, /sessions 0 running/);
  assert.match(text, /requests 0 in flight/);

  assert.ok(snapshot.memory.rss > 0, "rss must be observed");
  assert.ok(snapshot.load.cpus >= 1);
  // First report has no previous sample to diff against.
  assert.equal(snapshot.memory.rssDelta, undefined);
  assert.equal(snapshot.cache.hitRatio, undefined);
}

function testInFlightAndCompletionAccounting() {
  const clock: Clock = { value: 0 };
  const lines: string[] = [];
  const heartbeat = makeHeartbeat(clock, lines);

  const a = makeExchange("a", "sess-1", "openrouter");
  const b = makeExchange("b", "sess-2", "claude");
  heartbeat.trackRequest(a.request, a.reply);
  heartbeat.trackRequest(b.request, b.reply);
  // A duplicate onRequest must not double-count the same exchange.
  heartbeat.trackRequest(a.request, a.reply);

  clock.value = 41_000;
  let snapshot = heartbeat.snapshot();
  assert.equal(snapshot.requests.inFlight, 2);
  assert.equal(snapshot.sessions.running, 2);
  assert.equal(snapshot.requests.oldestInFlightMs, 41_000);

  clock.value = 43_000;
  a.close(200);
  clock.value = 44_000;
  b.close(500);

  snapshot = heartbeat.snapshot();
  assert.equal(snapshot.requests.inFlight, 0);
  assert.equal(snapshot.requests.completed, 2);
  assert.equal(snapshot.requests.failed, 1);
  assert.equal(snapshot.requests.p50Ms, 43_000);
  assert.equal(snapshot.requests.p95Ms, 44_000);
  assert.deepEqual(snapshot.requests.byProvider.map((p) => p.provider).sort(), [
    "claude",
    "openrouter",
  ]);
  assert.equal(
    snapshot.requests.byProvider.find((p) => p.provider === "claude")?.failed,
    1
  );
  // Sessions stay "active" for the window even after their request ends.
  assert.equal(snapshot.sessions.running, 0);
  assert.equal(snapshot.sessions.activeInWindow, 2);

  const text = heartbeat.formatLines(snapshot).join("\n");
  assert.match(text, /2 completed/);
  assert.match(text, /1 failed \(50\.0%\)/);
  assert.match(text, /upstream .*claude 0 ok \/ 1 failed/);

  // A request rejected before routing has no provider; counting it as one
  // would invent an upstream that was never contacted.
  const unrouted = makeExchange("c");
  heartbeat.trackRequest(unrouted.request, unrouted.reply);
  unrouted.close(401);
  const after = heartbeat.snapshot();
  assert.equal(after.requests.completed, 3);
  assert.equal(after.requests.failed, 2);
  assert.equal(after.requests.byProvider.length, 2);
}

function testSocketCloseReleasesAbortedRequests() {
  const clock: Clock = { value: 0 };
  const heartbeat = makeHeartbeat(clock, []);
  const exchange = makeExchange("aborted", "sess-x", "claude");
  heartbeat.trackRequest(exchange.request, exchange.reply);
  assert.equal(heartbeat.snapshot().requests.inFlight, 1);

  // Client hangup: the socket closes without a completed response.
  exchange.close(200);
  assert.equal(heartbeat.snapshot().requests.inFlight, 0);

  // A second close event must not double-count.
  exchange.close(200);
  assert.equal(heartbeat.snapshot().requests.completed, 1);
}

function testCacheRatioUsesAnthropicUsageShape() {
  const clock: Clock = { value: 0 };
  const heartbeat = makeHeartbeat(clock, []);

  // input_tokens excludes cached tokens, so the billable prompt is the sum.
  heartbeat.recordUsage("sess-1", {
    input_tokens: 10,
    cache_read_input_tokens: 80,
    cache_creation_input_tokens: 10,
    output_tokens: 5,
  });
  heartbeat.recordUsage("sess-2", {
    input_tokens: 100,
    cache_read_input_tokens: 0,
    cache_creation_input_tokens: 0,
  });

  clock.value = 600_000;
  const snapshot = heartbeat.snapshot();
  assert.equal(snapshot.cache.promptTokens, 200);
  assert.equal(snapshot.cache.cachedTokens, 80);
  assert.equal(snapshot.cache.writtenTokens, 10);
  assert.equal(snapshot.cache.hitRatio, 0.4);
  assert.equal(snapshot.sessions.activeInWindow, 2);

  const text = heartbeat.formatLines(snapshot).join("\n");
  assert.match(text, /cache 40\.0% prompt-cache hit/);

  // An empty usage object carries no signal and must not skew the ratio.
  heartbeat.recordUsage("sess-3", {});
  assert.equal(heartbeat.snapshot().cache.promptTokens, 200);
}

function testWindowResetsAfterEachReport() {
  const clock: Clock = { value: 0 };
  const lines: string[] = [];
  const heartbeat = makeHeartbeat(clock, lines);

  const exchange = makeExchange("a", "sess-1", "claude");
  heartbeat.trackRequest(exchange.request, exchange.reply);
  clock.value = 1_000;
  exchange.close(200);
  heartbeat.recordUsage("sess-1", {
    input_tokens: 10,
    cache_read_input_tokens: 90,
  });

  clock.value = 600_000;
  const first = heartbeat.report();
  assert.equal(first.requests.completed, 1);
  assert.equal(first.cache.promptTokens, 100);

  clock.value = 1_200_000;
  const second = heartbeat.report();
  assert.equal(second.requests.completed, 0);
  assert.equal(second.cache.promptTokens, 0);
  assert.equal(second.cache.hitRatio, undefined);
  assert.equal(second.windowMs, 600_000);
  // The previous report's rss becomes the baseline for the delta.
  assert.notEqual(second.memory.rssDelta, undefined);
  // Sessions idle for longer than the window are dropped.
  assert.equal(second.sessions.activeInWindow, 0);
}

function testDisabledHeartbeatDoesNothing() {
  const lines: string[] = [];
  const heartbeat = new HealthHeartbeat({
    intervalMs: 0,
    write: (line) => lines.push(line),
  });
  assert.equal(heartbeat.enabled, false);
  heartbeat.start();

  const exchange = makeExchange("a", "sess-1", "claude");
  heartbeat.trackRequest(exchange.request, exchange.reply);
  heartbeat.recordUsage("sess-1", { input_tokens: 10 });

  const snapshot = heartbeat.snapshot();
  assert.equal(snapshot.requests.inFlight, 0);
  assert.equal(snapshot.cache.promptTokens, 0);
  assert.deepEqual(lines, []);
  heartbeat.stop();
}

function testTimerNeverHoldsTheProcessOpen() {
  const lines: string[] = [];
  const heartbeat = new HealthHeartbeat({
    intervalMs: 50,
    write: (line) => lines.push(line),
  });
  heartbeat.start();
  assert.ok(lines.length > 0, "start() must emit an immediate report");

  // Timers are not reported by process._getActiveHandles(), so ask the timer.
  const timer = (heartbeat as any).timer;
  assert.equal(typeof timer?.hasRef, "function", "start() must create a timer");
  assert.equal(
    timer.hasRef(),
    false,
    "the heartbeat timer must be unref'd so it cannot keep the server alive"
  );

  heartbeat.stop();
  assert.equal((heartbeat as any).timer, undefined, "stop() must clear the timer");
}

function testStartupBaselineOmitsTrafficLines() {
  const clock: Clock = { value: 0 };
  const lines: string[] = [];
  const heartbeat = makeHeartbeat(clock, lines);

  // A window of a few milliseconds carries no traffic signal; reporting
  // "0 completed" there would read as an idle proxy rather than as no data.
  clock.value = 5;
  heartbeat.report();
  const text = lines.join("\n");
  assert.match(text, /memory rss /);
  assert.doesNotMatch(text, /requests /);
  assert.doesNotMatch(text, /sessions /);
}

async function testSnapshotFileHoldsCurrentStateOnly() {
  const dir = mkdtempSync(join(tmpdir(), "ccr-health-"));
  const file = join(dir, "health.json");
  try {
    const clock: Clock = { value: 0 };
    const heartbeat = new HealthHeartbeat({
      intervalMs: 600_000,
      write: () => {},
      now: () => clock.value,
      snapshotFile: file,
    });

    const exchange = makeExchange("a", "sess-1", "claude");
    heartbeat.trackRequest(exchange.request, exchange.reply);
    clock.value = 2_000;
    exchange.close(200);
    heartbeat.recordUsage("sess-1", {
      input_tokens: 20,
      cache_read_input_tokens: 80,
    });

    clock.value = 600_000;
    heartbeat.report();
    await heartbeat.whenPersisted();

    const state = JSON.parse(readFileSync(file, "utf-8"));
    assert.equal(state.version, 1);
    assert.equal(state.pid, process.pid);
    assert.equal(state.intervalMs, 600_000);
    assert.equal(state.updatedAt, 600_000);
    assert.equal(state.current.requests.completed, 1);
    assert.equal(state.current.cache.hitRatio, 0.8);
    assert.ok(state.current.memory.rss > 0);
    // Only the current window is persisted; no history accumulates.
    assert.equal("history" in state, false);

    clock.value = 1_200_000;
    heartbeat.report();
    await heartbeat.whenPersisted();
    const second = JSON.parse(readFileSync(file, "utf-8"));
    assert.equal(second.current.requests.completed, 0);
    assert.equal(second.updatedAt, 1_200_000);

    // The rename is atomic, so no temp file may survive a completed write.
    assert.deepEqual(readdirSync(dir), ["health.json"]);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}

function testGetStateDoesNotConsumeTheCpuWindow() {
  const clock: Clock = { value: 0 };
  const heartbeat = makeHeartbeat(clock, []);
  clock.value = 600_000;
  heartbeat.report();
  const baseline = (heartbeat as any).lastCpuUsage;
  assert.ok(baseline, "a report must establish the CPU baseline");

  // Reading the live state is what /health does on every poll; it must not
  // advance the baseline that the next report measures against.
  clock.value = 660_000;
  const state = heartbeat.getState();
  assert.equal(state.version, 1);
  assert.equal(state.updatedAt, 660_000);
  assert.equal(state.current.windowMs, 60_000);
  assert.equal((heartbeat as any).lastCpuUsage, baseline);

  clock.value = 1_200_000;
  heartbeat.report();
  assert.notEqual((heartbeat as any).lastCpuUsage, baseline);
}

function testPersistenceIsOptional() {
  const clock: Clock = { value: 0 };
  const heartbeat = makeHeartbeat(clock, []);
  clock.value = 600_000;
  // No snapshotFile configured: reporting still works, nothing is written.
  assert.doesNotThrow(() => heartbeat.report());
}

function testIntervalResolution() {
  assert.equal(resolveHeartbeatIntervalMs({}), 600_000);
  assert.equal(resolveHeartbeatIntervalMs({ HEARTBEAT_INTERVAL_MS: 0 }), 0);
  assert.equal(
    resolveHeartbeatIntervalMs({ HEARTBEAT_INTERVAL_MS: "30000" }),
    30_000
  );
  // Nonsense values fall back to the default rather than disabling reporting.
  assert.equal(
    resolveHeartbeatIntervalMs({ HEARTBEAT_INTERVAL_MS: "nope" }),
    600_000
  );
}

async function main() {
  testReportCoversTheRequestedFields();
  testInFlightAndCompletionAccounting();
  testSocketCloseReleasesAbortedRequests();
  testCacheRatioUsesAnthropicUsageShape();
  testWindowResetsAfterEachReport();
  testDisabledHeartbeatDoesNothing();
  testTimerNeverHoldsTheProcessOpen();
  testStartupBaselineOmitsTrafficLines();
  await testSnapshotFileHoldsCurrentStateOnly();
  testGetStateDoesNotConsumeTheCpuWindow();
  testPersistenceIsOptional();
  testIntervalResolution();
  console.log("health-heartbeat: PASS");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

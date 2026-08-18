import { rename, writeFile } from "node:fs/promises";
import { cpus, freemem, loadavg, totalmem } from "node:os";
import { monitorEventLoopDelay, type IntervalHistogram } from "node:perf_hooks";

/**
 * Periodic health report.
 *
 * The pino logger writes to a rotating file only, so a foreground `ccr start`
 * (or `docker compose logs -f ccr`) shows nothing between requests. This module
 * owns the one thing that is meant to reach stdout: a compact, greppable
 * snapshot of process health and routing activity since the previous report.
 *
 * Every report is also persisted to `~/.claude-code-router/health.json`, and
 * the same payload is attached to the existing `GET /health` liveness probe,
 * which is what the Web UI status bar reads. Only the current state is kept —
 * no history.
 */

const DEFAULT_INTERVAL_MS = 10 * 60 * 1000;

/** Schema version of the persisted file, so readers can reject old shapes. */
const SNAPSHOT_FILE_VERSION = 1;

/** Timer resolution of the event-loop histogram; also its measurement floor. */
const LOOP_RESOLUTION_MS = 10;

/** Bounded latency reservoir so a busy window cannot grow without limit. */
const MAX_LATENCY_SAMPLES = 2048;

const LINE_PREFIX = "[ccr:health]";

export interface HealthHeartbeatOptions {
  /** Report period. `0` (or negative) disables the heartbeat entirely. */
  intervalMs?: number;
  /** Pino instance; the same report is mirrored into the log file at info. */
  logger?: { info?: (payload: any, message?: string) => void };
  /** Console sink. Overridable for tests. */
  write?: (line: string) => void;
  /** Clock seam for tests. */
  now?: () => number;
  /**
   * Where to persist the snapshot + history. `null` disables persistence,
   * which is what tests that do not care about the file want.
   */
  snapshotFile?: string | null;
}

type InFlightRequest = {
  startedAt: number;
  request: any;
};

type ProviderCounters = {
  ok: number;
  failed: number;
};

export interface HealthSnapshot {
  uptimeMs: number;
  windowMs: number;
  memory: {
    rss: number;
    rssDelta: number | undefined;
    heapUsed: number;
    heapTotal: number;
    external: number;
    systemTotal: number;
    systemFree: number;
    /** True when the totals come from a cgroup limit rather than the host. */
    constrained: boolean;
  };
  load: {
    avg: [number, number, number];
    cpus: number;
    /** Process CPU time over the window, expressed in cores. */
    processCores: number | undefined;
    eventLoopMeanMs: number;
    eventLoopP99Ms: number;
  };
  sessions: {
    running: number;
    activeInWindow: number;
  };
  requests: {
    inFlight: number;
    oldestInFlightMs: number | undefined;
    completed: number;
    failed: number;
    p50Ms: number | undefined;
    p95Ms: number | undefined;
    byProvider: Array<{ provider: string } & ProviderCounters>;
  };
  cache: {
    promptTokens: number;
    cachedTokens: number;
    writtenTokens: number;
    hitRatio: number | undefined;
  };
}

/** Payload of `health.json`, and of the `vitals` field of `GET /health`. */
export interface HealthState {
  version: number;
  pid: number;
  node: string;
  updatedAt: number;
  intervalMs: number;
  current: HealthSnapshot;
}

function percentile(sorted: number[], fraction: number): number | undefined {
  if (!sorted.length) return undefined;
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil(fraction * sorted.length) - 1)
  );
  return sorted[index];
}

function formatBytes(bytes: number): string {
  const mb = bytes / 1024 / 1024;
  if (Math.abs(mb) >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function formatSignedBytes(bytes: number): string {
  const sign = bytes >= 0 ? "+" : "-";
  return `${sign}${formatBytes(Math.abs(bytes))}`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${String(minutes % 60).padStart(2, "0")}m`;
  return `${Math.floor(hours / 24)}d ${String(hours % 24).padStart(2, "0")}h`;
}

function formatCount(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

function num(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/** Node exposes cgroup limits only on some platforms; treat 0/absent as "host". */
function constrainedMemoryBytes(): number {
  const fn = (process as any).constrainedMemory;
  if (typeof fn !== "function") return 0;
  try {
    return num(fn.call(process));
  } catch {
    return 0;
  }
}

function availableMemoryBytes(): number {
  const fn = (process as any).availableMemory;
  if (typeof fn !== "function") return 0;
  try {
    return num(fn.call(process));
  } catch {
    return 0;
  }
}

export class HealthHeartbeat {
  private readonly intervalMs: number;
  private readonly logger: HealthHeartbeatOptions["logger"];
  private readonly write: (line: string) => void;
  private readonly now: () => number;
  private readonly snapshotFile: string | null;

  private readonly startedAt: number;
  private windowStartedAt: number;

  private readonly inFlight = new Map<any, InFlightRequest>();
  private readonly sessionsLastSeen = new Map<string, number>();
  private readonly byProvider = new Map<string, ProviderCounters>();
  private latencies: number[] = [];
  private completed = 0;
  private failed = 0;

  private promptTokens = 0;
  private cachedTokens = 0;
  private writtenTokens = 0;

  private lastRss: number | undefined;
  private lastCpuUsage: NodeJS.CpuUsage | undefined;

  private pendingPersist: Promise<void> = Promise.resolve();

  private loopDelay: IntervalHistogram | undefined;
  private timer: NodeJS.Timeout | undefined;

  constructor(options: HealthHeartbeatOptions = {}) {
    this.intervalMs = options.intervalMs ?? DEFAULT_INTERVAL_MS;
    this.logger = options.logger;
    this.write = options.write ?? ((line: string) => console.log(line));
    this.now = options.now ?? Date.now;
    this.snapshotFile =
      options.snapshotFile === undefined ? null : options.snapshotFile;
    this.startedAt = this.now();
    this.windowStartedAt = this.startedAt;
  }

  get enabled(): boolean {
    return this.intervalMs > 0;
  }

  /**
   * Begin reporting. The first report fires immediately so operators can see
   * the heartbeat is alive without waiting a full interval.
   */
  start(): void {
    if (!this.enabled || this.timer) return;

    this.loopDelay = monitorEventLoopDelay({ resolution: LOOP_RESOLUTION_MS });
    this.loopDelay.enable();
    this.lastCpuUsage = process.cpuUsage();

    this.write(
      `${LINE_PREFIX} reporting every ${formatDuration(this.intervalMs)}` +
        ` · set HEARTBEAT_INTERVAL_MS in config.json to change it (0 disables)`
    );
    // Baseline report: establishes the rss / cpu reference for the first window.
    this.report();

    this.timer = setInterval(() => this.report(), this.intervalMs);
    // A health report must never be the reason the process stays alive.
    this.timer.unref?.();
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
    this.loopDelay?.disable();
    this.loopDelay = undefined;
  }

  /**
   * Track a routed LLM request for its whole lifetime. Completion is observed
   * on the response socket closing, so client aborts release the slot too.
   */
  trackRequest(request: any, reply: any): void {
    if (!this.enabled || !request) return;
    const key = request.id ?? request;
    if (this.inFlight.has(key)) return;
    this.inFlight.set(key, { startedAt: this.now(), request });

    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      this.finishRequest(key, reply?.statusCode);
    };

    const raw = reply?.raw;
    if (raw && typeof raw.once === "function") {
      raw.once("close", finish);
    } else {
      finish();
    }
  }

  /** Called by the response-close listener; also safe to call directly. */
  finishRequest(key: any, statusCode?: number): void {
    const entry = this.inFlight.get(key);
    if (!entry) return;
    this.inFlight.delete(key);

    this.completed += 1;
    const isFailure = typeof statusCode === "number" && statusCode >= 400;
    if (isFailure) this.failed += 1;

    const duration = this.now() - entry.startedAt;
    if (this.latencies.length < MAX_LATENCY_SAMPLES) {
      this.latencies.push(duration);
    }

    // Requests rejected before routing (auth, validation) never reached an
    // upstream, so attributing them to a provider would invent one.
    const provider = entry.request?.provider;
    if (typeof provider === "string" && provider) {
      const counters = this.byProvider.get(provider) ?? { ok: 0, failed: 0 };
      if (isFailure) counters.failed += 1;
      else counters.ok += 1;
      this.byProvider.set(provider, counters);
    }

    this.touchSession(entry.request?.sessionId);
  }

  /**
   * Fold an Anthropic-shaped usage object into the window cache counters.
   * `input_tokens` excludes cached tokens, so the billable prompt is the sum.
   */
  recordUsage(sessionId: unknown, usage: any): void {
    if (!this.enabled || !usage || typeof usage !== "object") return;
    this.touchSession(sessionId);

    const input = num(usage.input_tokens);
    const cached = num(usage.cache_read_input_tokens);
    const written = num(usage.cache_creation_input_tokens);
    if (input === 0 && cached === 0 && written === 0) return;

    this.promptTokens += input + cached + written;
    this.cachedTokens += cached;
    this.writtenTokens += written;
  }

  private touchSession(sessionId: unknown): void {
    if (typeof sessionId !== "string" || !sessionId) return;
    this.sessionsLastSeen.set(sessionId, this.now());
  }

  snapshot(): HealthSnapshot {
    const now = this.now();
    const windowMs = Math.max(1, now - this.windowStartedAt);

    for (const [sessionId, seenAt] of this.sessionsLastSeen) {
      if (now - seenAt > this.intervalMs) this.sessionsLastSeen.delete(sessionId);
    }

    const runningSessions = new Set<string>();
    let oldestInFlightMs: number | undefined;
    for (const entry of this.inFlight.values()) {
      const sessionId = entry.request?.sessionId;
      if (typeof sessionId === "string" && sessionId) runningSessions.add(sessionId);
      const age = now - entry.startedAt;
      if (oldestInFlightMs === undefined || age > oldestInFlightMs) {
        oldestInFlightMs = age;
      }
    }

    const memory = process.memoryUsage();
    const limit = constrainedMemoryBytes();
    const available = availableMemoryBytes();
    const constrained = limit > 0 && limit < totalmem();

    // The CPU baseline is advanced by resetWindow(), not here, so that reading
    // a live snapshot (the /health poll) cannot consume the delta that the next
    // report is measuring.
    const cpuUsage = process.cpuUsage();
    let processCores: number | undefined;
    if (this.lastCpuUsage) {
      const micros =
        cpuUsage.user -
        this.lastCpuUsage.user +
        (cpuUsage.system - this.lastCpuUsage.system);
      processCores = micros / 1000 / windowMs;
    }

    // The histogram floor is the timer resolution itself; subtract it so an
    // idle loop reads as ~0 instead of ~10 ms.
    const meanNs = this.loopDelay?.mean;
    const p99Ns = this.loopDelay?.percentile(99);
    const toLagMs = (ns: number | undefined) =>
      Number.isFinite(ns as number)
        ? Math.max(0, (ns as number) / 1e6 - LOOP_RESOLUTION_MS)
        : 0;

    const sorted = [...this.latencies].sort((a, b) => a - b);
    const rss = memory.rss;
    const rssDelta = this.lastRss === undefined ? undefined : rss - this.lastRss;

    const [one, five, fifteen] = loadavg();

    return {
      uptimeMs: now - this.startedAt,
      windowMs,
      memory: {
        rss,
        rssDelta,
        heapUsed: memory.heapUsed,
        heapTotal: memory.heapTotal,
        external: memory.external,
        systemTotal: constrained ? limit : totalmem(),
        systemFree: constrained && available > 0 ? available : freemem(),
        constrained,
      },
      load: {
        avg: [one, five, fifteen],
        cpus: cpus().length || 1,
        processCores,
        eventLoopMeanMs: toLagMs(meanNs),
        eventLoopP99Ms: toLagMs(p99Ns),
      },
      sessions: {
        running: runningSessions.size,
        activeInWindow: this.sessionsLastSeen.size,
      },
      requests: {
        inFlight: this.inFlight.size,
        oldestInFlightMs,
        completed: this.completed,
        failed: this.failed,
        p50Ms: percentile(sorted, 0.5),
        p95Ms: percentile(sorted, 0.95),
        byProvider: [...this.byProvider.entries()]
          .map(([provider, counters]) => ({ provider, ...counters }))
          .sort((a, b) => b.ok + b.failed - (a.ok + a.failed)),
      },
      cache: {
        promptTokens: this.promptTokens,
        cachedTokens: this.cachedTokens,
        writtenTokens: this.writtenTokens,
        hitRatio: this.promptTokens
          ? Math.round((this.cachedTokens / this.promptTokens) * 1000) / 1000
          : undefined,
      },
    };
  }

  formatLines(snapshot: HealthSnapshot): string[] {
    const window = formatDuration(snapshot.windowMs);
    const lines: string[] = [];

    lines.push(
      `${LINE_PREFIX} uptime ${formatDuration(snapshot.uptimeMs)} · pid ${process.pid} · node ${process.version}`
    );

    const { memory } = snapshot;
    const rssDelta =
      memory.rssDelta === undefined
        ? ""
        : ` (${formatSignedBytes(memory.rssDelta)})`;
    const systemUsed = memory.systemTotal - memory.systemFree;
    lines.push(
      `${LINE_PREFIX} memory rss ${formatBytes(memory.rss)}${rssDelta}` +
        ` · heap ${formatBytes(memory.heapUsed)}/${formatBytes(memory.heapTotal)}` +
        ` · external ${formatBytes(memory.external)}` +
        ` · ${memory.constrained ? "container" : "system"} ${formatBytes(systemUsed)}/${formatBytes(memory.systemTotal)} used`
    );

    const { load } = snapshot;
    const loadText = load.avg.every((value) => value === 0)
      ? "load n/a"
      : `load ${load.avg.map((value) => value.toFixed(2)).join(" / ")}`;
    const cpuText =
      load.processCores === undefined
        ? ""
        : ` · proc cpu ${load.processCores.toFixed(2)} cores`;
    lines.push(
      `${LINE_PREFIX} ${loadText} (${load.cpus} cpus)${cpuText}` +
        ` · event loop mean ${load.eventLoopMeanMs.toFixed(1)} ms, p99 ${load.eventLoopP99Ms.toFixed(1)} ms`
    );

    // The startup baseline covers a window too short to carry traffic signal;
    // reporting "0 completed in 1ms" would read as an idle proxy, not as noise.
    if (snapshot.windowMs < 1000) return lines;

    lines.push(
      `${LINE_PREFIX} sessions ${snapshot.sessions.running} running` +
        ` · ${snapshot.sessions.activeInWindow} active in the last ${window}`
    );

    const { requests } = snapshot;
    const oldest =
      requests.oldestInFlightMs === undefined
        ? ""
        : ` (oldest ${formatDuration(requests.oldestInFlightMs)})`;
    const failureRate = requests.completed
      ? ((requests.failed / requests.completed) * 100).toFixed(1)
      : "0.0";
    const latency =
      requests.p50Ms === undefined
        ? ""
        : ` · p50 ${formatDuration(requests.p50Ms)} · p95 ${formatDuration(requests.p95Ms ?? requests.p50Ms)}`;
    lines.push(
      `${LINE_PREFIX} requests ${requests.inFlight} in flight${oldest}` +
        ` · ${requests.completed} completed in ${window}` +
        ` · ${requests.failed} failed (${failureRate}%)${latency}`
    );

    if (requests.byProvider.length) {
      lines.push(
        `${LINE_PREFIX} upstream ` +
          requests.byProvider
            .map((entry) => `${entry.provider} ${entry.ok} ok / ${entry.failed} failed`)
            .join(" · ")
      );
    }

    const { cache } = snapshot;
    if (cache.hitRatio !== undefined) {
      lines.push(
        `${LINE_PREFIX} cache ${(cache.hitRatio * 100).toFixed(1)}% prompt-cache hit` +
          ` · ${formatCount(cache.cachedTokens)} cached / ${formatCount(cache.promptTokens)} prompt tokens` +
          ` · ${formatCount(cache.writtenTokens)} written`
      );
    }

    return lines;
  }

  /**
   * Current state, as served by `GET /health`. This is a live read rather than
   * the last persisted report, so the UI is never up to one interval behind the
   * process it is describing.
   */
  getState(): HealthState {
    return this.toState(this.snapshot());
  }

  private toState(current: HealthSnapshot): HealthState {
    return {
      version: SNAPSHOT_FILE_VERSION,
      pid: process.pid,
      node: process.version,
      updatedAt: this.now(),
      intervalMs: this.intervalMs,
      current,
    };
  }

  /**
   * Persist the state next to the config. Written to a sibling temp file and
   * renamed so a reader never observes a half-written document.
   */
  private async persist(state: HealthState): Promise<void> {
    if (!this.snapshotFile) return;
    const temp = `${this.snapshotFile}.${process.pid}.tmp`;
    try {
      await writeFile(temp, JSON.stringify(state, null, 2), "utf-8");
      await rename(temp, this.snapshotFile);
    } catch {
      // The report is advisory; a read-only or missing directory must not
      // interrupt serving requests.
    }
  }

  /** Resolves once the most recent report has reached disk. */
  whenPersisted(): Promise<void> {
    return this.pendingPersist;
  }

  /** Emit one report and open a fresh window. */
  report(): HealthSnapshot {
    const snapshot = this.snapshot();

    for (const line of this.formatLines(snapshot)) {
      try {
        this.write(line);
      } catch {
        // A broken stdout must never take the server down.
      }
    }

    try {
      this.logger?.info?.({ type: "health heartbeat", ...snapshot });
    } catch {
      // ignore logger failures
    }

    // Writing must not delay the caller; `whenPersisted()` is the test seam.
    this.pendingPersist = this.persist(this.toState(snapshot));

    this.resetWindow(snapshot.memory.rss);
    return snapshot;
  }

  private resetWindow(rss: number): void {
    this.windowStartedAt = this.now();
    this.lastCpuUsage = process.cpuUsage();
    this.completed = 0;
    this.failed = 0;
    this.latencies = [];
    this.byProvider.clear();
    this.promptTokens = 0;
    this.cachedTokens = 0;
    this.writtenTokens = 0;
    this.lastRss = rss;
    this.loopDelay?.reset();
  }
}

/**
 * Resolve the configured interval. `HEARTBEAT_INTERVAL_MS: 0` disables it.
 */
export function resolveHeartbeatIntervalMs(config: any): number {
  const raw =
    config?.HEARTBEAT_INTERVAL_MS ?? process.env.CCR_HEARTBEAT_INTERVAL_MS;
  if (raw === undefined || raw === null || raw === "") return DEFAULT_INTERVAL_MS;
  const parsed = typeof raw === "number" ? raw : Number.parseInt(String(raw), 10);
  if (!Number.isFinite(parsed) || parsed < 0) return DEFAULT_INTERVAL_MS;
  return parsed;
}

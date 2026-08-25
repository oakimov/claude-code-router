import { readdir, stat, unlink } from "node:fs/promises";
import { join } from "node:path";

/** Matches RFS `maxFiles` in `index.ts`. */
export const DEFAULT_LOG_MAX_FILES = 3;

/** Matches RFS `maxSize: "50M"`. */
export const DEFAULT_LOG_MAX_FILE_BYTES = 50 * 1024 * 1024;

/** Default total budget = maxFiles × max file size. */
export const DEFAULT_LOG_MAX_TOTAL_BYTES =
  DEFAULT_LOG_MAX_FILES * DEFAULT_LOG_MAX_FILE_BYTES;

/** Once per day. */
export const DEFAULT_LOG_RETENTION_INTERVAL_MS = 24 * 60 * 60 * 1000;

/** Active (non-rotated) pino destination — never pruned. */
export const ACTIVE_SERVER_LOG_NAME = "ccr.log";

/** Stable RFS history file — never pruned by the daily sweeper. */
export const SERVER_LOG_HISTORY_NAME = "ccr-history.txt";

/** Rotated server logs: `ccr-YYYYMMDDHHmmss.log` or `ccr-YYYYMMDDHHmmss_N.log`. */
const ROTATED_LOG_RE = /^ccr-\d{14}(?:_\d+)?\.log$/;

/** Legacy per-run RFS history leftovers: `ccr-….log.txt`. */
const ORPHAN_HISTORY_RE = /^ccr-\d{14}(?:_\d+)?\.log\.txt$/;

export interface PruneServerLogsOptions {
  /** Keep at most this many rotated `ccr-*.log` files (newest first). */
  maxFiles?: number;
  /**
   * Keep at most this many bytes of rotated logs combined. Always keeps the
   * newest file even if it alone exceeds the budget.
   */
  maxTotalBytes?: number;
}

export interface PruneServerLogsResult {
  kept: string[];
  deleted: string[];
  errors: Array<{ file: string; error: string }>;
}

interface LogEntry {
  name: string;
  path: string;
  size: number;
  mtimeMs: number;
}

function isRotatedServerLog(name: string): boolean {
  return ROTATED_LOG_RE.test(name);
}

function isOrphanHistory(name: string): boolean {
  return ORPHAN_HISTORY_RE.test(name);
}

/**
 * Delete rotated `ccr-*.log` files beyond count/size budgets, plus orphaned
 * legacy `ccr-*.log.txt` history files from pre-stable-history runs.
 * Never touches the active `ccr.log` or stable `ccr-history.txt`.
 */
export async function pruneServerLogs(
  logDir: string,
  options: PruneServerLogsOptions = {}
): Promise<PruneServerLogsResult> {
  const maxFiles = options.maxFiles ?? DEFAULT_LOG_MAX_FILES;
  const maxTotalBytes = options.maxTotalBytes ?? DEFAULT_LOG_MAX_TOTAL_BYTES;
  const result: PruneServerLogsResult = { kept: [], deleted: [], errors: [] };

  let names: string[];
  try {
    names = await readdir(logDir);
  } catch (error: any) {
    if (error?.code === "ENOENT") return result;
    throw error;
  }

  const rotated: LogEntry[] = [];
  const orphanHistories: string[] = [];

  for (const name of names) {
    if (isOrphanHistory(name)) {
      orphanHistories.push(name);
      continue;
    }
    if (!isRotatedServerLog(name)) continue;
    const filePath = join(logDir, name);
    try {
      const info = await stat(filePath);
      if (!info.isFile()) continue;
      rotated.push({
        name,
        path: filePath,
        size: info.size,
        mtimeMs: info.mtimeMs,
      });
    } catch (error: any) {
      result.errors.push({
        file: name,
        error: error?.message || String(error),
      });
    }
  }

  rotated.sort((a, b) => {
    if (b.mtimeMs !== a.mtimeMs) return b.mtimeMs - a.mtimeMs;
    return b.name.localeCompare(a.name);
  });

  let keptBytes = 0;
  const toDelete: LogEntry[] = [];
  for (let i = 0; i < rotated.length; i++) {
    const entry = rotated[i];
    const withinCount = result.kept.length < maxFiles;
    const withinBytes =
      result.kept.length === 0 || keptBytes + entry.size <= maxTotalBytes;
    if (withinCount && withinBytes) {
      result.kept.push(entry.name);
      keptBytes += entry.size;
    } else {
      toDelete.push(entry);
    }
  }

  for (const entry of toDelete) {
    try {
      await unlink(entry.path);
      result.deleted.push(entry.name);
    } catch (error: any) {
      if (error?.code === "ENOENT") continue;
      result.errors.push({
        file: entry.name,
        error: error?.message || String(error),
      });
    }
  }

  for (const name of orphanHistories) {
    try {
      await unlink(join(logDir, name));
      result.deleted.push(name);
    } catch (error: any) {
      if (error?.code === "ENOENT") continue;
      result.errors.push({
        file: name,
        error: error?.message || String(error),
      });
    }
  }

  return result;
}

export interface LogRetentionSchedulerOptions {
  logDir: string;
  maxFiles?: number;
  maxTotalBytes?: number;
  intervalMs?: number;
  logger?: {
    info?: (payload: any, message?: string) => void;
    warn?: (payload: any, message?: string) => void;
  };
  prune?: typeof pruneServerLogs;
  setIntervalFn?: typeof setInterval;
  clearIntervalFn?: typeof clearInterval;
}

/**
 * Runs log retention immediately on start, then once per day.
 * The timer is unref'd so it never keeps the process alive alone.
 */
export class LogRetentionScheduler {
  private readonly logDir: string;
  private readonly maxFiles: number;
  private readonly maxTotalBytes: number;
  private readonly intervalMs: number;
  private readonly logger?: LogRetentionSchedulerOptions["logger"];
  private readonly prune: typeof pruneServerLogs;
  private readonly setIntervalFn: typeof setInterval;
  private readonly clearIntervalFn: typeof clearInterval;
  private timer: ReturnType<typeof setInterval> | null = null;
  private running = false;

  constructor(options: LogRetentionSchedulerOptions) {
    this.logDir = options.logDir;
    this.maxFiles = options.maxFiles ?? DEFAULT_LOG_MAX_FILES;
    this.maxTotalBytes = options.maxTotalBytes ?? DEFAULT_LOG_MAX_TOTAL_BYTES;
    this.intervalMs =
      options.intervalMs ?? DEFAULT_LOG_RETENTION_INTERVAL_MS;
    this.logger = options.logger;
    this.prune = options.prune ?? pruneServerLogs;
    this.setIntervalFn = options.setIntervalFn ?? setInterval;
    this.clearIntervalFn = options.clearIntervalFn ?? clearInterval;
  }

  get enabled(): boolean {
    return this.intervalMs > 0;
  }

  start(): void {
    if (!this.enabled || this.timer) return;
    void this.runOnce("startup");
    this.timer = this.setIntervalFn(() => {
      void this.runOnce("schedule");
    }, this.intervalMs);
    this.timer.unref?.();
  }

  stop(): void {
    if (!this.timer) return;
    this.clearIntervalFn(this.timer);
    this.timer = null;
  }

  async runOnce(reason: "startup" | "schedule" | "manual" = "manual"): Promise<PruneServerLogsResult> {
    if (this.running) {
      return { kept: [], deleted: [], errors: [] };
    }
    this.running = true;
    try {
      const result = await this.prune(this.logDir, {
        maxFiles: this.maxFiles,
        maxTotalBytes: this.maxTotalBytes,
      });
      if (result.deleted.length || result.errors.length) {
        this.logger?.info?.(
          {
            reason,
            kept: result.kept.length,
            deleted: result.deleted.length,
            errors: result.errors.length,
            files: result.deleted,
          },
          "Server log retention prune"
        );
      }
      if (result.errors.length) {
        this.logger?.warn?.(
          { reason, errors: result.errors },
          "Server log retention prune errors"
        );
      }
      return result;
    } finally {
      this.running = false;
    }
  }
}

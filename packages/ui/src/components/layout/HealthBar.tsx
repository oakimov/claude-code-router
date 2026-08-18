import { useCallback, useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Activity, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { HealthSnapshot, HealthVitals } from "@/types";

type Severity = "ok" | "warn" | "bad";

/** Never poll faster than this, whatever the server reports as its interval. */
const MIN_POLL_MS = 30_000;
const DEFAULT_POLL_MS = 10 * 60 * 1000;

const SEVERITY_DOT: Record<Severity, string> = {
  ok: "bg-emerald-500",
  warn: "bg-amber-500",
  bad: "bg-rose-500",
};

const SEVERITY_TEXT: Record<Severity, string> = {
  ok: "text-emerald-600 dark:text-emerald-400",
  warn: "text-amber-600 dark:text-amber-400",
  bad: "text-rose-600 dark:text-rose-400",
};

const SEVERITY_RANK: Record<Severity, number> = { ok: 0, warn: 1, bad: 2 };

/** Classify a value where higher is worse. */
function grade(value: number, warnAt: number, badAt: number): Severity {
  if (value >= badAt) return "bad";
  if (value >= warnAt) return "warn";
  return "ok";
}

function formatBytes(bytes: number): string {
  const mb = bytes / 1024 / 1024;
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${Math.round(mb)} MB`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

type Metric = {
  key: string;
  label: string;
  value: string;
  /** Meter fill, 0..1. Omitted for count-only metrics. */
  fill?: number;
  severity: Severity;
  detail: string;
  /** Informational metrics are shown but do not drive the overall state. */
  informational?: boolean;
};

function buildMetrics(
  snapshot: HealthSnapshot,
  t: (key: string, options?: Record<string, unknown>) => string,
): Metric[] {
  const { memory, load, sessions, requests, cache } = snapshot;

  const memoryUsed = memory.systemTotal - memory.systemFree;
  const memoryRatio = memory.systemTotal ? memoryUsed / memory.systemTotal : 0;

  // Node routes on a single thread, so one core is the ceiling that matters.
  const cores = load.processCores ?? 0;

  const failureRatio = requests.completed ? requests.failed / requests.completed : 0;

  const metrics: Metric[] = [
    {
      key: "memory",
      label: t("health.memory"),
      value: formatBytes(memory.rss),
      fill: memoryRatio,
      severity: grade(memoryRatio, 0.75, 0.9),
      detail: t("health.memory_detail", {
        heapUsed: formatBytes(memory.heapUsed),
        heapTotal: formatBytes(memory.heapTotal),
        scope: memory.constrained ? t("health.container") : t("health.system"),
        used: formatBytes(memoryUsed),
        total: formatBytes(memory.systemTotal),
      }),
    },
    {
      key: "cpu",
      label: t("health.cpu"),
      value: `${cores.toFixed(2)} ${t("health.cores")}`,
      fill: Math.min(1, cores),
      severity: grade(cores, 0.5, 0.8),
      detail: t("health.cpu_detail", {
        load: load.avg.map((value) => value.toFixed(2)).join(" / "),
        cpus: load.cpus,
      }),
    },
    {
      key: "loop",
      label: t("health.event_loop"),
      value: `${load.eventLoopP99Ms.toFixed(0)} ms`,
      fill: Math.min(1, load.eventLoopP99Ms / 250),
      severity: grade(load.eventLoopP99Ms, 50, 200),
      detail: t("health.event_loop_detail", {
        mean: load.eventLoopMeanMs.toFixed(1),
      }),
    },
    {
      key: "sessions",
      label: t("health.sessions"),
      value: String(sessions.running),
      severity: "ok",
      informational: true,
      detail: t("health.sessions_detail", {
        active: sessions.activeInWindow,
        window: formatDuration(snapshot.windowMs),
      }),
    },
    {
      key: "requests",
      label: t("health.requests"),
      value: t("health.requests_value", {
        inFlight: requests.inFlight,
        completed: requests.completed,
      }),
      fill: failureRatio,
      severity: requests.completed ? grade(failureRatio, 0.02, 0.1) : "ok",
      detail: t("health.requests_detail", {
        failed: requests.failed,
        rate: (failureRatio * 100).toFixed(1),
        p95: requests.p95Ms === undefined ? "—" : formatDuration(requests.p95Ms),
        oldest:
          requests.oldestInFlightMs === undefined
            ? "—"
            : formatDuration(requests.oldestInFlightMs),
      }),
    },
  ];

  if (cache.hitRatio !== undefined) {
    // Low cache reuse costs money, it does not break the proxy: shown in
    // colour, but kept out of the overall state.
    metrics.push({
      key: "cache",
      label: t("health.cache"),
      value: `${(cache.hitRatio * 100).toFixed(0)}%`,
      fill: cache.hitRatio,
      severity: grade(1 - cache.hitRatio, 0.4, 0.7),
      informational: true,
      detail: t("health.cache_detail", {
        cached: cache.cachedTokens.toLocaleString(),
        prompt: cache.promptTokens.toLocaleString(),
      }),
    });
  }

  return metrics;
}

function Meter({ fill, severity }: { fill: number; severity: Severity }) {
  return (
    <div className="h-1 w-12 overflow-hidden rounded-full bg-muted">
      <div
        className={cn("h-full rounded-full transition-all", SEVERITY_DOT[severity])}
        style={{ width: `${Math.max(2, Math.min(100, fill * 100))}%` }}
      />
    </div>
  );
}

export function HealthBar() {
  const { t } = useTranslation();
  const [vitals, setVitals] = useState<HealthVitals | null>(null);
  const [unavailable, setUnavailable] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [fetchedAt, setFetchedAt] = useState<number>(() => Date.now());
  const [now, setNow] = useState<number>(() => Date.now());
  const [pollMs, setPollMs] = useState<number>(DEFAULT_POLL_MS);

  const load = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const response = await api.getHealth();
      if (!response.vitals) {
        // Server predates the heartbeat, or reporting is disabled.
        setUnavailable(true);
        return;
      }
      setVitals(response.vitals);
      setFetchedAt(Date.now());
      setNow(Date.now());
      setUnavailable(false);
      // Follow the server's own cadence; a shorter one still gets a floor.
      setPollMs(Math.max(MIN_POLL_MS, response.vitals.intervalMs || DEFAULT_POLL_MS));
    } catch {
      setUnavailable(true);
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    load();
    const timer = setInterval(() => load(), pollMs);
    return () => clearInterval(timer);
  }, [load, pollMs]);

  // The report is minutes apart, so the "updated N ago" label needs its own
  // ticker to stay honest between polls.
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(timer);
  }, []);

  const metrics = useMemo(
    () => (vitals ? buildMetrics(vitals.current, t) : []),
    [vitals, t],
  );

  const overall = useMemo<Severity>(() => {
    return metrics
      .filter((metric) => !metric.informational)
      .reduce<Severity>(
        (worst, metric) =>
          SEVERITY_RANK[metric.severity] > SEVERITY_RANK[worst] ? metric.severity : worst,
        "ok",
      );
  }, [metrics]);

  // Nothing to say is better than an empty shell on top of every page.
  if (unavailable || !vitals) return null;

  const statusLabel = t(`health.state_${overall}`);

  return (
    <div className="mb-2 flex items-center gap-3 overflow-x-auto rounded-md border border-border bg-card px-3 py-1.5 text-xs">
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="flex shrink-0 items-center gap-2">
            <span className="relative flex h-2 w-2">
              {overall !== "ok" && (
                <span
                  className={cn(
                    "absolute inline-flex h-full w-full animate-ping rounded-full opacity-60",
                    SEVERITY_DOT[overall],
                  )}
                />
              )}
              <span
                className={cn("relative inline-flex h-2 w-2 rounded-full", SEVERITY_DOT[overall])}
              />
            </span>
            <span className={cn("font-medium", SEVERITY_TEXT[overall])}>{statusLabel}</span>
          </div>
        </TooltipTrigger>
        <TooltipContent>
          <p>
            {t("health.process_detail", {
              pid: vitals.pid,
              node: vitals.node,
              uptime: formatDuration(vitals.current.uptimeMs),
            })}
          </p>
        </TooltipContent>
      </Tooltip>

      <div className="h-4 w-px shrink-0 bg-border" />

      <div className="flex min-w-0 flex-1 items-center gap-4">
        {metrics.map((metric) => (
          <Tooltip key={metric.key}>
            <TooltipTrigger asChild>
              <div className="flex shrink-0 items-center gap-1.5">
                <span className="text-muted-foreground">{metric.label}</span>
                <span className={cn("font-medium tabular-nums", SEVERITY_TEXT[metric.severity])}>
                  {metric.value}
                </span>
                {metric.fill !== undefined && (
                  <Meter fill={metric.fill} severity={metric.severity} />
                )}
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>{metric.detail}</p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>

      <div className="flex shrink-0 items-center gap-2 text-muted-foreground">
        <Activity className="h-3 w-3" />
        <span className="tabular-nums">
          {t("health.updated", { ago: formatDuration(Math.max(0, now - fetchedAt)) })}
        </span>
        <button
          type="button"
          onClick={() => load()}
          className="rounded-sm p-1 hover:bg-muted hover:text-foreground"
          aria-label={t("health.refresh")}
        >
          <RefreshCw className={cn("h-3 w-3", isRefreshing && "animate-spin")} />
        </button>
      </div>
    </div>
  );
}

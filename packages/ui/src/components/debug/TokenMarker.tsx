import { useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TokenUsage } from "@/components/ai-elements/context";

export type DebugTokenUsage = TokenUsage & {
  cacheWrite?: number;
};

function num(value?: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

export function TokenMarker({ usage }: { usage: DebugTokenUsage }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const reads = num(usage.input);
  const writes = num(usage.output);
  const cacheReads = num(usage.cacheRead);
  const cacheWrites = num(usage.cacheWrite);
  const total = num(usage.total) || reads + writes;
  if (!total && !reads && !writes && !cacheReads && !cacheWrites) return null;

  return (
    <div className="mt-1 w-fit rounded-md border bg-background/80 text-[10px] leading-tight">
      <button
        type="button"
        className="flex items-center gap-1 px-1.5 py-0.5 text-muted-foreground hover:text-foreground"
        onClick={() => setOpen((value) => !value)}
      >
        <ChevronDown className={cn("size-3 transition-transform", open ? "" : "-rotate-90")} />
        <span className="font-mono tabular-nums">
          {total.toLocaleString()} {t("debug.token_total")}
        </span>
      </button>
      {open ? (
        <div className="space-y-0.5 border-t px-1.5 py-1 font-mono tabular-nums">
          <Row label={t("debug.token_reads")} value={reads} />
          <Row label={t("debug.token_writes")} value={writes} />
          <Row label={t("debug.token_cache_reads")} value={cacheReads} />
          <Row label={t("debug.token_cache_writes")} value={cacheWrites} />
        </div>
      ) : null}
    </div>
  );
}

function Row({ label, value }: { label: string; value: number }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-muted-foreground">{label}</span>
      <span>{value.toLocaleString()}</span>
    </div>
  );
}

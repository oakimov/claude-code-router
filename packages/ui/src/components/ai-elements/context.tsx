"use client";

import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type { ComponentProps } from "react";
import { createContext, useContext } from "react";

export type TokenUsage = {
  input?: number;
  output?: number;
  total?: number;
  cacheRead?: number;
  cacheWrite?: number;
  reasoning?: number;
};

type ContextValue = TokenUsage & {
  maxTokens?: number;
};

const ContextContext = createContext<ContextValue>({});

const useContextUsage = () => useContext(ContextContext);

export type ContextProps = ComponentProps<typeof HoverCard> & ContextValue;

export const Context = ({
  input,
  output,
  total,
  cacheRead,
  reasoning,
  maxTokens,
  children,
  ...props
}: ContextProps) => (
  <ContextContext.Provider
    value={{ input, output, total, cacheRead, reasoning, maxTokens }}
  >
    <HoverCard closeDelay={0} openDelay={0} {...props}>
      {children}
    </HoverCard>
  </ContextContext.Provider>
);

export type ContextTriggerProps = ComponentProps<typeof HoverCardTrigger>;

export const ContextTrigger = ({
  className,
  children,
  ...props
}: ContextTriggerProps) => {
  const { total, maxTokens, input, output } = useContextUsage();
  const used = total ?? (input ?? 0) + (output ?? 0);
  const percent = maxTokens ? Math.min(100, Math.round((used / maxTokens) * 100)) : undefined;
  return (
    <HoverCardTrigger asChild {...props}>
      {children ?? (
        <button
          type="button"
          className={cn(
            "inline-flex items-center gap-2 rounded-md px-1.5 py-0.5 text-xs text-muted-foreground hover:bg-muted",
            className
          )}
        >
          {percent != null && (
            <Progress className="h-1.5 w-10" value={percent} />
          )}
          {used.toLocaleString()}
          {maxTokens != null ? ` / ${maxTokens.toLocaleString()}` : ""}
        </button>
      )}
    </HoverCardTrigger>
  );
};

export type ContextContentProps = ComponentProps<typeof HoverCardContent>;

export const ContextContent = ({ className, children, ...props }: ContextContentProps) => (
  <HoverCardContent
    className={cn("min-w-48 space-y-1 text-xs", className)}
    {...props}
  >
    {children}
  </HoverCardContent>
);

function Row({ label, value }: { label: string; value?: number }) {
  if (value == null) return null;
  return (
    <div className="flex items-center justify-between gap-6">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono tabular-nums">{value.toLocaleString()}</span>
    </div>
  );
}

export function ContextUsageSummary({ className }: { className?: string }) {
  const usage = useContextUsage();
  return (
    <div className={cn("space-y-1", className)}>
      <Row label="Input" value={usage.input} />
      <Row label="Output" value={usage.output} />
      <Row label="Cache" value={usage.cacheRead} />
      <Row label="Reasoning" value={usage.reasoning} />
      <Row label="Total" value={usage.total} />
    </div>
  );
}

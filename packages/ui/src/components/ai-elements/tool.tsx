"use client";

import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  ChevronDownIcon,
  CircleIcon,
  ClockIcon,
  WrenchIcon,
  XCircleIcon,
} from "lucide-react";
import type { ComponentProps, ReactNode } from "react";
import { isValidElement } from "react";

export type ToolProps = ComponentProps<typeof Collapsible>;

export const Tool = ({ className, ...props }: ToolProps) => (
  <Collapsible
    className={cn("not-prose mb-3 w-full rounded-md border", className)}
    {...props}
  />
);

export type ToolHeaderProps = {
  title?: string;
  type: ToolUIPart["type"];
  state: ToolUIPart["state"];
  className?: string;
};

const STATUS: Partial<
  Record<ToolUIPart["state"], { label: string; icon: ReactNode }>
> = {
  "input-streaming": {
    label: "Pending",
    icon: <CircleIcon className="size-3.5 text-muted-foreground" />,
  },
  "input-available": {
    label: "Running",
    icon: <ClockIcon className="size-3.5 animate-pulse text-muted-foreground" />,
  },
  "output-available": {
    label: "Completed",
    icon: <CheckCircleIcon className="size-3.5 text-primary" />,
  },
  "output-error": {
    label: "Error",
    icon: <XCircleIcon className="size-3.5 text-destructive" />,
  },
};

export const ToolHeader = ({
  className,
  title,
  type,
  state,
  ...props
}: ToolHeaderProps) => {
  const status = STATUS[state] ?? {
    label: String(state),
    icon: <ClockIcon className="size-3.5 text-muted-foreground" />,
  };
  return (
    <CollapsibleTrigger
      className={cn(
        "flex w-full items-center justify-between gap-3 p-3 text-left",
        className
      )}
      {...props}
    >
      <div className="flex min-w-0 items-center gap-2">
        <WrenchIcon className="size-4 shrink-0 text-muted-foreground" />
        <span className="truncate text-sm font-medium">
          {title ?? type.replace(/^tool-/, "")}
        </span>
        <Badge variant="secondary" className="gap-1 font-normal">
          {status.icon}
          {status.label}
        </Badge>
      </div>
      <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
    </CollapsibleTrigger>
  );
};

export type ToolContentProps = ComponentProps<typeof CollapsibleContent>;

export const ToolContent = ({ className, ...props }: ToolContentProps) => (
  <CollapsibleContent
    className={cn("space-y-3 border-t p-3", className)}
    {...props}
  />
);

function JsonBlock({ value }: { value: unknown }) {
  return (
    <pre className="overflow-x-auto rounded-md bg-muted p-2 text-xs">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export type ToolInputProps = ComponentProps<"div"> & {
  input: ToolUIPart["input"];
};

export const ToolInput = ({ className, input, ...props }: ToolInputProps) => (
  <div className={cn("space-y-1", className)} {...props}>
    <h4 className="text-xs font-medium text-muted-foreground">Parameters</h4>
    <JsonBlock value={input} />
  </div>
);

export type ToolOutputProps = ComponentProps<"div"> & {
  output: ToolUIPart["output"];
  errorText?: ToolUIPart["errorText"];
};

export const ToolOutput = ({
  className,
  output,
  errorText,
  ...props
}: ToolOutputProps) => {
  if (output == null && !errorText) return null;

  let rendered: ReactNode = output as ReactNode;
  if (typeof output === "object" && !isValidElement(output)) {
    rendered = <JsonBlock value={output} />;
  } else if (typeof output === "string") {
    rendered = (
      <pre className="overflow-x-auto rounded-md bg-muted p-2 text-xs whitespace-pre-wrap">
        {output}
      </pre>
    );
  }

  return (
    <div className={cn("space-y-1", className)} {...props}>
      <h4 className="text-xs font-medium text-muted-foreground">
        {errorText ? "Error" : "Result"}
      </h4>
      {errorText ? (
        <p className="text-xs text-destructive">{errorText}</p>
      ) : (
        rendered
      )}
    </div>
  );
};

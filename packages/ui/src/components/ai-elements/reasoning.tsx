"use client";

import { useControllableState } from "@radix-ui/react-use-controllable-state";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { cn } from "@/lib/utils";
import { ChevronDownIcon } from "lucide-react";
import type { ComponentProps } from "react";
import { createContext, useContext } from "react";

type ReasoningContextValue = {
  isStreaming: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
};

const ReasoningContext = createContext<ReasoningContextValue | null>(null);

const useReasoning = () => {
  const ctx = useContext(ReasoningContext);
  if (!ctx) throw new Error("Reasoning components must be used within Reasoning");
  return ctx;
};

export type ReasoningProps = ComponentProps<typeof Collapsible> & {
  isStreaming?: boolean;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
};

export const Reasoning = ({
  className,
  isStreaming = false,
  open,
  defaultOpen = false,
  onOpenChange,
  children,
  ...props
}: ReasoningProps) => {
  const [isOpen, setIsOpen] = useControllableState({
    prop: open,
    defaultProp: defaultOpen,
    onChange: onOpenChange,
  });

  return (
    <ReasoningContext.Provider value={{ isStreaming, isOpen, setIsOpen }}>
      <Collapsible
        className={cn("not-prose", className)}
        onOpenChange={setIsOpen}
        open={isOpen}
        {...props}
      >
        {children}
      </Collapsible>
    </ReasoningContext.Provider>
  );
};

export type ReasoningTriggerProps = ComponentProps<typeof CollapsibleTrigger>;

export const ReasoningTrigger = ({
  className,
  children,
  ...props
}: ReasoningTriggerProps) => {
  const { isStreaming, isOpen } = useReasoning();
  return (
    <CollapsibleTrigger
      className={cn(
        "flex w-fit items-center gap-1 rounded-sm py-0.5 text-[11px] text-muted-foreground hover:text-foreground",
        className
      )}
      {...props}
    >
      <ChevronDownIcon
        className={cn("size-3.5 transition-transform", isOpen ? "" : "-rotate-90")}
      />
      <span className={cn("italic", isStreaming && "animate-pulse")}>
        {children ?? "Thinking..."}
      </span>
    </CollapsibleTrigger>
  );
};

export type ReasoningContentProps = ComponentProps<typeof CollapsibleContent> & {
  children: string;
};

export const ReasoningContent = ({
  className,
  children,
  ...props
}: ReasoningContentProps) => (
  <CollapsibleContent
    className={cn("outline-none", className)}
    {...props}
  >
    <div className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap border-l-2 border-border pl-2 text-[11px] leading-snug text-muted-foreground">
      {children}
    </div>
  </CollapsibleContent>
);

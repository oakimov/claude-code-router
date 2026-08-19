"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Loader2Icon, SendIcon, SquareIcon } from "lucide-react";
import type { ChatStatus } from "ai";
import {
  type ComponentProps,
  type FormEvent,
  type KeyboardEvent,
  useRef,
} from "react";

export type PromptInputMessage = {
  text: string;
};

export type PromptInputProps = Omit<ComponentProps<"form">, "onSubmit"> & {
  onSubmit: (message: PromptInputMessage, event: FormEvent<HTMLFormElement>) => void;
};

export const PromptInput = ({
  className,
  onSubmit,
  children,
  ...props
}: PromptInputProps) => (
  <form
    className={cn("w-full overflow-hidden rounded-xl border bg-background shadow-sm", className)}
    onSubmit={(event) => {
      event.preventDefault();
      const form = event.currentTarget;
      const textarea = form.querySelector("textarea");
      const text = textarea?.value.trim() ?? "";
      onSubmit({ text }, event);
    }}
    {...props}
  >
    {children}
  </form>
);

export type PromptInputBodyProps = ComponentProps<"div">;

export const PromptInputBody = ({ className, ...props }: PromptInputBodyProps) => (
  <div className={cn("flex flex-col", className)} {...props} />
);

export type PromptInputTextareaProps = ComponentProps<"textarea">;

export const PromptInputTextarea = ({
  className,
  onKeyDown,
  ...props
}: PromptInputTextareaProps) => {
  const ref = useRef<HTMLTextAreaElement>(null);
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    onKeyDown?.(event);
    if (event.defaultPrevented) return;
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      event.currentTarget.form?.requestSubmit();
    }
  };
  return (
    <textarea
      className={cn(
        "w-full resize-none border-none bg-transparent px-3 py-3 text-sm outline-none placeholder:text-muted-foreground",
        "field-sizing-content max-h-48 min-h-[48px]",
        className
      )}
      name="message"
      onKeyDown={handleKeyDown}
      ref={ref}
      rows={1}
      {...props}
    />
  );
};

export type PromptInputFooterProps = ComponentProps<"div">;

export const PromptInputFooter = ({ className, ...props }: PromptInputFooterProps) => (
  <div
    className={cn("flex items-center justify-between gap-2 px-2 pb-2", className)}
    {...props}
  />
);

export type PromptInputToolsProps = ComponentProps<"div">;

export const PromptInputTools = ({ className, ...props }: PromptInputToolsProps) => (
  <div className={cn("flex items-center gap-1", className)} {...props} />
);

export type PromptInputSubmitProps = ComponentProps<typeof Button> & {
  status?: ChatStatus;
};

export const PromptInputSubmit = ({
  className,
  status,
  children,
  ...props
}: PromptInputSubmitProps) => {
  const isBusy = status === "submitted" || status === "streaming";
  let icon = <SendIcon className="size-4" />;
  if (status === "submitted") icon = <Loader2Icon className="size-4 animate-spin" />;
  if (status === "streaming") icon = <SquareIcon className="size-4" />;
  return (
    <Button
      className={cn("size-8 rounded-lg", className)}
      size="icon"
      type={isBusy && props.onClick ? "button" : "submit"}
      variant="default"
      {...props}
    >
      {children ?? icon}
    </Button>
  );
};

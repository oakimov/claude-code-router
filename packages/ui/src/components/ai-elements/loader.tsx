import { cn } from "@/lib/utils";
import type { HTMLAttributes } from "react";

type LoaderIconProps = {
  size?: number;
};

const LoaderIcon = ({ size = 16 }: LoaderIconProps) => (
  <svg
    height={size}
    strokeLinejoin="round"
    viewBox="0 0 16 16"
    width={size}
    style={{ color: "currentcolor" }}
  >
    <title>Loader</title>
    <g fill="none" stroke="currentColor" strokeLinecap="round">
      <path d="M8 1.5v2.25" opacity="1" />
      <path d="M12.596 3.404l-1.591 1.591" opacity="0.8" />
      <path d="M14.5 8h-2.25" opacity="0.6" />
      <path d="M12.596 12.596l-1.591-1.591" opacity="0.4" />
      <path d="M8 14.5v-2.25" opacity="0.2" />
      <path d="M3.404 12.596l1.591-1.591" opacity="0.15" />
      <path d="M1.5 8h2.25" opacity="0.1" />
      <path d="M3.404 3.404l1.591 1.591" opacity="0.05" />
    </g>
  </svg>
);

export type LoaderProps = HTMLAttributes<HTMLDivElement> & {
  size?: number;
};

export const Loader = ({ className, size = 16, ...props }: LoaderProps) => (
  <div
    className={cn("inline-flex animate-spin items-center justify-center", className)}
    {...props}
  >
    <LoaderIcon size={size} />
  </div>
);

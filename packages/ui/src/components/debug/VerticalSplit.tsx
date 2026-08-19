import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

type VerticalSplitProps = {
  children: [ReactNode, ReactNode];
  defaultTopPercent?: number;
  minPercent?: number;
  onResize?: () => void;
};

export function VerticalSplit({
  children,
  defaultTopPercent = 85,
  minPercent = 12,
  onResize,
}: VerticalSplitProps) {
  const [topPercent, setTopPercent] = useState(defaultTopPercent);
  const splitRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const updateFromClientY = useCallback(
    (clientY: number) => {
      const root = splitRef.current;
      if (!root) return;
      const rect = root.getBoundingClientRect();
      if (rect.height <= 0) return;
      const next = ((clientY - rect.top) / rect.height) * 100;
      setTopPercent(Math.min(100 - minPercent, Math.max(minPercent, next)));
      onResize?.();
    },
    [minPercent, onResize]
  );

  useEffect(() => {
    const onMove = (event: PointerEvent) => {
      if (!dragging.current) return;
      event.preventDefault();
      updateFromClientY(event.clientY);
    };
    const onUp = () => {
      dragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      onUp();
    };
  }, [updateFromClientY]);

  return (
    <div ref={splitRef} className="flex min-h-0 flex-1 flex-col">
      <div
        className="flex min-h-0 flex-col overflow-hidden"
        style={{ flex: `0 0 ${topPercent}%` }}
      >
        {children[0]}
      </div>
      <div
        role="separator"
        aria-orientation="horizontal"
        aria-valuemin={minPercent}
        aria-valuemax={100 - minPercent}
        aria-valuenow={Math.round(topPercent)}
        tabIndex={0}
        className="group relative z-10 flex h-3 shrink-0 cursor-row-resize items-center justify-center bg-muted/40 hover:bg-muted touch-none select-none"
        aria-label="Resize request and response panes"
        onPointerDown={(event) => {
          event.preventDefault();
          dragging.current = true;
          document.body.style.cursor = "row-resize";
          document.body.style.userSelect = "none";
          updateFromClientY(event.clientY);
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowUp") {
            event.preventDefault();
            setTopPercent((value) => Math.max(minPercent, value - 3));
            onResize?.();
          }
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setTopPercent((value) => Math.min(100 - minPercent, value + 3));
            onResize?.();
          }
        }}
      >
        <div className="h-1 w-12 rounded-full bg-border group-hover:bg-primary group-focus-visible:bg-primary" />
      </div>
      <div className="flex min-h-0 flex-1 flex-col overflow-hidden">
        {children[1]}
      </div>
    </div>
  );
}

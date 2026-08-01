import { useEffect, useState, type ReactNode } from "react";
import { Outlet } from "react-router";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppSidebar } from "@/components/layout/AppSidebar";
import { ShellChromeProvider } from "@/components/layout/ShellChrome";
import { cn } from "@/lib/utils";

const COLLAPSE_KEY = "ccr-ui-sidebar-collapsed";

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSE_KEY) === "1";
  } catch {
    return false;
  }
}

type AppShellProps = {
  children?: ReactNode;
};

export function AppShell({ children }: AppShellProps) {
  const [collapsed, setCollapsed] = useState(readCollapsed);

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSE_KEY, collapsed ? "1" : "0");
    } catch {
      // ignore
    }
  }, [collapsed]);

  return (
    <TooltipProvider>
      <ShellChromeProvider>
        <div className="flex h-dvh overflow-hidden bg-background font-sans">
          <AppSidebar collapsed={collapsed} onCollapsedChange={setCollapsed} />
          <div
            className={cn(
              "content-container my-2 mr-2 min-w-0 flex-1 overflow-hidden rounded-md border border-border bg-card",
            )}
          >
            <main className="content-container-inner custom-scrollbar h-full overflow-auto p-4 md:p-6">
              {children ?? <Outlet />}
            </main>
          </div>
        </div>
      </ShellChromeProvider>
    </TooltipProvider>
  );
}

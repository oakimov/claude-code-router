import {
  Bug,
  FileCog,
  FileJson,
  FileText,
  LayoutDashboard,
  Languages,
  Settings,
  PanelLeftClose,
  PanelLeft,
} from "lucide-react";
import { useTranslation } from "react-i18next";
import { useLocation, useNavigate } from "react-router";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ThemeToggle } from "@/components/layout/ThemeToggle";
import { useShellChrome } from "@/components/layout/ShellChrome";
import { cn } from "@/lib/utils";

type NavItem =
  | { kind: "route"; id: string; labelKey: string; icon: typeof LayoutDashboard; path: string }
  | { kind: "action"; id: string; labelKey: string; icon: typeof FileText; action: "logs" | "settings" | "json" };

const NAV_ITEMS: NavItem[] = [
  { kind: "route", id: "dashboard", labelKey: "nav.dashboard", icon: LayoutDashboard, path: "/dashboard" },
  { kind: "route", id: "presets", labelKey: "nav.presets", icon: FileCog, path: "/presets" },
  { kind: "action", id: "logs", labelKey: "nav.logs", icon: FileText, action: "logs" },
  { kind: "route", id: "debug", labelKey: "nav.debug", icon: Bug, path: "/debug" },
];

const FOOTER_ACTIONS: NavItem[] = [
  { kind: "action", id: "json", labelKey: "nav.json", icon: FileJson, action: "json" },
  { kind: "action", id: "settings", labelKey: "nav.settings", icon: Settings, action: "settings" },
];

type AppSidebarProps = {
  collapsed: boolean;
  onCollapsedChange: (collapsed: boolean) => void;
};

export function AppSidebar({ collapsed, onCollapsedChange }: AppSidebarProps) {
  const { t, i18n } = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const chrome = useShellChrome();

  const runAction = (action: "logs" | "settings" | "json") => {
    if (action === "logs") chrome.openLogs();
    if (action === "settings") chrome.openSettings();
    if (action === "json") chrome.openJsonEditor();
  };

  const renderItem = (item: NavItem) => {
    const Icon = item.icon;
    const active =
      item.kind === "route" &&
      (location.pathname === item.path ||
        (item.path === "/dashboard" && location.pathname === "/"));

    const button = (
      <Button
        key={item.id}
        variant="ghost"
        size={collapsed ? "icon" : "sm"}
        className={cn(
          "w-full rounded-sm transition-colors",
          collapsed ? "justify-center" : "justify-start gap-2 px-2.5",
          active
            ? "bg-sidebar-accent text-primary border border-primary/20"
            : "text-muted-foreground hover:bg-sidebar-accent hover:text-foreground border border-transparent",
        )}
        onClick={() => {
          if (item.kind === "route") navigate(item.path);
          else runAction(item.action);
        }}
        aria-label={t(item.labelKey)}
      >
        <Icon className={cn("h-4 w-4 shrink-0", active && "text-primary")} />
        {!collapsed && <span className="truncate">{t(item.labelKey)}</span>}
      </Button>
    );

    if (!collapsed) return button;

    return (
      <Tooltip key={item.id}>
        <TooltipTrigger asChild>{button}</TooltipTrigger>
        <TooltipContent side="right">{t(item.labelKey)}</TooltipContent>
      </Tooltip>
    );
  };

  return (
    <aside
      className={cn(
        "flex h-[calc(100dvh-1rem)] flex-col border border-sidebar-border bg-sidebar text-sidebar-foreground rounded-md my-2 ml-2",
        collapsed ? "w-14" : "w-56",
      )}
    >
      <div className={cn("flex items-center gap-2 border-b border-sidebar-border p-3", collapsed && "justify-center px-2")}>
        {!collapsed && (
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-semibold tracking-tight">{t("app.title_short")}</div>
            <div className="truncate text-xs text-muted-foreground">{t("nav.console")}</div>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 shrink-0 rounded-sm text-muted-foreground"
          onClick={() => onCollapsedChange(!collapsed)}
          aria-label={collapsed ? t("nav.expand_sidebar") : t("nav.collapse_sidebar")}
        >
          {collapsed ? <PanelLeft className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
        </Button>
      </div>

      <nav className="flex flex-1 flex-col gap-1 overflow-y-auto p-2">
        {NAV_ITEMS.map(renderItem)}
      </nav>

      <div className="mt-auto space-y-1 border-t border-sidebar-border p-2">
        {FOOTER_ACTIONS.map(renderItem)}

        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size={collapsed ? "icon" : "sm"}
              className={cn(
                "w-full rounded-sm text-muted-foreground hover:text-foreground",
                collapsed ? "justify-center" : "justify-start gap-2",
              )}
              aria-label={t("app.language")}
            >
              <Languages className="h-4 w-4 shrink-0" />
              {!collapsed && <span className="truncate">{t("app.language")}</span>}
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-36 p-1" align="start" side="top">
            <div className="space-y-0.5">
              <Button
                variant={i18n.language.startsWith("en") ? "secondary" : "ghost"}
                size="sm"
                className="w-full justify-start rounded-sm"
                onClick={() => i18n.changeLanguage("en")}
              >
                English
              </Button>
              <Button
                variant={i18n.language.startsWith("zh") ? "secondary" : "ghost"}
                size="sm"
                className="w-full justify-start rounded-sm"
                onClick={() => i18n.changeLanguage("zh")}
              >
                中文
              </Button>
            </div>
          </PopoverContent>
        </Popover>

        <ThemeToggle collapsed={collapsed} />

        {!collapsed && (
          <div className="px-2 pt-1 text-[10px] text-muted-foreground">CCR UI</div>
        )}
      </div>
    </aside>
  );
}

import { Monitor, Moon, Sun } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { useTheme, type Theme } from "@/components/theme-provider";
import { cn } from "@/lib/utils";

const OPTIONS: { value: Theme; icon: typeof Sun; labelKey: string }[] = [
  { value: "light", icon: Sun, labelKey: "nav.theme_light" },
  { value: "dark", icon: Moon, labelKey: "nav.theme_dark" },
  { value: "system", icon: Monitor, labelKey: "nav.theme_system" },
];

export function ThemeToggle({ collapsed = false }: { collapsed?: boolean }) {
  const { t } = useTranslation();
  const { theme, setTheme, resolvedTheme } = useTheme();
  const ActiveIcon = resolvedTheme === "dark" ? Moon : Sun;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size={collapsed ? "icon" : "sm"}
          className={cn(
            "w-full justify-start gap-2 rounded-sm text-muted-foreground hover:text-foreground",
            collapsed && "justify-center",
          )}
          aria-label={t("nav.theme")}
        >
          <ActiveIcon className="h-4 w-4 shrink-0" />
          {!collapsed && <span className="truncate">{t("nav.theme")}</span>}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-40 p-1" align="start" side="top">
        <div className="space-y-0.5">
          {OPTIONS.map(({ value, icon: Icon, labelKey }) => (
            <Button
              key={value}
              variant={theme === value ? "secondary" : "ghost"}
              size="sm"
              className="w-full justify-start gap-2 rounded-sm"
              onClick={() => setTheme(value)}
            >
              <Icon className="h-3.5 w-3.5" />
              {t(labelKey)}
            </Button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

import { Pencil, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  isBuiltinWeatherTool,
  isToolEnabled,
  toolName,
  type DebugTool,
} from "@/lib/debugTools";
import { cn } from "@/lib/utils";

type ToolListProps = {
  tools: DebugTool[];
  onEdit: (index: number) => void;
  onRemove: (index: number) => void;
  onToggle: (index: number, enabled: boolean) => void;
};

function ToolCard({
  tool,
  index,
  onEdit,
  onRemove,
  onToggle,
}: {
  tool: DebugTool;
  index: number;
  onEdit: (index: number) => void;
  onRemove: (index: number) => void;
  onToggle: (index: number, enabled: boolean) => void;
}) {
  const { t } = useTranslation();
  const name = toolName(tool) || t("debug.tools_unnamed");
  const description = String(tool.function?.description || "");
  const builtin = isBuiltinWeatherTool(tool);
  const enabled = isToolEnabled(tool);

  return (
    <div
      className={cn(
        "inline-flex w-fit max-w-full shrink-0 items-center gap-1 rounded-md border bg-card px-1.5 py-1",
        !enabled && "opacity-60"
      )}
    >
      <Switch
        className="h-4 w-7 shrink-0 [&>span]:h-3 [&>span]:w-3 [&>span]:data-[state=checked]:translate-x-3 [&>span]:data-[state=unchecked]:translate-x-0"
        checked={enabled}
        onCheckedChange={(checked) => onToggle(index, Boolean(checked))}
        aria-label={t("debug.tools_enabled")}
      />
      <div className="w-max min-w-0 max-w-full">
        <p className="whitespace-nowrap text-[11px] font-medium leading-tight">{name}</p>
        {description ? (
          <p className="truncate text-[10px] leading-tight text-muted-foreground">{description}</p>
        ) : null}
      </div>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="size-6 shrink-0"
        onClick={() => onEdit(index)}
        aria-label={t("debug.tools_edit")}
      >
        <Pencil className="size-3" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="size-6 shrink-0 text-muted-foreground"
        onClick={() => onRemove(index)}
        disabled={builtin}
        title={builtin ? t("debug.tools_builtin_locked") : t("debug.tools_delete")}
        aria-label={t("debug.tools_delete")}
      >
        <Trash2 className="size-3" />
      </Button>
    </div>
  );
}

export function ToolList({ tools, onEdit, onRemove, onToggle }: ToolListProps) {
  const { t } = useTranslation();
  if (!tools.length) {
    return (
      <div className="w-fit rounded-md border px-2 py-3 text-center text-[11px] text-muted-foreground">
        {t("debug.tools_empty")}
      </div>
    );
  }

  return (
    <div className="flex flex-wrap content-start gap-1.5">
      {tools.map((tool, index) => (
        <ToolCard
          key={`${toolName(tool)}-${index}`}
          tool={tool}
          index={index}
          onEdit={onEdit}
          onRemove={onRemove}
          onToggle={onToggle}
        />
      ))}
    </div>
  );
}

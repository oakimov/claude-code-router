import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Plus, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";
import type { HeaderRow } from "@/lib/debugChat";
import { newHeaderRow } from "@/lib/debugChat";

type HeaderTableProps = {
  rows: HeaderRow[];
  onChange: (rows: HeaderRow[]) => void;
};

export function HeaderTable({ rows, onChange }: HeaderTableProps) {
  const { t } = useTranslation();

  const update = (id: string, patch: Partial<HeaderRow>) => {
    onChange(rows.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  };

  const remove = (id: string) => {
    const next = rows.filter((row) => row.id !== id);
    onChange(next.length > 0 ? next : [newHeaderRow()]);
  };

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="grid grid-cols-[2rem_1fr_1fr_2rem] gap-2 border-b px-1 pb-2 text-xs font-medium text-muted-foreground">
        <span />
        <span>{t("debug.header_key")}</span>
        <span>{t("debug.header_value")}</span>
        <span />
      </div>
      <div className="min-h-0 flex-1 space-y-2 overflow-auto py-2">
        {rows.map((row) => (
          <div key={row.id} className="grid grid-cols-[2rem_1fr_1fr_2rem] items-center gap-2">
            <Checkbox
              checked={row.enabled}
              onCheckedChange={(checked) => update(row.id, { enabled: Boolean(checked) })}
              aria-label={t("debug.header_enabled")}
            />
            <Input
              className="h-8"
              placeholder={t("debug.header_key")}
              value={row.key}
              onChange={(e) => update(row.id, { key: e.target.value })}
            />
            <Input
              className="h-8"
              placeholder={t("debug.header_value")}
              value={row.value}
              onChange={(e) => update(row.id, { value: e.target.value })}
            />
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="size-8 text-muted-foreground"
              onClick={() => remove(row.id)}
              aria-label={t("debug.header_delete")}
            >
              <Trash2 className="size-3.5" />
            </Button>
          </div>
        ))}
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="mt-2 self-start"
        onClick={() => onChange([...rows, newHeaderRow()])}
      >
        <Plus className="mr-1 size-3.5" />
        {t("debug.header_add")}
      </Button>
    </div>
  );
}

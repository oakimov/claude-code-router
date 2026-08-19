import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import MonacoEditor from "@monaco-editor/react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useMonacoTheme } from "@/hooks/useMonacoTheme";
import {
  isBuiltinWeatherTool,
  parseToolDefinition,
  stringifyTool,
  toolName,
  type DebugTool,
} from "@/lib/debugTools";

const TOOL_NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;

type ToolEditorDialogProps = {
  open: boolean;
  tool: DebugTool | null;
  existingNames: string[];
  isNew: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (tool: DebugTool) => void;
};

export function ToolEditorDialog({
  open,
  tool,
  existingNames,
  isNew,
  onOpenChange,
  onSave,
}: ToolEditorDialogProps) {
  const { t } = useTranslation();
  const monacoTheme = useMonacoTheme();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [definition, setDefinition] = useState("{}");
  const [error, setError] = useState<string | null>(null);
  const nameLocked = Boolean(tool && isBuiltinWeatherTool(tool));

  useEffect(() => {
    if (!open || !tool) return;
    setName(toolName(tool));
    setDescription(String(tool.function?.description || ""));
    setDefinition(stringifyTool(tool));
    setError(null);
  }, [open, tool]);

  const applyFieldsToDefinition = (nextName: string, nextDescription: string) => {
    const parsed = parseToolDefinition(definition);
    if (!parsed) return;
    parsed.function.name = nextName;
    parsed.function.description = nextDescription;
    setDefinition(stringifyTool(parsed));
  };

  const handleDefinitionChange = (value: string) => {
    const next = value || "{}";
    setDefinition(next);
    const parsed = parseToolDefinition(next);
    if (!parsed) return;
    setName(toolName(parsed));
    setDescription(String(parsed.function?.description || ""));
    setError(null);
  };

  const handleSave = () => {
    const parsed = parseToolDefinition(definition);
    if (!parsed) {
      setError(t("debug.tools_invalid_json"));
      return;
    }
    const nextName =
      nameLocked && tool
        ? toolName(tool)
        : toolName(parsed) || name.trim();
    if (!nextName) {
      setError(t("debug.tools_name_required"));
      return;
    }
    if (!TOOL_NAME_RE.test(nextName)) {
      setError(t("debug.tools_name_invalid"));
      return;
    }
    if (existingNames.includes(nextName)) {
      setError(t("debug.tools_name_duplicate"));
      return;
    }
    parsed.function.name = nextName;
    parsed.function.description = description.trim() || parsed.function.description;
    onSave(parsed);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[80vh] w-full max-w-2xl flex-col translate-x-0 translate-y-0 left-0 right-0 top-[8vh] mx-auto overflow-hidden data-[state=open]:slide-in-from-left-0 data-[state=open]:slide-in-from-top-2 data-[state=closed]:slide-out-to-left-0 data-[state=closed]:slide-out-to-top-2 data-[state=open]:zoom-in-100 data-[state=closed]:zoom-out-100">
        <DialogHeader>
          <DialogTitle>{isNew ? t("debug.tools_add") : t("debug.tools_edit")}</DialogTitle>
        </DialogHeader>
        <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-hidden">
          <div className="shrink-0 space-y-1.5">
            <Label htmlFor="debug-tool-name">{t("debug.tools_name")}</Label>
            <Input
              id="debug-tool-name"
              value={name}
              disabled={nameLocked}
              onChange={(event) => {
                const next = event.target.value;
                setName(next);
                applyFieldsToDefinition(next, description);
                setError(null);
              }}
            />
          </div>
          <div className="shrink-0 space-y-1.5">
            <Label htmlFor="debug-tool-description">{t("debug.tools_description")}</Label>
            <Textarea
              id="debug-tool-description"
              className="min-h-[72px]"
              value={description}
              onChange={(event) => {
                const next = event.target.value;
                setDescription(next);
                applyFieldsToDefinition(name, next);
                setError(null);
              }}
            />
          </div>
          <div className="flex min-h-0 flex-1 flex-col space-y-1.5">
            <Label>{t("debug.tools_definition")}</Label>
            <div className="h-[260px] min-h-[260px] overflow-hidden rounded-md border">
              <MonacoEditor
                height="260px"
                language="json"
                theme={monacoTheme}
                value={definition}
                onChange={(value) => handleDefinitionChange(value || "{}")}
                options={{
                  minimap: { enabled: false },
                  fontSize: 12,
                  wordWrap: "on",
                  automaticLayout: true,
                  folding: false,
                  smoothScrolling: false,
                  fixedOverflowWidgets: true,
                  scrollBeyondLastLine: false,
                  renderLineHighlight: "none",
                  overviewRulerLanes: 0,
                  scrollbar: {
                    verticalScrollbarSize: 8,
                    horizontalScrollbarSize: 8,
                    alwaysConsumeMouseWheel: true,
                  },
                }}
              />
            </div>
          </div>
          {error ? <p className="shrink-0 text-sm text-destructive">{error}</p> : null}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {t("app.cancel")}
          </Button>
          <Button onClick={handleSave}>{t("app.save")}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

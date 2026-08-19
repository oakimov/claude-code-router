import { cn } from "@/lib/utils";
import MonacoEditor from "@monaco-editor/react";
import { useMonacoTheme } from "@/hooks/useMonacoTheme";

const VIEWER_OPTIONS = {
  readOnly: true,
  domReadOnly: true,
  minimap: { enabled: false },
  fontSize: 11,
  wordWrap: "on" as const,
  automaticLayout: true,
  scrollBeyondLastLine: false,
  lineDecorationsWidth: 8,
  lineNumbersMinChars: 3,
  folding: true,
  renderLineHighlight: "none" as const,
  contextmenu: false,
};

export function JsonPane({
  value,
  onMount,
  error = false,
}: {
  value: string;
  onMount?: (editor: any) => void;
  error?: boolean;
}) {
  const theme = useMonacoTheme();
  return (
    <div className={cn("h-full min-h-0", error && "debug-response-error")}>
      <MonacoEditor
        height="100%"
        language="json"
        theme={theme}
        value={value || ""}
        onMount={(editor) => onMount?.(editor)}
        options={VIEWER_OPTIONS}
      />
    </div>
  );
}

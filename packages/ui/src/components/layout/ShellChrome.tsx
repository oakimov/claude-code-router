import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useTranslation } from "react-i18next";
import { SettingsDialog } from "@/components/SettingsDialog";
import { JsonEditor } from "@/components/JsonEditor";
import { LogViewer } from "@/components/LogViewer";
import { Toast } from "@/components/ui/toast";

type ShellChromeContextValue = {
  openLogs: () => void;
  openSettings: () => void;
  openJsonEditor: () => void;
  showToast: (message: string, type?: "success" | "error" | "warning") => void;
};

const ShellChromeContext = createContext<ShellChromeContextValue | null>(null);

export function ShellChromeProvider({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isJsonEditorOpen, setIsJsonEditorOpen] = useState(false);
  const [isLogViewerOpen, setIsLogViewerOpen] = useState(false);
  const [toast, setToast] = useState<{
    message: string;
    type: "success" | "error" | "warning";
  } | null>(null);

  const openLogs = useCallback(() => setIsLogViewerOpen(true), []);
  const openSettings = useCallback(() => setIsSettingsOpen(true), []);
  const openJsonEditor = useCallback(() => setIsJsonEditorOpen(true), []);
  const showToast = useCallback(
    (message: string, type: "success" | "error" | "warning" = "success") => {
      setToast({ message, type });
    },
    [],
  );

  const value = useMemo(
    () => ({ openLogs, openSettings, openJsonEditor, showToast }),
    [openLogs, openSettings, openJsonEditor, showToast],
  );

  return (
    <ShellChromeContext.Provider value={value}>
      {children}
      <SettingsDialog isOpen={isSettingsOpen} onOpenChange={setIsSettingsOpen} />
      <JsonEditor
        open={isJsonEditorOpen}
        onOpenChange={setIsJsonEditorOpen}
        showToast={(message, type) => setToast({ message, type })}
      />
      <LogViewer
        open={isLogViewerOpen}
        onOpenChange={setIsLogViewerOpen}
        showToast={(message, type) => setToast({ message, type })}
      />
      {toast && (
        <Toast
          message={toast.message || t("app.config_saved_success")}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </ShellChromeContext.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useShellChrome() {
  const ctx = useContext(ShellChromeContext);
  if (!ctx) {
    throw new Error("useShellChrome must be used within ShellChromeProvider");
  }
  return ctx;
}

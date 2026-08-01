import { useState, useEffect, useCallback, useRef } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router";
import { Transformers } from "@/components/Transformers";
import { Providers } from "@/components/Providers";
import { Router } from "@/components/Router";
import { Button } from "@/components/ui/button";
import { useConfig } from "@/components/ConfigProvider";
import { api } from "@/lib/api";
import { Save, RefreshCw, CircleArrowUp } from "lucide-react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { PageHeader } from "@/components/layout/PageHeader";
import { useShellChrome } from "@/components/layout/ShellChrome";
import "@/styles/animations.css";

function App() {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { config, error } = useConfig();
  const { showToast } = useShellChrome();
  const [isCheckingAuth, setIsCheckingAuth] = useState(true);
  const [isNewVersionAvailable, setIsNewVersionAvailable] = useState(false);
  const [isUpdateDialogOpen, setIsUpdateDialogOpen] = useState(false);
  const [newVersionInfo, setNewVersionInfo] = useState<{ version: string; changelog: string } | null>(null);
  const [isCheckingUpdate, setIsCheckingUpdate] = useState(false);
  const [hasCheckedUpdate, setHasCheckedUpdate] = useState(false);
  const [isUpdateFeatureAvailable, setIsUpdateFeatureAvailable] = useState(true);
  const hasAutoCheckedUpdate = useRef(false);

  const saveConfig = async () => {
    if (!config) {
      showToast(t("app.config_missing"), "error");
      return;
    }

    try {
      const response = await api.updateConfig(config);
      if (response && typeof response === "object" && "success" in response) {
        const apiResponse = response as unknown as { success: boolean; message?: string };
        if (apiResponse.success) {
          showToast(apiResponse.message || t("app.config_saved_success"), "success");
        } else {
          showToast(apiResponse.message || t("app.config_saved_failed"), "error");
        }
      } else {
        showToast(t("app.config_saved_success"), "success");
      }
    } catch (err) {
      console.error("Failed to save config:", err);
      showToast(t("app.config_saved_failed") + ": " + (err as Error).message, "error");
    }
  };

  const saveConfigAndRestart = async () => {
    if (!config) {
      showToast(t("app.config_missing"), "error");
      return;
    }

    try {
      const response = await api.updateConfig(config);

      let saveSuccessful = true;
      if (response && typeof response === "object" && "success" in response) {
        const apiResponse = response as unknown as { success: boolean; message?: string };
        if (!apiResponse.success) {
          saveSuccessful = false;
          showToast(apiResponse.message || t("app.config_saved_failed"), "error");
        }
      }

      if (saveSuccessful) {
        const restartResponse = await api.restartService();
        if (restartResponse && typeof restartResponse === "object" && "success" in restartResponse) {
          const apiResponse = restartResponse as unknown as { success: boolean; message?: string };
          if (apiResponse.success) {
            showToast(apiResponse.message || t("app.config_saved_restart_success"), "success");
          }
        } else {
          showToast(t("app.config_saved_restart_success"), "success");
        }
      }
    } catch (err) {
      console.error("Failed to save config and restart:", err);
      showToast(t("app.config_saved_restart_failed") + ": " + (err as Error).message, "error");
    }
  };

  const checkForUpdates = useCallback(
    async (showDialog: boolean = true) => {
      if (hasCheckedUpdate && isNewVersionAvailable) {
        if (showDialog) setIsUpdateDialogOpen(true);
        return;
      }

      setIsCheckingUpdate(true);
      try {
        const updateInfo = await api.checkForUpdates();

        if (updateInfo.hasUpdate && updateInfo.latestVersion && updateInfo.changelog) {
          setIsNewVersionAvailable(true);
          setNewVersionInfo({
            version: updateInfo.latestVersion,
            changelog: updateInfo.changelog,
          });
          if (showDialog) setIsUpdateDialogOpen(true);
        } else if (showDialog) {
          showToast(t("app.no_updates_available"), "success");
        }

        setHasCheckedUpdate(true);
      } catch (err) {
        console.error("Failed to check for updates:", err);
        setIsUpdateFeatureAvailable(false);
        if (showDialog) {
          showToast(t("app.update_check_failed") + ": " + (err as Error).message, "error");
        }
      } finally {
        setIsCheckingUpdate(false);
      }
    },
    [hasCheckedUpdate, isNewVersionAvailable, showToast, t],
  );

  useEffect(() => {
    const checkAuth = async () => {
      if (config) {
        setIsCheckingAuth(false);
        if (!hasCheckedUpdate && !hasAutoCheckedUpdate.current) {
          hasAutoCheckedUpdate.current = true;
          checkForUpdates(false);
        }
        return;
      }

      const apiKey = localStorage.getItem("apiKey");
      if (!apiKey) {
        setIsCheckingAuth(false);
        return;
      }

      try {
        await api.getConfig();
      } catch (err) {
        console.error("Error checking auth:", err);
        if ((err as Error).message === "Unauthorized") {
          navigate("/login");
        }
      } finally {
        setIsCheckingAuth(false);
        if (!hasCheckedUpdate && !hasAutoCheckedUpdate.current) {
          hasAutoCheckedUpdate.current = true;
          checkForUpdates(false);
        }
      }
    };

    checkAuth();

    const handleUnauthorized = () => {
      navigate("/login");
    };

    window.addEventListener("unauthorized", handleUnauthorized);
    return () => window.removeEventListener("unauthorized", handleUnauthorized);
  }, [config, navigate, hasCheckedUpdate, checkForUpdates]);

  const performUpdate = async () => {
    if (!newVersionInfo) return;

    try {
      const result = await api.performUpdate();

      if (result.success) {
        showToast(t("app.update_successful"), "success");
        setIsNewVersionAvailable(false);
        setIsUpdateDialogOpen(false);
        setHasCheckedUpdate(false);
      } else {
        showToast(t("app.update_failed") + ": " + result.message, "error");
      }
    } catch (err) {
      console.error("Failed to perform update:", err);
      showToast(t("app.update_failed") + ": " + (err as Error).message, "error");
    }
  };

  if (isCheckingAuth) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-muted-foreground">{t("nav.loading_app")}</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-destructive">
          {t("nav.error_prefix")}: {error.message}
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-muted-foreground">{t("nav.loading_config")}</div>
      </div>
    );
  }

  return (
    <>
      <PageHeader
        title={t("nav.dashboard")}
        description={t("nav.dashboard_description")}
        actions={
          <>
            {isUpdateFeatureAvailable && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => checkForUpdates(true)}
                    disabled={isCheckingUpdate}
                    className="relative h-8 w-8 rounded-sm"
                  >
                    <div className="relative">
                      <CircleArrowUp className="h-4 w-4" />
                      {isNewVersionAvailable && !isCheckingUpdate && (
                        <div className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full border-2 border-card bg-destructive" />
                      )}
                    </div>
                    {isCheckingUpdate && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
                      </div>
                    )}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{t("app.check_updates")}</p>
                </TooltipContent>
              </Tooltip>
            )}
            <Button
              onClick={saveConfig}
              variant="outline"
              size="sm"
              className="rounded-sm active:scale-[0.99]"
            >
              <Save className="mr-1.5 h-3.5 w-3.5" />
              {t("app.save")}
            </Button>
            <Button
              onClick={saveConfigAndRestart}
              size="sm"
              className="rounded-sm active:scale-[0.99]"
            >
              <RefreshCw className="mr-1.5 h-3.5 w-3.5" />
              {t("app.save_and_restart")}
            </Button>
          </>
        }
      />

      <div className="flex h-[calc(100%-4.5rem)] min-h-[28rem] gap-3 overflow-hidden">
        <div className="w-3/5 min-w-0">
          <Providers />
        </div>
        <div className="flex w-2/5 min-w-0 flex-col gap-3">
          <div className="h-3/5 min-h-0">
            <Router />
          </div>
          <div className="min-h-0 flex-1 overflow-hidden">
            <Transformers />
          </div>
        </div>
      </div>

      <Dialog open={isUpdateDialogOpen} onOpenChange={setIsUpdateDialogOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>
              {t("app.new_version_available")}
              {newVersionInfo && (
                <span className="ml-2 text-sm font-normal text-muted-foreground">
                  v{newVersionInfo.version}
                </span>
              )}
            </DialogTitle>
            <DialogDescription>{t("app.update_description")}</DialogDescription>
          </DialogHeader>
          <div className="max-h-96 overflow-y-auto py-4">
            {newVersionInfo?.changelog ? (
              <div className="whitespace-pre-wrap text-sm">{newVersionInfo.changelog}</div>
            ) : (
              <div className="text-muted-foreground">{t("app.no_changelog_available")}</div>
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsUpdateDialogOpen(false)}>
              {t("app.later")}
            </Button>
            <Button onClick={performUpdate}>{t("app.update_now")}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default App;

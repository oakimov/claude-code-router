import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLocation } from "react-router";
import { useTranslation } from "react-i18next";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage, type ToolUIPart } from "ai";
import { Copy, Maximize, Plus, RefreshCw, Send, Square } from "lucide-react";
import MonacoEditor from "@monaco-editor/react";
import { PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ChevronDown } from "lucide-react";
import { useMonacoTheme } from "@/hooks/useMonacoTheme";
import { useConfig } from "@/components/ConfigProvider";
import { api } from "@/lib/api";
import { HeaderTable } from "@/components/debug/HeaderTable";
import { VerticalSplit } from "@/components/debug/VerticalSplit";
import { TokenMarker, type DebugTokenUsage } from "@/components/debug/TokenMarker";
import { JsonPane } from "@/components/debug/JsonPane";
import { ToolList } from "@/components/debug/ToolList";
import { ToolEditorDialog } from "@/components/debug/ToolEditorDialog";
import { FileImportButton } from "@/components/debug/FileImportButton";
import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { Response } from "@/components/ai-elements/response";
import { Loader } from "@/components/ai-elements/loader";
import { Action, Actions } from "@/components/ai-elements/actions";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "@/components/ai-elements/reasoning";
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from "@/components/ai-elements/tool";
import {
  PromptInput,
  PromptInputBody,
  PromptInputTextarea,
} from "@/components/ai-elements/prompt-input";
import {
  buildEndpointBody,
  copyCurlCommand,
  guessInboundProtocol,
  headersToRows,
  loadDebugChat,
  loadDebugSystem,
  newHeaderRow,
  oauthRenewKind,
  parseHeadersJson,
  prettyJsonOrText,
  requestUrlForTarget,
  rowsToHeaders,
  saveDebugChat,
  saveDebugSystem,
  uiMessagesToWire,
  REASONING_EFFORTS,
  type CapturedExchange,
  type DebugTarget,
  type HeaderRow,
  type InboundProtocol,
} from "@/lib/debugChat";
import {
  isBuiltinWeatherTool,
  loadDebugTools,
  mergeDebugTools,
  nextSampleTool,
  parseToolsFile,
  saveDebugTools,
  toRequestTools,
  toolName,
  type DebugTool,
} from "@/lib/debugTools";
import { getProviderTitle } from "@/lib/providerMeta";
import type { Provider } from "@/types";

type DebugUIMessage = UIMessage<unknown, { "llm-exchange": CapturedExchange; usage: DebugTokenUsage }>;

function isToolPart(part: DebugUIMessage["parts"][number]): part is ToolUIPart {
  return typeof part.type === "string" && part.type.startsWith("tool-");
}

function statusTone(status: number): string {
  if (status >= 200 && status < 300) return "text-emerald-600 dark:text-emerald-400";
  if (status >= 400) return "text-red-600 dark:text-red-400";
  return "text-muted-foreground";
}

function isErrorStatus(status: number, body?: string): boolean {
  return status >= 400 || (status === 0 && Boolean(body));
}

export function DebugPage() {
  const { t } = useTranslation();
  const monacoTheme = useMonacoTheme();
  const location = useLocation();
  const { config } = useConfig();
  const providers = useMemo(() => config?.Providers ?? [], [config?.Providers]);

  const [target, setTarget] = useState<DebugTarget>("ccr");
  const [protocol, setProtocol] = useState<InboundProtocol>("chat_completions");
  const [providerName, setProviderName] = useState("");
  const [modelName, setModelName] = useState("");
  const [stream, setStream] = useState(true);
  const restoredChat = useMemo(() => loadDebugChat(), []);
  const [system, setSystem] = useState(() => loadDebugSystem());
  const [tools, setTools] = useState<DebugTool[]>(() => loadDebugTools());
  const [editingToolIndex, setEditingToolIndex] = useState<number | null>(null);
  const [editingTool, setEditingTool] = useState<DebugTool | null>(null);
  const [isNewTool, setIsNewTool] = useState(false);
  const [deletingToolIndex, setDeletingToolIndex] = useState<number | null>(null);
  const [reasoningEffort, setReasoningEffort] = useState("");
  const [headerRows, setHeaderRows] = useState<HeaderRow[]>([newHeaderRow()]);
  const [bodyJson, setBodyJson] = useState("{}");
  const [bodyDirty, setBodyDirty] = useState(false);
  const [requestTab, setRequestTab] = useState("chat");
  const [responseTab, setResponseTab] = useState("body");
  const [instructionsOpen, setInstructionsOpen] = useState(true);
  const [usageByMessage, setUsageByMessage] = useState<Record<string, DebugTokenUsage>>(
    () => (restoredChat.usageByMessage as Record<string, DebugTokenUsage>) ?? {}
  );
  const [input, setInput] = useState("");
  const [fullscreenEditor, setFullscreenEditor] = useState<"headers" | "body" | null>(null);
  const [rawLoading, setRawLoading] = useState(false);
  const [renewing, setRenewing] = useState(false);
  const [notice, setNotice] = useState<{ text: string; tone: "success" | "error" } | null>(null);
  const [responseData, setResponseData] = useState({
    status: 0,
    responseTime: 0,
    body: "",
    headers: "{}",
  });
  const rawAbortRef = useRef<AbortController | null>(null);
  const headersEditorRef = useRef<any>(null);
  const bodyEditorRef = useRef<any>(null);
  const responseBodyEditorRef = useRef<any>(null);
  const responseHeadersEditorRef = useRef<any>(null);
  const pendingUsageRef = useRef<DebugTokenUsage | null>(null);
  const messagesRef = useRef<DebugUIMessage[]>([]);
  const protocolTouched = useRef(false);

  const provider = useMemo(
    () => providers.find((p) => p.name === providerName) as Provider | undefined,
    [providers, providerName]
  );

  useEffect(() => {
    if (!providerName && providers[0]?.name) {
      setProviderName(providers[0].name);
    }
  }, [providers, providerName]);

  useEffect(() => {
    if (!provider) return;
    // Entering direct mode or changing its provider should start from that
    // provider's native protocol; the user can still override it afterward.
    if (target === "direct" || !protocolTouched.current) {
      setProtocol(guessInboundProtocol(provider));
    }
    if (!provider.models?.includes(modelName)) {
      setModelName(provider.models?.[0] || "");
    }
    setHeaderRows((rows) => {
      const next = rows.filter((row) => {
        const key = row.key.trim().toLowerCase();
        return key !== "authorization" && key !== "x-api-key";
      });
      return next.length > 0 ? next : [newHeaderRow()];
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providerName, target]);

  const requestUrl = requestUrlForTarget(target, protocol, provider);
  const oauthKind = oauthRenewKind(provider);
  const toolsJson = useMemo(() => JSON.stringify(toRequestTools(tools)), [tools]);

  const updateTools = useCallback((next: DebugTool[]) => {
    setTools(next);
    saveDebugTools(next);
  }, []);

  const settingsRef = useRef({
    target,
    protocol,
    provider: providerName,
    model: modelName,
    system,
    toolsJson,
    stream,
    reasoningEffort,
    headerRows,
  });
  settingsRef.current = {
    target,
    protocol,
    provider: providerName,
    model: modelName,
    system,
    toolsJson,
    stream,
    reasoningEffort,
    headerRows,
  };

  const attachUsage = useCallback((usage: DebugTokenUsage) => {
    const lastAssistant = [...messagesRef.current]
      .reverse()
      .find((message) => message.role === "assistant");
    if (!lastAssistant) {
      pendingUsageRef.current = usage;
      return;
    }
    pendingUsageRef.current = null;
    setUsageByMessage((current) => ({ ...current, [lastAssistant.id]: usage }));
  }, []);

  const applyExchange = useCallback((exchange: CapturedExchange) => {
    setResponseData({
      status: exchange.status,
      responseTime: 0,
      body: prettyJsonOrText(exchange.responseBody || ""),
      headers: prettyJsonOrText(exchange.responseHeaders ?? {}),
    });
    if (exchange.usage) attachUsage(exchange.usage);
  }, [attachUsage]);
  const applyExchangeRef = useRef(applyExchange);
  applyExchangeRef.current = applyExchange;

  const transport = useMemo(
    () =>
      new DefaultChatTransport<DebugUIMessage>({
        api: "/api/debug/chat",
        credentials: "include",
        fetch: async (input, init) => {
          const res = await fetch(input, init);
          if (res.ok) return res;
          const text = await res.text();
          const payload: Record<string, unknown> = (() => {
            try {
              return JSON.parse(text);
            } catch {
              return { error: text };
            }
          })();
          const status =
            typeof payload.status === "number" && payload.status > 0
              ? payload.status
              : res.status;
          const headers =
            payload.headers &&
            typeof payload.headers === "object" &&
            !Array.isArray(payload.headers)
              ? (payload.headers as Record<string, string>)
              : (() => {
                  const out: Record<string, string> = {};
                  res.headers.forEach((value, key) => {
                    out[key] = value;
                  });
                  return out;
                })();
          const body =
            typeof payload.body === "string" && payload.body
              ? prettyJsonOrText(payload.body)
              : prettyJsonOrText(
                  payload.error != null ? { error: payload.error } : payload
                );
          applyExchangeRef.current({
            url: typeof input === "string" ? input : "",
            method: "POST",
            requestHeaders: {},
            requestBody: undefined,
            status,
            responseHeaders: headers,
            responseBody: body,
            streaming: false,
          });
          return new globalThis.Response(text, {
            status: res.status,
            statusText: res.statusText,
            headers: res.headers,
          });
        },
        prepareSendMessagesRequest: ({ id, messages, body }) => {
          const s = settingsRef.current;
          let tools: unknown;
          try {
            tools = JSON.parse(s.toolsJson || "[]");
          } catch {
            tools = [];
          }
          return {
            body: {
              ...body,
              id,
              messages,
              target: s.target,
              protocol: s.protocol,
              provider: s.provider,
              model: s.model,
              system: s.system,
              tools,
              stream: true,
              reasoningEffort: s.reasoningEffort || undefined,
              headers: rowsToHeaders(s.headerRows),
            },
          };
        },
      }),
    []
  );

  const { messages, sendMessage, status, stop } = useChat<DebugUIMessage>({
    id: "ccr-debug",
    transport,
    messages: restoredChat.messages as DebugUIMessage[],
    onData: (part) => {
      if (part.type === "data-llm-exchange") {
        applyExchange(part.data as CapturedExchange);
      }
      if (part.type === "data-usage") {
        attachUsage(part.data as DebugTokenUsage);
      }
    },
    onError: (err) => {
      setResponseData((current) => {
        if (current.body) return current;
        return {
          status: current.status || 0,
          responseTime: current.responseTime,
          body: prettyJsonOrText({ error: err.message }),
          headers: current.headers || "{}",
        };
      });
    },
  });

  messagesRef.current = messages;

  useEffect(() => {
    saveDebugSystem(system);
  }, [system]);

  useEffect(() => {
    if (status === "submitted" || status === "streaming") return;
    saveDebugChat({ messages, usageByMessage });
  }, [messages, usageByMessage, status]);

  useEffect(() => {
    const usage = pendingUsageRef.current;
    if (!usage) return;
    attachUsage(usage);
  }, [messages, attachUsage]);

  const wireModel =
    target === "ccr" && providerName && modelName
      ? `${providerName},${modelName}`
      : modelName;

  const prerenderedBody = useMemo(
    () =>
      JSON.stringify(
        buildEndpointBody({
          protocol,
          model: wireModel,
          system,
          messages: uiMessagesToWire(messages, input),
          toolsJson,
          stream,
          reasoningEffort,
        }),
        null,
        2
      ),
    [protocol, wireModel, system, messages, input, toolsJson, stream, reasoningEffort]
  );

  useEffect(() => {
    if (bodyDirty) return;
    setBodyJson(prerenderedBody);
  }, [prerenderedBody, bodyDirty]);

  const layoutEditors = useCallback(() => {
    headersEditorRef.current?.layout();
    bodyEditorRef.current?.layout();
    responseBodyEditorRef.current?.layout();
    responseHeadersEditorRef.current?.layout();
  }, []);

  const updateBodyJson = (value: string) => {
    const next = value || "{}";
    setBodyJson(next);
    setBodyDirty(next !== prerenderedBody);
  };

  const busy = status === "submitted" || status === "streaming" || rawLoading;

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const logDataParam = params.get("logData");
    if (!logDataParam) return;
    try {
      const parsedData = JSON.parse(decodeURIComponent(logDataParam));
      const url = parsedData.url || parsedData.requestUrl || parsedData.endpoint || "";
      let headers: Record<string, string> = {};
      if (parsedData.headers) {
        headers =
          typeof parsedData.headers === "string"
            ? parseHeadersJson(parsedData.headers)
            : parsedData.headers;
      }
      let body: unknown = parsedData.body ?? parsedData.request?.body ?? {};
      if (typeof body === "string") {
        try {
          body = JSON.parse(body);
        } catch {
          body = { raw: body };
        }
      }
      setHeaderRows(headersToRows(headers));
      setBodyJson(JSON.stringify(body ?? {}, null, 2));
      setBodyDirty(true);
      setRequestTab("body");
      if (typeof url === "string" && url.includes("/messages")) {
        protocolTouched.current = true;
        setProtocol("messages");
      }
      if (typeof url === "string" && url.includes("/responses")) {
        protocolTouched.current = true;
        setProtocol("responses");
      }
    } catch (err) {
      console.error("Failed to parse log data:", err);
    }
  }, [location.search]);

  const toggleFullscreen = (editorType: "headers" | "body") => {
    const entering = fullscreenEditor !== editorType;
    setFullscreenEditor(entering ? editorType : null);
    setTimeout(() => {
      headersEditorRef.current?.layout();
      bodyEditorRef.current?.layout();
    }, 300);
  };

  const sendRawRequest = async () => {
    if (!providerName) return;
    try {
      setRawLoading(true);
      setResponseData({ status: 0, responseTime: 0, body: "", headers: "{}" });
      const headers = rowsToHeaders(headerRows);
      let body: unknown;
      try {
        body = JSON.parse(bodyJson || "{}");
      } catch (err) {
        setResponseData({
          status: 0,
          responseTime: 0,
          body: prettyJsonOrText({
            error: `Request body is not valid JSON: ${
              err instanceof Error ? err.message : "parse error"
            }`,
          }),
          headers: "{}",
        });
        return;
      }
      const start = Date.now();
      rawAbortRef.current = new AbortController();
      const response = await fetch("/api/debug/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          target,
          protocol,
          provider: providerName,
          model: modelName,
          headers,
          body,
        }),
        signal: rawAbortRef.current.signal,
        credentials: "include",
      });
      const responseTime = Date.now() - start;
      const payload = (await response.json().catch(() => ({}))) as {
        error?: string;
        status?: number;
        headers?: Record<string, string>;
        body?: string;
      };
      const responseHeaders =
        payload.headers && typeof payload.headers === "object"
          ? payload.headers
          : {};
      const upstreamStatus =
        typeof payload.status === "number"
          ? payload.status
          : response.ok
            ? 0
            : response.status;
      const responseBody =
        typeof payload.body === "string"
          ? payload.body
          : payload.error != null
            ? prettyJsonOrText({ error: payload.error })
            : upstreamStatus >= 400 || !response.ok
              ? prettyJsonOrText({ error: `Request failed: ${upstreamStatus || response.status}` })
              : "";
      setResponseData({
        status: upstreamStatus,
        responseTime,
        body: prettyJsonOrText(responseBody),
        headers: prettyJsonOrText(responseHeaders),
      });
    } catch (err) {
      setResponseData({
        status: 0,
        responseTime: 0,
        body: prettyJsonOrText({
          error: err instanceof Error ? err.message : "Unknown error",
        }),
        headers: "{}",
      });
    } finally {
      setRawLoading(false);
      rawAbortRef.current = null;
    }
  };

  const sendChat = (text?: string) => {
    if (status === "submitted" || status === "streaming") return;
    const payload = (text ?? input).trim();
    if (!payload || !providerName || !modelName) return;
    setResponseData({ status: 0, responseTime: 0, body: "", headers: "{}" });
    sendMessage({ text: payload });
    setInput("");
  };

  const handleLoadSystemFile = (text: string) => {
    setSystem(text);
    setInstructionsOpen(true);
    setNotice({ text: t("debug.system_loaded"), tone: "success" });
  };

  const handleLoadToolsFile = (text: string) => {
    try {
      const incoming = parseToolsFile(text);
      if (!incoming.length) {
        setNotice({ text: t("debug.tools_load_empty"), tone: "error" });
        return;
      }
      updateTools(mergeDebugTools(tools, incoming));
      setInstructionsOpen(true);
      setNotice({ text: t("debug.tools_loaded"), tone: "success" });
    } catch {
      setNotice({ text: t("debug.tools_load_failed"), tone: "error" });
    }
  };

  const handleLoadUserFile = (text: string) => {
    const payload = text.trim();
    if (!payload) return;
    if (busy || !providerName || !modelName) {
      setInput(payload);
      setNotice({ text: t("debug.user_loaded_pending"), tone: "success" });
      return;
    }
    sendChat(payload);
  };

  const handleFooterSend = () => {
    if (requestTab === "chat") sendChat();
    else void sendRawRequest();
  };

  const handleStop = () => {
    if (status === "submitted" || status === "streaming") stop();
    rawAbortRef.current?.abort();
  };

  const handleAddTool = () => {
    setEditingTool(nextSampleTool(tools));
    setEditingToolIndex(tools.length);
    setIsNewTool(true);
  };

  const handleEditTool = (index: number) => {
    const tool = tools[index];
    if (!tool) return;
    setEditingTool(structuredClone(tool));
    setEditingToolIndex(index);
    setIsNewTool(false);
  };

  const handleSaveTool = (tool: DebugTool) => {
    const next = [...tools];
    const original =
      editingToolIndex != null && editingToolIndex < next.length
        ? next[editingToolIndex]
        : undefined;
    if (original && isBuiltinWeatherTool(original)) {
      tool.function.name = toolName(original);
    }
    const enabled =
      isNewTool || editingToolIndex == null || editingToolIndex >= next.length
        ? true
        : next[editingToolIndex]?.enabled !== false;
    const saved = { ...tool, enabled };
    if (isNewTool || editingToolIndex == null || editingToolIndex >= next.length) {
      next.push(saved);
    } else {
      next[editingToolIndex] = saved;
    }
    updateTools(next);
    setEditingTool(null);
    setEditingToolIndex(null);
    setIsNewTool(false);
  };

  const handleToggleTool = (index: number, enabled: boolean) => {
    const tool = tools[index];
    if (!tool) return;
    updateTools(tools.map((item, current) => (current === index ? { ...item, enabled } : item)));
  };

  const handleRemoveTool = (index: number) => {
    const tool = tools[index];
    if (!tool || isBuiltinWeatherTool(tool)) return;
    updateTools(tools.filter((_, current) => current !== index));
    setDeletingToolIndex(null);
  };

  const copyCurl = async () => {
    try {
      let body: unknown;
      try {
        body = JSON.parse(bodyJson || prerenderedBody || "{}");
      } catch (error) {
        setNotice({
          text: `${t("debug.body_invalid_json")}: ${
            error instanceof Error ? error.message : "parse error"
          }`,
          tone: "error",
        });
        return;
      }
      const curl = copyCurlCommand({
        url: requestUrl,
        method: "POST",
        headers: rowsToHeaders(headerRows),
        body,
      });
      await navigator.clipboard.writeText(curl);
      setNotice({ text: t("debug.copied"), tone: "success" });
    } catch {
      setNotice({ text: t("debug.copy_failed"), tone: "error" });
    }
  };

  const renewOAuth = async () => {
    if (!providerName) return;
    setRenewing(true);
    try {
      await api.refreshOAuth(providerName);
      setNotice({ text: t("debug.renew_ok"), tone: "success" });
    } catch (err) {
      setNotice({
        text: err instanceof Error ? err.message : t("debug.renew_failed"),
        tone: "error",
      });
    } finally {
      setRenewing(false);
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden text-[11px] leading-snug font-sans">
      <PageHeader
        className="mb-2 pb-2 [&_h1]:text-sm [&_p]:text-[11px]"
        title={t("debug.title")}
        description={t("debug.description")}
        actions={
          <Button variant="outline" size="sm" className="h-7 text-[11px]" onClick={() => void copyCurl()}>
            <Copy className="mr-1.5 h-3 w-3" />
            {t("debug.copy_curl")}
          </Button>
        }
      />

      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden pb-1">
        <div className="flex flex-wrap items-end gap-2 rounded-lg border bg-card p-2">
          <div className="space-y-0.5">
            <Label htmlFor="debug-target" className="text-[11px]">{t("debug.target")}</Label>
            <Select value={target} onValueChange={(v) => setTarget(v as DebugTarget)}>
              <SelectTrigger id="debug-target" className="h-7 w-24 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="ccr">{t("debug.target_ccr")}</SelectItem>
                <SelectItem value="direct">{t("debug.target_direct")}</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {target === "ccr" && (
            <div className="space-y-0.5">
              <Label htmlFor="debug-protocol" className="text-[11px]">{t("debug.protocol")}</Label>
              <Select
                value={protocol}
                onValueChange={(v) => {
                  protocolTouched.current = true;
                  setProtocol(v as InboundProtocol);
                }}
              >
                <SelectTrigger id="debug-protocol" className="h-7 w-44 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="chat_completions">{t("debug.protocol_chat")}</SelectItem>
                  <SelectItem value="messages">{t("debug.protocol_messages")}</SelectItem>
                  <SelectItem value="responses">{t("debug.protocol_responses")}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
          <div className="space-y-0.5">
            <Label htmlFor="debug-provider" className="text-[11px]">{t("debug.provider")}</Label>
            <Select value={providerName} onValueChange={setProviderName}>
              <SelectTrigger id="debug-provider" className="h-7 w-44 text-xs">
                <SelectValue placeholder={t("debug.select_provider")} />
              </SelectTrigger>
              <SelectContent>
                {providers.map((p) => (
                  <SelectItem key={p.name} value={p.name}>
                    {getProviderTitle(p)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-0.5">
            <Label htmlFor="debug-model" className="text-[11px]">{t("debug.model")}</Label>
            <Select value={modelName} onValueChange={setModelName}>
              <SelectTrigger id="debug-model" className="h-7 w-52 text-xs">
                <SelectValue placeholder={t("debug.select_model")} />
              </SelectTrigger>
              <SelectContent>
                {(provider?.models || []).map((m) => (
                  <SelectItem key={m} value={m}>
                    {m}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {requestTab !== "chat" ? (
            <div className="flex h-7 items-center gap-2">
              <Switch checked={stream} onCheckedChange={setStream} id="debug-stream" />
              <Label htmlFor="debug-stream" className="text-[11px]">{t("debug.stream")}</Label>
            </div>
          ) : null}
          {oauthKind && (
            <Button variant="outline" size="sm" className="h-7" onClick={() => void renewOAuth()} disabled={renewing}>
              <RefreshCw className={`mr-1.5 h-3.5 w-3.5 ${renewing ? "animate-spin" : ""}`} />
              {renewing ? t("debug.renewing") : t("debug.renew_oauth")}
            </Button>
          )}
        </div>

        <p className="truncate px-1 text-[11px] text-muted-foreground">
          {t("debug.url_hint")}: {requestUrl || "—"}
        </p>
        {notice && (
          <p
            className={`px-1 text-[11px] ${
              notice.tone === "error"
                ? "text-destructive"
                : "text-emerald-600 dark:text-emerald-400"
            }`}
          >
            {notice.text}
          </p>
        )}
        {providers.length === 0 && (
          <p className="px-1 text-[11px] text-muted-foreground">{t("debug.no_providers")}</p>
        )}

        <div className="flex min-h-0 flex-1 flex-col">
        <VerticalSplit defaultTopPercent={68} minPercent={18} onResize={layoutEditors}>
          <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-lg border bg-card">
            <Tabs value={requestTab} onValueChange={(value) => {
              setRequestTab(value);
              setTimeout(layoutEditors, 50);
            }} className="flex min-h-0 flex-1 flex-col">
              <div className="flex items-center justify-between border-b px-3 py-2">
                <TabsList className="h-7">
                  <TabsTrigger className="h-6 px-2 text-[11px]" value="chat">{t("debug.tab_chat")}</TabsTrigger>
                  <TabsTrigger className="h-6 px-2 text-[11px]" value="headers">{t("debug.tab_headers")}</TabsTrigger>
                  <TabsTrigger className="h-6 px-2 text-[11px]" value="body">{t("debug.tab_body")}</TabsTrigger>
                </TabsList>
                {requestTab !== "chat" && (
                  <div className="flex items-center gap-1">
                    {requestTab === "body" && bodyDirty && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7"
                        onClick={() => {
                          setBodyJson(prerenderedBody);
                          setBodyDirty(false);
                        }}
                      >
                        {t("debug.reset_body")}
                      </Button>
                    )}
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-8"
                      aria-label={t("debug.fullscreen")}
                      onClick={() => toggleFullscreen(requestTab === "headers" ? "headers" : "body")}
                    >
                      <Maximize className="size-3.5" />
                    </Button>
                  </div>
                )}
              </div>
              <TabsContent value="chat" className="mt-0 flex min-h-0 flex-1 flex-col overflow-hidden">
                <Conversation className="min-h-0 flex-1 text-[11px]">
                  <ConversationContent className="gap-3 p-2 min-h-0">
                    {messages.length === 0 && (
                      <ConversationEmptyState
                        title={t("debug.empty_title")}
                        description={t("debug.empty_description")}
                      />
                    )}
                    {messages.map((message, messageIndex) => {
                      const isLastAssistant =
                        message.role === "assistant" && messageIndex === messages.length - 1;
                      const reasoningText = message.parts
                        .filter((part) => part.type === "reasoning")
                        .map((part) => (part as { text?: string }).text || "")
                        .join("\n");
                      const isReasoningStreaming =
                        isLastAssistant &&
                        status === "streaming" &&
                        message.parts.at(-1)?.type === "reasoning";
                      const showReasoning = Boolean(reasoningText) || isReasoningStreaming;
                      return (
                        <Message from={message.role} key={message.id} className="text-[11px]">
                          {showReasoning ? (
                            <Reasoning isStreaming={isReasoningStreaming} defaultOpen={false}>
                              <ReasoningTrigger>{t("debug.thinking")}</ReasoningTrigger>
                              <ReasoningContent>{reasoningText}</ReasoningContent>
                            </Reasoning>
                          ) : null}
                          {message.parts.map((part, i) => {
                            if (part.type === "text") {
                              return (
                                <MessageContent key={`${message.id}-${i}`} className="text-[11px] px-3 py-2">
                                  <Response>{part.text}</Response>
                                  {isLastAssistant && status !== "streaming" && status !== "submitted" && (
                                    <Actions>
                                      <Action
                                        label={t("debug.copy")}
                                        tooltip={t("debug.copy")}
                                        onClick={() => void navigator.clipboard.writeText(part.text)}
                                      >
                                        <Copy className="size-3.5" />
                                      </Action>
                                    </Actions>
                                  )}
                                </MessageContent>
                              );
                            }
                            if (part.type === "reasoning") return null;
                            if (isToolPart(part)) {
                              return (
                                <Tool key={`${message.id}-${i}`} defaultOpen>
                                  <ToolHeader type={part.type} state={part.state} title={part.type.replace(/^tool-/, "")} />
                                  <ToolContent>
                                    {"input" in part ? <ToolInput input={part.input} /> : null}
                                    {"output" in part || "errorText" in part ? (
                                      <ToolOutput
                                        output={"output" in part ? part.output : undefined}
                                        errorText={"errorText" in part ? part.errorText : undefined}
                                      />
                                    ) : null}
                                  </ToolContent>
                                </Tool>
                              );
                            }
                            return null;
                          })}
                          {message.role === "assistant" && usageByMessage[message.id] ? (
                            <TokenMarker usage={usageByMessage[message.id]} />
                          ) : null}
                        </Message>
                      );
                    })}
                    {status === "submitted" && <Loader />}
                  </ConversationContent>
                  <ConversationScrollButton />
                </Conversation>
                <div className="border-t p-2">
                  <Collapsible open={instructionsOpen} onOpenChange={setInstructionsOpen}>
                    <CollapsibleTrigger className="mb-2 flex items-center gap-1 text-[11px] text-muted-foreground">
                      <ChevronDown className={`size-3.5 transition-transform ${instructionsOpen ? "" : "-rotate-90"}`} />
                      {t("debug.instructions")}
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mb-3 space-y-2">
                      <div className="space-y-1">
                        <Label htmlFor="debug-reasoning-effort" className="text-[11px]">{t("debug.reasoning_effort")}</Label>
                        <Select
                          value={reasoningEffort || "default"}
                          onValueChange={(value) =>
                            setReasoningEffort(value === "default" ? "" : value)
                          }
                        >
                          <SelectTrigger id="debug-reasoning-effort" className="h-7 w-52 text-xs">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="default">{t("debug.reasoning_default")}</SelectItem>
                            {REASONING_EFFORTS.map((level) => (
                              <SelectItem key={level} value={level}>
                                {level}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="grid gap-2 md:grid-cols-2">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between">
                            <Label className="text-[11px]">{t("debug.system_prompt")}</Label>
                            <FileImportButton
                              accept=".txt,.md,text/plain"
                              label={t("debug.load_system")}
                              onLoad={handleLoadSystemFile}
                            />
                          </div>
                          <Textarea
                            className="min-h-[88px] text-[11px]"
                            placeholder={t("debug.system_placeholder")}
                            value={system}
                            onChange={(e) => setSystem(e.target.value)}
                          />
                        </div>
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between">
                            <Label className="text-[11px]">{t("debug.tools")}</Label>
                            <div className="flex items-center gap-1">
                              <FileImportButton
                                accept=".json,application/json"
                                label={t("debug.load_tools")}
                                onLoad={handleLoadToolsFile}
                              />
                              <Button
                                type="button"
                                variant="outline"
                                size="icon"
                                className="size-6"
                                aria-label={t("debug.tools_add")}
                                onClick={handleAddTool}
                              >
                                <Plus className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          </div>
                          <div className="max-h-[132px] overflow-y-auto">
                            <ToolList
                              tools={tools}
                              onEdit={handleEditTool}
                              onRemove={(index) => {
                                if (isBuiltinWeatherTool(tools[index])) return;
                                setDeletingToolIndex(index);
                              }}
                              onToggle={handleToggleTool}
                            />
                          </div>
                        </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                  <PromptInput
                    onSubmit={() => {
                      sendChat();
                    }}
                  >
                    <PromptInputBody>
                      <div className="flex items-start">
                        <PromptInputTextarea
                          className="min-w-0 flex-1 px-2 py-2 text-[11px]"
                          placeholder={t("debug.composer_placeholder")}
                          value={input}
                          onChange={(e) => setInput(e.target.value)}
                        />
                        <div className="shrink-0 p-1">
                          <FileImportButton
                            accept=".txt,.md,text/plain"
                            label={t("debug.load_user")}
                            disabled={busy}
                            onLoad={handleLoadUserFile}
                          />
                        </div>
                      </div>
                    </PromptInputBody>
                  </PromptInput>
                </div>
              </TabsContent>
              <TabsContent value="headers" className="mt-0 min-h-0 flex-1 overflow-auto p-3">
                <HeaderTable rows={headerRows} onChange={setHeaderRows} />
              </TabsContent>
              <TabsContent value="body" className="mt-0 flex min-h-0 flex-1 flex-col">
                <p className="shrink-0 border-b px-3 py-1.5 text-[11px] text-muted-foreground">
                  {t("debug.body_hint")}
                </p>
                <div className="min-h-0 flex-1">
                  <MonacoEditor
                    height="100%"
                    language="json"
                    theme={monacoTheme}
                    value={bodyJson}
                    onChange={(value) => updateBodyJson(value || "{}")}
                    onMount={(editor) => {
                      bodyEditorRef.current = editor;
                    }}
                    options={{ minimap: { enabled: false }, fontSize: 11, wordWrap: "on", automaticLayout: true }}
                  />
                </div>
              </TabsContent>
              <div className="flex shrink-0 items-center justify-end gap-2 border-t px-3 py-2">
                {busy ? (
                  <Button size="sm" className="h-7" variant="outline" onClick={handleStop}>
                    <Square className="mr-1.5 h-3.5 w-3.5" />
                    {t("debug.stop")}
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    className="h-7"
                    onClick={handleFooterSend}
                    disabled={
                      !providerName ||
                      (requestTab === "chat" && (!input.trim() || !modelName))
                    }
                  >
                    <Send className="mr-1.5 h-3.5 w-3.5" />
                    {t("debug.send_request")}
                  </Button>
                )}
              </div>
            </Tabs>
          </div>

          <div className="flex min-h-[12rem] flex-1 flex-col overflow-hidden rounded-lg border bg-card">
            <Tabs value={responseTab} onValueChange={(value) => {
              setResponseTab(value);
              setTimeout(layoutEditors, 50);
            }} className="flex min-h-0 flex-1 flex-col">
              <div className="flex items-center gap-2 border-b px-2 py-1">
                <TabsList className="h-7">
                  <TabsTrigger className="h-6 px-2 text-[11px]" value="body">
                    {t("debug.response_body")}
                  </TabsTrigger>
                  <TabsTrigger className="h-6 px-2 text-[11px]" value="headers">
                    {t("debug.response_headers")}
                  </TabsTrigger>
                </TabsList>
                <span className="text-[11px] font-medium text-muted-foreground">{t("debug.response")}</span>
                {responseData.status > 0 ? (
                  <span className={`font-mono text-[11px] font-semibold ${statusTone(responseData.status)}`}>
                    {responseData.status}
                    {responseData.responseTime > 0 ? (
                      <span className="ml-1 font-normal text-muted-foreground">
                        · {responseData.responseTime}ms
                      </span>
                    ) : null}
                  </span>
                ) : responseData.body ? (
                  <span className="font-mono text-[11px] font-semibold text-red-600 dark:text-red-400">
                    ERR
                  </span>
                ) : null}
              </div>
              <TabsContent value="body" className="mt-0 min-h-0 flex-1 overflow-hidden">
                <JsonPane
                  error={isErrorStatus(responseData.status, responseData.body)}
                  value={responseData.body}
                  onMount={(editor) => {
                    responseBodyEditorRef.current = editor;
                  }}
                />
              </TabsContent>
              <TabsContent value="headers" className="mt-0 min-h-0 flex-1 overflow-hidden">
                <JsonPane
                  error={isErrorStatus(responseData.status, responseData.body)}
                  value={responseData.headers}
                  onMount={(editor) => {
                    responseHeadersEditorRef.current = editor;
                  }}
                />
              </TabsContent>
            </Tabs>
          </div>
        </VerticalSplit>
        </div>
      </div>

      {fullscreenEditor && (
        <div className="fixed inset-0 z-50 bg-background p-4">
          <div className="mb-2 flex justify-end">
            <Button variant="outline" size="sm" onClick={() => setFullscreenEditor(null)}>
              {t("app.cancel")}
            </Button>
          </div>
          {fullscreenEditor === "headers" ? (
            <HeaderTable rows={headerRows} onChange={setHeaderRows} />
          ) : (
            <MonacoEditor
              height="calc(100vh - 80px)"
              language="json"
              theme={monacoTheme}
              value={bodyJson}
              onChange={(value) => updateBodyJson(value || "{}")}
              options={{ minimap: { enabled: false }, fontSize: 13, wordWrap: "on" }}
            />
          )}
        </div>
      )}

      <ToolEditorDialog
        open={editingToolIndex !== null}
        tool={editingTool}
        isNew={isNewTool}
        existingNames={tools
          .map(toolName)
          .filter((name, index) => name && (isNewTool || index !== editingToolIndex))}
        onOpenChange={(open) => {
          if (!open) {
            setEditingToolIndex(null);
            setEditingTool(null);
            setIsNewTool(false);
          }
        }}
        onSave={handleSaveTool}
      />

      <Dialog
        open={deletingToolIndex !== null}
        onOpenChange={(open) => {
          if (!open) setDeletingToolIndex(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("debug.tools_delete")}</DialogTitle>
            <DialogDescription>{t("debug.tools_delete_confirm")}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setDeletingToolIndex(null)}>
              {t("app.cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={() => deletingToolIndex !== null && handleRemoveTool(deletingToolIndex)}
            >
              {t("app.delete")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

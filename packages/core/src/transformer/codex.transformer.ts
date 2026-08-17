import { randomUUID } from "crypto";
import { execFileSync } from "child_process";
import { arch as osArch, platform as osPlatform, release as osRelease } from "os";
import { UnifiedChatRequest, MessageContent } from "@/types/llm";
import { Transformer } from "@/types/transformer";
import { resolveCodexPat } from "@caeliq/ccr-shared";
import { sanitizeResponsesCallId } from "@/utils/toolCallId";
import {
  applyRequestCacheKey,
  validateOpenAIToolCalls,
} from "../utils/openai.util";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "../utils/stream";
import {
  getValidAccessToken,
  toCodexOAuthAuth,
} from "../utils/codex-auth";
import {
  isReasoningDisabled,
  normalizeReasoningEffort,
} from "../utils/reasoning-effort";
import {
  CUSTOM_TOOL_INPUT_KEY,
  assistantTurnHasText,
  canonicalAssistantTurn,
  isResponsesReasoningItemId,
  recordReasoningSummaryDelta,
  responsesEncryptedContentFrom,
  responsesReasoningItemFromThinking,
  responsesTextFormatFromResponseFormat,
  thinkingForLateReasoningItem,
  thinkingFromResponsesReasoningItem,
  thinkingFromUnifiedAssistant,
  unwrapCustomToolInput,
} from "../utils/openai.responses.util";

const PAT_METADATA_TTL_MS = 5 * 60 * 1000;
const whoamiCache = new Map<
  string,
  { value: PatAuth; expiresAt: number }
>();
const whoamiRequests = new Map<string, Promise<PatAuth>>();
const CODEX_CLI_VERSION = "0.145.0";
const CODEX_ORIGINATOR = "codex_cli_rs";

interface WhoamiResponse {
  chatgpt_account_id?: string;
  chatgpt_account_is_fedramp?: boolean;
  chatgpt_user_id?: string;
  chatgpt_plan_type?: string;
}

interface PatAuth {
  mode: "pat";
  token: string;
  accountId: string;
  isFedramp: boolean;
}

type ResolvedCodexAuth =
  | PatAuth
  | ReturnType<typeof toCodexOAuthAuth>;

interface ResponsesAPIOutputItem {
  type: string;
  id?: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  /** Freeform text for a `custom_tool_call` item (parallel to `arguments`
   * for `function_call`). */
  input?: string;
  content?: Array<{
    type: string;
    text?: string;
    image_url?: string;
    mime_type?: string;
    image_base64?: string;
    annotations?: Array<{
      type?: string;
      url?: string;
      title?: string;
      start_index?: number;
      end_index?: number;
    }>;
  }>;
  reasoning?: string;
}

interface ResponsesAPIPayload {
  id: string;
  object: string;
  model: string;
  created_at: number;
  output: ResponsesAPIOutputItem[];
  usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    input_tokens_details?: {
      cached_tokens?: number;
      cache_write_tokens?: number;
    };
  };
}

interface ResponsesStreamEvent {
  type: string;
  item_id?: string;
  output_index?: number;
  text?: string;
  /** Full freeform text on a `response.custom_tool_call_input.done` event. */
  input?: string;
  delta?:
    | string
    | {
        url?: string;
        b64_json?: string;
        mime_type?: string;
      };
  item?: {
    id?: string;
    type?: string;
    call_id?: string;
    name?: string;
    /** Freeform text for a `custom_tool_call` item (parallel to `arguments`
     * for `function_call`). */
    input?: string;
    content?: Array<{
      type: string;
      text?: string;
      image_url?: string;
    }>;
    reasoning?: string;
  };
  response?: {
    id?: string;
    model?: string;
    output?: Array<{
      type: string;
    }>;
    usage?: {
      input_tokens: number;
      output_tokens: number;
      total_tokens: number;
      input_tokens_details?: {
        cached_tokens?: number;
        cache_write_tokens?: number;
      };
    };
  };
  reasoning_summary?: string;
  annotation?: {
    url?: string;
    title?: string;
    start_index?: number;
    end_index?: number;
  };
  part?: any;
}

interface CodexTurnMetadata {
  session_id: string;
  thread_id: string;
  turn_id: string;
  window_id: string;
  request_kind: "turn";
  turn_started_at_unix_ms: number;
}

const CODEX_TURN_METADATA_HEADER = "x-codex-turn-metadata";
const CODEX_WINDOW_ID_HEADER = "x-codex-window-id";
const CODEX_PARENT_THREAD_ID_HEADER = "x-codex-parent-thread-id";
const CODEX_SESSION_ID_KEY = "session_id";
const CODEX_THREAD_ID_KEY = "thread_id";
const CODEX_TURN_ID_KEY = "turn_id";
const CODEX_WINDOW_ID_KEY = "window_id";

/**
 * Reads the set of tool names the client's inbound `type: "custom"` tools
 * were flattened from, set by OpenAIResponsesTransformer.transformRequestOut
 * (openai.responses.transformer.ts:130-145). Mirrors the same 3-way context
 * fallback chain that transformer's own transformResponseIn already uses
 * (openai.responses.transformer.ts:163-167), so this works whether the
 * request arrived via the normal route pipeline or a direct transformer
 * call. Real OpenAI natively supports `type: "custom"` (unlike xAI/Bedrock/
 * other Responses-compatible backends), so the codex provider is the one
 * destination where reconstructing the native tool shape is a pure win.
 */
function createToolIndexFor() {
  const toolIndexByKey = new Map<string, number>();
  let nextToolIndex = 0;
  return (data: ResponsesStreamEvent): number => {
    const key =
      data.item_id ||
      data.item?.id ||
      (typeof data.output_index === "number"
        ? `output:${data.output_index}`
        : undefined);
    if (key !== undefined) {
      const existing = toolIndexByKey.get(key);
      if (existing !== undefined) return existing;
      const index = nextToolIndex++;
      toolIndexByKey.set(key, index);
      return index;
    }
    return nextToolIndex++;
  };
}

function customToolInputKey(data: ResponsesStreamEvent): string {
  return data.item_id || data.item?.id || data.item?.call_id || "";
}

function getCustomToolNames(context: any): Set<string> {
  return (
    context?.responsesCustomToolNames ||
    context?.protocolContext?.responsesCustomToolNames ||
    context?.req?.protocolContext?.responsesCustomToolNames ||
    new Set<string>()
  );
}

function getCodexOsType(): string {
  switch (osPlatform()) {
    case "darwin":
      return "Mac OS";
    case "linux":
      return "Linux";
    case "win32":
      return "Windows";
    default:
      return osPlatform() || "unknown";
  }
}

function getCodexArchitecture(): string {
  switch (osArch()) {
    case "x64":
      return "x86_64";
    case "arm64":
      return "arm64";
    default:
      return osArch() || "unknown";
  }
}

function getCodexOsVersion(): string {
  if (osPlatform() === "darwin") {
    try {
      return execFileSync("sw_vers", ["-productVersion"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      }).trim();
    } catch {
      // Fall through to the Node runtime version if sw_vers is unavailable.
    }
  }

  return osRelease() || "unknown";
}

function getCodexUserAgent(): string {
  return `${CODEX_ORIGINATOR}/${CODEX_CLI_VERSION} (${getCodexOsType()} ${getCodexOsVersion()}; ${getCodexArchitecture()})`;
}

function appendCodexClientVersion(url: string): string {
  const parsedUrl = new URL(url);
  if (!parsedUrl.searchParams.has("client_version")) {
    parsedUrl.searchParams.set("client_version", CODEX_CLI_VERSION);
  }
  return parsedUrl.toString();
}

function buildCodexTurnMetadata(sessionId: string): CodexTurnMetadata {
  return {
    session_id: sessionId,
    thread_id: sessionId,
    turn_id: randomUUID(),
    window_id: `${sessionId}:0`,
    request_kind: "turn",
    turn_started_at_unix_ms: Date.now(),
  };
}

function buildCodexClientMetadata(turnMetadata: CodexTurnMetadata): Record<string, string> {
  return {
    [CODEX_SESSION_ID_KEY]: turnMetadata.session_id,
    [CODEX_THREAD_ID_KEY]: turnMetadata.thread_id,
    [CODEX_TURN_ID_KEY]: turnMetadata.turn_id,
    [CODEX_WINDOW_ID_KEY]: turnMetadata.window_id,
    [CODEX_TURN_METADATA_HEADER]: JSON.stringify(turnMetadata),
  };
}

function mergeClientMetadata(
  existing: unknown,
  additions: Record<string, string>
): Record<string, string> {
  if (!existing || typeof existing !== "object" || Array.isArray(existing)) {
    return { ...additions };
  }

  return {
    ...(existing as Record<string, string>),
    ...additions,
  };
}

function buildCodexCompatibilityHeaders(turnMetadata: CodexTurnMetadata): Record<string, string> {
  return {
    "session-id": turnMetadata.session_id,
    "thread-id": turnMetadata.thread_id,
    "x-client-request-id": turnMetadata.thread_id,
    [CODEX_WINDOW_ID_HEADER]: turnMetadata.window_id,
    [CODEX_TURN_METADATA_HEADER]: JSON.stringify(turnMetadata),
    [CODEX_PARENT_THREAD_ID_HEADER]: turnMetadata.thread_id,
  };
}

function inferSessionIdFromMetadata(request: UnifiedChatRequest): string | undefined {
  const userId = (request as any)?.metadata?.user_id;
  if (typeof userId !== "string" || !userId) {
    return undefined;
  }

  const parts = userId.split("_session_");
  if (parts.length > 1 && parts[1]) {
    return parts[1];
  }

  try {
    const parsed = JSON.parse(userId);
    if (parsed && typeof parsed.session_id === "string" && parsed.session_id) {
      return parsed.session_id;
    }
  } catch {
    // Ignore non-JSON user_id formats.
  }

  return undefined;
}

function getCodexSessionId(context: any, request: UnifiedChatRequest): string | undefined {
  return context?.req?.sessionId || inferSessionIdFromMetadata(request);
}

function applyCodexTurnMetadata(request: UnifiedChatRequest, headers: Record<string, string>, sessionId?: string): void {
  if (!sessionId) {
    return;
  }

  const turnMetadata = buildCodexTurnMetadata(sessionId);
  (request as any).client_metadata = mergeClientMetadata(
    (request as any).client_metadata,
    buildCodexClientMetadata(turnMetadata)
  );
  Object.assign(headers, buildCodexCompatibilityHeaders(turnMetadata));
}

export class CodexTransformer implements Transformer {
  name = "codex";
  logger?: any;
  // Per-request streaming intent. The codex API requires stream:true, so the
  // outgoing call is always streaming — but we need to know whether the
  // caller wants streaming or non-streaming so transformResponseOut can
  // return the right shape. Non-streaming callers (e.g. the Anthropic SDK's
  // client.beta.messages.create with stream:false, used by Claude Code CLI's
  // model-validator) get a single JSON ChatCompletion back instead of an SSE
  // stream the SDK can't accumulate into a flat BetaMessage.
  private streamIntent: Map<string, boolean> = new Map();

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any,
    context?: any
  ): Promise<Record<string, any>> {
    const reqId = context?.req?.id;
    if (reqId) {
      // Record the caller's streaming intent. The codex upstream only supports
      // streaming, so the outgoing call is always stream:true (set on `request`
      // below at `(request as any).stream = true`). But when the caller did
      // NOT ask for a stream — e.g. the Anthropic SDK's non-streaming
      // client.beta.messages.create used by Claude Code's model-validator,
      // which omits `stream` entirely — transformResponseOut must convert the
      // SSE stream back into a flat JSON ChatCompletion so the SDK receives a
      // parseable BetaMessage. Streaming is opt-in (=== true): an omitted
      // `stream` field is a non-streaming request, matching the Anthropic SDK
      // default. We must NOT mutate context.req.body.stream here — that object
      // is the original request body read by formatResponse, and forcing it to
      // true would send a flat JSON body with text/event-stream headers.
      this.streamIntent.set(reqId, request.stream === true);
    }
    const resolvedAuth = await this.resolveAuth(provider);

    delete request.temperature;
    delete request.max_tokens;

    // Effort comes from the unified reasoning field (populated by anthropic.transformer
    // from the user's /effort setting via output_config.effort or request.effort).
    const reasoningDisabled = isReasoningDisabled(
      request.reasoning,
      request.thinking
    );
    const effort = reasoningDisabled
      ? "none"
      : normalizeReasoningEffort(request.reasoning?.effort) ||
        normalizeReasoningEffort(provider?.reasoningEffort);
    if (effort) {
      const reasoning: Record<string, any> = { effort };
      const VALID_SUMMARIES = ["auto", "detailed", "none"];
      const summary = provider?.reasoningSummary;
      if (effort !== "none") {
        if (summary && VALID_SUMMARIES.includes(summary)) {
          if (summary !== "none") {
            reasoning.summary = summary;
          }
        } else {
          // Default to detailed reasoning summaries when effort is enabled so
          // Claude Code reliably receives visible thinking unless the provider
          // explicitly disables them with reasoningSummary: "none".
          reasoning.summary = "detailed";
        }
      }
      (request as any).reasoning = reasoning;
    } else {
      delete (request as any).reasoning;
    }

    // Pass through verbosity when configured (Codex models accept low|medium|high)
    const VALID_VERBOSITIES = ["low", "medium", "high"];
    if (provider?.verbosity && VALID_VERBOSITIES.includes(provider.verbosity)) {
      (request as any).verbosity = provider.verbosity;
    }

    // Restore native structured output for the real OpenAI backend. The
    // client's Responses `text.format` was converted to the Chat Completions
    // `response_format` shape on the inbound leg (openai.responses.util.ts)
    // since Unified is Chat-Completions-shaped; only codex reconstructs it
    // back — same restore-only-for-codex pattern as the custom tool type.
    const format = responsesTextFormatFromResponseFormat(
      (request as any).response_format
    );
    if (format) {
      (request as any).text = { ...(request as any).text, format };
    }
    delete (request as any).response_format;

    request = applyRequestCacheKey(request, context);
    const messages = validateOpenAIToolCalls(request.messages);
    request.messages = messages;

    const input: any[] = [];
    let lastWasTool = false;
    const customToolNames = getCustomToolNames(context);
    // Populated while walking assistant tool_calls below (messages arrive in
    // causal order, so a custom tool's call is always seen before its
    // matching tool-result message) so the tool-role branch knows whether to
    // emit custom_tool_call_output vs. function_call_output — a Unified
    // tool-role message only carries tool_call_id, not the tool name.
    const customToolCallIds = new Set<string>();

    const systemMessages = request.messages.filter(
      (msg) => msg.role === "system"
    );
    if (systemMessages.length > 0) {
      const firstSystem = systemMessages[0];
      let instructionsText: string;
      if (Array.isArray(firstSystem.content)) {
        instructionsText = firstSystem.content
          .map((item) => {
            if (typeof item === "string") return item;
            if (item && typeof item === "object" && "text" in item)
              return (item as { text: string }).text;
            return "";
          })
          .filter(Boolean)
          .join("\n");
      } else {
        instructionsText = firstSystem.content as string;
      }
      if (instructionsText) {
        (request as any).instructions = instructionsText;
      }
    }
    for (const extraSystem of systemMessages.slice(1)) {
      const text = Array.isArray(extraSystem.content)
        ? extraSystem.content
            .map((item: any) =>
              typeof item === "string"
                ? item
                : item && typeof item === "object" && "text" in item
                  ? (item as { text: string }).text
                  : ""
            )
            .filter(Boolean)
            .join("\n")
        : String(extraSystem.content || "");
      if (text) {
        input.push({
          type: "message",
          role: "system",
          content: [{ type: "input_text", text }],
        });
      }
    }

    // Codex API requires instructions — provide a default if none set
    if (!(request as any).instructions) {
      (request as any).instructions = "";
    }

    request.messages.forEach((message) => {
      if (message.role === "system") return;

      if (Array.isArray(message.content)) {
        const convertedContent = message.content
          .map((content) => this.normalizeRequestContent(content, message.role))
          .filter(
            (content): content is Record<string, unknown> => content !== null
          );

        if (convertedContent.length > 0) {
          (message as any).content = convertedContent;
        } else {
          delete (message as any).content;
        }
      }

      if (message.role === "tool") {
        const callId =
          sanitizeResponsesCallId(message.tool_call_id) ?? message.tool_call_id;
        if (callId && customToolCallIds.has(callId)) {
          input.push({
            type: "custom_tool_call_output",
            call_id: callId,
            output: message.content,
          });
          lastWasTool = true;
          return;
        }
        const toolMessage: any = { ...message };
        toolMessage.type = "function_call_output";
        toolMessage.call_id = callId;
        toolMessage.output = message.content;
        delete toolMessage.cache_control;
        delete toolMessage.role;
        delete toolMessage.tool_call_id;
        delete toolMessage.content;
        input.push(toolMessage);
        lastWasTool = true;
        return;
      }

      if (message.role === "assistant") {
        const turn = canonicalAssistantTurn(message);
        if (turn.thinking || turn.toolCalls.length) {
          lastWasTool = false;
          const reasoningItem = responsesReasoningItemFromThinking(turn.thinking);
          if (reasoningItem) input.push(reasoningItem);
          if (assistantTurnHasText(turn) || turn.images.length) {
            input.push({
              type: "message",
              role: "assistant",
              content:
                turn.images.length > 0
                  ? message.content
                  : [
                      {
                        type: "output_text",
                        text: turn.texts.map((part) => part.text).join("\n"),
                      },
                    ],
            });
          }
          for (const tool of turn.toolCalls) {
            const callId = sanitizeResponsesCallId(tool.id) ?? tool.id;
            if (customToolNames.has(tool.function.name)) {
              customToolCallIds.add(callId);
              input.push({
                type: "custom_tool_call",
                name: tool.function.name,
                call_id: callId,
                input: unwrapCustomToolInput(tool.function.arguments),
              });
              continue;
            }
            input.push({
              type: "function_call",
              arguments: tool.function.arguments,
              name: tool.function.name,
              call_id: callId,
            });
          }
          if (turn.toolCalls.length || turn.thinking) return;
        }
      }

      if (lastWasTool && message.role === "user") {
        input.push({
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "" }],
        });
      }
      lastWasTool = false;

      if (typeof message.content === "string") {
        input.push({
          type: "message",
          role: message.role,
          content: [
            {
              type: message.role === "assistant" ? "output_text" : "input_text",
              text: message.content,
            },
          ],
        });
      } else {
        input.push({
          type: "message",
          role: message.role,
          content: message.content,
        });
      }
    });

    (request as any).input = input;
    delete (request as any).messages;

    // Codex API requires store to be false
    (request as any).store = false;

    if (Array.isArray(request.tools)) {
      const webSearch = request.tools.find(
        (tool) => tool.function.name === "web_search"
      );

      (request as any).tools = request.tools
        .filter((tool) => tool.function.name !== "web_search")
        .map((tool) => {
          if (customToolNames.has(tool.function.name)) {
            // Real OpenAI natively supports the freeform `custom` tool type
            // (unlike xAI/Bedrock/other Responses-compatible backends) —
            // restore it instead of the generic JSON-schema function-tool
            // proxy normalizeResponsesTools built for the inbound leg. No
            // `parameters` key: custom tools take unconstrained freeform
            // text, not a JSON schema.
            return {
              type: "custom",
              name: tool.function.name,
              description: tool.function.description,
            };
          }
          if (tool.function.name === "WebSearch") {
            delete tool.function.parameters.properties.allowed_domains;
          }
          if (tool.function.name === "Edit") {
            return {
              type: tool.type,
              name: tool.function.name,
              description: tool.function.description,
              parameters: {
                ...tool.function.parameters,
                required: [
                  "file_path",
                  "old_string",
                  "new_string",
                  "replace_all",
                ],
              },
              strict: true,
            };
          }
          return {
            type: tool.type,
            name: tool.function.name,
            description: tool.function.description,
            parameters: tool.function.parameters,
          };
        });

      if (webSearch) {
        (request as any).tools.push({
          type: "web_search",
        });
      }
    }

    // Default to serial tool calls for safety; provider can opt-in to parallel via config
    request.parallel_tool_calls = provider?.parallelToolCalls === true;
    (request as any).stream = true;

    const headers = this.buildAuthHeaders(resolvedAuth);

    applyCodexTurnMetadata(
      request,
      headers,
      (request as any).prompt_cache_key ||
        getCodexSessionId(context, request)
    );

    const baseUrl = provider?.baseUrl || "https://chatgpt.com/backend-api/codex";

    return {
      body: request,
      config: {
        url: appendCodexClientVersion(`${baseUrl}/responses`),
        headers,
        __authRecovery: () =>
          this.recoverUnauthorizedAuth(provider, resolvedAuth),
      },
    };
  }

  async auth(
    request: any,
    provider: any
  ): Promise<any> {
    const resolvedAuth = await this.resolveAuth(provider);
    const headers = this.buildAuthHeaders(resolvedAuth);

    const baseUrl = provider?.baseUrl || "https://chatgpt.com/backend-api/codex";

    return {
      body: request,
      config: {
        url: appendCodexClientVersion(`${baseUrl}/responses`),
        headers,
        __authRecovery: () =>
          this.recoverUnauthorizedAuth(provider, resolvedAuth),
      },
    };
  }

  private async resolveAuth(provider: any): Promise<ResolvedCodexAuth> {
    const pat = resolveCodexPat(provider?.apiKey, { allowBareEnvName: true });
    if (pat) {
      return this.resolvePatAuth(pat);
    }
    return toCodexOAuthAuth(await getValidAccessToken());
  }

  private buildAuthHeaders(auth: ResolvedCodexAuth): Record<string, string> {
    const headers: Record<string, string> = {
      Authorization: `Bearer ${auth.token}`,
      originator: CODEX_ORIGINATOR,
      "User-Agent": getCodexUserAgent(),
    };
    if (auth.accountId) headers["ChatGPT-Account-ID"] = auth.accountId;
    if (auth.isFedramp) headers["X-OpenAI-Fedramp"] = "true";
    return headers;
  }

  private async recoverUnauthorizedAuth(
    provider: any,
    previous: ResolvedCodexAuth
  ): Promise<Record<string, string> | null> {
    if (previous.mode === "pat") {
      whoamiCache.delete(previous.token);
      return null;
    }

    if (resolveCodexPat(provider?.apiKey, { allowBareEnvName: true })) {
      return null;
    }

    const refreshed = await getValidAccessToken({
      force: true,
      previousAccessToken: previous.token,
      expectedAccountId: previous.accountId,
    });
    const next = toCodexOAuthAuth(refreshed);
    if (
      previous.accountId &&
      next.accountId &&
      previous.accountId !== next.accountId
    ) {
      throw new Error(
        "Codex OAuth account changed during unauthorized recovery."
      );
    }
    return this.buildAuthHeaders(next);
  }

  private async resolvePatAuth(pat: string): Promise<PatAuth> {
    const cached = whoamiCache.get(pat);
    if (cached && cached.expiresAt > Date.now()) return cached.value;
    whoamiCache.delete(pat);

    const pending = whoamiRequests.get(pat);
    if (pending) return pending;

    const request = this.requestPatAuth(pat).finally(() => {
      whoamiRequests.delete(pat);
    });
    whoamiRequests.set(pat, request);
    return request;
  }

  private async requestPatAuth(pat: string): Promise<PatAuth> {
    const authApiBaseUrl = (
      process.env.CODEX_AUTHAPI_BASE_URL ||
      "https://auth.openai.com/api/accounts"
    ).replace(/\/+$/, "");
    const response = await fetch(
      `${authApiBaseUrl}/v1/user-auth-credential/whoami`,
      {
        headers: { Authorization: `Bearer ${pat}` },
        signal: AbortSignal.timeout(15_000),
      }
    );

    if (!response.ok) {
      throw new Error(
        `Codex PAT metadata request failed (${response.status}). Verify that the configured at- token is valid and has Codex access.`
      );
    }

    const data: WhoamiResponse = await response.json();
    if (
      !data.chatgpt_account_id ||
      !data.chatgpt_user_id ||
      !data.chatgpt_plan_type
    ) {
      throw new Error(
        "Codex PAT metadata response is missing required account, user, or plan information."
      );
    }

    const result: PatAuth = {
      mode: "pat",
      token: pat,
      accountId: data.chatgpt_account_id,
      isFedramp: data.chatgpt_account_is_fedramp === true,
    };
    whoamiCache.set(pat, {
      value: result,
      expiresAt: Date.now() + PAT_METADATA_TTL_MS,
    });
    return result;
  }

  async transformResponseOut(
    response: Response,
    context?: { req?: { id?: string } }
  ): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";
    const reqId = context?.req?.id;
    // Wrap the body in try/finally to guarantee streamIntent cleanup on any
    // unexpected throw (e.g. response.json() parse error, downstream buffer
    // failure). Without this, an exception mid-flow would leak the per-request
    // entry in the map for the lifetime of the process.
    const prevIntent = reqId ? this.streamIntent.get(reqId) : undefined;
    try {
      return await this.transformResponseOutInner(response, contentType, this, reqId, prevIntent);
    } finally {
      if (reqId) this.streamIntent.delete(reqId);
    }
  }

  private async transformResponseOutInner(
    response: Response,
    contentType: string,
    codexTransformer: CodexTransformer,
    reqId: string | undefined,
    prevIntent: boolean | undefined,
  ): Promise<Response> {
    const wantsStream = reqId ? prevIntent !== false : true;

    // Codex streams omit the `model` field on every chunk except the final
    // `response.completed` event. Downstream consumers (notably Claude Code
    // CLI's post-call model validator) read `message_start.message.model` and
    // throw when it's "unknown". Track the model as soon as we see it so
    // every chunk we forward can carry a real value.
    let observedModel: string | undefined = undefined;
    const resolveModel = (eventModel: string | undefined): string | undefined => {
      if (eventModel) {
        observedModel = eventModel;
        return eventModel;
      }
      return observedModel;
    };

    // Cloudflare strips Content-Type from SSE responses — treat missing content-type as SSE
    if (!contentType || contentType.includes("text/event-stream")) {
      if (!response.body) {
        return response;
      }

      // The codex API occasionally returns a single JSON object
      // (chat.completion or ResponsesAPIPayload) with text/event-stream
      // Content-Type — e.g. for short completions it skips streaming and
      // returns a flat JSON. Peek at the first non-whitespace character of
      // a clone of the body: a real SSE stream starts with `data:` or
      // `event:`, while a raw JSON object starts with `{` or `[`. (Model
      // message text is always inside a JSON event's `delta` field, never
      // as raw stream text, so this disambiguation is unambiguous.)
      const peek = await this.readBodyAndPeek(response.clone().body!);
      if (peek && (peek.firstChar === "{" || peek.firstChar === "[")) {
        // Codex returned a flat JSON object (chat.completion or
        // ResponsesAPIPayload) instead of an SSE stream — it does this for
        // short/trivial completions even when stream:true was requested.
        // Normalize it to an OpenAI ChatCompletion, then dispatch by the
        // caller's streaming intent: non-streaming callers get the flat JSON
        // (the SDK reads a single JSON object); streaming callers get that
        // same ChatCompletion re-emitted as a one-shot SSE stream so their
        // SSE parser stays happy.
        let parsed: any;
        try {
          parsed = JSON.parse(peek.text);
        } catch {
          if (!wantsStream) {
            return new Response(peek.text, {
              status: response.status,
              statusText: response.statusText,
              headers: new Headers({ "Content-Type": "application/json" }),
            });
          }
          return new Response(this.jsonToSseStream(peek.text), {
            status: response.status,
            statusText: response.statusText,
            headers: new Headers({ "Content-Type": "text/event-stream" }),
          });
        }
        const chatCompletion =
          parsed?.object === "response" && parsed.output
            ? this.convertResponseToChat(parsed)
            : parsed;
        const bodyJson = JSON.stringify(chatCompletion);
        if (!wantsStream) {
          return new Response(bodyJson, {
            status: response.status,
            statusText: response.statusText,
            headers: new Headers({ "Content-Type": "application/json" }),
          });
        }
        return new Response(this.jsonToSseStream(bodyJson), {
          status: response.status,
          statusText: response.statusText,
          headers: new Headers({ "Content-Type": "text/event-stream" }),
        });
      }

      // Non-streaming caller (e.g. Anthropic SDK's client.beta.messages.create
      // with stream:false, used by Claude Code CLI's model-validator): the
      // codex API only streams, so consume the full SSE stream and assemble
      // a single OpenAI ChatCompletion JSON so the SDK can parse it.
      if (!wantsStream) {
        const completion = await this.collectSseIntoChatCompletion(
          response,
          resolveModel
        );
        return new Response(JSON.stringify(completion), {
          status: response.status,
          statusText: response.statusText,
          headers: { "Content-Type": "application/json" },
        });
      }

      let currentIndex = -1;
      let lastEventType = "";
      const customToolInputByItemId = new Map<string, string>();
      const thinkingByItemId = new Map<string, string>();
      const toolIndexFor = createToolIndexFor();

      const getCurrentIndex = (eventType: string) => {
        if (eventType !== lastEventType) {
          currentIndex++;
          lastEventType = eventType;
        }
        return currentIndex;
      };

      const reader = createSSEStreamReader(
        response,
        (line: string, ctx: StreamContext) => {
          if (!line.trim()) return;

          if (line.startsWith("event: ")) return;

          if (line.startsWith("data: ")) {
            const dataStr = line.slice(5).trim();
            if (dataStr === "[DONE]") {
              ctx.controller.enqueue(encodeSSEData("[DONE]", ctx.encoder));
              return;
            }

            try {
              const data: ResponsesStreamEvent = JSON.parse(dataStr);
              if (data.response?.model) {
                observedModel = data.response.model;
              }
              const chunk = codexTransformer.convertStreamEvent(data, getCurrentIndex, resolveModel, customToolInputByItemId, toolIndexFor, thinkingByItemId);
              if (chunk) {
                ctx.controller.enqueue(encodeSSEData(JSON.stringify(chunk), ctx.encoder));
              }
            } catch {
              ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
            }
          } else {
            ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          }
        }
      );
      // Ensure cleanup if the stream is cancelled before [DONE]. Preserve the
      // text/event-stream Content-Type (createSSEStreamReader's Response carried
      // it; the codex upstream's headers have no content-type since Cloudflare
      // strips it, so re-using response.headers here would make the downstream
      // anthropic transformer take the JSON branch and break streaming).
      const liveStream = reader.body!;
      // Track the stream reader for cleanup. The original stream from
      // createSSEStreamReader handles cancel, but our wrapping Response
      // won't fire [DONE] cleanup on disconnect — add a cancel handler
      // directly to the returned ReadableStream, and propagate the cancel
      // upstream so we don't leak the inner reader. The pump must also
      // guard against enqueue/close on an already-closed controller (which
      // throws "Controller is already closed" and would surface as an
      // unhandled promise rejection).
      let cancelled = false;
      let innerReader: ReadableStreamDefaultReader<Uint8Array> | null = null;
      const trackedStream = new ReadableStream<Uint8Array>({
        start(controller) {
          innerReader = liveStream.getReader();
          const safeEnqueue = (value: Uint8Array) => {
            if (cancelled) return;
            try {
              controller.enqueue(value);
            } catch {
              // controller was closed (e.g. by upstream cancel); stop pumping
              cancelled = true;
            }
          };
          const safeClose = () => {
            if (cancelled) return;
            try {
              controller.close();
            } catch {
              // already closed
            }
          };
          const safeError = (err: any) => {
            if (cancelled) return;
            try {
              controller.error(err);
            } catch {
              // already closed
            }
          };
          const pump = () => {
            if (cancelled || !innerReader) return;
            innerReader.read().then(({ done, value }) => {
              if (done) {
                safeClose();
                return;
              }
              if (value) safeEnqueue(value);
              pump();
            }).catch((err) => {
              if (cancelled) return;
              safeError(err);
              // The outer try/finally in transformResponseOut already deleted
              // the streamIntent entry; no need to clean up here.
            });
          };
          pump();
        },
        cancel() {
          cancelled = true;
          // Propagate cancel upstream so the inner reader's read() resolves
          // with done=true instead of hanging or erroring. The outer
          // try/finally in transformResponseOut already deleted the
          // streamIntent entry by the time this fires, so no cleanup here.
          if (innerReader) {
            try {
              innerReader.cancel().catch(() => {
                // best-effort — the stream may already be closed
              });
            } catch {
              // already released
            }
          }
        },
      });
      return new Response(trackedStream, {
        status: response.status,
        statusText: response.statusText,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else if (contentType.includes("application/json")) {
      const jsonResponse: any = await response.json();

      // Streaming response: JSON array of events
      if (Array.isArray(jsonResponse)) {
        const encoder = new TextEncoder();
        let index = -1;
        let lastEventType = "";
        const customToolInputByItemId = new Map<string, string>();
        const thinkingByItemId = new Map<string, string>();
        const toolIndexFor = createToolIndexFor();
        const getCurrentIndex = (eventType: string) => {
          if (eventType !== lastEventType) {
            index++;
            lastEventType = eventType;
          }
          return index;
        };

        const stream = new ReadableStream({
          start(controller) {
            for (const event of jsonResponse) {
              if (event?.response?.model) {
                observedModel = event.response.model;
              }
              const chunk = codexTransformer.convertStreamEvent(event, getCurrentIndex, resolveModel, customToolInputByItemId, toolIndexFor, thinkingByItemId);
              if (chunk) {
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(chunk)}\n\n`)
                );
              }
            }
            controller.enqueue(encoder.encode("data: [DONE]\n\n"));
            controller.close();
          },
        });

        return new Response(stream, {
          status: response.status,
          statusText: response.statusText,
          headers: { "Content-Type": "text/event-stream" },
        });
      }

      if (jsonResponse.object === "response" && jsonResponse.output) {
        const chatResponse = this.convertResponseToChat(jsonResponse);
        const bodyJson = JSON.stringify(chatResponse);
        if (reqId && wantsStream) {
          return new Response(this.jsonToSseStream(bodyJson), {
            status: response.status,
            statusText: response.statusText,
            headers: new Headers({ "Content-Type": "text/event-stream" }),
          });
        }
        return new Response(bodyJson, {
          status: response.status,
          statusText: response.statusText,
          headers: new Headers({ "Content-Type": "application/json" }),
        });
      }

      const bodyJson = JSON.stringify(jsonResponse);
      if (reqId && wantsStream) {
        return new Response(this.jsonToSseStream(bodyJson), {
          status: response.status,
          statusText: response.statusText,
          headers: new Headers({ "Content-Type": "text/event-stream" }),
        });
      }
      return new Response(bodyJson, {
        status: response.status,
        statusText: response.statusText,
        headers: new Headers({ "Content-Type": "application/json" }),
      });

    }

    return response;
  }

  private convertStreamEvent(
    data: ResponsesStreamEvent,
    getCurrentIndex: (type: string) => number,
    resolveModel?: (eventModel: string | undefined) => string | undefined,
    customToolInputByItemId?: Map<string, string>,
    toolIndexFor?: (data: ResponsesStreamEvent) => number,
    thinkingByItemId?: Map<string, string>
  ): any | null {
    const toolIndex = () => (toolIndexFor ? toolIndexFor(data) : 0);
    const deliveredThinking = thinkingByItemId ?? new Map<string, string>();
    const fallback = (eventModel: string | undefined): string | undefined =>
      resolveModel ? resolveModel(eventModel) : eventModel;
    const modelForChunk = (eventModel: string | undefined) =>
      eventModel || fallback(undefined);
    if (data.type === "response.output_text.delta") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {
              content: data.delta || "",
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_item.added" && data.item?.type === "function_call") {
      return {
        id: data.item.call_id || data.item.id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index: toolIndex(),
                  id:
                    sanitizeResponsesCallId(
                      data.item.call_id || data.item.id
                    ) || data.item.call_id || data.item.id,
                  function: {
                    name: data.item.name || "",
                    arguments: "",
                  },
                  type: "function",
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_item.added" && data.item?.type === "custom_tool_call") {
      const itemId = customToolInputKey(data);
      if (itemId) customToolInputByItemId?.set(itemId, "");
      return {
        id: data.item.call_id || data.item.id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {
              role: "assistant",
              tool_calls: [
                {
                  index: toolIndex(),
                  id:
                    sanitizeResponsesCallId(
                      data.item.call_id || data.item.id
                    ) || data.item.call_id || data.item.id,
                  function: {
                    name: data.item.name || "",
                    arguments: "",
                  },
                  type: "function",
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.custom_tool_call_input.delta") {
      // Custom tools carry unconstrained freeform text, not JSON — a partial
      // string can't be JSON-wrapped mid-stream (CUSTOM_TOOL_INPUT_KEY
      // convention requires valid JSON per chunk). Accumulate and emit the
      // complete wrapped argument once, on .done, mirroring how
      // unifiedChunkToResponsesEvents treats custom tools as
      // non-streaming-argument on the reverse path
      // (openai.responses.util.ts:1017-1035).
      const itemId = customToolInputKey(data);
      if (customToolInputByItemId && itemId) {
        const prev = customToolInputByItemId.get(itemId) ?? "";
        const delta = typeof data.delta === "string" ? data.delta : "";
        customToolInputByItemId.set(itemId, prev + delta);
      }
      return null;
    }

    if (data.type === "response.custom_tool_call_input.done") {
      const itemId = customToolInputKey(data);
      const fullInput =
        (typeof data.input === "string" ? data.input : undefined) ??
        (itemId ? customToolInputByItemId?.get(itemId) : undefined) ??
        "";
      if (itemId) customToolInputByItemId?.delete(itemId);
      return {
        id: data.item_id || data.item?.id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            // Same index bucket the function_call arguments delta uses, so a
            // custom tool call's argument block transitions identically to
            // a function tool call's.
            index: getCurrentIndex("response.function_call_arguments.delta"),
            delta: {
              tool_calls: [
                {
                  index: toolIndex(),
                  function: {
                    arguments: JSON.stringify({
                      [CUSTOM_TOOL_INPUT_KEY]: fullInput,
                    }),
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_item.added" && data.item?.type === "message") {
      const contentItems: MessageContent[] = [];
      (data.item.content || []).forEach((item: any) => {
        if (item.type === "output_text") {
          contentItems.push({
            type: "text",
            text: item.text || "",
          });
        }
      });

      const delta: any = { role: "assistant" };
      if (contentItems.length === 1 && contentItems[0].type === "text") {
        delta.content = contentItems[0].text;
      } else if (contentItems.length > 0) {
        delta.content = contentItems;
      }
      if (delta.content) {
        return {
          id: data.item.id || "chatcmpl-" + Date.now(),
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: modelForChunk(data.response?.model),
          choices: [
            {
              index: getCurrentIndex(data.type),
              delta,
              finish_reason: null,
            },
          ],
        };
      }
      return null;
    }

    if (data.type === "response.output_text.annotation.added") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {
              annotations: [
                {
                  type: "url_citation",
                  url_citation: {
                    url: data.annotation?.url || "",
                    title: data.annotation?.title || "",
                    content: "",
                    start_index: data.annotation?.start_index || 0,
                    end_index: data.annotation?.end_index || 0,
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.function_call_arguments.delta") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {
              tool_calls: [
                {
                  index: toolIndex(),
                  function: {
                    arguments: data.delta || "",
                  },
                },
              ],
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.completed") {
      const finishReason = data.response?.output?.some(
        (item: any) => item.type === "function_call" || item.type === "custom_tool_call"
      )
        ? "tool_calls"
        : "stop";

      // Ciphertext often arrives only on the completed reasoning item.
      // Omit content when summary deltas already streamed it — Unified
      // `delta.thinking.content` is additive.
      const reasoningThinking = thinkingForLateReasoningItem(
        data.response?.output?.find((item: any) => item.type === "reasoning"),
        deliveredThinking
      );

      const chunk: any = {
        id: data.response?.id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: 0,
            delta: reasoningThinking ? { thinking: reasoningThinking } : {},
            finish_reason: finishReason,
          },
        ],
      };

      if (data.response?.usage) {
        chunk.usage = {
          prompt_tokens: data.response.usage.input_tokens || 0,
          completion_tokens: data.response.usage.output_tokens || 0,
          total_tokens: data.response.usage.total_tokens || 0,
          prompt_tokens_details: {
            cached_tokens:
              data.response.usage.input_tokens_details?.cached_tokens || 0,
            cache_write_tokens:
              data.response.usage.input_tokens_details?.cache_write_tokens || 0,
          },
        };
      }

      return chunk;
    }

    if (data.type === "response.reasoning_summary_text.delta") {
      recordReasoningSummaryDelta(deliveredThinking, data.item_id, data.delta);
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {
              thinking: {
                content: data.delta || "",
              },
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.reasoning_summary_text.done") {
      // Preserve the reasoning item id for replay. Never store it as
      // thinking.signature — that field used to be copied into
      // encrypted_content and Codex rejected the forged ciphertext.
      if (typeof data.item_id !== "string" || !data.item_id) return null;
      return {
        id: data.item_id,
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {
              thinking: {
                id: data.item_id,
              },
            },
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_text.done") {
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {},
            finish_reason: null,
          },
        ],
      };
    }

    if (data.type === "response.output_item.done") {
      if (data.item?.type === "reasoning") {
        const thinking = thinkingForLateReasoningItem(
          data.item,
          deliveredThinking
        );
        if (!thinking) return null;
        return {
          id: data.item.id || data.item_id || "chatcmpl-" + Date.now(),
          object: "chat.completion.chunk",
          created: Math.floor(Date.now() / 1000),
          model: modelForChunk(data.response?.model),
          choices: [
            {
              index: getCurrentIndex(data.type),
              delta: { thinking },
              finish_reason: null,
            },
          ],
        };
      }
      return {
        id: data.item_id || "chatcmpl-" + Date.now(),
        object: "chat.completion.chunk",
        created: Math.floor(Date.now() / 1000),
        model: modelForChunk(data.response?.model),
        choices: [
          {
            index: getCurrentIndex(data.type),
            delta: {},
            finish_reason: null,
          },
        ],
      };
    }

    return null;
  }

  /**
   * Consume a Codex SSE response fully and return a single OpenAI
   * ChatCompletion JSON. Used for non-streaming Anthropic SDK calls
   * (e.g. client.beta.messages.create with stream:false) where the
   * SDK expects a flat BetaMessage. The SDK accumulates the response
   * by reading a single JSON object, not by parsing SSE, so we
   * have to materialize the response here.
   */

  /**
   * Re-emit a single OpenAI ChatCompletion JSON as a one-shot SSE stream of
   * `chat.completion.chunk` events, so a streaming caller (which expects SSE)
   * still receives the content when codex returned a flat JSON instead of a
   * stream. The downstream anthropic transformer's stream reader parses each
   * `data:` line as a chunk, so we split the full message into proper delta
   * chunks (text content first, then a final chunk with finish_reason + usage)
   * — emitting the raw chat.completion would put the text under `message`
   * instead of `delta` and the content would be dropped.
   */
  private jsonToSseStream(chatCompletionJson: string): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    let parsed: any;
    try {
      parsed = JSON.parse(chatCompletionJson);
    } catch {
      parsed = null;
    }
    const emit = (obj: any) => encoder.encode(`data: ${JSON.stringify(obj)}\n\n`);
    return new ReadableStream<Uint8Array>({
      start(controller) {
        if (!parsed) {
          controller.close();
          return;
        }
        const base = {
          id: parsed.id || "chatcmpl-" + Date.now(),
          object: "chat.completion.chunk",
          created: parsed.created || Math.floor(Date.now() / 1000),
          model: parsed.model || "unknown",
        };
        const choice = parsed.choices?.[0];
        if (choice) {
          const message = choice.message || {};
          // Content chunk(s)
          if (typeof message.content === "string" && message.content) {
            controller.enqueue(emit({
              ...base,
              choices: [{
                index: 0,
                delta: { role: "assistant", content: message.content },
                finish_reason: null,
              }],
            }));
          }
          if (
            message.thinking?.content ||
            message.thinking?.encrypted_content ||
            message.thinking?.id
          ) {
            controller.enqueue(emit({
              ...base,
              choices: [{
                index: 0,
                delta: {
                  thinking: {
                    ...(message.thinking?.content
                      ? { content: message.thinking.content }
                      : {}),
                    ...(message.thinking?.encrypted_content
                      ? { encrypted_content: message.thinking.encrypted_content }
                      : {}),
                    ...(message.thinking?.id
                      ? { id: message.thinking.id }
                      : {}),
                  },
                },
                finish_reason: null,
              }],
            }));
          }
          if (Array.isArray(message.tool_calls)) {
            for (const tc of message.tool_calls) {
              controller.enqueue(emit({
                ...base,
                choices: [{
                  index: 0,
                  delta: {
                    role: "assistant",
                    tool_calls: [{
                      index: 0,
                      id: tc.id,
                      type: "function",
                      function: { name: tc.function?.name || "", arguments: tc.function?.arguments || "" },
                    }],
                  },
                  finish_reason: null,
                }],
              }));
            }
          }
          // Final chunk with finish_reason + usage
          const finalChunk: any = {
            ...base,
            choices: [{
              index: 0,
              delta: {},
              finish_reason: choice.finish_reason || "stop",
            }],
          };
          if (parsed.usage) {
            finalChunk.usage = {
              prompt_tokens: parsed.usage.prompt_tokens || 0,
              completion_tokens: parsed.usage.completion_tokens || 0,
              total_tokens: parsed.usage.total_tokens || 0,
              ...(parsed.usage.prompt_tokens_details
                ? {
                    prompt_tokens_details: {
                      ...parsed.usage.prompt_tokens_details,
                    },
                  }
                : {}),
            };
          }
          controller.enqueue(emit(finalChunk));
        }
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });
  }

  /**
   * Peek at the first non-whitespace character of a cloned response body to
   * distinguish a real SSE stream (starts with `data:` or `event:`, first
   * char `d` or `e`) from a flat JSON body (first char `{` or `[`) that the
   * codex API sometimes returns even with text/event-stream Content-Type.
   *
   * - **For flat JSON** (`{` / `[`): reads the full clone body and returns
   *   `{ firstChar, text }` so the caller can parse and handle it.
   * - **For SSE** (anything else): returns `null` immediately after reading
   *   only the first chunk from the clone. The original `response.body` on
   *   the other side of the tee is untouched and can still be streamed live
   *   by the SSE reader — avoiding buffering the entire response.
   *
   * Callers must pass `response.clone().body!` so the original body is not
   * consumed by the peek.
   */
  private async readBodyAndPeek(
    body: ReadableStream<Uint8Array>
  ): Promise<{ firstChar: string; text: string } | null> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    // Wrap all early returns so the reader lock is always released on the
    // clone's body — otherwise the clone is left in a locked state and the
    // GC can't reclaim it. The original response.body (on the other side of
    // response.clone()) is independent and unaffected.
    const releaseReader = () => {
      try {
        reader.releaseLock();
      } catch {
        // already released
      }
    };
    let firstRead: { done: boolean; value?: Uint8Array };
    try {
      firstRead = await reader.read();
    } catch (err) {
      releaseReader();
      throw err;
    }
    if (firstRead.done) {
      releaseReader();
      return null;
    }
    const firstChunk = decoder.decode(firstRead.value!, { stream: true });
    const trimmed = firstChunk.trimStart();
    const firstChar = trimmed.charAt(0) || "";
    // SSE starts with "d" (data:) or "e" (event:) — not JSON → bail fast
    if (firstChar !== "{" && firstChar !== "[") {
      releaseReader();
      return null;
    }
    // Flat JSON — drain the rest of the clone body
    let buffer = firstChunk;
    try {
      while (true) {
        const r = await reader.read();
        if (r.done) break;
        buffer += decoder.decode(r.value, { stream: true });
      }
    } finally {
      buffer += decoder.decode();
      releaseReader();
    }
    return { firstChar, text: buffer };
  }

  private async collectSseIntoChatCompletion(
    response: Response,
    resolveModel: (eventModel: string | undefined) => string | undefined,
  ): Promise<any> {
    const reader = response.body!.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let model = "unknown";
    let id = "chatcmpl-" + Date.now();
    let created = Math.floor(Date.now() / 1000);

    // Accumulator state — mirrors convertStreamEvent output shape
    let contentText = "";
    let reasoningText = "";
    let thinkingEncryptedContent: string | undefined;
    let thinkingId: string | undefined;
    const annotations: any[] = [];
    const toolCallsMap = new Map<number, any>();
    let finishReason: string | null = null;
    let usage: any = undefined;

    let currentIndex = -1;
    let lastEventType = "";
    const customToolInputByItemId = new Map<string, string>();
    const thinkingByItemId = new Map<string, string>();
    const toolIndexFor = createToolIndexFor();

    const getCurrentIndex = (eventType: string) => {
      if (eventType !== lastEventType) {
        currentIndex++;
        lastEventType = eventType;
      }
      return currentIndex;
    };

    const finalize = (): any => {
      const message: any = { role: "assistant" };
      if (reasoningText || thinkingEncryptedContent || thinkingId) {
        message.thinking = {
          content: reasoningText,
          ...(thinkingEncryptedContent
            ? { encrypted_content: thinkingEncryptedContent }
            : {}),
          ...(thinkingId ? { id: thinkingId } : {}),
        };
      }
      const toolCalls = Array.from(toolCallsMap.values());
      if (toolCalls.length) {
        message.tool_calls = toolCalls;
      }
      if (annotations.length) {
        message.annotations = annotations;
      }
      if (contentText) {
        message.content = contentText;
      } else if (toolCalls.length === 0 && !reasoningText) {
        message.content = null;
      } else {
        message.content = contentText || null;
      }

      const choice: any = {
        index: 0,
        message,
        logprobs: null,
        finish_reason: toolCalls.length ? "tool_calls" : (finishReason || "stop"),
      };

      const completion: any = {
        id,
        object: "chat.completion",
        created,
        model: model || "unknown",
        choices: [choice],
        usage: usage || null,
      };
      return completion;
    };

    const processChunk = (chunk: any) => {
      if (!chunk) return;
      if (chunk.id) id = chunk.id;
      if (chunk.model) model = chunk.model;
      if (chunk.created) created = chunk.created;
      if (chunk.usage) usage = chunk.usage;

      const choice = chunk.choices?.[0];
      if (!choice) return;

      if (choice.finish_reason) finishReason = choice.finish_reason;
      const delta = choice.delta || {};
      if (delta.role) {
        // First chunk — already set
      }
      if (typeof delta.content === "string" && delta.content) {
        contentText += delta.content;
      }
      if (delta.thinking?.content) {
        reasoningText += delta.thinking.content;
      }
      const encrypted = responsesEncryptedContentFrom(
        delta.thinking?.encrypted_content
      );
      if (encrypted) thinkingEncryptedContent = encrypted;
      if (typeof delta.thinking?.id === "string" && delta.thinking.id) {
        thinkingId = delta.thinking.id;
      } else if (isResponsesReasoningItemId(delta.thinking?.signature)) {
        thinkingId = delta.thinking.signature;
      }
      if (Array.isArray(delta.annotations)) {
        for (const a of delta.annotations) {
          annotations.push(a);
        }
      }
      if (Array.isArray(delta.tool_calls)) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index ?? 0;
          const existing = toolCallsMap.get(idx) || {
            index: idx,
            id: tc.id,
            type: "function",
            function: { name: tc.function?.name || "", arguments: "" },
          };
          if (tc.id) existing.id = tc.id;
          if (tc.function?.name) existing.function.name = tc.function.name;
          if (typeof tc.function?.arguments === "string") {
            existing.function.arguments =
              (existing.function.arguments || "") + tc.function.arguments;
          }
          toolCallsMap.set(idx, existing);
        }
      }
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const dataStr = line.slice(5).trim();
        if (!dataStr || dataStr === "[DONE]") continue;
        try {
          const event = JSON.parse(dataStr) as ResponsesStreamEvent;
          if (event.response?.model) {
            const resolved = resolveModel(event.response.model);
            if (resolved) model = resolved;
          }
          const chunk = this.convertStreamEvent(event, getCurrentIndex, resolveModel, customToolInputByItemId, toolIndexFor, thinkingByItemId);
          processChunk(chunk);
        } catch {
          // ignore malformed line
        }
      }
    }
    return finalize();
  }

  private normalizeRequestContent(content: any, role: string | undefined) {
    if (content.type === "text") {
      return {
        type: role === "assistant" ? "output_text" : "input_text",
        text: content.text,
      };
    }

    if (content.type === "image_url") {
      this.logger?.debug(content);
      const imagePayload: Record<string, unknown> = {
        type: role === "assistant" ? "output_image" : "input_image",
      };

      if (typeof content.image_url?.url === "string") {
        imagePayload.image_url = content.image_url.url;
      }

      return imagePayload;
    }

    return null;
  }

  private convertResponseToChat(responseData: ResponsesAPIPayload): any {
    const messageOutput = responseData.output?.find(
      (item) => item.type === "message"
    );
    const toolCallOutputs = (responseData.output ?? []).filter(
      (item) => item.type === "function_call" || item.type === "custom_tool_call"
    );
    let annotations;
    if (
      messageOutput?.content?.length &&
      messageOutput?.content[0].annotations
    ) {
      annotations = messageOutput.content[0].annotations.map((item) => {
        return {
          type: "url_citation",
          url_citation: {
            url: item.url || "",
            title: item.title || "",
            content: "",
            start_index: item.start_index || 0,
            end_index: item.end_index || 0,
          },
        };
      });
    }

    this.logger.debug({
      data: annotations,
      type: "url_citation",
    });

    let messageContent: string | MessageContent[] | null = null;
    let toolCalls = null;
    let thinking =
      thinkingFromResponsesReasoningItem(
        responseData.output?.find((item) => item.type === "reasoning")
      ) || null;

    if (!thinking && messageOutput && messageOutput.reasoning) {
      thinking = {
        content: messageOutput.reasoning,
      };
    }

    if (messageOutput && messageOutput.content) {
      const textParts: string[] = [];
      const imageParts: MessageContent[] = [];

      messageOutput.content.forEach((item: any) => {
        if (item.type === "output_text") {
          textParts.push(item.text || "");
        } else if (item.type === "output_image") {
          const imageContent = this.buildImageContent({
            url: item.image_url,
            mime_type: item.mime_type,
          });
          if (imageContent) {
            imageParts.push(imageContent);
          }
        } else if (item.type === "output_image_base64") {
          const imageContent = this.buildImageContent({
            b64_json: item.image_base64,
            mime_type: item.mime_type,
          });
          if (imageContent) {
            imageParts.push(imageContent);
          }
        }
      });

      if (imageParts.length > 0) {
        const contentArray: MessageContent[] = [];
        if (textParts.length > 0) {
          contentArray.push({
            type: "text",
            text: textParts.join(""),
          });
        }
        contentArray.push(...imageParts);
        messageContent = contentArray;
      } else {
        messageContent = textParts.join("");
      }
    }

    if (toolCallOutputs.length > 0) {
      // Every function_call / custom_tool_call survives, in source order —
      // parallel calls must not collapse into the first one found.
      toolCalls = toolCallOutputs.map((call) => ({
        id:
          sanitizeResponsesCallId(call.call_id || call.id) ||
          call.call_id ||
          call.id,
        function: {
          name: call.name,
          arguments:
            call.type === "custom_tool_call"
              ? JSON.stringify({
                  [CUSTOM_TOOL_INPUT_KEY]: call.input || "",
                })
              : call.arguments,
        },
        type: "function",
      }));
    }

    return {
      id: responseData.id || "chatcmpl-" + Date.now(),
      object: "chat.completion",
      created: responseData.created_at,
      model: responseData.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: messageContent || null,
            tool_calls: toolCalls,
            thinking: thinking,
            annotations: annotations,
          },
          logprobs: null,
          finish_reason: toolCalls ? "tool_calls" : "stop",
        },
      ],
      usage: responseData.usage
        ? {
            prompt_tokens: responseData.usage.input_tokens || 0,
            completion_tokens: responseData.usage.output_tokens || 0,
            total_tokens: responseData.usage.total_tokens || 0,
            prompt_tokens_details: {
              cached_tokens:
                responseData.usage.input_tokens_details?.cached_tokens || 0,
              cache_write_tokens:
                responseData.usage.input_tokens_details?.cache_write_tokens ||
                0,
            },
          }
        : null,
    };
  }

  private buildImageContent(source: {
    url?: string;
    b64_json?: string;
    mime_type?: string;
  }): MessageContent | null {
    if (!source) return null;

    if (source.url || source.b64_json) {
      return {
        type: "image_url",
        image_url: {
          url: source.url || "",
          b64_json: source.b64_json,
        },
        media_type: source.mime_type,
      } as MessageContent;
    }

    return null;
  }
}

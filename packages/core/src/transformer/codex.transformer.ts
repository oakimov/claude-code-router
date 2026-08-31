import { randomUUID } from "crypto";
import { execFileSync } from "child_process";
import { arch as osArch, platform as osPlatform, release as osRelease } from "os";
import { Transformer } from "@/types/transformer";
import { resolveCodexPat } from "@caeliq/ccr-shared";
import { deriveCacheSessionKey, extractClientSessionId } from "@/utils/cacheControl";
import { createSSEStreamReader, StreamContext, encodeSSELine } from "../utils/stream";
import { peekResponseBody } from "../utils/stream-peek";
import {
  getValidAccessToken,
  toCodexOAuthAuth,
} from "../utils/codex-auth";
import { unwrapCustomToolInput } from "../utils/openai.responses.util";

const PAT_METADATA_TTL_MS = 5 * 60 * 1000;
const whoamiCache = new Map<
  string,
  { value: PatAuth; expiresAt: number }
>();
const whoamiRequests = new Map<string, Promise<PatAuth>>();
const CODEX_CLI_VERSION = "0.145.0";
const CODEX_ORIGINATOR = "codex_cli_rs";
const CODEX_REQUIRES_RESPONSES =
  'codex requires openai-responses in transformer.use (e.g. ["openai-responses", "codex"]). Codex is ChatGPT auth/headers middleware on the Responses wire and does not convert Chat Completions bodies.';

function isCodexSystemRole(role: unknown): boolean {
  return role === "system" || role === "developer";
}

function textFromInputContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    return content == null ? "" : String(content);
  }
  return content
    .map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object" && "text" in item) {
        return String((item as { text: unknown }).text ?? "");
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

/**
 * Codex rejects `role: "system"|"developer"` in Responses `input` with
 * `{"detail":"System messages are not allowed"}`. Fold those items (and any
 * pre-set instructions) into the top-level `instructions` string in source
 * order. Operates on the Responses wire — conversion from Chat/Anthropic is
 * owned by openai-responses.
 */
function foldSystemItemsIntoInstructions(request: Record<string, any>): void {
  const parts: string[] = [];
  const existing = request.instructions;
  if (typeof existing === "string" && existing) {
    parts.push(existing);
  }
  const input = request.input;
  if (!Array.isArray(input)) {
    request.instructions = parts.join("\n\n");
    return;
  }
  const next: any[] = [];
  for (const item of input) {
    if (item && isCodexSystemRole(item.role)) {
      const text = textFromInputContent(item.content);
      if (text) parts.push(text);
      continue;
    }
    next.push(item);
  }
  request.instructions = parts.join("\n\n");
  request.input = next;
}

function getCustomToolNames(context: any): Set<string> {
  return (
    context?.responsesCustomToolNames ||
    context?.protocolContext?.responsesCustomToolNames ||
    context?.req?.protocolContext?.responsesCustomToolNames ||
    new Set<string>()
  );
}

function restoreCustomTools(
  request: Record<string, any>,
  customToolNames: Set<string>
): void {
  if (customToolNames.size === 0) return;

  if (Array.isArray(request.tools)) {
    request.tools = request.tools.map((tool: any) => {
      const name = tool?.name || tool?.function?.name;
      if (typeof name !== "string" || !customToolNames.has(name)) return tool;
      return {
        type: "custom",
        name,
        description: tool.description || tool.function?.description,
      };
    });
  }

  if (!Array.isArray(request.input)) return;
  const customToolCallIds = new Set<string>();
  request.input = request.input.map((item: any) => {
    if (
      item?.type === "function_call" &&
      typeof item.name === "string" &&
      customToolNames.has(item.name)
    ) {
      const callId = item.call_id;
      if (typeof callId === "string" && callId) customToolCallIds.add(callId);
      return {
        type: "custom_tool_call",
        name: item.name,
        call_id: callId,
        input: unwrapCustomToolInput(
          typeof item.arguments === "string" ? item.arguments : ""
        ),
      };
    }
    if (
      item?.type === "function_call_output" &&
      typeof item.call_id === "string" &&
      customToolCallIds.has(item.call_id)
    ) {
      return {
        type: "custom_tool_call_output",
        call_id: item.call_id,
        output: item.output,
      };
    }
    return item;
  });
}

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

function getCodexSessionId(context: any, request: Record<string, any>): string | undefined {
  return (
    context?.protocolContext?.sessionId ||
    context?.req?.protocolContext?.sessionId ||
    context?.req?.sessionId ||
    extractClientSessionId({
      body: request,
      headers: context?.req?.headers,
    })
  );
}

function applyCodexTurnMetadata(
  request: Record<string, any>,
  headers: Record<string, string>,
  sessionId?: string
): void {
  if (!sessionId) {
    return;
  }

  const turnMetadata = buildCodexTurnMetadata(sessionId);
  request.client_metadata = mergeClientMetadata(
    request.client_metadata,
    buildCodexClientMetadata(turnMetadata)
  );
  Object.assign(headers, buildCodexCompatibilityHeaders(turnMetadata));
}

function stripCodexCacheBreakpoints(value: unknown): void {
  if (!value || typeof value !== "object") return;
  if (Array.isArray(value)) {
    for (const item of value) stripCodexCacheBreakpoints(item);
    return;
  }
  const record = value as Record<string, unknown>;
  delete record.prompt_cache_breakpoint;
  delete record.cache_control;
  for (const nested of Object.values(record)) {
    if (nested && typeof nested === "object") stripCodexCacheBreakpoints(nested);
  }
}

function applyCodexWireConstraints(
  request: Record<string, any>,
  provider: any,
  context: any
): void {
  delete request.temperature;
  delete request.max_tokens;
  delete request.max_completion_tokens;
  delete request.max_output_tokens;

  foldSystemItemsIntoInstructions(request);
  restoreCustomTools(request, getCustomToolNames(context));
  stripCodexCacheBreakpoints(request.input);
  stripCodexCacheBreakpoints(request.tools);

  const VALID_VERBOSITIES = ["low", "medium", "high"];
  if (provider?.verbosity && VALID_VERBOSITIES.includes(provider.verbosity)) {
    request.verbosity = provider.verbosity;
  }

  request.store = false;
  request.stream = true;
  request.parallel_tool_calls = provider?.parallelToolCalls === true;

  if (!request.prompt_cache_key) {
    const cacheKey = deriveCacheSessionKey(context, request as any);
    if (cacheKey) request.prompt_cache_key = cacheKey;
  }
}

function responsesJsonToSse(payload: any): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      const output = Array.isArray(payload?.output) ? payload.output : [];
      for (const item of output) {
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: "response.output_item.added", item })}\n\n`
          )
        );
        controller.enqueue(
          encoder.encode(
            `data: ${JSON.stringify({ type: "response.output_item.done", item })}\n\n`
          )
        );
      }
      controller.enqueue(
        encoder.encode(
          `data: ${JSON.stringify({ type: "response.completed", response: payload })}\n\n`
        )
      );
      controller.close();
    },
  });
}

/**
 * ChatGPT/Codex backend auth + Responses-wire constraints.
 *
 * Body conversion is owned by `openai-responses`. Configure
 * `transformer.use: ["openai-responses", "codex"]`. Same-protocol
 * Responses clients keep `input[]` (including `reasoning.encrypted_content`);
 * this transformer only stamps auth, Codex headers, `store: false`, and
 * `stream: true`.
 */
export class CodexTransformer implements Transformer {
  name = "codex";
  requestPhase = "headers" as const;
  logger?: any;
  // Per-request streaming intent. The Codex API requires stream:true, so the
  // outgoing call is always streaming — but we need to know whether the
  // caller wants streaming or non-streaming so transformResponseOut can
  // return JSON vs SSE. Streaming is opt-in (=== true).
  private streamIntent: Map<string, boolean> = new Map();

  async transformRequestIn(
    request: any,
    provider: any,
    context?: any
  ): Promise<Record<string, any>> {
    const body = structuredClone(request?.body && request?.config ? request.body : request);
    if (Array.isArray(body?.messages) && !Array.isArray(body?.input)) {
      throw new Error(CODEX_REQUIRES_RESPONSES);
    }

    const reqId = context?.req?.id;
    if (reqId) {
      this.streamIntent.set(reqId, body.stream === true);
    }

    const resolvedAuth = await this.resolveAuth(provider);
    applyCodexWireConstraints(body, provider, context);

    const headers = this.buildAuthHeaders(resolvedAuth);
    applyCodexTurnMetadata(
      body,
      headers,
      body.prompt_cache_key || getCodexSessionId(context, body)
    );

    const baseUrl = provider?.baseUrl || "https://chatgpt.com/backend-api/codex";

    return {
      body,
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

  /**
   * Transport quirks only. Do not convert Responses → Chat — openai-responses
   * owns that, and same-protocol keep must forward native `input[]` /
   * `encrypted_content` events unchanged.
   */
  async transformResponseOut(
    response: Response,
    context?: { req?: { id?: string } }
  ): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";
    const reqId = context?.req?.id;
    const prevIntent = reqId ? this.streamIntent.get(reqId) : undefined;
    try {
      return await this.normalizeCodexTransport(
        response,
        contentType,
        reqId ? prevIntent !== false : true
      );
    } finally {
      if (reqId) this.streamIntent.delete(reqId);
    }
  }

  private async normalizeCodexTransport(
    response: Response,
    contentType: string,
    wantsStream: boolean
  ): Promise<Response> {
    const asJson = (payload: unknown, status = response.status, statusText = response.statusText) =>
      new Response(JSON.stringify(payload), {
        status,
        statusText,
        headers: new Headers({ "Content-Type": "application/json" }),
      });
    const asSse = (stream: ReadableStream<Uint8Array>) =>
      new Response(stream, {
        status: response.status,
        statusText: response.statusText,
        headers: new Headers({
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        }),
      });

    // Cloudflare strips Content-Type from SSE responses — treat missing
    // content-type as SSE, then peek so a flat JSON object is not fed to
    // the SSE parser.
    if (!contentType || contentType.includes("text/event-stream")) {
      if (!response.body) return response;
      const peek = await peekResponseBody(response);
      if (peek.kind === "json") {
        let parsed: any;
        try {
          parsed = JSON.parse(peek.text);
        } catch {
          return wantsStream
            ? asSse(this.jsonToSseBytes(peek.text))
            : new Response(peek.text, {
                status: response.status,
                statusText: response.statusText,
                headers: new Headers({ "Content-Type": "application/json" }),
              });
        }
        if (parsed?.object === "response") {
          return wantsStream ? asSse(responsesJsonToSse(parsed)) : asJson(parsed);
        }
        return wantsStream ? asSse(this.jsonToSseBytes(peek.text)) : asJson(parsed);
      }
      if (peek.kind === "empty") {
        return new Response(null, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }
      response = peek.response;
      if (!wantsStream) {
        const payload = await this.collectSseIntoResponses(response);
        return asJson(payload);
      }
      return this.ensureSseContentType(response);
    }

    if (contentType.includes("application/json")) {
      const jsonResponse: any = await response.json();
      if (Array.isArray(jsonResponse)) {
        if (!wantsStream) {
          const completed = jsonResponse.find(
            (event: any) => event?.type === "response.completed"
          )?.response;
          if (completed) return asJson(completed);
        }
        const encoder = new TextEncoder();
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            for (const event of jsonResponse) {
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(event)}\n\n`)
              );
            }
            controller.close();
          },
        });
        if (!wantsStream) {
          const payload = await this.collectSseIntoResponses(
            new Response(stream, {
              headers: { "Content-Type": "text/event-stream" },
            })
          );
          return asJson(payload);
        }
        return asSse(stream);
      }
      if (jsonResponse?.object === "response") {
        return wantsStream ? asSse(responsesJsonToSse(jsonResponse)) : asJson(jsonResponse);
      }
      return asJson(jsonResponse);
    }

    return response;
  }

  private ensureSseContentType(response: Response): Response {
    const headers = new Headers(response.headers);
    if (!headers.get("Content-Type")) {
      headers.set("Content-Type", "text/event-stream");
    }
    return new Response(response.body, {
      status: response.status,
      statusText: response.statusText,
      headers,
    });
  }

  private jsonToSseBytes(text: string): ReadableStream<Uint8Array> {
    const encoder = new TextEncoder();
    return new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(`data: ${text}\n\n`));
        controller.close();
      },
    });
  }

  private async collectSseIntoResponses(response: Response): Promise<any> {
    if (!response.body) {
      return { object: "response", output: [] };
    }
    const reader = createSSEStreamReader(
      response,
      (line: string, ctx: StreamContext) => {
        ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
      }
    );
    const text = await new Response(reader.body).text();
    let completed: any;
    const output: any[] = [];
    for (const line of text.split("\n")) {
      if (!line.startsWith("data: ")) continue;
      const dataStr = line.slice(5).trim();
      if (!dataStr || dataStr === "[DONE]") continue;
      try {
        const event = JSON.parse(dataStr);
        if (event?.type === "response.output_item.done" && event.item) {
          output.push(event.item);
        }
        if (event?.type === "response.completed" && event.response) {
          completed = event.response;
        }
      } catch {
        // ignore malformed line
      }
    }
    if (completed) return completed;
    return { object: "response", output };
  }
}


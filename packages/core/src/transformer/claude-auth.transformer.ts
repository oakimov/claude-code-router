import { randomUUID } from "crypto";
import { UnifiedChatRequest } from "@/types/llm";
import { Transformer, TransformerContext } from "@/types/transformer";
import {
  getValidAccessToken,
  loadOrCreateDeviceId,
  loadTokens,
  refreshTokens,
  saveTokens,
} from "../utils/claude-auth";
import { HeaderRecord } from "../utils/headers";
import {
  applyClaudeBillingSystemBlock,
  applyClaudeSystemIdentity,
  prefixClaudeToolNames,
  relocateForeignSystemContent,
  unprefixClaudeToolNames,
  CC_USER_AGENT_ENTRYPOINT,
  CC_VERSION,
  normalizeSystemToArray,
} from "../utils/claude-billing";
import { createSSEStreamReader, StreamContext } from "../utils/stream";
import { buildAnthropicMessagesUrl } from "@/utils/anthropic-url";
import {
  AnthropicClientKind,
  readHeaderValue as readPolicyHeaderValue,
} from "../utils/anthropic-client-policy";
import {
  ClaudeModelCatalogEntry,
  catalogEntryHasCapability,
  catalogEntrySupportsThinking,
  lookupClaudeModelCatalogEntry,
  stripOneMillionContextMarker,
} from "../utils/claude-model-catalog";

/** Anthropic beta required for Claude subscription / Claude Code OAuth Bearer auth. */
export const CLAUDE_OAUTH_REQUIRED_BETA = "oauth-2025-04-20";

export function mergeAnthropicBetaValues(
  ...values: Array<string | undefined | null>
): string {
  const seen = new Set<string>();
  const merged: string[] = [];
  for (const value of values) {
    if (!value) continue;
    for (const part of value.split(",")) {
      const token = part.trim();
      if (!token) continue;
      const key = token.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(token);
    }
  }
  return merged.join(",");
}

/** Read a named header value from a Fastify/Node headers object (case-insensitive). */
export function readHeaderValue(
  headers: Record<string, unknown> | undefined,
  name: string
): string | undefined {
  if (!headers) return undefined;
  const want = name.toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() !== want) continue;
    if (value == null) return undefined;
    if (Array.isArray(value)) {
      const parts = value.filter((v) => v != null && String(v).length > 0);
      return parts.length ? parts.map(String).join(", ") : undefined;
    }
    const s = String(value);
    return s.length ? s : undefined;
  }
  return undefined;
}

/** True when the client's User-Agent identifies it as the genuine Claude Code CLI. */
export function isClaudeCodeClient(userAgent: string | undefined): boolean {
  return userAgent?.startsWith("claude-cli/") ?? false;
}

/**
 * Build outbound anthropic-beta for Claude subscription OAuth.
 *
 * - If the client sent anthropic-beta (e.g. Claude Code), merge with
 *   oauth-2025-04-20 (deduped, case-insensitive).
 * - Otherwise, send only oauth-2025-04-20.
 */
export function resolveClaudeAuthAnthropicBeta(input: {
  clientBeta?: string;
}): string {
  if (input.clientBeta?.trim()) {
    return mergeAnthropicBetaValues(
      input.clientBeta,
      CLAUDE_OAUTH_REQUIRED_BETA
    );
  }

  return CLAUDE_OAUTH_REQUIRED_BETA;
}

/**
 * Build outbound anthropic-beta for the non-Claude-Code (full synthesis)
 * branch, mirroring Claude Code's current model-driven beta selection. Model
 * capabilities come from the catalog; the CLI's one family-level exception
 * (the ordinary Haiku profile omits the Claude Code beta) is retained from the
 * decompiled `cui()` branch. Current CLI `ANTHROPIC_BETAS` values are appended
 * to the profile; OAuth callers still add their required OAuth beta at the auth
 * boundary.
 */
export function resolveClaudeAuthBetas(
  modelId: string | undefined,
  opts?: {
    envBeta?: string;
    includeOAuthBeta?: boolean;
    includeToolSearch?: boolean;
    includeEffort?: boolean;
    includeFallbackCredit?: boolean;
  }
): string {
  const { requestedOneMillion } = stripOneMillionContextMarker(modelId);
  const entry = lookupClaudeModelCatalogEntry(modelId);
  const cap = (capability: string) => catalogEntryHasCapability(entry, capability);

  const normalizedModel = stripOneMillionContextMarker(modelId).modelId.toLowerCase();
  // Claude Code's current beta catalog omits the attribution beta for Haiku
  // unless an agentic query explicitly re-adds it. The emulation path has no
  // agent-query marker, so follow the ordinary request profile.
  const betas: string[] = normalizedModel.includes("haiku")
    ? []
    : ["claude-code-20250219"];
  if (opts?.includeOAuthBeta !== false) betas.push("oauth-2025-04-20");
  if (requestedOneMillion) betas.push("context-1m-2025-08-07");
  if (catalogEntrySupportsThinking(entry)) {
    betas.push("interleaved-thinking-2025-05-14", "thinking-token-count-2026-05-13");
  }
  if (cap("context_management")) betas.push("context-management-2025-06-27");
  betas.push("prompt-caching-scope-2026-01-05");
  if (cap("mid_conv_system")) betas.push("mid-conversation-system-2026-04-07");
  if (opts?.includeToolSearch) betas.push("advanced-tool-use-2025-11-20");
  if (opts?.includeEffort && cap("effort")) betas.push("effort-2025-11-24");
  if (opts?.includeFallbackCredit) betas.push("fallback-credit-2026-06-01");

  const configuredBetas = opts?.envBeta ?? process.env.ANTHROPIC_BETAS;
  for (const beta of configuredBetas?.split(",") || []) {
    const normalized = beta.trim();
    if (normalized && !betas.includes(normalized)) betas.push(normalized);
  }

  return betas.join(",");
}

/**
 * Recreate the SDK's context marker only for models whose 1M window is a beta.
 * Native-1M models still accept the gateway picker suffix, but must not receive
 * the legacy context beta upstream.
 */
export function modelIdForRequestedOneMillionBeta(
  modelId: string | undefined,
  requestedOneMillion: boolean | undefined
): string | undefined {
  if (!modelId || !requestedOneMillion) return modelId;
  const bareModelId = stripOneMillionContextMarker(modelId).modelId;
  const entry = lookupClaudeModelCatalogEntry(bareModelId);
  return entry?.nativeOneMillion ? bareModelId : `${bareModelId}[1m]`;
}

/**
 * Reshape a built Anthropic body's `thinking`/`output_config`/`max_tokens`
 * to what the resolved model actually supports, replacing a hand-rolled
 * per-model denylist with a single catalog-driven pass. Operates on the
 * post-build Anthropic body (not the Unified request) because
 * `buildAnthropicBody` may synthesize `thinking`/`output_config` itself.
 */
export function applyClaudeModelCapabilityAdjustments(
  anthropicBody: Record<string, any>,
  entry: ClaudeModelCatalogEntry | undefined
): void {
  const cap = (capability: string) => catalogEntryHasCapability(entry, capability);

  const stripEffort = (container: Record<string, any> | undefined) => {
    if (container && typeof container === "object" && !cap("effort")) {
      delete container.effort;
    }
  };
  stripEffort(anthropicBody.thinking);
  stripEffort(anthropicBody.output_config);

  if (
    anthropicBody.thinking &&
    typeof anthropicBody.thinking === "object" &&
    anthropicBody.thinking.type !== "disabled"
  ) {
    anthropicBody.thinking = cap("adaptive_thinking")
      ? { type: "adaptive", display: "omitted" }
      : {
          type: "enabled",
          ...(anthropicBody.thinking.budget_tokens !== undefined
            ? { budget_tokens: anthropicBody.thinking.budget_tokens }
            : {}),
          display: "omitted",
        };
  }

  if (entry && typeof anthropicBody.max_tokens === "number") {
    anthropicBody.max_tokens = Math.min(anthropicBody.max_tokens, entry.maxOutputTokens.upper);
  }
}

const STAINLESS_PACKAGE_VERSION = "0.94.0";

function stainlessArch(): string {
  switch (process.arch) {
    case "arm64":
      return "arm64";
    case "x64":
      return "x64";
    default:
      return process.arch;
  }
}

function stainlessOs(): string {
  switch (process.platform) {
    case "darwin":
      return "MacOS";
    case "linux":
      return "Linux";
    case "win32":
      return "Windows";
    default:
      return process.platform;
  }
}

let cachedSessionId: string | undefined;

/** Module-level session id, cached per process like Claude Code's own. */
function claudeAuthSessionId(): string {
  if (!cachedSessionId) cachedSessionId = randomUUID();
  return cachedSessionId;
}

const SYNTHESIZED_CUSTOM_HEADER_DENYLIST = new Set([
  "authorization",
  "x-api-key",
  "anthropic-beta",
  "cookie",
  "set-cookie",
  "host",
  "content-length",
  "content-encoding",
  "connection",
  "transfer-encoding",
  "upgrade",
  "te",
  "trailer",
]);

/** Parse the current CLI's newline-delimited custom application headers. */
function readSynthesizedCustomHeaders(): HeaderRecord {
  const custom: HeaderRecord = {};
  for (const line of (process.env.ANTHROPIC_CUSTOM_HEADERS || "").split(/\r?\n/)) {
    if (!line.trim()) continue;
    const separator = line.indexOf(":");
    if (separator < 0) continue;
    const name = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    if (!name || !value || SYNTHESIZED_CUSTOM_HEADER_DENYLIST.has(name.toLowerCase())) {
      continue;
    }
    custom[name] = value;
  }
  return custom;
}

/** Test-only reset hook so session-id state doesn't leak across test cases. */
export function __resetClaudeAuthTransformerStateForTests(): void {
  cachedSessionId = undefined;
}

/** Synthesized Claude Code identity headers for the non-Claude-Code branch. */
export function buildSynthesizedIdentityHeaders(): HeaderRecord {
  const userAgentSuffix = [
    CC_USER_AGENT_ENTRYPOINT,
    process.env.CLAUDE_AGENT_SDK_VERSION
      ? `agent-sdk/${process.env.CLAUDE_AGENT_SDK_VERSION}`
      : undefined,
    process.env.CLAUDE_AGENT_SDK_CLIENT_APP
      ? `client-app/${process.env.CLAUDE_AGENT_SDK_CLIENT_APP}`
      : undefined,
  ]
    .filter(Boolean)
    .join(", ");
  const userAgent =
    process.env.ANTHROPIC_USER_AGENT ||
    `claude-cli/${CC_VERSION} (external, ${userAgentSuffix})`;
  return {
    "User-Agent": userAgent,
    "x-app": "cli",
    "anthropic-dangerous-direct-browser-access": "true",
    "X-Claude-Code-Session-Id": claudeAuthSessionId(),
    "x-client-request-id": randomUUID(),
    "x-stainless-arch": stainlessArch(),
    "x-stainless-lang": "js",
    "x-stainless-os": stainlessOs(),
    "x-stainless-package-version": STAINLESS_PACKAGE_VERSION,
    "x-stainless-retry-count": "0",
    "x-stainless-runtime": "node",
    "x-stainless-runtime-version": process.version,
    "x-stainless-timeout": "600",
    ...(process.env.CLAUDE_CODE_CONTAINER_ID
      ? { "x-claude-remote-container-id": process.env.CLAUDE_CODE_CONTAINER_ID }
      : {}),
    ...(process.env.CLAUDE_CODE_REMOTE_SESSION_ID
      ? { "x-claude-remote-session-id": process.env.CLAUDE_CODE_REMOTE_SESSION_ID }
      : {}),
    ...(process.env.CLAUDE_AGENT_SDK_CLIENT_APP
      ? { "x-client-app": process.env.CLAUDE_AGENT_SDK_CLIENT_APP }
      : {}),
    ...(process.env.CLAUDE_CODE_ADDITIONAL_PROTECTION &&
    !["0", "false", "no", "off"].includes(
      process.env.CLAUDE_CODE_ADDITIONAL_PROTECTION.trim().toLowerCase()
    )
      ? { "x-anthropic-additional-protection": "true" }
      : {}),
    ...readSynthesizedCustomHeaders(),
  };
}

export function buildSynthesizedUserMetadata(): Record<string, string> {
  return {
    user_id: JSON.stringify({
      device_id: loadOrCreateDeviceId(),
      account_uuid: "",
      session_id: claudeAuthSessionId(),
    }),
  };
}

export class ClaudeAuthTransformer implements Transformer {
  name = "claude-auth";
  logger?: any;

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any,
    context?: TransformerContext
  ): Promise<Record<string, any>> {
    const creds = await getValidAccessToken();

    const clientHeaders = context?.req?.headers as Record<string, unknown> | undefined || {};
    const clientUserAgent = readHeaderValue(clientHeaders, "user-agent");
    const clientKind: AnthropicClientKind =
      context?.protocolContext?.anthropicClientKind ||
      (isClaudeCodeClient(clientUserAgent) ? "claude_code" : "other");
    const isClaudeCode = clientKind === "claude_code";
    // The route pipeline always supplies destination scope. Direct legacy
    // transformer callers have no destination context, so retain their
    // historical behavior for compatibility; routed non-Anthropic requests
    // explicitly fail this predicate and receive no Claude system synthesis.
    const mayApplyLegacyPolicy =
      !context?.protocolContext ||
      context.protocolContext.anthropicDestinationInScope === true;

    if (
      clientKind === "other" &&
      mayApplyLegacyPolicy &&
      !context?.protocolContext?.anthropicPolicyApplied
    ) {
      // Non-Claude-Code branch runs on the Unified body, before
      // AnthropicTransformer builds the wire body, so system[] ends up as
      // the single source of truth (buildAnthropicBody prefers
      // request.system over a role:"system" message and would otherwise
      // silently drop one).
      // Billing must land at system[0] before identity is inserted at
      // system[1] — applyClaudeSystemIdentity's insertion index assumes
      // billing is already in place, otherwise the caller's first entry
      // ends up sandwiched between billing and identity instead of
      // following it (see plan Step 3's [billing, identity, ...caller]
      // invariant).
      const system = normalizeSystemToArray(request);
      applyClaudeBillingSystemBlock(system, request.messages);
      applyClaudeSystemIdentity(system);
      // Anthropic's OAuth billing validator rejects requests whose system[]
      // carries a foreign harness prompt past the identity block; relocate
      // it into the first user message so it still reaches the model.
      relocateForeignSystemContent(system, request.messages);
      // Claude Code's OAuth validator also expects tool names in its
      // mcp_PascalCase spelling. Keep a request-local reverse map for the
      // response transformer so the caller receives its original names.
      const toolNameMap = new Map<string, string>();
      prefixClaudeToolNames(request, toolNameMap);
      if (context) {
        context.claudeAuthToolNameMap = toolNameMap;
        if (context.protocolContext) {
          context.protocolContext.claudeAuthToolNameMap = toolNameMap;
        }
      }
    } else if (clientKind === "other") {
      const toolNameMap = context?.protocolContext?.claudeAuthToolNameMap;
      if (toolNameMap && context) context.claudeAuthToolNameMap = toolNameMap;
    }

    const clientBeta = isClaudeCode
      ? readHeaderValue(clientHeaders, "anthropic-beta")
      : undefined;
    // Claude Code passes `betas` to the Anthropic SDK, whose Messages resource
    // removes that SDK-only option from the JSON body and serializes it as the
    // `anthropic-beta` header. CCR builds the raw HTTP request itself, so the
    // synthesized profile must reproduce the resulting wire shape directly.
    const anthropicBeta = isClaudeCode
      ? resolveClaudeAuthAnthropicBeta({ clientBeta })
      : resolveClaudeAuthAnthropicBeta({
          clientBeta: resolveClaudeAuthBetas(
            modelIdForRequestedOneMillionBeta(
              request.model,
              context?.protocolContext?.requestedOneMillion
            )
          ),
        });

    const headers: HeaderRecord = {
      Authorization: `Bearer ${creds.access_token}`,
      "anthropic-beta": anthropicBeta,
    };

    if (isClaudeCode) {
      // Forward Claude Code identity headers verbatim.
      for (const name of [
        "user-agent",
        "x-app",
        "x-claude-code-session-id",
        "anthropic-dangerous-direct-browser-access",
        "x-client-request-id",
        "x-stainless-arch",
        "x-stainless-lang",
        "x-stainless-os",
        "x-stainless-package-version",
        "x-stainless-retry-count",
        "x-stainless-runtime",
        "x-stainless-runtime-version",
        "x-stainless-timeout",
      ]) {
        const value = readHeaderValue(clientHeaders, name);
        if (value) headers[name] = value;
      }
    } else {
      Object.assign(headers, buildSynthesizedIdentityHeaders());

      // AnthropicTransformer owns building the wire body; hand it the
      // catalog-driven capability clamp and synthesized user_id metadata to
      // apply immediately after, preserving today's post-build ordering.
      const catalogEntry = lookupClaudeModelCatalogEntry(request.model);
      if (context) {
        context.claudeAuthPostBuildHook = (anthropicBody: Record<string, any>) => {
          applyClaudeModelCapabilityAdjustments(anthropicBody, catalogEntry);
          anthropicBody.metadata = {
            ...(anthropicBody.metadata || {}),
            ...buildSynthesizedUserMetadata(),
          };
        };
      }
    }

    return {
      body: request,
      config: {
        headers,
        __authRecovery: () => this.recoverUnauthorizedAuth(creds.access_token),
      },
    };
  }

  /**
   * Auth-only path used by native Desktop/CLI raw-wire requests. It keeps the
   * original body and application headers untouched while still replacing the
   * caller credential with CCR's OAuth token.
   */
  async auth(
    request: any,
    provider: any,
    context?: TransformerContext
  ): Promise<any> {
    const creds = await getValidAccessToken();
    const clientHeaders =
      (context?.req?.headers as Record<string, unknown> | undefined) || {};
    const clientBeta = readPolicyHeaderValue(clientHeaders, "anthropic-beta");
    return {
      body: request,
      config: {
        url: buildAnthropicMessagesUrl(provider?.baseUrl),
        headers: {
          Authorization: `Bearer ${creds.access_token}`,
          "anthropic-beta": resolveClaudeAuthAnthropicBeta({ clientBeta }),
        },
        __authRecovery: () => this.recoverUnauthorizedAuth(creds.access_token),
      },
    };
  }

  /**
   * Body/URL/wire-format conversion belong to AnthropicTransformer's
   * provider pair, which already ran (response-side order is reversed, so
   * it runs before this stage). This stage only inspects the resulting
   * response for subscription-specific overage observability.
   */
  async transformResponseOut(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    // Response processing receives a fresh context object, while the request
    // context's protocolContext is deliberately carried forward. Read the
    // request-local map from both locations for direct transformer callers and
    // the normal route pipeline respectively.
    const nameMap = (context?.claudeAuthToolNameMap ??
      context?.protocolContext?.claudeAuthToolNameMap) as
      | Map<string, string>
      | undefined;
    if (nameMap?.size && response.ok) {
      const contentType = response.headers.get("Content-Type") || "";
      if (contentType.includes("application/json")) {
        const body = await response.json();
        unprefixClaudeToolNames(body, nameMap);
        return new Response(JSON.stringify(body), {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        });
      }
      if (contentType.includes("text/event-stream") && response.body) {
        return createSSEStreamReader(
          response,
          (line: string, streamContext: StreamContext) => {
            if (!line.startsWith("data: ") || line.trim() === "data: [DONE]") {
              streamContext.controller.enqueue(streamContext.encoder.encode(line + "\n"));
              return;
            }
            try {
              const payload = JSON.parse(line.slice(6));
              unprefixClaudeToolNames(payload, nameMap);
              streamContext.controller.enqueue(
                streamContext.encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)
              );
            } catch {
              streamContext.controller.enqueue(streamContext.encoder.encode(line + "\n"));
            }
          },
          { logger: this.logger }
        );
      }
    }

    const overageInUse = response.headers.get(
      "anthropic-ratelimit-unified-overage-in-use"
    );
    if (overageInUse) {
      this.logger?.debug?.(
        {
          overageInUse,
          overageStatus: response.headers.get(
            "anthropic-ratelimit-unified-overage-status"
          ),
        },
        "claude-auth: subscription overage in use for this request"
      );
    }
    return response;
  }

  /**
   * 401 recovery: reload the token file in case another process (e.g. a
   * concurrent `ccr claude-auth` re-login) rotated it externally, otherwise
   * refresh and persist. Never falls through to an unauthenticated request.
   */
  private async recoverUnauthorizedAuth(
    previousAccessToken: string
  ): Promise<Record<string, string> | null> {
    const reloaded = loadTokens();
    if (reloaded?.access_token && reloaded.access_token !== previousAccessToken) {
      return { Authorization: `Bearer ${reloaded.access_token}` };
    }

    if (!reloaded?.refresh_token) return null;

    const refreshed = await refreshTokens(reloaded.refresh_token);
    saveTokens(refreshed);
    return { Authorization: `Bearer ${refreshed.access_token}` };
  }
}

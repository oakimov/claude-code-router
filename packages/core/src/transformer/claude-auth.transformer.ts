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
  applyClaudeSystemIdentity,
  fillClaudeBillingSystemBlock,
  reserveClaudeBillingSystemBlock,
  prefixClaudeToolNames,
  relocateForeignSystemContent,
  restoreClaudeToolNamesInResponse,
  CC_USER_AGENT_ENTRYPOINT,
  CC_VERSION,
  normalizeSystemToArray,
} from "../utils/claude-billing";
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

/**
 * Beta tokens a third-party client sent for its Anthropic-defined computer use
 * tool (e.g. `computer-use-2025-11-24` for `computer_20251124`). The emulated
 * profile otherwise ignores client betas; the client already chose the one
 * matching its tool version, so only that family is carried over, and only
 * when the request actually declares a `computer_*` tool.
 */
export function clientComputerUseBetas(
  clientBeta: string | undefined,
  typedTools: Array<Record<string, any>> | undefined
): string | undefined {
  if (
    !clientBeta ||
    !typedTools?.some((tool) => String(tool?.type).startsWith("computer_"))
  ) {
    return undefined;
  }
  const tokens = clientBeta
    .split(",")
    .map((part) => part.trim())
    .filter((token) => token.toLowerCase().startsWith("computer-use-"));
  return tokens.length ? tokens.join(",") : undefined;
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
  entry: ClaudeModelCatalogEntry | undefined,
  logger?: any
): void {
  const cap = (capability: string) => catalogEntryHasCapability(entry, capability);
  const constraints = entry?.apiConstraints;
  // Read before stripEffort: a model without the effort capability still needs
  // the requested effort to size a manual thinking budget below.
  const requestedEffort =
    anthropicBody.output_config?.effort ?? anthropicBody.thinking?.effort;

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
    anthropicBody.thinking.type === "disabled" &&
    constraints?.thinkingAlwaysOn
  ) {
    // These models return 400 for disabled thinking (and for manual
    // enabled-budget thinking, which the branch below already normalizes):
    // thinking is always on there, effort is the only knob.
    anthropicBody.thinking = { type: "adaptive", display: "summarized" };
  }

  if (constraints?.noSamplingParams) {
    // Sampling knobs are removed on these models and return 400 whether or
    // not thinking is on; Claude Code never sends them.
    const dropped = ["temperature", "top_p", "top_k"].filter(
      (field) => anthropicBody[field] !== undefined
    );
    for (const field of dropped) delete anthropicBody[field];
    if (dropped.length > 0) {
      logger?.debug?.(
        { model: anthropicBody.model, dropped },
        "claude-auth: dropped sampling params the model rejects"
      );
    }
  }

  if (
    anthropicBody.tool_choice &&
    typeof anthropicBody.tool_choice === "object" &&
    (anthropicBody.tool_choice.type === "any" ||
      anthropicBody.tool_choice.type === "tool") &&
    constraints?.noForcedToolChoice
  ) {
    // Forced tool use returns 400 on these models. `auto` is the only
    // accepted mode, so the forcing guarantee is lost: the model may answer
    // in text or pick another tool. Anthropic's suggested substitute (an
    // instruction naming the tool) is not injected here: a per-request edit
    // to the user turn would be absent from the client's replayed history,
    // which preserved thinking rejects as an edited transcript.
    const forced = anthropicBody.tool_choice;
    logger?.warn?.(
      {
        model: anthropicBody.model,
        toolChoice: forced.type,
        tool: forced.type === "tool" ? forced.name : undefined,
      },
      "claude-auth: forced tool_choice unsupported by model; downgraded to auto"
    );
    const { disable_parallel_tool_use } = forced;
    anthropicBody.tool_choice = {
      ...(typeof disable_parallel_tool_use === "boolean"
        ? { disable_parallel_tool_use }
        : {}),
      type: "auto",
    };
  }

  if (entry && typeof anthropicBody.max_tokens === "number") {
    anthropicBody.max_tokens = Math.min(anthropicBody.max_tokens, entry.maxOutputTokens.upper);
  }

  if (
    anthropicBody.thinking &&
    typeof anthropicBody.thinking === "object" &&
    anthropicBody.thinking.type !== "disabled"
  ) {
    // Claude Code uses display:"summarized". "omitted" returns signature-only
    // empty thinking blocks — Chat Completions / OpenAI-compatible clients then
    // see no reasoning_content. Keep summarized so OAuth-proxied third parties
    // receive visible thinking text.
    if (cap("adaptive_thinking")) {
      anthropicBody.thinking = { type: "adaptive", display: "summarized" };
    } else {
      // Manual thinking requires budget_tokens (400 "thinking.enabled.
      // budget_tokens: Field required"). Effort-only clients (Responses/Chat
      // reasoning.effort) arrive as adaptive with no budget, so derive one.
      const budget = manualThinkingBudget(
        anthropicBody.thinking.budget_tokens,
        requestedEffort,
        anthropicBody.max_tokens
      );
      if (budget === undefined) {
        delete anthropicBody.thinking;
      } else {
        anthropicBody.thinking = {
          type: "enabled",
          budget_tokens: budget,
          display: "summarized",
        };
      }
    }
  }
}

/** Anthropic's minimum manual thinking budget. */
const MIN_THINKING_BUDGET = 1024;

/**
 * Budget for manual (`enabled`) thinking: the client's own budget when sent,
 * else an effort share of max_tokens (as in gemini-thinking's budget dialect).
 * It must stay below max_tokens and at or above the 1024 floor; when both
 * cannot hold, thinking is dropped rather than sent invalid.
 */
function manualThinkingBudget(
  clientBudget: unknown,
  effort: unknown,
  maxTokens: unknown
): number | undefined {
  const max = typeof maxTokens === "number" ? maxTokens : undefined;
  let budget: number;
  if (typeof clientBudget === "number") {
    budget = clientBudget;
  } else {
    if (max === undefined) return undefined;
    const share =
      effort === "minimal" ? 0.1 : effort === "low" ? 0.25 : effort === "medium" ? 0.5 : 1;
    budget = Math.round((max - 1) * share);
  }
  if (max !== undefined && budget >= max) budget = max - 1;
  if (budget < MIN_THINKING_BUDGET) {
    return max !== undefined && max - 1 >= MIN_THINKING_BUDGET
      ? MIN_THINKING_BUDGET
      : undefined;
  }
  return budget;
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
      const billingBlock = reserveClaudeBillingSystemBlock(system);
      applyClaudeSystemIdentity(system);
      // Anthropic's OAuth billing validator rejects requests whose system[]
      // carries a foreign harness prompt past the identity block; relocate
      // it into the first user message so it still reaches the model.
      // Unlike applyThirdPartyAnthropicPolicy, this legacy path (direct
      // transformer callers without route policy) authors no cache profile,
      // so a string first user message is prefixed in place and the relocated
      // prompt gets no breakpoint of its own.
      relocateForeignSystemContent(system, request.messages);
      // Compute the billing suffix from the relocated first user text (the
      // body actually sent), which stays stable when early turns are dropped.
      fillClaudeBillingSystemBlock(system, billingBlock, request.messages);
      // Claude Code's OAuth validator also expects tool names in its
      // mcp_PascalCase spelling. Keep a request-local reverse map for the
      // response transformer so the caller receives its original names.
      const toolNameMap = new Map<string, string>();
      prefixClaudeToolNames(
        request,
        toolNameMap,
        (request.anthropic_tools ?? []).map((tool) => tool?.name)
      );
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
          clientBeta: mergeAnthropicBetaValues(
            resolveClaudeAuthBetas(
              modelIdForRequestedOneMillionBeta(
                request.model,
                context?.protocolContext?.requestedOneMillion
              )
            ),
            clientComputerUseBetas(
              readHeaderValue(clientHeaders, "anthropic-beta"),
              context?.protocolContext?.anthropicSource?.tools ??
                request.anthropic_tools
            )
          ),
        });

    const headers: HeaderRecord = {
      Authorization: `Bearer ${creds.access_token}`,
      "anthropic-beta": anthropicBeta,
    };

    if (isClaudeCode) {
      // Forward Claude Code identity headers verbatim, including the
      // opt-in LLM-gateway hint headers (2.1.273+, CLAUDE_CODE_GATEWAY_HINT_HEADERS=1).
      for (const name of [
        "user-agent",
        "x-app",
        "x-claude-code-session-id",
        "x-claude-code-request-class",
        "x-claude-code-agent-type",
        "x-claude-code-prev-tool-durations",
        "x-claude-code-compaction",
        "x-claude-code-context-compacted",
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
          applyClaudeModelCapabilityAdjustments(anthropicBody, catalogEntry, this.logger);
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
   * it runs before this stage). This stage inspects the resulting response
   * for subscription-specific overage observability and restores tool names
   * its own legacy branch renamed.
   */
  async transformResponseOut(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    // Names renamed by the route's third-party policy are restored by the
    // route itself (it alone knows whether this response is Unified or exact
    // Anthropic wire). Only the legacy branch's own renames are undone here.
    // Response processing receives a fresh context object, so read the map
    // from both locations for direct callers and protocol-context callers.
    const nameMap = context?.protocolContext?.anthropicPolicyApplied
      ? undefined
      : ((context?.claudeAuthToolNameMap ??
          context?.protocolContext?.claudeAuthToolNameMap) as
          | Map<string, string>
          | undefined);

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
    return nameMap
      ? restoreClaudeToolNamesInResponse(response, nameMap, this.logger)
      : response;
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

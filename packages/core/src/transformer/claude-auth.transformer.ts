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
  CC_ENTRYPOINT,
  CC_VERSION,
  normalizeSystemToArray,
} from "../utils/claude-billing";
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
 * branch, mirroring Claude Code's own model-driven beta selection. Never
 * derived from model-name prefix matching — always from the capability
 * catalog. `ANTHROPIC_BETA_FLAGS` replaces the list wholesale.
 */
export function resolveClaudeAuthBetas(
  modelId: string | undefined,
  opts?: { envBeta?: string }
): string {
  const envBeta = opts?.envBeta ?? process.env.ANTHROPIC_BETA_FLAGS;
  if (envBeta?.trim()) return envBeta.trim();

  const { requestedOneMillion } = stripOneMillionContextMarker(modelId);
  const entry = lookupClaudeModelCatalogEntry(modelId);
  const cap = (capability: string) => catalogEntryHasCapability(entry, capability);

  const betas: string[] = ["claude-code-20250219", "oauth-2025-04-20"];
  if (requestedOneMillion) betas.push("context-1m-2025-08-07");
  if (catalogEntrySupportsThinking(entry)) {
    betas.push("interleaved-thinking-2025-05-14", "thinking-token-count-2026-05-13");
  }
  if (cap("context_management")) betas.push("context-management-2025-06-27");
  betas.push("prompt-caching-scope-2026-01-05");
  if (cap("mid_conv_system")) betas.push("mid-conversation-system-2026-04-07");
  betas.push("advanced-tool-use-2025-11-20");
  if (cap("effort")) betas.push("effort-2025-11-24");
  if (cap("fast_mode")) betas.push("fallback-credit-2026-06-01");

  return betas.join(",");
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

/** Test-only reset hook so session-id state doesn't leak across test cases. */
export function __resetClaudeAuthTransformerStateForTests(): void {
  cachedSessionId = undefined;
}

/** Synthesized Claude Code identity headers for the non-Claude-Code branch. */
function buildSynthesizedIdentityHeaders(): HeaderRecord {
  const userAgent =
    process.env.ANTHROPIC_USER_AGENT || `claude-cli/${CC_VERSION} (external, ${CC_ENTRYPOINT})`;
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
    const isClaudeCode = isClaudeCodeClient(clientUserAgent);

    if (!isClaudeCode) {
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
    }

    const clientBeta = isClaudeCode
      ? readHeaderValue(clientHeaders, "anthropic-beta")
      : undefined;
    const anthropicBeta = isClaudeCode
      ? resolveClaudeAuthAnthropicBeta({ clientBeta })
      : resolveClaudeAuthBetas(request.model);

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
          const deviceId = loadOrCreateDeviceId();
          anthropicBody.metadata = {
            ...(anthropicBody.metadata || {}),
            user_id: JSON.stringify({
              device_id: deviceId,
              account_uuid: "",
              session_id: claudeAuthSessionId(),
            }),
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
   * Body/URL/wire-format conversion belong to AnthropicTransformer's
   * provider pair, which already ran (response-side order is reversed, so
   * it runs before this stage). This stage only inspects the resulting
   * response for subscription-specific overage observability.
   */
  async transformResponseOut(response: Response): Promise<Response> {
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

import { UnifiedChatRequest } from "@/types/llm";
import { Transformer, TransformerContext } from "@/types/transformer";
import { getValidAccessToken } from "../utils/claude-auth";
import { transformResponseOut } from "../utils/vertex-claude.util";
import { AnthropicTransformer } from "./anthropic.transformer";

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

/**
 * Build outbound anthropic-beta for Claude subscription OAuth.
 *
 * - If the client sent anthropic-beta (e.g. Claude Code), merge with
 *   oauth-2025-04-20 (deduped, case-insensitive).
 * - Otherwise, send only oauth-2025-04-20. Do not synthesise Claude Code
 *   betas — Anthropic validates the attestation on claude-code-20250219
 *   and subscription OAuth works without it for non-Claude-Code clients.
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

export class ClaudeAuthTransformer implements Transformer {
  name = "claude-auth";
  logger?: any;

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any,
    context?: TransformerContext
  ): Promise<Record<string, any>> {
    const creds = await getValidAccessToken();
    const baseUrl =
      provider?.api_base_url ?? provider?.baseUrl ?? "https://api.anthropic.com";
    const url = baseUrl.endsWith("/v1/messages")
      ? baseUrl
      : `${baseUrl.replace(/\/$/, "")}/v1/messages`;

    // Safely append ?beta=true using URL constructor to handle existing query params
    const requestUrl = new URL(url);
    requestUrl.searchParams.set("beta", "true");

    const anthropicBody = AnthropicTransformer.buildAnthropicBody(
      request,
      this.logger
    );

    const clientHeaders = context?.req?.headers as Record<string, unknown> | undefined || {};
    const clientUserAgent = readHeaderValue(clientHeaders, "user-agent");
    const isClaudeCode = clientUserAgent?.startsWith("claude-cli/") ?? false;

    const clientBeta = isClaudeCode
      ? readHeaderValue(clientHeaders, "anthropic-beta")
      : undefined;
    const anthropicBeta = resolveClaudeAuthAnthropicBeta({
      clientBeta,
    });

    const headers: Record<string, string> = {
      Authorization: `Bearer ${creds.access_token}`,
      "Content-Type": "application/json",
      "anthropic-version":
        (isClaudeCode && readHeaderValue(clientHeaders, "anthropic-version")) ||
        "2023-06-01",
      "anthropic-beta": anthropicBeta,
    };

    if (isClaudeCode) {
      // Forward Claude Code identity headers verbatim.
      for (const name of [
        "user-agent",
        "x-app",
        "x-claude-code-session-id",
        "anthropic-dangerous-direct-browser-access",
        "x-anthropic-billing-header",
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
      // Non-Claude-Code client: send only required headers.
      // Do NOT synthesise Claude Code identity headers — Anthropic validates
      // attestation tokens server-side, so mocked headers still get 429'd.
      // The subscription OAuth Bearer token is sufficient for authentication.
      if (clientUserAgent) headers["User-Agent"] = clientUserAgent;
    }

    return {
      body: anthropicBody,
      config: {
        url: requestUrl.toString(),
        headers,
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    return transformResponseOut(response, this.name, this.logger);
  }
}

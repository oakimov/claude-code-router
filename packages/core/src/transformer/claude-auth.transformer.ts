import { UnifiedChatRequest } from "@/types/llm";
import { Transformer } from "@/types/transformer";
import { getValidAccessToken } from "../utils/claude-auth";
import { transformResponseOut } from "../utils/vertex-claude.util";
import { AnthropicTransformer } from "./anthropic.transformer";

/** Anthropic beta required for Claude subscription / Claude Code OAuth Bearer auth. */
const CLAUDE_OAUTH_REQUIRED_BETA = "oauth-2025-04-20";

function hasCacheControl(value: unknown): boolean {
  if (!value || typeof value !== "object") {
    return false;
  }

  if (Array.isArray(value)) {
    return value.some((item) => hasCacheControl(item));
  }

  if ((value as Record<string, any>).cache_control) {
    return true;
  }

  return Object.values(value as Record<string, unknown>).some((item) =>
    hasCacheControl(item)
  );
}

function mergeAnthropicBetaValues(
  ...values: Array<string | undefined>
): string {
  const seen = new Set<string>();
  const merged: string[] = [];
  for (const value of values) {
    if (!value) continue;
    for (const part of value.split(",")) {
      const token = part.trim();
      if (!token || seen.has(token)) continue;
      seen.add(token);
      merged.push(token);
    }
  }
  return merged.join(",");
}

export class ClaudeAuthTransformer implements Transformer {
  name = "claude-auth";
  logger?: any;

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any
  ): Promise<Record<string, any>> {
    const creds = await getValidAccessToken();
    const baseUrl = provider?.api_base_url ?? provider?.baseUrl ?? "https://api.anthropic.com";
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

    const headers: Record<string, string> = {
      Authorization: `Bearer ${creds.access_token}`,
      "anthropic-version": "2023-06-01",
      "Content-Type": "application/json",
      "User-Agent": "claude-cli/2.1.195 (external, cli)",
    };

    const usesThinking =
      request.thinking?.type === "enabled" ||
      request.thinking?.type === "adaptive" ||
      request.enable_thinking ||
      request.anthropic_thinking;
    const usesPromptCaching = hasCacheControl(anthropicBody);

    const featureBetas =
      usesThinking || usesPromptCaching
        ? "interleaved-thinking-2025-05-14,effort-2025-11-24,prompt-caching-scope-2026-01-05"
        : undefined;

    // Always declare OAuth subscription beta for Bearer auth; merge feature betas when needed.
    headers["anthropic-beta"] = mergeAnthropicBetaValues(
      featureBetas,
      CLAUDE_OAUTH_REQUIRED_BETA
    );

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

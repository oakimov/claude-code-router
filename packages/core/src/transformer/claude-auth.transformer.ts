import { UnifiedChatRequest } from "@/types/llm";
import { Transformer } from "@/types/transformer";
import { getValidAccessToken } from "../utils/claude-auth";
import { transformResponseOut } from "../utils/vertex-claude.util";
import { AnthropicTransformer } from "./anthropic.transformer";

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

    // Add anthropic-beta when the rebuilt Anthropic payload uses beta-only features.
    if (usesThinking || usesPromptCaching) {
      headers["anthropic-beta"] =
        "interleaved-thinking-2025-05-14,effort-2025-11-24,prompt-caching-scope-2026-01-05";
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

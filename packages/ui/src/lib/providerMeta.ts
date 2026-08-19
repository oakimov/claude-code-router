import type { Provider } from "@/types";

export function getNormalizedHost(apiBaseUrl?: string): string {
  if (!apiBaseUrl) return "";
  try {
    const url = new URL(apiBaseUrl);
    return url.host || apiBaseUrl;
  } catch {
    return apiBaseUrl;
  }
}

export function getProviderTitle(provider: Provider): string {
  return provider.display_name?.trim() || provider.name || "";
}

export function getProviderDescription(provider: Provider): string {
  return provider.description?.trim() || "";
}

export function getAuthHintKey(provider: Provider):
  | "claude_oauth"
  | "qwen_jwt"
  | "codex_pat"
  | "codex_oauth"
  | "antigravity_oauth"
  | "xai_oauth"
  | "missing_api_key"
  | "env_auth"
  | null {
  const apiKey = provider.api_key?.trim() || "";
  const transformers = Array.isArray(provider.transformer?.use)
    ? provider.transformer.use.map((item) =>
        Array.isArray(item) ? String(item[0]) : String(item)
      )
    : [];

  if (transformers.includes("claude-auth")) {
    return "claude_oauth";
  }
  if (transformers.includes("qwen-auth")) {
    return "qwen_jwt";
  }
  if (transformers.includes("antigravity-auth")) {
    return "antigravity_oauth";
  }
  if (transformers.includes("xai-auth")) {
    return "xai_oauth";
  }
  if (transformers.includes("codex")) {
    return apiKey.startsWith("at-") ? "codex_pat" : "codex_oauth";
  }
  if (!apiKey) {
    return "missing_api_key";
  }
  if (apiKey.startsWith("$") || apiKey.startsWith("${")) {
    return "env_auth";
  }
  return null;
}

export function getProviderTags(provider: Provider, authHint?: string | null): string[] {
  const tags = Array.isArray(provider.tags) ? provider.tags.filter(Boolean) : [];
  if (authHint) {
    tags.unshift(authHint);
  }
  return Array.from(new Set(tags)).slice(0, 4);
}

export function getTemplateOptionLabel(provider: Provider, fallbackNoUrl: string): string {
  const host = getNormalizedHost(provider.api_base_url);
  const title = getProviderTitle(provider);
  return host && host !== fallbackNoUrl ? `${title} — ${host}` : title;
}

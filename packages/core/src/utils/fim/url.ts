/** Resolve Codestral/Mistral native FIM URL from provider base. */
export function resolveFimMistralUrl(baseUrl: string): URL {
  const url = new URL(baseUrl);
  const path = url.pathname.replace(/\/$/, "") || "";
  if (path.endsWith("/fim/completions")) {
    return url;
  }
  return new URL("/v1/fim/completions", baseUrl);
}

/**
 * DeepSeek hosted FIM uses beta completions.
 * Prefer explicit …/beta/completions; otherwise derive from host.
 */
export function resolveFimDeepseekUrl(baseUrl: string): URL {
  const url = new URL(baseUrl);
  const path = url.pathname.replace(/\/$/, "") || "";
  if (path.endsWith("/completions") && path.includes("/beta")) {
    return url;
  }
  if (path.endsWith("/beta") || path === "/beta") {
    url.pathname = `${path}/completions`.replace(/\/+/g, "/");
    return url;
  }
  // Chat or bare origin → force beta completions on same host.
  return new URL("/beta/completions", `${url.protocol}//${url.host}`);
}

/**
 * Qwen / LM Studio / DashScope: OpenAI legacy completions.
 * Trust api_base_url when it already ends with /completions.
 */
export function resolveFimQwenCompletionsUrl(baseUrl: string): URL {
  const url = new URL(baseUrl);
  const path = url.pathname.replace(/\/$/, "") || "";
  if (path.endsWith("/completions") && !path.endsWith("/chat/completions")) {
    return url;
  }
  if (
    path.endsWith("/v1") ||
    path.endsWith("/compatible-mode/v1") ||
    /\/compatible-mode\/v1$/.test(path)
  ) {
    url.pathname = `${path}/completions`.replace(/\/+/g, "/");
    return url;
  }
  return new URL("/v1/completions", baseUrl);
}

export function bearerAuthHeaders(
  apiKey: string | undefined
): Record<string, string> {
  if (!apiKey) return {};
  return { Authorization: `Bearer ${apiKey}` };
}

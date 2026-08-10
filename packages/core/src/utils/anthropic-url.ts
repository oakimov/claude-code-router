export function buildAnthropicMessagesUrl(baseUrl: string | undefined): string {
  const url = new URL(baseUrl || "https://api.anthropic.com");
  const path = url.pathname.replace(/\/$/, "");
  if (!path.endsWith("/v1/messages")) {
    url.pathname = `${path}/v1/messages`.replace(/\/+/g, "/");
  }
  url.searchParams.set("beta", "true");
  return url.toString();
}

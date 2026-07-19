/**
 * Resolve a Cursor dashboard API key.
 * Priority: provider api_key (crsr_...) → CURSOR_API_KEY env
 */
const UNRESOLVED_ENV_PATTERN = /^\$\{?[A-Z_][A-Z0-9_]*\}?$/;

export function resolveCursorApiKey(provider?: {
  apiKey?: string;
}): string {
  const fromProvider =
    typeof provider?.apiKey === "string" ? provider.apiKey.trim() : "";
  if (
    fromProvider &&
    fromProvider.startsWith("crsr_") &&
    !UNRESOLVED_ENV_PATTERN.test(fromProvider) &&
    !fromProvider.includes("$CURSOR_API_KEY") &&
    !fromProvider.includes("${")
  ) {
    return fromProvider;
  }

  const fromEnv = process.env.CURSOR_API_KEY?.trim();
  if (fromEnv) return fromEnv;

  throw Object.assign(
    new Error(
      "Cursor API key not found. Set Providers[].api_key to a crsr_ key, " +
        "or export CURSOR_API_KEY."
    ),
    { statusCode: 401, code: "provider_response_error", type: "api_error" }
  );
}

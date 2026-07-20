/**
 * Privacy-safe redaction for upstream errors and debug logs.
 * Keep enough diagnostic signal (status, error type) without leaking secrets.
 */

const SENSITIVE_JSON_KEYS =
  "(?:proxy[-_])?authorization|x[-_]?api[-_]?key|api[-_]?key|access[-_]?token|refresh[-_]?token|id[-_]?token|client[-_]?secret|secret|password|passwd|token|cookie|set[-_]?cookie";

export function sanitizeUpstreamErrorText(value: string): string {
  return value
    .replace(/\b(?:https?|wss?):\/\/[^\s"'<>]+/gi, "[redacted-url]")
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]{8,}/gi, "Bearer [redacted]")
    .replace(/\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}/gi, "[redacted-secret]")
    .replace(
      new RegExp(
        `"(${SENSITIVE_JSON_KEYS})"\\s*:\\s*"(?:\\\\.|[^"\\\\])*"`,
        "gi"
      ),
      '"$1":"[redacted]"'
    )
    .replace(
      /\b(?:authorization|x-api-key|api-key|api_key|access_token|token)\s*[:=]\s*["']?[^\s,;"']+/gi,
      (match) => `${match.split(/\s*[:=]/, 1)[0]}=[redacted]`
    )
    .replace(/\[[0-9a-f:]+\](?::\d+)?/gi, "[redacted-address]")
    .replace(/\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b/g, "[redacted-address]")
    .replace(
      /\b(connect(?:ing)?(?:\s+to)?|from|to)\s+(?:localhost|(?:[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?\.)+[a-z]{2,63})(?::\d+)?\b/gi,
      "$1 [redacted-address]"
    )
    .replace(/([?&](?:api_?key|key|token|access_token|auth)=)[^&\s]+/gi, "$1[redacted]")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 240);
}

function sanitizeMultiline(value: string, maxLength: number): string {
  return value
    .split("\n")
    .map((line) => sanitizeUpstreamErrorText(line))
    .filter(Boolean)
    .join("\n")
    .slice(0, maxLength);
}

export function sanitizeErrorForLog(error: unknown): {
  message: string;
  code?: string;
  name?: string;
  statusCode?: number;
  stack?: string;
  cause?: string;
} {
  const err = error as any;
  const rawMessage =
    typeof err?.message === "string"
      ? err.message
      : typeof error === "string"
        ? error
        : String(error);
  const rawCause =
    typeof err?.cause?.message === "string"
      ? err.cause.message
      : typeof err?.cause === "string"
        ? err.cause
        : undefined;
  return {
    message: sanitizeUpstreamErrorText(rawMessage) || "Unknown error",
    code: typeof err?.code === "string" ? err.code : undefined,
    name: typeof err?.name === "string" ? err.name : undefined,
    statusCode:
      typeof err?.statusCode === "number" ? err.statusCode : undefined,
    stack:
      typeof err?.stack === "string"
        ? sanitizeMultiline(err.stack, 4_000)
        : undefined,
    cause: rawCause ? sanitizeUpstreamErrorText(rawCause) : undefined,
  };
}

/** Strip auth headers from a headers object for debug logging. */
export function sanitizeHeadersForLog(
  headers: Headers | Record<string, unknown> | undefined
): Record<string, string> {
  const out: Record<string, string> = {};
  if (!headers) return out;

  const entries: Array<[string, unknown]> =
    typeof (headers as Headers).forEach === "function" &&
    typeof (headers as any).entries === "function"
      ? Array.from((headers as any).entries() as Iterable<[string, unknown]>)
      : Object.entries(headers as Record<string, unknown>);

  for (const [key, value] of entries) {
    if (value == null) continue;
    const lower = key.toLowerCase();
    if (
      lower === "authorization" ||
      lower === "x-api-key" ||
      lower === "api-key" ||
      lower === "cookie" ||
      lower === "set-cookie" ||
      lower.includes("token") ||
      lower.includes("secret")
    ) {
      out[key] = "[redacted]";
      continue;
    }
    out[key] = String(value);
  }
  return out;
}

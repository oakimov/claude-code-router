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

const SENSITIVE_BODY_KEY = new RegExp(`^(?:${SENSITIVE_JSON_KEYS})$`, "i");

const MAX_BODY_DEPTH = 6;
const MAX_BODY_ARRAY_ITEMS = 20;

/**
 * Deep-sanitize a parsed upstream error body before it is logged or
 * serialized to a client: secret-bearing keys are redacted outright and
 * every string value passes through sanitizeUpstreamErrorText. Depth and
 * array length are bounded so an adversarial body cannot balloon.
 */
export function sanitizeUpstreamErrorBody(
  value: unknown,
  depth: number = 0
): unknown {
  if (value == null) return value;
  if (depth > MAX_BODY_DEPTH) return undefined;
  if (typeof value === "string") return sanitizeUpstreamErrorText(value);
  if (Array.isArray(value)) {
    return value
      .slice(0, MAX_BODY_ARRAY_ITEMS)
      .map((item) => sanitizeUpstreamErrorBody(item, depth + 1));
  }
  if (typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(
      value as Record<string, unknown>
    )) {
      out[key] = SENSITIVE_BODY_KEY.test(key)
        ? "[redacted]"
        : sanitizeUpstreamErrorBody(entry, depth + 1);
    }
    return out;
  }
  return value;
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


const SENSITIVE_HEADER_NAME =
  /^(?:authorization|x-api-key|api-key|cookie|set-cookie)$|token|secret|password|passwd/i;

function isSensitiveHeaderName(name: string): boolean {
  return SENSITIVE_HEADER_NAME.test(name);
}

/** Flatten Fastify/Node header values to a single string for logging. */
export function normalizeHeaderValue(value: unknown): string | undefined {
  if (value == null) return undefined;
  if (Array.isArray(value)) {
    const parts = value
      .filter((v) => v != null && String(v).length > 0)
      .map((v) => String(v));
    return parts.length ? parts.join(", ") : undefined;
  }
  const s = String(value);
  return s.length ? s : undefined;
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
    const normalized = normalizeHeaderValue(value);
    if (normalized == null) continue;
    if (isSensitiveHeaderName(key)) {
      out[key] = "[redacted]";
      continue;
    }
    out[key] = normalized;
  }
  return out;
}

/**
 * Compare two header maps (case-insensitive names).
 * Values are sanitized the same way as sanitizeHeadersForLog.
 */
export function diffHeadersForLog(
  left: Headers | Record<string, unknown> | undefined,
  right: Headers | Record<string, unknown> | undefined
): {
  onlyInLeft: Record<string, string>;
  onlyInRight: Record<string, string>;
  changed: Record<string, { from: string; to: string }>;
  sameCount: number;
} {
  const a = sanitizeHeadersForLog(left);
  const b = sanitizeHeadersForLog(right);
  const aMap = new Map(
    Object.entries(a).map(([k, v]) => [k.toLowerCase(), { key: k, value: v }])
  );
  const bMap = new Map(
    Object.entries(b).map(([k, v]) => [k.toLowerCase(), { key: k, value: v }])
  );

  const onlyInLeft: Record<string, string> = {};
  const onlyInRight: Record<string, string> = {};
  const changed: Record<string, { from: string; to: string }> = {};
  let sameCount = 0;

  for (const [lk, { key, value }] of aMap) {
    const other = bMap.get(lk);
    if (!other) {
      onlyInLeft[key] = value;
      continue;
    }
    if (other.value === value) {
      sameCount += 1;
    } else {
      changed[key] = { from: value, to: other.value };
    }
  }
  for (const [rk, { key, value }] of bMap) {
    if (!aMap.has(rk)) {
      onlyInRight[key] = value;
    }
  }

  return { onlyInLeft, onlyInRight, changed, sameCount };
}

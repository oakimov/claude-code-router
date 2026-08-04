export type HeaderRecord = Record<string, string | undefined>;

function findHeaderKey(headers: HeaderRecord, name: string): string | undefined {
  const normalized = name.toLowerCase();
  return Object.keys(headers).find((key) => key.toLowerCase() === normalized);
}

export function mergeHeadersCaseInsensitive(
  base: HeaderRecord | undefined,
  update: HeaderRecord | undefined
): HeaderRecord {
  const merged: HeaderRecord = { ...(base || {}) };
  if (!update) return merged;

  for (const [name, value] of Object.entries(update)) {
    const existing = findHeaderKey(merged, name);
    if (existing) delete merged[existing];
    if (value !== undefined && value !== "undefined") {
      merged[name] = value;
    }
  }
  return merged;
}

/**
 * Headers that describe the previous representation (framing, length,
 * encoding) and must never survive onto a reshaped response body — the new
 * body has different framing/length than what upstream sent.
 */
const NON_FORWARDABLE_RESPONSE_HEADERS = new Set([
  "content-length",
  "content-encoding",
  "transfer-encoding",
  "connection",
  "content-type",
  "keep-alive",
]);

/**
 * Carry non-representational upstream response headers (rate-limit/usage
 * metadata, request ids, etc.) onto a reshaped `Response`, so overage
 * observability (`anthropic-ratelimit-unified-*`) survives SSE re-framing.
 */
export function preserveUpstreamResponseHeaders(
  headers: Headers | undefined
): Record<string, string> {
  const preserved: Record<string, string> = {};
  if (!headers) return preserved;
  headers.forEach((value, key) => {
    if (NON_FORWARDABLE_RESPONSE_HEADERS.has(key.toLowerCase())) return;
    preserved[key] = value;
  });
  return preserved;
}

/**
 * Strict allowlist of upstream response headers that may cross the final
 * server boundary to the client: retry and observability metadata only.
 * Everything else — credentials, cookies, hop-by-hop, proxy (`server`,
 * `via`), and representation headers whose values are recomputed after body
 * conversion — is dropped by omission. Names are matched case-insensitively
 * and emitted lowercase; first value wins so duplicates cannot accumulate.
 */
const DOWNSTREAM_SAFE_EXACT_HEADERS = new Set([
  "retry-after",
  "request-id",
  "x-request-id",
  "openai-request-id",
]);

const DOWNSTREAM_SAFE_HEADER_PREFIXES = [
  "anthropic-ratelimit-",
  "x-ratelimit-",
];

export function selectSafeDownstreamHeaders(
  headers: Headers | HeaderRecord | undefined
): Record<string, string> {
  const safe: Record<string, string> = {};
  if (!headers) return safe;

  const entries: Array<[string, unknown]> =
    typeof (headers as Headers).forEach === "function" &&
    typeof (headers as any).entries === "function"
      ? Array.from((headers as any).entries() as Iterable<[string, unknown]>)
      : Object.entries(headers as HeaderRecord);

  for (const [name, value] of entries) {
    if (typeof value !== "string" || !value) continue;
    const lower = name.toLowerCase();
    const allowed =
      DOWNSTREAM_SAFE_EXACT_HEADERS.has(lower) ||
      DOWNSTREAM_SAFE_HEADER_PREFIXES.some((prefix) =>
        lower.startsWith(prefix)
      );
    if (!allowed || safe[lower] !== undefined) continue;
    safe[lower] = value;
  }
  return safe;
}

export function canonicalizeOutboundHeaders(
  headers: HeaderRecord | undefined,
  fallbackBearer?: string
): Record<string, string> {
  const canonical: HeaderRecord = {};
  for (const [name, value] of Object.entries(headers || {})) {
    if (value === undefined || value === "undefined") continue;
    const normalized = name.toLowerCase();
    const outputName =
      normalized === "authorization"
        ? "Authorization"
        : normalized === "x-api-key"
          ? "x-api-key"
          : normalized;
    const existing = findHeaderKey(canonical, outputName);
    if (existing) delete canonical[existing];
    canonical[outputName] = value;
  }

  if (canonical["x-api-key"]) {
    delete canonical.Authorization;
  } else if (!canonical.Authorization && fallbackBearer) {
    canonical.Authorization = `Bearer ${fallbackBearer}`;
  }

  return Object.fromEntries(
    Object.entries(canonical).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string"
    )
  );
}

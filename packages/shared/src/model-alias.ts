const CLAUDE_MODEL_ALIAS_PREFIX = "claude-";
const ONE_MILLION_SUFFIX = "[1m]";
const LOWERCASE_HEX = /^[0-9a-f]+$/;

/** Whether a canonical CCR model id already passes Claude's prefix filter. */
export function modelIdNeedsClaudeAlias(modelId: string): boolean {
  return !/^(?:claude|anthropic)/i.test(modelId.trim());
}

/** Encode `provider,model` as a Claude-filter-safe, reversible model id. */
export function encodeClaudeModelAlias(modelId: string): string {
  return `${CLAUDE_MODEL_ALIAS_PREFIX}${Buffer.from(modelId, "utf8").toString("hex")}`;
}

/**
 * Decode a `claude-<lowercase UTF-8 hex>` alias. A trailing Desktop `[1m]`
 * selector remains outside the encoded payload and is restored on the result.
 */
export function decodeClaudeModelAlias(modelId: string): string | null {
  const trimmed = modelId.trim();
  const hasOneMillionSuffix = trimmed.endsWith(ONE_MILLION_SUFFIX);
  const withoutSuffix = hasOneMillionSuffix
    ? trimmed.slice(0, -ONE_MILLION_SUFFIX.length)
    : trimmed;
  if (!withoutSuffix.startsWith(CLAUDE_MODEL_ALIAS_PREFIX)) return null;

  const hex = withoutSuffix.slice(CLAUDE_MODEL_ALIAS_PREFIX.length);
  if (!hex || hex.length % 2 !== 0 || !LOWERCASE_HEX.test(hex)) return null;

  const decoded = Buffer.from(hex, "hex").toString("utf8");
  if (Buffer.from(decoded, "utf8").toString("hex") !== hex) return null;

  const comma = decoded.indexOf(",");
  if (comma <= 0 || comma === decoded.length - 1) return null;
  if (decoded.trim() !== decoded) return null;

  return `${decoded}${hasOneMillionSuffix ? ONE_MILLION_SUFFIX : ""}`;
}

/** Return the canonical id for equality checks, preserving ordinary ids. */
export function canonicalClaudeModelId(modelId: string): string {
  return decodeClaudeModelAlias(modelId) || modelId.trim();
}

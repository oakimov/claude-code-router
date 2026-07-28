/**
 * Anthropic validates `tool_use.id` / `tool_result.tool_use_id` against
 * `^[a-zA-Z0-9_-]+$` and rejects the whole request with a 400
 * `invalid_request_error` if any id violates it.
 *
 * The Cursor SDK supplies `SDKCustomToolContext.toolCallId` as two ids joined
 * by a newline — the upstream OpenAI-format call id and the SDK's own tracking
 * id, e.g. `call-<uuid>-3\nfc_<uuid>_0`. That value is emitted as the
 * `tool_calls[].id`, so it lands in the assistant turn and is then echoed back
 * by the client on every later request for the life of the conversation.
 */
const ANTHROPIC_TOOL_ID_ALLOWED = /^[a-zA-Z0-9_-]+$/;
const ANTHROPIC_TOOL_ID_DISALLOWED = /[^a-zA-Z0-9_-]+/g;

/** Anthropic rejects ids above this length. */
const MAX_TOOL_ID_LENGTH = 256;

function isConformingId(id: string): boolean {
  return (
    id.length > 0 &&
    id.length <= MAX_TOOL_ID_LENGTH &&
    ANTHROPIC_TOOL_ID_ALLOWED.test(id)
  );
}

export function isValidAnthropicToolCallId(id: unknown): id is string {
  return typeof id === "string" && isConformingId(id);
}

/**
 * Coerce an id into Anthropic's allowed alphabet.
 *
 * Runs of invalid characters collapse to a single `_` rather than being
 * deleted: deletion can merge two distinct ids into one (`a\nb` and `ab` both
 * become `ab`), and a duplicate `tool_use.id` is a worse failure than a long
 * one — it silently mispairs tool results.
 *
 * Must be deterministic and idempotent. The same id arrives from two
 * independent directions — emitted on the response and echoed back by the
 * client on the next request — and both must map to the same value or the
 * `tool_use` / `tool_result` pair stops matching.
 */
export function sanitizeToolCallId(id: unknown): string | undefined {
  if (typeof id !== "string" || id.length === 0) return undefined;
  if (isConformingId(id)) return id;

  const cleaned = id
    .replace(ANTHROPIC_TOOL_ID_DISALLOWED, "_")
    .slice(0, MAX_TOOL_ID_LENGTH)
    // A trailing "_" from truncation carries no information.
    .replace(/_+$/, "");

  return cleaned.length ? cleaned : undefined;
}

/**
 * Thought signatures for tool calls, carried across turns on behalf of clients
 * that cannot round-trip them.
 *
 * Gemini 3 / Antigravity sign the reasoning behind every `functionCall` and
 * expect that signature back when the call is replayed. Anthropic's `tool_use`
 * block has no field to carry it — and must not grow one, since Claude Code's
 * wire format is fixed — so the signature is lost the moment a turn reaches the
 * client. The documented `skip_thought_signature_validator` sentinel stops the
 * request from 400ing, but Google is explicit that it is a last resort that
 * "will negatively impact model performance": the model is handed back its own
 * earlier tool calls stripped of the signed reasoning that produced them.
 *
 * Observed consequence in a long Claude Code session: upstream signatures shrank
 * from ~1.2 KB to a 56-byte stub over six replayed tool calls, then the model
 * emitted an empty `functionCall` and the turn died with
 * `MALFORMED_FUNCTION_CALL` ("Function call is empty - no input to parse").
 *
 * So CCR keeps the signatures here instead of discarding them, keyed by the one
 * identifier that does survive the round trip:
 *
 *   upstream functionCall.id → Anthropic tool_use.id
 *     → tool_result.tool_use_id → Unified tool_calls[].id
 *
 * Entries are scoped per provider: a signature minted by one upstream is not
 * valid at another, so a conversation re-routed mid-session misses the cache and
 * falls back to the sentinel rather than replaying a foreign signature.
 */

const MAX_ENTRIES = 4096;
const TTL_MS = 6 * 60 * 60 * 1000;

type Entry = { signature: string; expiresAt: number };

// Insertion-ordered: the oldest key is the first one Map iteration yields.
const cache = new Map<string, Entry>();

const keyOf = (scope: string | undefined, id: string): string =>
  `${scope || "default"}${id}`;

export function rememberThoughtSignature(
  scope: string | undefined,
  id: string | undefined,
  signature: string | undefined
): void {
  if (!id || !signature) return;

  const key = keyOf(scope, id);
  // Re-insert so refreshed entries count as most recently used.
  cache.delete(key);
  cache.set(key, { signature, expiresAt: Date.now() + TTL_MS });

  while (cache.size > MAX_ENTRIES) {
    const oldest = cache.keys().next();
    if (oldest.done) break;
    cache.delete(oldest.value);
  }
}

export function recallThoughtSignature(
  scope: string | undefined,
  id: string | undefined
): string | undefined {
  if (!id) return undefined;

  const key = keyOf(scope, id);
  const entry = cache.get(key);
  if (!entry) return undefined;
  if (entry.expiresAt <= Date.now()) {
    cache.delete(key);
    return undefined;
  }
  return entry.signature;
}

/** Test hook. */
export function clearThoughtSignatureCache(): void {
  cache.clear();
}

/** Test hook. */
export function thoughtSignatureCacheSize(): number {
  return cache.size;
}

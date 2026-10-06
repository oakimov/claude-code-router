import { createHash, randomBytes } from "crypto";
import {
  getPersistedSession,
  putPersistedSession,
  updatePersistedSessions,
  type PersistedSession,
} from "@/session-registry";
import { extractClientSessionId } from "@/utils/cacheControl";
import {
  firstSubstantiveUserText,
  isForkOpeningText,
  isHarnessUserNoise,
  substantiveUserTexts,
  userMessageTextParts,
} from "@/utils/nested-agent";
import { COMPLETED_TURN_TTL_MS } from "./turn-output";

/**
 * Internal Cursor conversation id.
 *
 * Inbound clients never send or receive `x-ccr-cursor-session`. Each protocol
 * already names its conversation; CCR mints an opaque id from that name and
 * stores it under a hashed key. A later request of the same conversation
 * resolves to the same Cursor agent.
 *
 * Native identity, first match:
 *
 * - anthropic_messages — `metadata.user_id` (`session_id`, `_session_` suffix,
 *   or the raw string), else `x-claude-code-session-id`.
 * - openai_chat_completions — client header `x-opencode-session`,
 *   `x-kilocode-taskid`, `x-kilo-session`, `x-grok-session-id`,
 *   `x-grok-conv-id`, `x-conversation-id`, `session-id`, `x-session-id`,
 *   `x-session-affinity`; else body `conversation` / `conversation_id`; else
 *   `prompt_cache_key`.
 * - openai_responses — those headers, else `prompt_cache_key`. The Responses
 *   `conversation` field is rejected by the adapter, so it is not an identity.
 * - openai_fim_completions — no conversation; the prompt text is the opening.
 *
 * The first substantive user text is part of every native key. Clients send
 * side calls under the conversation id (OpenCode's title request carries the
 * session's `prompt_cache_key`; Claude Code Tasks share the parent id), and
 * one Cursor agent per id would let them supersede the main turn.
 *
 * Openings (no assistant turn or tool result yet) and follow-ups use
 * different keys. An opening that matches an unclaimed opening row reuses it
 * while that agent's turn is still running or finished within the
 * completed-turn replay window, so a client retry joins or replays. Later,
 * the same opening is a new conversation and gets a fresh agent.
 *
 * A follow-up key adds the conversation's lineage: its first assistant turn
 * (tool-call ids, else text). The first follow-up claims the opening row's
 * agent; after that, a repeated opening can only replace the opening row,
 * never an ongoing conversation. When an identical opening replaced an
 * unclaimed one, a follow-up cannot tell the two apart and gets a fresh
 * agent (the runner replays the transcript) rather than the other's history.
 *
 * No native id: the substantive user-text list, as a chain of prefix hashes.
 * A follow-up that extends a stored list keeps the session and re-keys the
 * registry to the longer list; a sibling that only shares an earlier prefix
 * does not. System text is not part of the id. Parent-lineage headers
 * (`x-parent-session-id`, `x-kilocode-parent-taskid`) are never the id.
 *
 * Worker forks use the suffix beginning at their latest explicit fork marker
 * for identity and lineage, including all opening user instructions. Inherited
 * parent turns still reach the SDK, but never identify the worker's agent.
 * Nested contexts have a separate namespace from parent contexts.
 *
 * Only hashes are persisted, never conversation text.
 */

const FAMILY = "cursor-inbound";

/** An unchanged binding's TTL is refreshed at most this often. */
const REFRESH_INTERVAL_MS = 3600_000;

export type InternalCursorSession = {
  /** Opaque id passed to the Cursor SDK session key. Never a client header. */
  internalId: string;
  /** Registry key: protocol identity, or the anonymous opening. */
  inboundKey: string;
};

function hash(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 32);
}

/** Drop a client-supplied cursor header so it cannot become the session. */
function stripCursorSessionHeader(headers: unknown): void {
  if (!headers || typeof headers !== "object") return;
  const record = headers as Record<string, unknown>;
  for (const name of Object.keys(record)) {
    if (name.toLowerCase() === "x-ccr-cursor-session") delete record[name];
  }
}

function stringField(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function nativeIdOf(input: {
  request: any;
  context: any;
  sourceSessionIdentity?: string;
}): string | undefined {
  const captured =
    stringField(input.context?.protocolContext?.sessionId) ||
    stringField(input.context?.req?.protocolContext?.sessionId) ||
    stringField(input.context?.req?.sessionId);
  if (captured) return captured;

  const fromWire = extractClientSessionId({
    body: input.request,
    headers: input.context?.req?.headers,
  });
  if (fromWire) return fromWire;

  if (input.sourceSessionIdentity) {
    const fromMetadata = extractClientSessionId({
      body: { metadata: { user_id: input.sourceSessionIdentity } },
    });
    if (fromMetadata) return fromMetadata;
  }


  // Routes strip prompt_cache_key from the provider body when Cursor is not
  // the Responses owner. The client value remains on the original request.
  const cacheKey =
    stringField(input.request?.prompt_cache_key) ||
    stringField(input.context?.req?.unifiedBody?.prompt_cache_key) ||
    stringField(input.context?.req?.originalClientBody?.prompt_cache_key) ||
    stringField(input.context?.req?.body?.prompt_cache_key);
  if (cacheKey) return cacheKey;
  return undefined;
}

/** Request items in transcript order: Unified `messages` or Responses `input`. */
function itemsOf(request: any): any[] {
  if (Array.isArray(request?.messages)) return request.messages;
  if (Array.isArray(request?.input)) return request.input;
  return [];
}

/** An item past the opening: assistant output, tool traffic, or a typed item. */
function isPastOpeningItem(item: any): boolean {
  if (!item || typeof item !== "object") return false;
  if (item.role === "assistant" || item.role === "tool") return true;
  // Responses items without a role: function_call(_output), reasoning, …
  return !item.role && typeof item.type === "string" && item.type !== "message";
}

/** Index of the first item past the opening, or -1 for an opening. */
function firstReplyIndex(request: any): number {
  return itemsOf(request).findIndex(isPastOpeningItem);
}

/**
 * No assistant turn, tool result, or stored previous response yet: a new
 * conversation, or a client retry of one.
 */
function isOpeningRequest(request: any): boolean {
  if (stringField(request?.previous_response_id)) return false;
  return firstReplyIndex(request) < 0;
}

function substantiveTexts(request: any): string[] {
  const texts = substantiveUserTexts(request);
  if (texts.length) return texts;
  const prompt = stringField(request?.prompt);
  return prompt ? [prompt] : [];
}

function assistantTextParts(content: unknown): string[] {
  if (typeof content === "string") return content ? [content] : [];
  if (!Array.isArray(content)) return [];
  return content
    .map((part: any) =>
      typeof part === "string"
        ? part
        : (part?.type === "text" || part?.type === "output_text") &&
            typeof part.text === "string"
          ? part.text
          : ""
    )
    .filter(Boolean);
}

/**
 * Hash of the conversation's first assistant turn: what tells two
 * conversations with the same opening apart. Tool-call ids are unique per
 * response and survive client-side content edits, so they win over text.
 */
function lineageOf(items: any[], start: number): string | undefined {
  const ids: string[] = [];
  const texts: string[] = [];
  for (let i = start; i >= 0 && i < items.length; i += 1) {
    const item = items[i];
    if (!item || typeof item !== "object") continue;
    if (item.role === "assistant") {
      for (const call of Array.isArray(item.tool_calls) ? item.tool_calls : []) {
        const id = stringField(call?.id);
        if (id) ids.push(id);
      }
      texts.push(...assistantTextParts(item.content));
      continue;
    }
    if (item.role || typeof item.type !== "string") break;
    // Tool results end the first assistant turn.
    if (item.type.endsWith("_output") || item.type === "message") break;
    const id = stringField(item.call_id) || stringField(item.id);
    if (id) ids.push(id);
  }
  if (ids.length) return hash(`ids\0${ids.join("\0")}`);
  const text = texts.join("");
  return text ? hash(`text\0${text}`) : undefined;
}

/** Hashes of every prefix of `texts`: entry k covers texts[0..k]. */
function prefixHashes(texts: string[]): string[] {
  const out: string[] = [];
  let previous = "";
  for (const text of texts) {
    previous = hash(`${previous}\0${Buffer.byteLength(text, "utf8")}:${text}`);
    out.push(previous);
  }
  return out;
}

function mint(): string {
  return `ccrs_${randomBytes(16).toString("hex")}`;
}

/**
 * Store `key → internalId` when the row is missing, points elsewhere, or is
 * due for a TTL refresh. Unchanged rows are not rewritten on every turn.
 */
function keep(
  key: string,
  internalId: string,
  existing: PersistedSession | undefined,
  now: number
): void {
  if (
    existing?.sessionId === internalId &&
    now - existing.updatedAt <= REFRESH_INTERVAL_MS
  ) {
    return;
  }
  putPersistedSession(
    FAMILY,
    key,
    { ...(existing?.sessionId === internalId ? existing : {}), sessionId: internalId },
    now
  );
}

/**
 * Opening: reuse an unclaimed opening row while its agent is running or
 * finished within the replay window (a client retry), else mint. A reused
 * row keeps its write time, so repeats cannot extend the window on their own.
 */
function openingSession(
  key: string,
  now: number,
  isActive: (internalId: string) => boolean
): InternalCursorSession {
  const existing = getPersistedSession(FAMILY, key, now);
  if (
    existing &&
    !existing.progressed &&
    (now - existing.updatedAt <= COMPLETED_TURN_TTL_MS ||
      isActive(existing.sessionId))
  ) {
    return { internalId: existing.sessionId, inboundKey: key };
  }
  const internalId = mint();
  putPersistedSession(
    FAMILY,
    key,
    {
      sessionId: internalId,
      // An unclaimed earlier opening may still get its follow-up.
      ...(existing && !existing.progressed ? { contested: true } : {}),
    },
    now
  );
  return { internalId, inboundKey: key };
}

/**
 * Follow-up: the longest stored key wins (re-keyed to `currentKey` when it is
 * shorter). Otherwise claim the opening row's agent, unless it is contested
 * or already claimed by another conversation; else mint.
 */
function followUpSession(input: {
  openingKey: string;
  currentKey: string;
  /** Stored keys this request may continue, longest first. */
  candidateKeys: string[];
  now: number;
}): InternalCursorSession {
  const { openingKey, currentKey, now } = input;
  for (const key of input.candidateKeys) {
    const existing = getPersistedSession(FAMILY, key, now);
    if (!existing) continue;
    if (key === currentKey) {
      keep(key, existing.sessionId, existing, now);
    } else {
      // Drop the shorter key so a sibling sharing only it starts fresh.
      updatePersistedSessions(
        FAMILY,
        {
          put: [{ key: currentKey, value: { sessionId: existing.sessionId } }],
          remove: [key],
        },
        now
      );
    }
    return { internalId: existing.sessionId, inboundKey: currentKey };
  }

  const opening = getPersistedSession(FAMILY, openingKey, now);
  if (opening && !opening.progressed && !opening.contested) {
    updatePersistedSessions(
      FAMILY,
      {
        put: [
          { key: currentKey, value: { sessionId: opening.sessionId } },
          {
            key: openingKey,
            value: { sessionId: opening.sessionId, progressed: true },
          },
        ],
      },
      now
    );
    return { internalId: opening.sessionId, inboundKey: currentKey };
  }
  const internalId = mint();
  putPersistedSession(FAMILY, currentKey, { sessionId: internalId }, now);
  return { internalId, inboundKey: currentKey };
}

/**
 * Follow-up without a first assistant turn to compare (stored Responses
 * state, or a transcript without assistant content): the conversation key
 * itself is the identity.
 */
function unlinkedFollowUpSession(
  key: string,
  now: number
): InternalCursorSession {
  const existing = getPersistedSession(FAMILY, key, now);
  if (existing) {
    if (!existing.progressed) {
      putPersistedSession(
        FAMILY,
        key,
        { sessionId: existing.sessionId, progressed: true },
        now
      );
    } else {
      keep(key, existing.sessionId, existing, now);
    }
    return { internalId: existing.sessionId, inboundKey: key };
  }
  const internalId = mint();
  putPersistedSession(FAMILY, key, { sessionId: internalId, progressed: true }, now);
  return { internalId, inboundKey: key };
}

/**
 * Protocol conversation → opaque Cursor session id.
 *
 * `isActive` reports whether the agent behind an internal id is running a
 * turn or finished one within the replay window; it extends an opening's
 * retry window past the time of its first request.
 */
export function resolveInternalCursorSession(input: {
  protocol?: string;
  request: unknown;
  context: any;
  model: string;
  sourceSessionIdentity?: string;
  isActive?: (internalId: string) => boolean;
}): InternalCursorSession {
  stripCursorSessionHeader(input.context?.req?.headers);
  const protocol = input.protocol || "unknown";
  const request = input.request as any;
  const now = Date.now();
  const isActive = input.isActive ?? (() => false);
  const nativeId = nativeIdOf({
    request,
    context: input.context,
    sourceSessionIdentity: input.sourceSessionIdentity,
  });
  // A fork can inherit the parent's first user and assistant turns. Only
  // its own suffix is identity material; the SDK still receives the full
  // request. The latest boundary also isolates forks of workers.
  const sourceItems = itemsOf(request);
  let forkAt = -1;
  for (let i = sourceItems.length - 1; i >= 0; i -= 1) {
    const item = sourceItems[i];
    if (
      item?.role === "user" &&
      userMessageTextParts(item.content).some(
        (text) => !isHarnessUserNoise(text) && isForkOpeningText(text)
      )
    ) {
      forkAt = i;
      break;
    }
  }
  const items = forkAt < 0 ? sourceItems : sourceItems.slice(forkAt);
  const identityRequest = forkAt < 0 ? request : { ...request, messages: items };
  const protocolContext = input.context?.protocolContext || input.context?.req?.protocolContext;
  const nested = forkAt >= 0 || protocolContext?.nestedAgent === true ||
    protocolContext?.claudeCodeSubagent === true;
  const replyAt = firstReplyIndex(identityRequest);
  const opening = isOpeningRequest(identityRequest);
  const lineage = opening ? undefined : lineageOf(items, replyAt);

  if (nativeId) {
    const firstUser = firstSubstantiveUserText(identityRequest);
    // Shared fork boilerplate may be a separate user message. Include all
    // worker opening instructions, stopping before the worker's own reply.
    const openingIdentity = forkAt >= 0
      ? prefixHashes(substantiveUserTexts({
          messages: replyAt < 0 ? items : items.slice(0, replyAt),
        })).at(-1)
      : firstUser ? hash(firstUser) : undefined;
    const nativeKey = `native\0${protocol}\0${nativeId}${nested ? "\0nested" : ""}`;
    const openingKey = openingIdentity ? `${nativeKey}\0${openingIdentity}` : nativeKey;
    if (opening) return openingSession(openingKey, now, isActive);
    if (!lineage) return unlinkedFollowUpSession(openingKey, now);
    const currentKey = `${openingKey}\0L${lineage}`;
    return followUpSession({
      openingKey,
      currentKey,
      candidateKeys: [currentKey],
      now,
    });
  }

  const base = `anon\0${protocol}\0${input.model || ""}\0${nested ? "nested\0" : ""}`;
  const texts = substantiveTexts(identityRequest);
  const prefixes = prefixHashes(texts);
  const openingCount =
    replyAt < 0
      ? texts.length
      : substantiveUserTexts({ messages: items.slice(0, replyAt) }).length;
  const openingKey = `${base}${prefixes[openingCount - 1] ?? ""}`;
  if (opening) return openingSession(openingKey, now, isActive);
  const currentText = prefixes[prefixes.length - 1] ?? "";
  if (!lineage) return unlinkedFollowUpSession(`${base}${currentText}`, now);
  const candidateKeys: string[] = [];
  for (let k = prefixes.length; k >= Math.max(openingCount, 1); k -= 1) {
    candidateKeys.push(`${base}${prefixes[k - 1]}\0L${lineage}`);
  }
  const currentKey = candidateKeys[0] ?? `${base}\0L${lineage}`;
  if (!candidateKeys.length) candidateKeys.push(currentKey);
  return followUpSession({ openingKey, currentKey, candidateKeys, now });
}

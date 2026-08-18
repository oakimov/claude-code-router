import { createHash } from "crypto";

const MAX_SNAPSHOTS = 512;
const HASH_CHARS = 16;
const CHARS_PER_TOKEN = 4;

/**
 * Pipeline position a snapshot was taken at. `client` is the Unified/exact-wire
 * body handed to the provider chain; `wire` is what actually left for upstream.
 * Comparing the two attributes a broken prefix to the client or to our own
 * transformer chain without a bisect.
 */
export type CachePrefixStage = "client" | "wire";

export type CacheAffinityHeaders = {
  sessionId?: string;
  threadId?: string;
  clientRequestId?: string;
};

export type CachePrefixSegment = {
  path: string;
  role?: string;
  type?: string;
  hash: string;
  /** Rough token weight (JSON chars / 4) — enough to rank misses by cost. */
  approxTokens: number;
  breakpoints: number;
  reasoningId?: string;
};

export type CachePrefixSnapshot = {
  createdAt: number;
  model?: string;
  prompt_cache_key?: string;
  session_id?: string;
  affinity?: CacheAffinityHeaders;
  lastAssistantBlockOrder?: string[];
  breakpointPaths: string[];
  systemHash?: string;
  toolsHash?: string;
  approxTokens: number;
  unstableIds: string[];
  segments: CachePrefixSegment[];
};

export type CachePrefixChange = "none" | "appended" | "modified" | "removed";

export type CachePrefixIdSource = "session" | "cache_key" | "fingerprint";

export type CachePrefixDiff = {
  conversationId: string;
  conversationIdSource: CachePrefixIdSource;
  stage: CachePrefixStage;
  firstTurn: boolean;
  prefixIntact: boolean;
  change: CachePrefixChange;
  unchangedPrefixCount: number;
  previousSegmentCount: number;
  currentSegmentCount: number;
  /** Approximate tokens of prefix that stayed byte-identical. */
  unchangedPrefixApproxTokens: number;
  /** Approximate tokens the provider must re-read because the prefix moved. */
  approxPrefixTokensLost: number;
  /** Gap since the previous turn — separates a real break from TTL expiry. */
  msSinceLastTurn?: number;
  prompt_cache_keyChanged: boolean;
  affinityChanged: boolean;
  lastAssistantBlockOrderChanged: boolean;
  modelChanged: boolean;
  systemHashChanged: boolean;
  toolsHashChanged: boolean;
  breakpointsMoved: boolean;
  /** Ids carrying an embedded timestamp — they rewrite the prefix every turn. */
  unstableIds?: string[];
  firstDivergencePath?: string;
  firstDivergence?: {
    previous?: DivergenceSide;
    current?: DivergenceSide;
  };
  appendedPaths?: string[];
  removedPaths?: string[];
  prompt_cache_key?: { previous?: string; current?: string };
  affinity?: {
    previous?: CacheAffinityHeaders;
    current?: CacheAffinityHeaders;
  };
  lastAssistantBlockOrder?: {
    previous?: string[];
    current?: string[];
  };
};

export type CachePrefixDiffOptions = {
  stage?: CachePrefixStage;
  provider?: string;
  model?: string;
  /**
   * Persist this snapshot as the baseline for the next turn. Pass false when
   * upstream rejected the request — a body that was never cached must not
   * become the baseline the following turn is judged against.
   */
  commit?: boolean;
};

const snapshots = new Map<string, CachePrefixSnapshot>();

export function __resetCachePrefixSnapshotsForTests(): void {
  snapshots.clear();
}

function digest(value: unknown): { hash: string; approxTokens: number } {
  const json = JSON.stringify(stableValue(value)) ?? "";
  return {
    hash: createHash("sha256").update(json).digest("hex").slice(0, HASH_CHARS),
    approxTokens: Math.ceil(json.length / CHARS_PER_TOKEN),
  };
}

function fingerprint(value: unknown): string {
  return digest(value).hash;
}

function stableValue(value: unknown): unknown {
  if (value === null || typeof value !== "object") return value;
  if (Array.isArray(value)) return value.map(stableValue);
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(value as object).sort()) {
    out[key] = stableValue((value as Record<string, unknown>)[key]);
  }
  return out;
}

function isEphemeral(value: unknown): boolean {
  if (!value || typeof value !== "object") return false;
  const cc = (value as any).cache_control;
  return Boolean(cc && typeof cc === "object" && cc.type === "ephemeral");
}

function hasPromptBreakpoint(value: unknown): boolean {
  return Boolean(
    value &&
      typeof value === "object" &&
      (value as any).prompt_cache_breakpoint
  );
}

function countBreakpointsOn(value: unknown): number {
  if (!value || typeof value !== "object") return 0;
  let n = 0;
  if (isEphemeral(value) || hasPromptBreakpoint(value)) n += 1;
  if (Array.isArray(value)) {
    for (const item of value) n += countBreakpointsOn(item);
    return n;
  }
  for (const child of Object.values(value as Record<string, unknown>)) {
    if (child && typeof child === "object") n += countBreakpointsOn(child);
  }
  return n;
}

function collectBreakpointPaths(value: unknown, path: string, out: string[]): void {
  if (!value || typeof value !== "object") return;
  if (isEphemeral(value) || hasPromptBreakpoint(value)) out.push(path);
  if (Array.isArray(value)) {
    value.forEach((item, index) =>
      collectBreakpointPaths(item, `${path}[${index}]`, out)
    );
    return;
  }
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if (child && typeof child === "object") {
      collectBreakpointPaths(child, `${path}.${key}`, out);
    }
  }
}

function assistantBlockOrder(content: unknown): string[] | undefined {
  if (!Array.isArray(content)) return undefined;
  const order: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const type = String((part as any).type || "");
    if (!type) continue;
    if (order[order.length - 1] !== type) order.push(type);
  }
  return order.length ? order : undefined;
}

function reasoningIdOf(item: any): string | undefined {
  if (!item || typeof item !== "object") return undefined;
  if (item.type === "reasoning" && typeof item.id === "string" && item.id) {
    return item.id;
  }
  const thinkingId = item.thinking?.id;
  return typeof thinkingId === "string" && thinkingId ? thinkingId : undefined;
}

/**
 * An id carrying a wall-clock timestamp (`rs_1735689600000`) is minted fresh on
 * every convert, so the prefix can never match. Hashed and provider-issued ids
 * do not look like this.
 */
function looksUnstableId(value: unknown): boolean {
  return typeof value === "string" && /\d{12,}/.test(value);
}

function collectUnstableIds(value: unknown, out: Set<string>, depth = 0): void {
  if (!value || typeof value !== "object" || depth > 8) return;
  if (Array.isArray(value)) {
    for (const item of value) collectUnstableIds(item, out, depth + 1);
    return;
  }
  for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
    if ((key === "id" || key === "call_id") && looksUnstableId(child)) {
      out.add(String(child));
    } else if (child && typeof child === "object") {
      collectUnstableIds(child, out, depth + 1);
    }
  }
}

function segment(
  path: string,
  value: unknown,
  extra?: { role?: string; type?: string }
): CachePrefixSegment {
  const item = value as any;
  const { hash, approxTokens } = digest(value);
  return {
    path,
    ...(extra?.role ? { role: extra.role } : {}),
    ...(extra?.type ? { type: extra.type } : {}),
    ...(typeof item?.role === "string" && !extra?.role ? { role: item.role } : {}),
    ...(typeof item?.type === "string" && !extra?.type ? { type: item.type } : {}),
    hash,
    approxTokens,
    breakpoints: countBreakpointsOn(value),
    ...(reasoningIdOf(item) ? { reasoningId: reasoningIdOf(item) } : {}),
  };
}

function remember(key: string, snapshot: CachePrefixSnapshot): void {
  if (snapshots.has(key)) snapshots.delete(key);
  snapshots.set(key, snapshot);
  while (snapshots.size > MAX_SNAPSHOTS) {
    const oldest = snapshots.keys().next().value;
    if (oldest === undefined) break;
    snapshots.delete(oldest);
  }
}

/**
 * Compact, content-free snapshot of the outbound fields prompt caching uses.
 */
export function snapshotOutboundCachePrefix(
  body: Record<string, any> | null | undefined,
  affinity?: CacheAffinityHeaders
): CachePrefixSnapshot | null {
  if (!body || typeof body !== "object") return null;

  const segments: CachePrefixSegment[] = [];
  const breakpointPaths: string[] = [];
  let systemHash: string | undefined;
  let toolsHash: string | undefined;

  if (body.system != null) {
    const item = segment("system", body.system, { type: "system" });
    systemHash = item.hash;
    segments.push(item);
    collectBreakpointPaths(body.system, "system", breakpointPaths);
  } else if (typeof body.instructions === "string" && body.instructions) {
    const item = segment("instructions", body.instructions, {
      type: "instructions",
    });
    systemHash = item.hash;
    segments.push(item);
  }

  if (Array.isArray(body.tools)) {
    const item = segment("tools", body.tools, { type: "tools" });
    toolsHash = item.hash;
    segments.push(item);
    collectBreakpointPaths(body.tools, "tools", breakpointPaths);
  }

  const conversation = Array.isArray(body.messages)
    ? { key: "messages", items: body.messages }
    : Array.isArray(body.input)
      ? { key: "input", items: body.input }
      : Array.isArray(body.contents)
        ? { key: "contents", items: body.contents }
        : undefined;

  let lastAssistantBlockOrder: string[] | undefined;
  if (conversation) {
    conversation.items.forEach((item: any, index: number) => {
      const path = `${conversation.key}[${index}]`;
      segments.push(segment(path, item));
      collectBreakpointPaths(item, path, breakpointPaths);
      if (item?.role === "assistant" || item?.role === "model") {
        lastAssistantBlockOrder =
          assistantBlockOrder(item.content) ||
          assistantBlockOrder(item.parts) ||
          lastAssistantBlockOrder;
      }
    });
  }

  const prompt_cache_key =
    typeof body.prompt_cache_key === "string" && body.prompt_cache_key
      ? body.prompt_cache_key
      : typeof body.session_id === "string" && body.session_id
        ? body.session_id
        : undefined;

  const cleanedAffinity = affinity
    ? Object.fromEntries(
        Object.entries(affinity).filter(
          ([, value]) => typeof value === "string" && value
        )
      )
    : undefined;

  if (
    segments.length === 0 &&
    !prompt_cache_key &&
    !(cleanedAffinity && Object.keys(cleanedAffinity).length)
  ) {
    return null;
  }

  const unstable = new Set<string>();
  collectUnstableIds(conversation?.items, unstable);

  return {
    createdAt: Date.now(),
    ...(typeof body.model === "string" ? { model: body.model } : {}),
    ...(prompt_cache_key ? { prompt_cache_key } : {}),
    ...(typeof body.session_id === "string" ? { session_id: body.session_id } : {}),
    ...(cleanedAffinity && Object.keys(cleanedAffinity).length
      ? { affinity: cleanedAffinity }
      : {}),
    ...(lastAssistantBlockOrder ? { lastAssistantBlockOrder } : {}),
    breakpointPaths,
    ...(systemHash ? { systemHash } : {}),
    ...(toolsHash ? { toolsHash } : {}),
    approxTokens: segments.reduce((sum, item) => sum + item.approxTokens, 0),
    unstableIds: [...unstable],
    segments,
  };
}

type DivergenceSide = Pick<
  CachePrefixSegment,
  "path" | "role" | "type" | "approxTokens" | "breakpoints" | "reasoningId"
>;

function publicSegment(
  segmentValue: CachePrefixSegment | undefined
): DivergenceSide | undefined {
  if (!segmentValue) return undefined;
  return {
    path: segmentValue.path,
    ...(segmentValue.role ? { role: segmentValue.role } : {}),
    ...(segmentValue.type ? { type: segmentValue.type } : {}),
    approxTokens: segmentValue.approxTokens,
    breakpoints: segmentValue.breakpoints,
    ...(segmentValue.reasoningId
      ? { reasoningId: segmentValue.reasoningId }
      : {}),
  };
}

function affinityFingerprint(affinity?: CacheAffinityHeaders): string {
  if (!affinity) return "";
  return fingerprint(affinity);
}

function sumApproxTokens(segmentList: CachePrefixSegment[]): number {
  return segmentList.reduce((sum, item) => sum + item.approxTokens, 0);
}

export function diffCachePrefixSnapshots(
  conversationId: string,
  previous: CachePrefixSnapshot | undefined,
  current: CachePrefixSnapshot,
  meta?: { stage?: CachePrefixStage; conversationIdSource?: CachePrefixIdSource }
): CachePrefixDiff {
  const stage = meta?.stage ?? "wire";
  const conversationIdSource = meta?.conversationIdSource ?? "session";
  const unstableIds = current.unstableIds.length
    ? { unstableIds: current.unstableIds.slice(0, 8) }
    : {};

  if (!previous) {
    return {
      conversationId,
      conversationIdSource,
      stage,
      firstTurn: true,
      prefixIntact: true,
      change: "none",
      unchangedPrefixCount: current.segments.length,
      previousSegmentCount: 0,
      currentSegmentCount: current.segments.length,
      unchangedPrefixApproxTokens: current.approxTokens,
      approxPrefixTokensLost: 0,
      prompt_cache_keyChanged: false,
      affinityChanged: false,
      lastAssistantBlockOrderChanged: false,
      modelChanged: false,
      systemHashChanged: false,
      toolsHashChanged: false,
      breakpointsMoved: false,
      ...unstableIds,
    };
  }

  const prompt_cache_keyChanged =
    (previous.prompt_cache_key || "") !== (current.prompt_cache_key || "");
  const affinityChanged =
    affinityFingerprint(previous.affinity) !==
    affinityFingerprint(current.affinity);
  const lastAssistantBlockOrderChanged =
    JSON.stringify(previous.lastAssistantBlockOrder || []) !==
    JSON.stringify(current.lastAssistantBlockOrder || []);
  const modelChanged = (previous.model || "") !== (current.model || "");
  const systemHashChanged = (previous.systemHash || "") !== (current.systemHash || "");
  const toolsHashChanged = (previous.toolsHash || "") !== (current.toolsHash || "");
  const breakpointsMoved =
    JSON.stringify(previous.breakpointPaths) !==
    JSON.stringify(current.breakpointPaths);

  const prevSegs = previous.segments;
  const currSegs = current.segments;
  const shared = Math.min(prevSegs.length, currSegs.length);
  let unchanged = 0;
  while (
    unchanged < shared &&
    prevSegs[unchanged].path === currSegs[unchanged].path &&
    prevSegs[unchanged].hash === currSegs[unchanged].hash
  ) {
    unchanged += 1;
  }

  const prefixIntact =
    unchanged === prevSegs.length && currSegs.length >= prevSegs.length;
  let change: CachePrefixChange = "none";
  let firstDivergencePath: string | undefined;
  let appendedPaths: string[] | undefined;
  let removedPaths: string[] | undefined;
  let firstDivergence: CachePrefixDiff["firstDivergence"];

  if (unchanged < shared) {
    change = "modified";
    firstDivergencePath = currSegs[unchanged]?.path || prevSegs[unchanged]?.path;
    firstDivergence = {
      previous: publicSegment(prevSegs[unchanged]),
      current: publicSegment(currSegs[unchanged]),
    };
  } else if (currSegs.length > prevSegs.length) {
    change = "appended";
    appendedPaths = currSegs.slice(unchanged).map((item) => item.path);
  } else if (currSegs.length < prevSegs.length) {
    change = "removed";
    removedPaths = prevSegs.slice(unchanged).map((item) => item.path);
    firstDivergencePath = prevSegs[unchanged]?.path;
  }

  if (
    change === "none" &&
    (prompt_cache_keyChanged ||
      affinityChanged ||
      lastAssistantBlockOrderChanged ||
      modelChanged ||
      breakpointsMoved)
  ) {
    change = "modified";
  }

  if (!firstDivergencePath) {
    if (prompt_cache_keyChanged) firstDivergencePath = "prompt_cache_key";
    else if (affinityChanged) firstDivergencePath = "affinity";
    else if (modelChanged) firstDivergencePath = "model";
    else if (lastAssistantBlockOrderChanged) {
      firstDivergencePath = "lastAssistantBlockOrder";
    } else if (breakpointsMoved) {
      firstDivergencePath = "breakpointPaths";
    }
  }

  // A prefix break forces the provider to re-read everything from the
  // divergence onward; an append costs nothing that was already cached.
  const approxPrefixTokensLost =
    change === "modified" || change === "removed"
      ? sumApproxTokens(prevSegs.slice(unchanged))
      : 0;

  return {
    conversationId,
    conversationIdSource,
    stage,
    firstTurn: false,
    prefixIntact:
      prefixIntact &&
      !prompt_cache_keyChanged &&
      !affinityChanged &&
      !modelChanged,
    change,
    unchangedPrefixCount: unchanged,
    previousSegmentCount: prevSegs.length,
    currentSegmentCount: currSegs.length,
    unchangedPrefixApproxTokens: sumApproxTokens(currSegs.slice(0, unchanged)),
    approxPrefixTokensLost,
    msSinceLastTurn: Math.max(0, current.createdAt - previous.createdAt),
    prompt_cache_keyChanged,
    affinityChanged,
    lastAssistantBlockOrderChanged,
    modelChanged,
    systemHashChanged,
    toolsHashChanged,
    breakpointsMoved,
    ...unstableIds,
    ...(firstDivergencePath ? { firstDivergencePath } : {}),
    ...(firstDivergence ? { firstDivergence } : {}),
    ...(appendedPaths ? { appendedPaths } : {}),
    ...(removedPaths ? { removedPaths } : {}),
    ...(prompt_cache_keyChanged
      ? {
          prompt_cache_key: {
            previous: previous.prompt_cache_key,
            current: current.prompt_cache_key,
          },
        }
      : {}),
    ...(affinityChanged
      ? {
          affinity: {
            previous: previous.affinity,
            current: current.affinity,
          },
        }
      : {}),
    ...(lastAssistantBlockOrderChanged
      ? {
          lastAssistantBlockOrder: {
            previous: previous.lastAssistantBlockOrder,
            current: current.lastAssistantBlockOrder,
          },
        }
      : {}),
  };
}

/**
 * Structural identity for clients that send no session metadata. Keyed on the
 * system prompt plus the opening turn, both of which are fixed for the life of
 * a conversation and differ between conversations — unlike a shared literal,
 * which would make concurrent conversations diff against each other.
 */
function fingerprintConversation(snapshot: CachePrefixSnapshot): string {
  const opening = snapshot.segments.find((item) =>
    /^(messages|input|contents)\[0\]$/.test(item.path)
  );
  return `fp_${fingerprint([
    snapshot.systemHash || "",
    snapshot.toolsHash || "",
    opening?.hash || "",
  ])}`;
}

function resolveConversationId(
  conversationId: string | undefined,
  snapshot: CachePrefixSnapshot
): { id: string; source: CachePrefixIdSource } {
  if (typeof conversationId === "string" && conversationId) {
    return { id: conversationId, source: "session" };
  }
  if (snapshot.prompt_cache_key) {
    return { id: snapshot.prompt_cache_key, source: "cache_key" };
  }
  return { id: fingerprintConversation(snapshot), source: "fingerprint" };
}

/**
 * Compare this outbound body to the last one for the conversation, then
 * remember the current snapshot. Returns null when there is nothing cacheable.
 *
 * Snapshots are keyed per stage and per provider/model: a routed fallback to a
 * different destination is a legitimate cache miss, not prefix corruption, and
 * must not be reported as one.
 */
export function rememberAndDiffOutboundCachePrefix(
  conversationId: string | undefined,
  body: Record<string, any> | null | undefined,
  affinity?: CacheAffinityHeaders,
  options?: CachePrefixDiffOptions
): CachePrefixDiff | null {
  const current = snapshotOutboundCachePrefix(body, affinity);
  if (!current) return null;

  const stage = options?.stage ?? "wire";
  const { id, source } = resolveConversationId(conversationId, current);
  const key = [
    stage,
    options?.provider || "-",
    options?.model || current.model || "-",
    id,
  ].join("|");

  const previous = snapshots.get(key);
  const diff = diffCachePrefixSnapshots(id, previous, current, {
    stage,
    conversationIdSource: source,
  });
  if (options?.commit !== false) remember(key, current);
  return diff;
}

/**
 * Attribute a broken wire prefix to the stage that broke it. The client leg is
 * checked first: if Claude Code itself rewrote history there is nothing our
 * transformer chain could have preserved.
 */
export function attributeDivergenceStage(
  clientDiff: CachePrefixDiff | null | undefined,
  wireDiff: CachePrefixDiff | null | undefined
): CachePrefixStage | "none" | undefined {
  if (!wireDiff || wireDiff.firstTurn) return undefined;
  if (wireDiff.prefixIntact) return "none";
  if (!clientDiff || clientDiff.firstTurn) return undefined;
  return clientDiff.prefixIntact ? "wire" : "client";
}

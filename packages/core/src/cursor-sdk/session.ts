import { createHash } from "crypto";
import { existsSync, mkdirSync, readdirSync, rmSync, statSync } from "fs";
import { join } from "path";
import {
  Agent,
  type ModelSelection,
  type Run,
  type SDKAgent,
  type SDKMessage,
} from "@cursor/sdk";
import type { OpenAiUsage } from "./usage";
import type { CursorTranscriptCommit } from "./turn-identity";
import { sanitizeToolCallId } from "@/utils/toolCallId";
import {
  deletePersistedSession,
  getPersistedSession,
  putPersistedSession,
} from "@/session-registry";
import { ensureDenyHooksWorkspace } from "./hooks-template";
import { EMPTY_HOST_ENVIRONMENT, type HostEnvironment } from "./host-env";
import { installCursorAuthExchangeCache } from "./auth-exchange-cache";
import {
  CURSOR_SDK_WORKSPACES_ROOT,
  ORPHAN_WORKSPACE_TTL_MS,
  SESSION_IDLE_TTL_MS,
  SESSION_LRU_MAX,
  WORKSPACE_SWEEP_INTERVAL_MS,
  hashSessionFingerprint,
  isManagedWorkspacePath,
  type CursorSdkMode,
} from "./shared";

export type ParkedTool = {
  /** Original Cursor SDK tool-call id (never emitted to the client). */
  id: string;
  name: string;
  args: Record<string, unknown>;
  runToken?: symbol;
  resolve: (result: string) => void;
  reject: (err: Error) => void;
  promise: Promise<string>;
};

export type CursorSdkSession = {
  key: string;
  agent: SDKAgent;
  agentId: string;
  mode: CursorSdkMode;
  workspaceDir: string;
  /** Host machine facts for this session, refreshed each turn. */
  hostEnv: HostEnvironment;
  run?: Run;
  streamIterator?: AsyncIterator<any>;
  /**
   * The one outstanding `iterator.next()` owned by this SDK run. It survives
   * the HTTP tool-call boundary so a continuation never starts a second
   * consumer on Cursor's single-consumer iterator.
   */
  streamNext?: Promise<IteratorResult<any>>;
  streamNextRunToken?: symbol;
  activeRunToken?: symbol;
  lastSdkUsageRaw?: OpenAiUsage;
  parked: ParkedTool[];
  /**
   * Per-session bidirectional map between raw Cursor SDK tool-call ids
   * (possibly newline-joined / overlong) and the host-safe aliases emitted
   * to the client. Parked records keep originals; the wire carries aliases;
   * incoming echoes are translated back before matching. Dies with the
   * session; bounded below so long-lived agents cannot grow it forever.
   */
  toolIdAliases: {
    byOriginal: Map<string, string>;
    byAlias: Map<string, string>;
  };
  /** Tool calls waiting to be emitted on the current SSE response. */
  pendingEmit: Array<{
    id: string;
    name: string;
    args: Record<string, unknown>;
    runToken?: symbol;
  }>;
  emitWaiters: Array<() => void>;
  /** SDK raw delta callbacks waiting to be merged into the current SSE response. */
  pendingSdkMessages: Array<{
    message: SDKMessage;
    runToken?: symbol;
    source: "delta";
  }>;
  sdkMessageWaiters: Array<() => void>;
  /**
   * Serializes cancel/send so a follow-up compact/retry cannot call
   * `agent.send` while a prior run is still marked active in the SDK store.
   */
  sendChain: Promise<void>;
  /** True after at least one successful `agent.send` on this session. */
  hasSentPrompt: boolean;
  /**
   * Exact host-visible transcript represented by this agent after the last
   * completed response. Unknown means tail-only reuse is forbidden.
   */
  transcriptCommit?: CursorTranscriptCommit;
  /** Agent/model/workspace/tool configuration paired with transcriptCommit. */
  compatibilityStamp?: string;
  /** Fingerprint of the host env baked into the workspace rules/deny hooks. */
  guidanceFingerprint?: string;
  /**
   * Bridge preamble last actually sent on this agent (host env + tool catalog).
   * Unchanged follow-ups omit the preamble so history does not stack copies.
   */
  lastBridgePromptGuidanceFingerprint?: string;
  /**
   * Set when the local SDK handles are no longer trustworthy. The manager must
   * not hand this agent out for a future request.
   */
  poisoned?: boolean;
  poisonReason?: string;
  lastActiveAt: number;
  metrics: {
    customToolCalls: number;
    builtinToolCallsSeen: number;
    /** Host tool calls whose arguments pointed at the scratch workspace. */
    scratchPathViolations: number;
    /** Violations answered with a corrective result instead of execution. */
    scratchPathCorrections: number;
  };
  parkHostTool: (tool: {
    id: string;
    name: string;
    args: Record<string, unknown>;
  }) => Promise<string>;
  notifyEmit: () => void;
  waitForEmit: () => Promise<void>;
  enqueueSdkMessage: (message: SDKMessage, runToken?: symbol) => void;
  notifySdkMessage: () => void;
  waitForSdkMessage: () => Promise<void>;
};

export type CancelActiveRunResult = {
  skipped: boolean;
  hadRun: boolean;
  hadIterator: boolean;
  runCancelFailed: boolean;
  iteratorReturnFailed: boolean;
  failed: boolean;
  timedOut: boolean;
};

const CURSOR_AGENT_DISPOSE_TIMEOUT_MS = 2_000;

function isAbortError(error: unknown): boolean {
  const candidate = error as { name?: unknown; message?: unknown };
  return (
    candidate?.name === "AbortError" ||
    String(candidate?.message || error).toLowerCase() ===
      "this operation was aborted"
  );
}

/**
 * Host-emitted tool-call ids must survive both Anthropic (`tool_use.id`,
 * charset + 256 chars) and Responses (`call_id`, 64 chars) validation, and
 * must be unique within the session so results cannot mispair. Conforming
 * SDK ids pass through unchanged; anything else gets a stable `ct_<hash>`
 * alias, disambiguated on collision. Deterministic per original so re-emits
 * are stable for the life of the session.
 */
const TOOL_ID_ALIAS_MAX_LENGTH = 64;
const TOOL_ID_ALIAS_PREFIX = "ct";
const TOOL_ID_ALIAS_MAP_MAX = 1000;

function hashToolIdAlias(value: string): string {
  return createHash("sha256")
    .update(value, "utf8")
    .digest("base64url")
    .slice(0, 20);
}

function evictOldestToolIdAlias(session: CursorSdkSession): void {
  const oldest = session.toolIdAliases.byOriginal.keys().next().value;
  if (oldest === undefined) return;
  const alias = session.toolIdAliases.byOriginal.get(oldest);
  session.toolIdAliases.byOriginal.delete(oldest);
  if (alias !== undefined && session.toolIdAliases.byAlias.get(alias) === oldest) {
    session.toolIdAliases.byAlias.delete(alias);
  }
}

export function aliasHostToolId(
  session: CursorSdkSession,
  original: unknown
): string | undefined {
  if (typeof original !== "string" || original.length === 0) return undefined;
  const maps = session.toolIdAliases;
  if (!maps) return sanitizeToolCallId(original);
  const existing = maps.byOriginal.get(original);
  if (existing) return existing;
  const base = sanitizeToolCallId(original);
  let n = 0;
  let alias = "";
  for (;;) {
    const candidate =
      n === 0 && base && base.length <= TOOL_ID_ALIAS_MAX_LENGTH
        ? base
        : `${TOOL_ID_ALIAS_PREFIX}_${hashToolIdAlias(n === 0 ? original : `${original}#${n}`)}`;
    const owner = maps.byAlias.get(candidate);
    if (owner === undefined || owner === original) {
      alias = candidate;
      break;
    }
    n += 1;
  }
  while (maps.byOriginal.size >= TOOL_ID_ALIAS_MAP_MAX) {
    evictOldestToolIdAlias(session);
  }
  maps.byOriginal.set(original, alias);
  maps.byAlias.set(alias, original);
  return alias;
}

/**
 * Translate a client-echoed tool-call id back to the parked SDK original.
 * Unknown ids pass through unchanged so legacy / unmapped turns mismatch
 * exactly as before instead of collapsing onto something unrelated.
 */
export function resolveHostToolId(
  session: CursorSdkSession,
  echoed: unknown
): string {
  if (typeof echoed !== "string" || !echoed) return "";
  return session.toolIdAliases?.byAlias.get(echoed) ?? echoed;
}

export function markSessionPoisoned(  session: CursorSdkSession,
  reason: string
): void {
  session.poisoned = true;
  session.poisonReason = reason;
}

/**
 * Drop local run handles and await SDK `run.cancel()` when needed.
 *
 * Cursor persists `activeRunId` in its local store and throws
 * `Agent … already has active run` on the next `agent.send` until the prior
 * run is terminal. Client disconnect / compact / retry often leave a live run.
 */
export async function cancelActiveRun(
  session: CursorSdkSession,
  options: {
    rejectParked?: boolean;
    reason?: string;
    timeoutMs?: number;
    onlyRunToken?: symbol;
    poisonOnFailure?: boolean;
  } = {}
): Promise<CancelActiveRunResult> {
  const result: CancelActiveRunResult = {
    skipped: false,
    hadRun: Boolean(session.run),
    hadIterator: Boolean(session.streamIterator),
    runCancelFailed: false,
    iteratorReturnFailed: false,
    failed: false,
    timedOut: false,
  };
  if (options.onlyRunToken && session.activeRunToken !== options.onlyRunToken) {
    return { ...result, skipped: true };
  }
  const run = session.run;
  const iterator = session.streamIterator;
  const pendingNext = session.streamNext;
  session.run = undefined;
  session.streamIterator = undefined;
  session.streamNext = undefined;
  session.streamNextRunToken = undefined;
  session.activeRunToken = undefined;
  session.pendingEmit = [];
  session.pendingSdkMessages = [];
  session.notifyEmit();
  session.notifySdkMessage?.();
  // The iterator cancellation below should settle this read. Keep a rejection
  // handler attached after dropping session ownership.
  void pendingNext?.catch(() => undefined);

  if (options.rejectParked && session.parked.length) {
    const reason = options.reason || "cursor-sdk run cancelled";
    for (const parked of session.parked.splice(0, session.parked.length)) {
      try {
        parked.reject(new Error(reason));
      } catch {
        // ignore
      }
    }
  }

  const settleCancellation = async (
    operation: () => Promise<unknown>,
    timeoutMessage: string,
    operationKind: "run" | "iterator"
  ): Promise<void> => {
    try {
      await withTimeout(
        Promise.resolve().then(operation),
        options.timeoutMs ?? 2_000,
        timeoutMessage
      );
    } catch (err: any) {
      result.failed = true;
      if (operationKind === "run") result.runCancelFailed = true;
      if (operationKind === "iterator") result.iteratorReturnFailed = true;
      result.timedOut =
        result.timedOut ||
        String(err?.message || err).toLowerCase().includes("timed out");
    }
  };

  const cancellations: Promise<void>[] = [];
  // Cancel the run before closing its async iterator. A real async generator's
  // return() queues behind an outstanding next(), while run.cancel() is what
  // releases that next(). Running both together also keeps one bounded deadline.
  if (run?.status === "running") {
    cancellations.push(
      settleCancellation(
        () => run.cancel(),
        "cursor-sdk run cancel timed out",
        "run"
      )
    );
  }
  if (iterator && typeof iterator.return === "function") {
    cancellations.push(
      settleCancellation(
        () => iterator.return!(undefined),
        "cursor-sdk stream iterator return timed out",
        "iterator"
      )
    );
  }
  await Promise.all(cancellations);

  if (result.failed && options.poisonOnFailure) {
    markSessionPoisoned(
      session,
      options.reason || "cursor-sdk run cancellation did not complete"
    );
  }
  return result;
}

export async function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  message: string
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error(message)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** Run `fn` exclusively against this session's cancel/send critical section. */
export async function withSessionSendLock<T>(
  session: CursorSdkSession,
  fn: () => Promise<T>
): Promise<T> {
  const previous = session.sendChain;
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  session.sendChain = previous.then(() => gate).catch(() => gate);
  await previous.catch(() => undefined);
  try {
    return await fn();
  } finally {
    release();
  }
}

/**
 * Cursor local sandbox requires a supported host (typically desktop with the
 * sandbox helper). Docker/Alpine/CI do not support it and throw:
 * "Local SDK sandboxing was requested, but sandboxing is not supported…".
 *
 * Model D relies on deny-hooks + customTools park, not sandbox. Default OFF;
 * opt in with transformer `sandboxEnabled: true` or `CCR_CURSOR_SANDBOX=1`
 * only on supported hosts.
 */
export function shouldEnableCursorSandbox(requested?: boolean): boolean {
  if (requested === false) return false;

  const env = process.env.CCR_CURSOR_SANDBOX?.trim().toLowerCase();
  if (env === "0" || env === "false" || env === "off") return false;

  const wants =
    requested === true || env === "1" || env === "true" || env === "on";
  if (!wants) return false;

  // Even if requested, refuse common unsupported environments.
  if (existsSync("/.dockerenv") || process.env.container === "docker") {
    return false;
  }
  return true;
}

function ensureWorkspace(sessionKey: string, hostEnv?: HostEnvironment): string {
  if (!existsSync(CURSOR_SDK_WORKSPACES_ROOT)) {
    mkdirSync(CURSOR_SDK_WORKSPACES_ROOT, { recursive: true });
  }
  const dir = join(CURSOR_SDK_WORKSPACES_ROOT, sessionKey);
  mkdirSync(dir, { recursive: true });
  ensureDenyHooksWorkspace(dir, hostEnv);
  return dir;
}

/**
 * Adopt this turn's host facts and re-stamp workspace rules when they changed.
 * Cheap: files are only rewritten when their content actually differs.
 *
 * Cursor loads workspace rules when the agent's rules service is constructed,
 * i.e. once per session — verified by appending a canary line to a live
 * session's AGENTS.md, which the model did not see. So this rewrite serves the
 * *next* agent created against this directory; the live turn picks up changed
 * host facts through the prompt, and `session.hostEnv` (used by the scratch
 * path correction) is updated here regardless.
 */
export function refreshWorkspaceGuidance(
  session: CursorSdkSession,
  hostEnv: HostEnvironment
): boolean {
  // A turn that carries no environment block must not erase known host facts.
  if (!hostEnv.known && session.hostEnv?.known) return false;

  const changed = session.guidanceFingerprint !== hostEnv.fingerprint;
  session.hostEnv = hostEnv;
  if (session.mode !== "bridge" || !changed) return false;

  try {
    ensureDenyHooksWorkspace(session.workspaceDir, hostEnv);
    session.guidanceFingerprint = hostEnv.fingerprint;
    return true;
  } catch {
    // Never fail a turn over guidance refresh — the prompt still carries it.
    return false;
  }
}

/** Remove a scratch workspace, but only one this module created. */
function removeManagedWorkspace(
  dir: string,
  logger?: any,
  root: string = CURSOR_SDK_WORKSPACES_ROOT
): boolean {
  if (!isManagedWorkspacePath(dir, root)) return false;
  try {
    rmSync(dir, { recursive: true, force: true });
    return true;
  } catch (err) {
    logger?.debug?.({ err, dir }, "cursor-sdk workspace cleanup failed");
    return false;
  }
}

function sessionIdFromMetadataUserId(metadataUserId: string): string {
  try {
    const parsed = JSON.parse(metadataUserId);
    if (parsed && typeof parsed.session_id === "string" && parsed.session_id) {
      return parsed.session_id;
    }
  } catch {
    // Non-JSON metadata.user_id is allowed.
  }
  const parts = metadataUserId.split("_session_");
  if (parts.length > 1 && parts[1]) {
    return parts[1];
  }
  return metadataUserId;
}

function hashSessionKeyMaterial(value: string): string {
  return createHash("sha256").update(value).digest("hex").slice(0, 32);
}

function parentSessionIdentity(input: {
  headerSession?: string;
  clientSessionId?: string;
  metadataUserId?: string;
}): string | undefined {
  if (input.headerSession) return input.headerSession;
  if (input.clientSessionId) return input.clientSessionId;
  if (input.metadataUserId) {
    return sessionIdFromMetadataUserId(input.metadataUserId);
  }
  return undefined;
}

/**
 * Cursor SDK session directory key. Prefer explicit client conversation ids
 * over hashing prompt text — never include system / harness version.
 *
 * Claude Code subagents share the parent session id. Mix first-user text so
 * parallel Task agents each get their own Cursor Agent instead of collapsing
 * onto one turn registry slot.
 */
export function buildSessionKey(input: {
  headerSession?: string;
  /** Inbound-captured Claude/OpenCode session id (protocolContext / req). */
  clientSessionId?: string;
  metadataUserId?: string;
  model?: string;
  /** Anonymous clients only: first user text (not system). */
  firstUserText?: string;
  /** @deprecated Use firstUserText. Kept for call-site compatibility. */
  systemAndFirstUser?: string;
  /**
   * Claude Code Task / subagent turn. Parent session id alone is not unique
   * across parallel subagents.
   */
  isSubagent?: boolean;
}): string {
  const parent = parentSessionIdentity(input);
  const firstUser = input.firstUserText || input.systemAndFirstUser || "";
  if (parent && firstUser) {
    // Mix opening user text for every harness. Claude Code Tasks share the
    // parent session id; OpenCode sends a child session plus x-parent-session-id
    // but first-user mix is still stable. Skip reminder/caveat at the call site.
    return hashSessionKeyMaterial(`${parent}\n${firstUser}`);
  }
  if (parent) {
    return hashSessionKeyMaterial(parent);
  }
  // Anonymous: model + first user text only (never system / cc_version).
  return hashSessionFingerprint([input.model || "", firstUser]);
}

/** True while a session has a live run, open stream, or unresolved host tools. */
export function isSessionInFlight(session: CursorSdkSession): boolean {
  return Boolean(
    session.streamIterator ||
      session.streamNext ||
      session.activeRunToken ||
      session.parked.length > 0 ||
      session.pendingSdkMessages.length > 0 ||
      session.run?.status === "running"
  );
}

export function touchSession(session: CursorSdkSession): void {
  session.lastActiveAt = Date.now();
}

export class SessionManager {
  private sessions = new Map<string, CursorSdkSession>();
  /** Per-key teardown barriers so reconnects cannot overtake SDK cancellation. */
  private retirements = new Map<
    string,
    {
      tail: Promise<void>;
      pending: Map<CursorSdkSession, Promise<boolean>>;
    }
  >();
  /** Per-key creation barriers so retirement waiters cannot create sibling agents. */
  private creations = new Map<string, Promise<void>>();
  private cleanupTimer?: ReturnType<typeof setInterval>;
  /** Epoch 0 so the first eviction tick performs a sweep. */
  private lastSweepAt = 0;

  constructor(private logger?: any) {
    this.cleanupTimer = setInterval(() => this.evictIdle(), 60_000);
    if (typeof this.cleanupTimer.unref === "function") {
      this.cleanupTimer.unref();
    }
  }

  setLogger(logger: any): void {
    this.logger = logger;
  }

  get(key: string): CursorSdkSession | undefined {
    const session = this.sessions.get(key);
    if (session?.poisoned) {
      const reason =
        session.poisonReason || "cursor-sdk session poisoned";
      void this.retireSession(session, reason, async () => {
        await this.invalidate(session, reason);
        return true;
      }).catch((err) => {
        this.logger?.warn?.(
          { err, sessionKey: session.key, agentId: session.agentId, reason },
          "cursor-sdk poisoned session retirement failed"
        );
      });
      return undefined;
    }
    if (session) touchSession(session);
    return session;
  }

  async retireSession(
    session: CursorSdkSession,
    reason: string,
    cleanup: () => Promise<boolean>
  ): Promise<boolean> {
    // Detach immediately even when another stale object with the same key is
    // still cleaning up. A queued retirement must never leave this session
    // available for another send.
    markSessionPoisoned(session, reason);
    if (this.sessions.get(session.key) === session) {
      this.sessions.delete(session.key);
    }

    let queue = this.retirements.get(session.key);
    if (!queue) {
      queue = {
        tail: Promise.resolve(),
        pending: new Map<CursorSdkSession, Promise<boolean>>(),
      };
      this.retirements.set(session.key, queue);
    }
    const duplicate = queue.pending.get(session);
    if (duplicate) {
      await duplicate.catch(() => undefined);
      return false;
    }

    const hasPendingCleanup = queue.pending.size > 0;
    const previous = queue.tail;
    const runCleanup = async () => {
      this.logger?.debug?.(
        {
          sessionKey: session.key,
          agentId: session.agentId,
          reason,
        },
        "cursor-sdk session retirement started"
      );
      return cleanup();
    };
    // Preserve the existing synchronous-start behavior for the first cleanup.
    // Later different-session objects are chained immediately into `tail`, so
    // getOrCreate cannot slip into the gap between their cleanup operations.
    const task = hasPendingCleanup
      ? previous.then(runCleanup)
      : runCleanup();
    queue.pending.set(session, task);
    const barrier = task.then(
      () => undefined,
      () => undefined
    );
    queue.tail = barrier;

    try {
      return await task;
    } finally {
      if (queue.pending.get(session) === task) {
        queue.pending.delete(session);
      }
      if (
        this.retirements.get(session.key) === queue &&
        queue.tail === barrier &&
        queue.pending.size === 0
      ) {
        this.retirements.delete(session.key);
      }
    }
  }

  async getOrCreate(options: {
    key: string;
    apiKey: string;
    model: ModelSelection;
    mode: CursorSdkMode;
    cursorCwd?: string;
    sandboxEnabled?: boolean;
    hostEnv?: HostEnvironment;
  }): Promise<CursorSdkSession> {
    installCursorAuthExchangeCache();
    // A client can reconnect before ReadableStream cancellation finishes, and
    // several reconnects can wake together. Wait for both retirement and any
    // in-progress replacement creation, then re-check the manager state.
    while (true) {
      const retirement = this.retirements.get(options.key);
      if (retirement) {
        await retirement.tail;
        continue;
      }
      const existing = this.get(options.key);
      if (existing) return existing;
      // get() starts a retirement when it discovers a poisoned session.
      // Re-check before creating so disposal cannot be overtaken.
      const discoveredRetirement = this.retirements.get(options.key);
      if (discoveredRetirement) {
        await discoveredRetirement.tail;
        continue;
      }
      const creation = this.creations.get(options.key);
      if (!creation) break;
      await creation;
    }

    let releaseCreation!: () => void;
    const creation = new Promise<void>((resolve) => {
      releaseCreation = resolve;
    });
    this.creations.set(options.key, creation);

    try {
      this.evictIfNeeded();

      const hostEnv = options.hostEnv || EMPTY_HOST_ENVIRONMENT;
      const workspaceDir =
        options.mode === "agent" && options.cursorCwd
          ? options.cursorCwd
          : ensureWorkspace(options.key, hostEnv);

      if (options.mode === "bridge") {
        ensureDenyHooksWorkspace(workspaceDir, hostEnv);
      }

      const agentMode = options.mode === "plan" ? "plan" : "agent";
      const sandboxEnabled = shouldEnableCursorSandbox(options.sandboxEnabled);

      const agent =
        (await this.rehydratePersistedAgent(
          options,
          workspaceDir,
          sandboxEnabled
        )) ||
        (await Agent.create({
          apiKey: options.apiKey,
          model: options.model,
          name: `ccr-${options.key.slice(0, 8)}`,
          mode: agentMode,
          local: {
            cwd: workspaceDir,
            settingSources: [],
            sandboxOptions: { enabled: sandboxEnabled },
            enableAgentRetries: true,
          },
        }));
      putPersistedSession("cursor", options.key, {
        sessionId: agent.agentId,
        workspaceDir,
        model: options.model.id,
      });

      const session = this.createSessionRecord({
        key: options.key,
        agent,
        mode: options.mode,
        workspaceDir,
        hostEnv,
        guidanceFingerprint: hostEnv.fingerprint,
      });

      this.sessions.set(options.key, session);
      this.logger?.info?.(
        {
          sessionKey: options.key,
          agentId: agent.agentId,
          mode: options.mode,
          sandboxEnabled,
        },
        "cursor-sdk session created"
      );
      return session;
    } finally {
      if (this.creations.get(options.key) === creation) {
        this.creations.delete(options.key);
      }
      releaseCreation();
    }
  }

  async resume(options: {
    key: string;
    agentId: string;
    apiKey: string;
    model?: ModelSelection;
    mode: CursorSdkMode;
    workspaceDir: string;
    sandboxEnabled?: boolean;
    hostEnv?: HostEnvironment;
  }): Promise<CursorSdkSession> {
    installCursorAuthExchangeCache();
    const sandboxEnabled = shouldEnableCursorSandbox(options.sandboxEnabled);
    const hostEnv = options.hostEnv || EMPTY_HOST_ENVIRONMENT;
    if (options.mode === "bridge") {
      ensureDenyHooksWorkspace(options.workspaceDir, hostEnv);
    }
    const agent = await Agent.resume(options.agentId, {
      apiKey: options.apiKey,
      model: options.model,
      local: {
        cwd: options.workspaceDir,
        settingSources: [],
        sandboxOptions: { enabled: sandboxEnabled },
      },
    });
    const session = this.createSessionRecord({
      key: options.key,
      agent,
      mode: options.mode,
      workspaceDir: options.workspaceDir,
      hostEnv,
      guidanceFingerprint: hostEnv.fingerprint,
    });
    this.sessions.set(options.key, session);
    return session;
  }

  resolveParkedTools(
    session: CursorSdkSession,
    results: Array<{ toolCallId: string; content: string }>
  ): number {
    let resolved = 0;
    const unmatched: string[] = [];

    // Ids are sanitized for Anthropic's tool_use.id alphabet, so two distinct
    // upstream ids can in principle collapse onto one. findIndex+splice would
    // then pair a result with the wrong call and never report it.
    const duplicates = session.parked
      .map((p) => p.id)
      .filter((id, i, all) => id && all.indexOf(id) !== i);
    if (duplicates.length) {
      this.logger?.warn?.(
        { sessionKey: session.key, duplicates: [...new Set(duplicates)] },
        "cursor-sdk parked tools share an id; results may mispair"
      );
    }

    for (const result of results) {
      let idx = -1;
      if (result.toolCallId) {
        idx = session.parked.findIndex((p) => p.id === result.toolCallId);
      } else if (results.length === 1 && session.parked.length === 1) {
        // Empty id is only safe when the mapping is unambiguous.
        idx = 0;
      }
      if (idx === -1) {
        unmatched.push(result.toolCallId || "(empty)");
        continue;
      }
      const [parked] = session.parked.splice(idx, 1);
      parked.resolve(result.content);
      resolved++;
    }
    if (unmatched.length) {
      this.logger?.warn?.(
        {
          unmatched,
          parkedRemaining: session.parked.map((p) => p.id),
          resolved,
        },
        "cursor-sdk tool results did not match parked tools"
      );
    }
    touchSession(session);
    return resolved;
  }

  async dispose(key: string): Promise<void> {
    const session = this.sessions.get(key);
    if (!session) return;
    const reason = "cursor-sdk session disposed";
    await this.retireSession(session, reason, async () => {
      await cancelActiveRun(session, {
        rejectParked: true,
        reason,
      });
      await this.disposeAgent(session, reason);
      // Scratch workspaces are per-session and hold only generated files. Left
      // behind they accumulate forever on the mounted config volume.
      removeManagedWorkspace(session.workspaceDir, this.logger);
      return true;
    });
  }

  invalidate(
    sessionOrKey: CursorSdkSession | string,
    reason: string
  ): Promise<void> {
    const session =
      typeof sessionOrKey === "string"
        ? this.sessions.get(sessionOrKey)
        : sessionOrKey;
    if (!session) return Promise.resolve();

    markSessionPoisoned(session, reason);
    session.run = undefined;
    session.streamIterator = undefined;
    void session.streamNext?.catch(() => undefined);
    session.streamNext = undefined;
    session.streamNextRunToken = undefined;
    session.activeRunToken = undefined;
    session.pendingEmit = [];
    session.pendingSdkMessages = [];
    if (session.parked.length) {
      for (const parked of session.parked.splice(0, session.parked.length)) {
        try {
          parked.reject(new Error(reason));
        } catch {
          // ignore
        }
      }
    }
    session.notifyEmit();
    session.notifySdkMessage();
    if (this.sessions.get(session.key) === session) {
      this.sessions.delete(session.key);
    }
    // In-process lifecycle is unchanged (fresh agent next): only the durable
    // binding is forgotten. A process death skips invalidate entirely, so the
    // binding survives restarts — that is the whole point of persistence.
    deletePersistedSession("cursor", session.key);
    this.logger?.warn?.(
      {
        sessionKey: session.key,
        agentId: session.agentId,
        reason,
      },
      "cursor-sdk session invalidated"
    );
    return this.disposeAgent(session, reason);
  }

  private async disposeAgent(
    session: CursorSdkSession,
    reason: string
  ): Promise<void> {
    try {
      // SDKAgent.close() is fire-and-forget and leaves releaseExecutorLease()
      // unobserved. The SDK's canonical async disposer owns that promise.
      // Attach a catch before racing the timeout so a late AbortError cannot
      // become an unhandledRejection after withTimeout has already rejected.
      const disposal = session.agent[Symbol.asyncDispose]();
      void disposal.catch(() => undefined);
      await withTimeout(
        disposal,
        CURSOR_AGENT_DISPOSE_TIMEOUT_MS,
        "cursor-sdk agent disposal timed out"
      );
    } catch (err) {
      const details = {
        err,
        sessionKey: session.key,
        agentId: session.agentId,
        reason,
      };
      if (isAbortError(err)) {
        this.logger?.debug?.(
          details,
          "cursor-sdk agent disposal observed expected cancellation"
        );
      } else {
        this.logger?.warn?.(details, "cursor-sdk agent disposal failed");
      }
    }
  }

  private async rehydratePersistedAgent(
    options: {
      key: string;
      apiKey: string;
      model: ModelSelection;
    },
    workspaceDir: string,
    sandboxEnabled: boolean
  ): Promise<SDKAgent | undefined> {
    // Restart survival: the binding outlives the process (registry file on the
    // mounted volume) while the live agent does not. Agent.resume reattaches
    // to the SDK's own on-disk agent store, preserving the server-side
    // conversation prefix. Anything unusable falls through to a fresh create.
    const binding = getPersistedSession("cursor", options.key);
    if (!binding) return undefined;
    if (binding.model && binding.model !== options.model.id) return undefined;
    if (
      binding.workspaceDir &&
      binding.workspaceDir !== workspaceDir &&
      !existsSync(binding.workspaceDir)
    ) {
      return undefined;
    }
    try {
      const agent = await Agent.resume(binding.sessionId, {
        apiKey: options.apiKey,
        model: options.model,
        local: {
          cwd: workspaceDir,
          settingSources: [],
          sandboxOptions: { enabled: sandboxEnabled },
          enableAgentRetries: true,
        },
      });
      if (!agent || agent.agentId !== binding.sessionId) return undefined;
      this.logger?.info?.(
        { sessionKey: options.key, agentId: binding.sessionId },
        "cursor-sdk session resumed from persisted binding"
      );
      return agent;
    } catch (err) {
      this.logger?.debug?.(
        { err, sessionKey: options.key },
        "cursor-sdk persisted agent resume failed; creating fresh"
      );
      deletePersistedSession("cursor", options.key);
      return undefined;
    }
  }

  private createSessionRecord(input: {
    key: string;
    agent: SDKAgent;
    mode: CursorSdkMode;
    workspaceDir: string;
    hostEnv: HostEnvironment;
    guidanceFingerprint?: string;
  }): CursorSdkSession {
    const session: CursorSdkSession = {
      key: input.key,
      agent: input.agent,
      agentId: input.agent.agentId,
      mode: input.mode,
      workspaceDir: input.workspaceDir,
      hostEnv: input.hostEnv,
      parked: [],
      toolIdAliases: { byOriginal: new Map(), byAlias: new Map() },
      pendingEmit: [],
      emitWaiters: [],
      pendingSdkMessages: [],
      sdkMessageWaiters: [],
      sendChain: Promise.resolve(),
      hasSentPrompt: false,
      guidanceFingerprint: input.guidanceFingerprint,
      lastActiveAt: Date.now(),
      metrics: {
        customToolCalls: 0,
        builtinToolCallsSeen: 0,
        scratchPathViolations: 0,
        scratchPathCorrections: 0,
      },
      parkHostTool: ({ id, name, args }) => {
        let resolve!: (result: string) => void;
        let reject!: (err: Error) => void;
        const promise = new Promise<string>((res, rej) => {
          resolve = res;
          reject = rej;
        });
        const runToken = session.activeRunToken;
        // Park the original SDK id; emit the host-safe alias. Incoming
        // echoes are translated back before matching (see resolveHostToolId).
        const alias = aliasHostToolId(session, id) ?? id;
        session.parked.push({ id, name, args, runToken, resolve, reject, promise });
        session.pendingEmit.push({ id: alias, name, args, runToken });
        session.notifyEmit();
        return promise;
      },
      notifyEmit: () => {
        const waiters = session.emitWaiters.splice(0, session.emitWaiters.length);
        for (const w of waiters) w();
      },
      waitForEmit: () =>
        new Promise<void>((resolve) => {
          if (session.pendingEmit.length) {
            resolve();
            return;
          }
          session.emitWaiters.push(resolve);
        }),
      enqueueSdkMessage: (message, runToken = session.activeRunToken) => {
        session.pendingSdkMessages.push({
          message,
          runToken,
          source: "delta",
        });
        session.notifySdkMessage();
      },
      notifySdkMessage: () => {
        const waiters = session.sdkMessageWaiters.splice(
          0,
          session.sdkMessageWaiters.length
        );
        for (const w of waiters) w();
      },
      waitForSdkMessage: () =>
        new Promise<void>((resolve) => {
          if (session.pendingSdkMessages.length) {
            resolve();
            return;
          }
          session.sdkMessageWaiters.push(resolve);
        }),
    };
    return session;
  }

  private evictIfNeeded(): void {
    while (this.sessions.size >= SESSION_LRU_MAX) {
      let oldestKey: string | undefined;
      let oldestAt = Infinity;
      for (const [key, session] of this.sessions) {
        // Never dispose mid-stream / parked-tool sessions — lastActiveAt can be
        // stale during long SSE turns and must not make them look "oldest".
        if (isSessionInFlight(session)) continue;
        if (session.lastActiveAt < oldestAt) {
          oldestAt = session.lastActiveAt;
          oldestKey = key;
        }
      }
      if (!oldestKey) {
        this.logger?.warn?.(
          { size: this.sessions.size, max: SESSION_LRU_MAX },
          "cursor-sdk session capacity reached with all sessions in-flight; deferring eviction"
        );
        break;
      }
      void this.dispose(oldestKey);
    }
  }

  private evictIdle(): void {
    const now = Date.now();
    for (const [key, session] of this.sessions) {
      if (isSessionInFlight(session)) continue;
      if (now - session.lastActiveAt > SESSION_IDLE_TTL_MS) {
        this.logger?.info?.({ sessionKey: key }, "cursor-sdk idle session evicted");
        void this.dispose(key);
      }
    }
    this.sweepOrphanWorkspaces(now);
  }

  /**
   * Remove scratch workspaces left by earlier processes. `dispose` handles the
   * live case; this collects what a crash, kill, or pre-cleanup build left on
   * the volume. Rate limited and restricted to managed directories.
   */
  sweepOrphanWorkspaces(
    now = Date.now(),
    root: string = CURSOR_SDK_WORKSPACES_ROOT
  ): number {
    if (now - this.lastSweepAt < WORKSPACE_SWEEP_INTERVAL_MS) return 0;
    this.lastSweepAt = now;

    let entries: string[];
    try {
      entries = readdirSync(root);
    } catch {
      return 0;
    }

    const live = new Set(
      [...this.sessions.values()].map((session) => session.workspaceDir)
    );
    let removed = 0;

    for (const entry of entries) {
      const dir = join(root, entry);
      if (live.has(dir) || !isManagedWorkspacePath(dir, root)) continue;
      try {
        const stat = statSync(dir);
        if (!stat.isDirectory()) continue;
        if (now - stat.mtimeMs <= ORPHAN_WORKSPACE_TTL_MS) continue;
      } catch {
        continue;
      }
      if (removeManagedWorkspace(dir, this.logger, root)) removed++;
    }

    if (removed) {
      this.logger?.info?.({ removed, root }, "cursor-sdk orphan workspaces swept");
    }
    return removed;
  }
}

export const globalSessionManager = new SessionManager();

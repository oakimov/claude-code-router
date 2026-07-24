import { createHash } from "crypto";
import { existsSync, mkdirSync } from "fs";
import { join } from "path";
import { Agent, type ModelSelection, type Run, type SDKAgent } from "@cursor/sdk";
import type { OpenAiUsage } from "./usage";
import { ensureDenyHooksWorkspace } from "./hooks-template";
import {
  CURSOR_SDK_WORKSPACES_ROOT,
  SESSION_IDLE_TTL_MS,
  SESSION_LRU_MAX,
  hashSessionFingerprint,
  type CursorSdkMode,
} from "./shared";

export type ParkedTool = {
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
  run?: Run;
  streamIterator?: AsyncIterator<any>;
  activeRunToken?: symbol;
  lastSdkUsageRaw?: OpenAiUsage;
  parked: ParkedTool[];
  /** Tool calls waiting to be emitted on the current SSE response. */
  pendingEmit: Array<{
    id: string;
    name: string;
    args: Record<string, unknown>;
    runToken?: symbol;
  }>;
  emitWaiters: Array<() => void>;
  /**
   * Serializes cancel/send so a follow-up compact/retry cannot call
   * `agent.send` while a prior run is still marked active in the SDK store.
   */
  sendChain: Promise<void>;
  /** True after at least one successful `agent.send` on this session. */
  hasSentPrompt: boolean;
  lastActiveAt: number;
  metrics: {
    customToolCalls: number;
    builtinToolCallsSeen: number;
  };
  parkHostTool: (tool: {
    id: string;
    name: string;
    args: Record<string, unknown>;
  }) => Promise<string>;
  notifyEmit: () => void;
  waitForEmit: () => Promise<void>;
};

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
  } = {}
): Promise<void> {
  if (options.onlyRunToken && session.activeRunToken !== options.onlyRunToken) {
    return;
  }
  const run = session.run;
  const iterator = session.streamIterator;
  session.run = undefined;
  session.streamIterator = undefined;
  session.activeRunToken = undefined;
  session.pendingEmit = [];
  session.notifyEmit();

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

  if (iterator && typeof iterator.return === "function") {
    try {
      await withTimeout(
        iterator.return(undefined),
        options.timeoutMs ?? 2_000,
        "cursor-sdk stream iterator return timed out"
      );
    } catch {
      // ignore
    }
  }

  if (!run) return;
  if (run.status !== "running") return;

  try {
    await withTimeout(
      run.cancel(),
      options.timeoutMs ?? 2_000,
      "cursor-sdk run cancel timed out"
    );
  } catch {
    // Best-effort — next send may still succeed once store marks terminal.
  }
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

function ensureWorkspace(sessionKey: string): string {
  if (!existsSync(CURSOR_SDK_WORKSPACES_ROOT)) {
    mkdirSync(CURSOR_SDK_WORKSPACES_ROOT, { recursive: true });
  }
  const dir = join(CURSOR_SDK_WORKSPACES_ROOT, sessionKey);
  mkdirSync(dir, { recursive: true });
  ensureDenyHooksWorkspace(dir);
  return dir;
}

export function buildSessionKey(input: {
  headerSession?: string;
  metadataUserId?: string;
  model?: string;
  systemAndFirstUser?: string;
}): string {
  if (input.headerSession) {
    return createHash("sha256").update(input.headerSession).digest("hex").slice(0, 32);
  }
  if (input.metadataUserId) {
    const parts = input.metadataUserId.split("_session_");
    if (parts.length > 1 && parts[1]) {
      return createHash("sha256").update(parts[1]).digest("hex").slice(0, 32);
    }
    return createHash("sha256").update(input.metadataUserId).digest("hex").slice(0, 32);
  }
  return hashSessionFingerprint([
    input.model || "",
    input.systemAndFirstUser || "",
  ]);
}

/** True while a session has a live run, open stream, or unresolved host tools. */
export function isSessionInFlight(session: CursorSdkSession): boolean {
  return Boolean(
    session.streamIterator ||
      session.parked.length > 0 ||
      session.run?.status === "running"
  );
}

export function touchSession(session: CursorSdkSession): void {
  session.lastActiveAt = Date.now();
}

export class SessionManager {
  private sessions = new Map<string, CursorSdkSession>();
  private cleanupTimer?: ReturnType<typeof setInterval>;

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
    if (session) touchSession(session);
    return session;
  }

  async getOrCreate(options: {
    key: string;
    apiKey: string;
    model: ModelSelection;
    mode: CursorSdkMode;
    cursorCwd?: string;
    sandboxEnabled?: boolean;
  }): Promise<CursorSdkSession> {
    const existing = this.get(options.key);
    if (existing) return existing;

    this.evictIfNeeded();

    const workspaceDir =
      options.mode === "agent" && options.cursorCwd
        ? options.cursorCwd
        : ensureWorkspace(options.key);

    if (options.mode === "bridge") {
      ensureDenyHooksWorkspace(workspaceDir);
    }

    const agentMode = options.mode === "plan" ? "plan" : "agent";
    const sandboxEnabled = shouldEnableCursorSandbox(options.sandboxEnabled);

    const agent = await Agent.create({
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
    });

    const session = this.createSessionRecord({
      key: options.key,
      agent,
      mode: options.mode,
      workspaceDir,
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
  }

  async resume(options: {
    key: string;
    agentId: string;
    apiKey: string;
    model?: ModelSelection;
    mode: CursorSdkMode;
    workspaceDir: string;
    sandboxEnabled?: boolean;
  }): Promise<CursorSdkSession> {
    const sandboxEnabled = shouldEnableCursorSandbox(options.sandboxEnabled);
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
    this.sessions.delete(key);
    await cancelActiveRun(session, {
      rejectParked: true,
      reason: "cursor-sdk session disposed",
    });
    try {
      session.agent.close();
    } catch {
      // ignore
    }
  }

  private createSessionRecord(input: {
    key: string;
    agent: SDKAgent;
    mode: CursorSdkMode;
    workspaceDir: string;
  }): CursorSdkSession {
    const session: CursorSdkSession = {
      key: input.key,
      agent: input.agent,
      agentId: input.agent.agentId,
      mode: input.mode,
      workspaceDir: input.workspaceDir,
      parked: [],
      pendingEmit: [],
      emitWaiters: [],
      sendChain: Promise.resolve(),
      hasSentPrompt: false,
      lastActiveAt: Date.now(),
      metrics: { customToolCalls: 0, builtinToolCallsSeen: 0 },
      parkHostTool: ({ id, name, args }) => {
        let resolve!: (result: string) => void;
        let reject!: (err: Error) => void;
        const promise = new Promise<string>((res, rej) => {
          resolve = res;
          reject = rej;
        });
        const runToken = session.activeRunToken;
        session.parked.push({ id, name, args, runToken, resolve, reject, promise });
        session.pendingEmit.push({ id, name, args, runToken });
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
  }
}

export const globalSessionManager = new SessionManager();

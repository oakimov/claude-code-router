import { mkdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { CCR_HOME } from "./cursor-sdk/shared";

/**
 * Persistent registry for every session identity CCR mints.
 *
 * Several families, one mechanism — the stable lookup key already exists in each
 * case (Zen conversation id, cursor buildSessionKey hash); only the minted
 * value used to live in process memory and died on restart:
 *
 * - "zen":    conversationKey -> x-opencode-session (ses_…)
 * - "cursor": sessionKey      -> { agentId, workspaceDir, model }
 * - "cursor-inbound": hashed protocol conversation key -> internal Cursor id
 *   (cursor-sdk/inbound-session.ts; written on mint, claim, new user text,
 *   and at most hourly as a TTL refresh)
 *
 * The file lives under CCR_HOME (the mounted ~/.claude-code-router volume),
 * so bindings survive both restarts and image rebuilds. Plain JSON: values
 * are short strings, not blobs (unlike cursor-opencode-provider's pb.gz,
 * which persists raw Cursor protocol state we never see — the SDK owns that).
 *
 * Synchronous API, tiny file (entries capped per family), persistence on
 * writes only — never on the read path.
 */

export type PersistedSession = {
  /** The fixed CCR-minted id (ses_… for zen; SDK agentId for cursor). */
  sessionId: string;
  workspaceDir?: string;
  model?: string;
  /** `id|sorted params` from cursorModelFingerprint. Missing means pre-variant binding. */
  modelFingerprint?: string;
  /**
   * Cursor inbound opening row: a follow-up turn claimed it. A repeated
   * opening then needs a fresh id.
   */
  progressed?: boolean;
  /**
   * Cursor inbound opening row that replaced an unclaimed opening of an
   * identical earlier conversation. A follow-up cannot tell the two apart,
   * so it gets a fresh agent instead of claiming this one.
   */
  contested?: boolean;
  updatedAt: number;
};

export const SESSION_REGISTRY_TTL_MS = 24 * 3600 * 1000;
const SESSION_REGISTRY_MAX_ENTRIES = 256;
const SESSION_REGISTRY_FILE = "ccr-sessions.json";

function registryDir(): string {
  const override = process.env.CCR_SESSION_REGISTRY_DIR?.trim();
  if (override) return override;
  try {
    return CCR_HOME;
  } catch {
    return tmpdir();
  }
}

function registryFile(): string {
  return join(registryDir(), SESSION_REGISTRY_FILE);
}

type RegistryFileShape = {
  version: 1;
  families: Record<string, Record<string, PersistedSession>>;
};

let cache: Map<string, PersistedSession> | undefined;
let cacheFile = "";
let startupPrunedFile = "";

function compositeKey(family: string, key: string): string {
  return `${family}\0${key}`;
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value ? value : undefined;
}

function loadLocked(now: number): Map<string, PersistedSession> {
  const file = registryFile();
  if (cache && cacheFile === file) return cache;
  const loaded = new Map<string, PersistedSession>();
  try {
    const raw = readFileSync(file, "utf-8");
    const parsed = JSON.parse(raw) as Partial<RegistryFileShape>;
    const families = parsed?.families;
    if (families && typeof families === "object") {
      for (const [family, entries] of Object.entries(families)) {
        if (!entries || typeof entries !== "object") continue;
        for (const [key, value] of Object.entries(entries)) {
          if (
            !value ||
            typeof value !== "object" ||
            typeof (value as any).sessionId !== "string" ||
            !(value as any).sessionId
          ) {
            continue;
          }
          const binding = value as PersistedSession;
          if (
            typeof binding.updatedAt !== "number" ||
            now - binding.updatedAt > SESSION_REGISTRY_TTL_MS
          ) {
            continue;
          }
          const workspaceDir = optionalString(binding.workspaceDir);
          const model = optionalString(binding.model);
          const modelFingerprint = optionalString(binding.modelFingerprint);
          loaded.set(compositeKey(family, key), {
            sessionId: binding.sessionId,
            ...(workspaceDir ? { workspaceDir } : {}),
            ...(model ? { model } : {}),
            ...(modelFingerprint ? { modelFingerprint } : {}),
            ...(binding.progressed === true ? { progressed: true } : {}),
            ...(binding.contested === true ? { contested: true } : {}),
            updatedAt: binding.updatedAt,
          });
        }
      }
    }
  } catch {
    // Missing or corrupt file: start empty (a corrupt file is replaced on
    // the next write rather than crashing request handling).
  }
  cache = loaded;
  cacheFile = file;
  // Once per process: entries expired at load time are already excluded from
  // memory above — rewrite the file itself so it cannot accumulate stale
  // bindings when no writes ever trigger a rewrite.
  if (startupPrunedFile !== file) {
    startupPrunedFile = file;
    persistLocked(loaded);
  }
  return loaded;
}

function persistLocked(entries: Map<string, PersistedSession>): void {
  const file = registryFile();
  const families: Record<string, Record<string, PersistedSession>> = {};
  // The cap applies per family, so churn in one family (Cursor inbound
  // bindings) cannot evict another's (Cursor agents, Zen sessions).
  // Oldest-first so the cap evicts stale bindings, not fresh ones.
  const byFamily = new Map<string, Array<[string, PersistedSession]>>();
  for (const entry of entries) {
    const family = entry[0].slice(0, entry[0].indexOf("\0"));
    const list = byFamily.get(family);
    if (list) list.push(entry);
    else byFamily.set(family, [entry]);
  }
  const kept: Array<[string, PersistedSession]> = [];
  for (const [family, list] of byFamily) {
    list.sort((a, b) => a[1].updatedAt - b[1].updatedAt);
    for (const entry of list.slice(-SESSION_REGISTRY_MAX_ENTRIES)) {
      kept.push(entry);
      const key = entry[0].slice(family.length + 1);
      (families[family] ||= {})[key] = entry[1];
    }
  }
  const payload = JSON.stringify({ version: 1, families });
  try {
    mkdirSync(registryDir(), { recursive: true, mode: 0o700 });
  } catch {
    return;
  }
  const tmp = `${file}.${process.pid}.tmp`;
  try {
    writeFileSync(tmp, payload, { mode: 0o600 });
    renameSync(tmp, file);
  } catch {
    return;
  } finally {
    try {
      unlinkSync(tmp);
    } catch {
      // Already renamed or never written.
    }
  }
  if (kept.length !== entries.size) {
    cache = new Map(kept);
  }
}

/** Fixed id previously minted for this family+key, or undefined. */
export function getPersistedSession(
  family: string,
  key: string,
  now = Date.now()
): PersistedSession | undefined {
  if (!family || !key) return undefined;
  const entries = loadLocked(now);
  const binding = entries.get(compositeKey(family, key));
  if (!binding) return undefined;
  if (now - binding.updatedAt > SESSION_REGISTRY_TTL_MS) {
    entries.delete(compositeKey(family, key));
    persistLocked(entries);
    return undefined;
  }
  return { ...binding };
}

/** Record a newly minted fixed id. Overwrites any prior binding. */
export function putPersistedSession(
  family: string,
  key: string,
  value: Omit<PersistedSession, "updatedAt">,
  now = Date.now()
): PersistedSession {
  const entries = loadLocked(now);
  const binding: PersistedSession = { ...value, updatedAt: now };
  entries.set(compositeKey(family, key), binding);
  persistLocked(entries);
  return { ...binding };
}

/**
 * Apply several writes to one family with a single file rewrite: `put`
 * entries are stored (overwriting), then `remove` keys are dropped.
 */
export function updatePersistedSessions(
  family: string,
  changes: {
    put?: Array<{ key: string; value: Omit<PersistedSession, "updatedAt"> }>;
    remove?: string[];
  },
  now = Date.now()
): void {
  if (!family) return;
  const entries = loadLocked(now);
  let changed = false;
  for (const { key, value } of changes.put || []) {
    if (!key) continue;
    entries.set(compositeKey(family, key), { ...value, updatedAt: now });
    changed = true;
  }
  for (const key of changes.remove || []) {
    if (key && entries.delete(compositeKey(family, key))) changed = true;
  }
  if (changed) persistLocked(entries);
}

/** Forget a binding (session retire, Zen bucket re-roll). */
export function deletePersistedSession(family: string, key: string): void {
  if (!family || !key) return;
  const entries = loadLocked(Date.now());
  if (entries.delete(compositeKey(family, key))) {
    persistLocked(entries);
  }
}

/** Drop expired bindings; returns the number pruned. */
export function pruneSessionRegistry(now = Date.now()): number {
  const entries = loadLocked(now);
  let pruned = 0;
  for (const [composite, binding] of entries) {
    if (now - binding.updatedAt > SESSION_REGISTRY_TTL_MS) {
      entries.delete(composite);
      pruned += 1;
    }
  }
  if (pruned) persistLocked(entries);
  return pruned;
}

/** Test hook: drop in-memory state (the file itself is keyed by env dir). */
export function resetSessionRegistryForTests(): void {
  cache = undefined;
  cacheFile = "";
  startupPrunedFile = "";
}

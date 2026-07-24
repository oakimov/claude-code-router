import {
  closeSync,
  existsSync,
  fstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  renameSync,
  statSync,
  unlinkSync,
  writeFileSync,
} from "fs";
import { dirname, join } from "path";
import { homedir } from "os";

const DEFAULT_CODEX_AUTH_FILE = join(
  homedir(),
  ".claude-code-router",
  "codex_auth.json"
);
const REFRESH_WINDOW_SECONDS = 5 * 60;
const REFRESH_FALLBACK_SECONDS = 8 * 24 * 60 * 60;
const REFRESH_LOCK_STALE_MS = 2 * 60_000;
const REFRESH_LOCK_WAIT_MS = 35_000;

const OAUTH_CONFIG = {
  client_id: "app_EMoamEEZ73f0CkXaXp7hrann",
  token_endpoint: "https://auth.openai.com/oauth/token",
};

export interface CodexTokens {
  access_token: string;
  refresh_token?: string;
  id_token?: string;
  token_type: string;
  scope?: string;
  expires_at: number;
  account_id?: string;
  last_refresh?: number;
}

export interface CodexOAuthAuth {
  mode: "oauth";
  token: string;
  accountId?: string;
  isFedramp: boolean;
}

interface ChatGptAuthClaims {
  chatgpt_account_id?: string;
  chatgpt_account_is_fedramp?: boolean;
}

interface JwtPayload {
  exp?: number;
  "https://api.openai.com/auth"?: ChatGptAuthClaims;
}

export interface RefreshOptions {
  force?: boolean;
  previousAccessToken?: string;
  expectedAccountId?: string;
}

let refreshPromise: Promise<CodexTokens> | null = null;

function getAuthFilePath(): string {
  return process.env.CCR_CODEX_AUTH_FILE || DEFAULT_CODEX_AUTH_FILE;
}

function getRefreshLockPath(): string {
  return `${getAuthFilePath()}.refresh.lock`;
}

function getTokenEndpoint(): string {
  return process.env.CODEX_OAUTH_TOKEN_ENDPOINT || OAUTH_CONFIG.token_endpoint;
}

function decodeJwtPayload(token: string | undefined): JwtPayload | null {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 3 || !parts[1]) return null;

  try {
    return JSON.parse(
      Buffer.from(parts[1], "base64url").toString("utf8")
    ) as JwtPayload;
  } catch {
    return null;
  }
}

function tokenAccountId(tokens: CodexTokens): string | undefined {
  const explicit = tokens.account_id?.trim();
  if (explicit) return explicit;
  return decodeJwtPayload(tokens.id_token)?.[
    "https://api.openai.com/auth"
  ]?.chatgpt_account_id;
}

function tokenIsFedramp(tokens: CodexTokens): boolean {
  return (
    decodeJwtPayload(tokens.id_token)?.["https://api.openai.com/auth"]
      ?.chatgpt_account_is_fedramp === true
  );
}

function tokenExpiresAt(tokens: CodexTokens): number | undefined {
  const jwtExpiry = decodeJwtPayload(tokens.access_token)?.exp;
  if (Number.isFinite(jwtExpiry)) return jwtExpiry;
  return Number.isFinite(tokens.expires_at) ? tokens.expires_at : undefined;
}

function normalizeTokens(tokens: CodexTokens): CodexTokens {
  const accountId = tokenAccountId(tokens);
  const expiresAt = tokenExpiresAt(tokens);
  return {
    ...tokens,
    ...(accountId ? { account_id: accountId } : {}),
    ...(expiresAt ? { expires_at: expiresAt } : {}),
  };
}

export function loadTokens(): CodexTokens | null {
  try {
    const authFile = getAuthFilePath();
    if (!existsSync(authFile)) return null;
    const tokens = JSON.parse(readFileSync(authFile, "utf8")) as CodexTokens;
    if (!tokens.access_token) return null;
    return normalizeTokens(tokens);
  } catch {
    return null;
  }
}

export function saveTokens(tokens: CodexTokens): void {
  const authFile = getAuthFilePath();
  mkdirSync(dirname(authFile), { recursive: true });

  const normalized = normalizeTokens(tokens);
  const tempFile = `${authFile}.${process.pid}.${Date.now()}.tmp`;
  try {
    writeFileSync(tempFile, JSON.stringify(normalized, null, 2), {
      mode: 0o600,
      encoding: "utf8",
    });
    renameSync(tempFile, authFile);
  } finally {
    try {
      if (existsSync(tempFile)) unlinkSync(tempFile);
    } catch {
      // The atomic rename already succeeded or cleanup is best-effort.
    }
  }
}

export function isTokenExpired(
  tokens: CodexTokens,
  leewaySeconds = REFRESH_WINDOW_SECONDS
): boolean {
  const expiresAt = tokenExpiresAt(tokens);
  if (expiresAt) {
    return Date.now() / 1000 + leewaySeconds >= expiresAt;
  }
  return Boolean(
    tokens.last_refresh &&
      Date.now() / 1000 >= tokens.last_refresh + REFRESH_FALLBACK_SECONDS
  );
}

function assertSameAccount(
  expectedAccountId: string | undefined,
  tokens: CodexTokens
): void {
  const currentAccountId = tokenAccountId(tokens);
  if (
    expectedAccountId &&
    expectedAccountId !== currentAccountId
  ) {
    throw new Error(
      "Codex OAuth account changed while credentials were being refreshed. Retry the request."
    );
  }
}

function sanitizeTokenEndpointError(body: string): string {
  try {
    const parsed = JSON.parse(body);
    const code = parsed?.error?.code || parsed?.error;
    const message = parsed?.error?.message || parsed?.error_description;
    return [code, message].filter((value) => typeof value === "string").join(": ");
  } catch {
    return body.slice(0, 240).replace(
      /(access_token|refresh_token|id_token)["']?\s*[:=]\s*["']?[^"',\s}]+/gi,
      "$1=<redacted>"
    );
  }
}

async function requestRefreshedTokens(
  current: CodexTokens
): Promise<CodexTokens> {
  if (!current.refresh_token) {
    throw new Error(
      "Codex OAuth token cannot be refreshed because no refresh token is available. Run `ccr codex-auth` again."
    );
  }

  const response = await fetch(getTokenEndpoint(), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      client_id: OAUTH_CONFIG.client_id,
      grant_type: "refresh_token",
      refresh_token: current.refresh_token,
    }),
    signal: AbortSignal.timeout(30_000),
  });

  const responseText = await response.text();
  if (!response.ok) {
    throw new Error(
      `Codex OAuth token refresh failed (${response.status}): ${sanitizeTokenEndpointError(
        responseText
      )}`
    );
  }

  const data = JSON.parse(responseText);
  if (typeof data.access_token !== "string" || !data.access_token) {
    throw new Error("Codex OAuth token refresh response did not include an access token.");
  }

  const now = Date.now() / 1000;
  return normalizeTokens({
    ...current,
    access_token: data.access_token,
    refresh_token: data.refresh_token || current.refresh_token,
    id_token: data.id_token || current.id_token,
    token_type: data.token_type || current.token_type || "Bearer",
    scope: data.scope || current.scope,
    expires_at:
      data.expires_at ||
      now + (typeof data.expires_in === "number" ? data.expires_in : 3600),
    account_id: data.account_id || current.account_id,
    last_refresh: now,
  });
}

function tryAcquireRefreshLock(): number | null {
  const lockPath = getRefreshLockPath();
  try {
    return openSync(lockPath, "wx", 0o600);
  } catch (error: any) {
    if (error?.code !== "EEXIST") throw error;
    try {
      if (Date.now() - statSync(lockPath).mtimeMs > REFRESH_LOCK_STALE_MS) {
        unlinkSync(lockPath);
      }
    } catch {
      // Another process may have released or replaced the lock.
    }
    return null;
  }
}

async function withRefreshFileLock<T>(operation: () => Promise<T>): Promise<T> {
  const deadline = Date.now() + REFRESH_LOCK_WAIT_MS;
  let fd: number | null = null;
  while (fd === null) {
    fd = tryAcquireRefreshLock();
    if (fd !== null) break;
    if (Date.now() >= deadline) {
      throw new Error("Timed out waiting for another CCR process to refresh Codex OAuth credentials.");
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  try {
    return await operation();
  } finally {
    const lockInode = fstatSync(fd).ino;
    closeSync(fd);
    try {
      if (statSync(getRefreshLockPath()).ino === lockInode) {
        unlinkSync(getRefreshLockPath());
      }
    } catch {
      // Another process may already have removed or replaced a stale lock.
    }
  }
}

async function refreshTokensInternal(
  initial: CodexTokens,
  options: RefreshOptions
): Promise<CodexTokens> {
  return withRefreshFileLock(async () => {
    const current = loadTokens();
    if (!current) {
      throw new Error(
        "Codex OAuth credentials disappeared while refreshing. Run `ccr codex-auth` again."
      );
    }

    assertSameAccount(options.expectedAccountId, current);
    const previousAccessToken =
      options.previousAccessToken || initial.access_token;
    if (current.access_token !== previousAccessToken) {
      return current;
    }
    if (!options.force && !isTokenExpired(current)) {
      return current;
    }

    const refreshed = await requestRefreshedTokens(current);
    assertSameAccount(options.expectedAccountId, refreshed);

    const latest = loadTokens();
    if (latest && latest.access_token !== current.access_token) {
      assertSameAccount(options.expectedAccountId, latest);
      return latest;
    }

    saveTokens(refreshed);
    return refreshed;
  });
}

export async function refreshTokens(
  refreshToken: string
): Promise<CodexTokens> {
  const current = loadTokens();
  if (!current) {
    throw new Error("No Codex OAuth credentials are available to refresh.");
  }
  return requestRefreshedTokens({ ...current, refresh_token: refreshToken });
}

export async function getValidAccessToken(
  options: RefreshOptions = {}
): Promise<CodexTokens> {
  const tokens = loadTokens();
  if (!tokens) {
    throw new Error(
      "No Codex OAuth tokens found. Run `ccr codex-auth` to authenticate."
    );
  }

  if (!options.force && !isTokenExpired(tokens)) return tokens;

  if (!refreshPromise) {
    refreshPromise = refreshTokensInternal(tokens, {
      ...options,
      expectedAccountId:
        options.expectedAccountId || tokenAccountId(tokens),
    }).finally(() => {
      refreshPromise = null;
    });
  }
  return refreshPromise;
}

export function toCodexOAuthAuth(tokens: CodexTokens): CodexOAuthAuth {
  const normalized = normalizeTokens(tokens);
  return {
    mode: "oauth",
    token: normalized.access_token,
    accountId: tokenAccountId(normalized),
    isFedramp: tokenIsFedramp(normalized),
  };
}

export { getAuthFilePath, OAUTH_CONFIG };

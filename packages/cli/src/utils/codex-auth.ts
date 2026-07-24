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
} from "node:fs";
import { execFileSync } from "node:child_process";
import { homedir, arch, platform, release } from "node:os";
import { dirname, join } from "node:path";

const DEFAULT_AUTH_FILE = join(
  homedir(),
  ".claude-code-router",
  "codex_auth.json"
);
const CODEX_CLIENT_ORIGINATOR = "codex_cli_rs";
const CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const REFRESH_WINDOW_SECONDS = 5 * 60;
const REFRESH_FALLBACK_SECONDS = 8 * 24 * 60 * 60;
const LOCK_STALE_MS = 2 * 60_000;
const LOCK_WAIT_MS = 35_000;

interface CodexTokens {
  access_token: string;
  refresh_token?: string;
  id_token?: string;
  token_type?: string;
  scope?: string;
  expires_at?: number;
  account_id?: string;
  last_refresh?: number;
}

interface AuthClaims {
  chatgpt_account_id?: string;
  chatgpt_account_is_fedramp?: boolean;
}

interface JwtPayload {
  exp?: number;
  "https://api.openai.com/auth"?: AuthClaims;
}

interface WhoamiResponse {
  chatgpt_account_id?: string;
  chatgpt_account_is_fedramp?: boolean;
  chatgpt_user_id?: string;
  chatgpt_plan_type?: string;
}

export interface CliCodexAuth {
  mode: "pat" | "oauth";
  token: string;
  accountId?: string;
  isFedramp: boolean;
}

let refreshPromise: Promise<CodexTokens> | null = null;

function authFilePath(): string {
  return process.env.CCR_CODEX_AUTH_FILE || DEFAULT_AUTH_FILE;
}

function decodeJwt(token: string | undefined): JwtPayload | null {
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

function accountId(tokens: CodexTokens): string | undefined {
  return (
    tokens.account_id?.trim() ||
    decodeJwt(tokens.id_token)?.["https://api.openai.com/auth"]
      ?.chatgpt_account_id
  );
}

function isFedramp(tokens: CodexTokens): boolean {
  return (
    decodeJwt(tokens.id_token)?.["https://api.openai.com/auth"]
      ?.chatgpt_account_is_fedramp === true
  );
}

function expiresAt(tokens: CodexTokens): number | undefined {
  const jwtExpiry = decodeJwt(tokens.access_token)?.exp;
  if (typeof jwtExpiry === "number") return jwtExpiry;
  return tokens.expires_at;
}

function normalize(tokens: CodexTokens): CodexTokens {
  const resolvedAccountId = accountId(tokens);
  const resolvedExpiry = expiresAt(tokens);
  return {
    ...tokens,
    ...(resolvedAccountId ? { account_id: resolvedAccountId } : {}),
    ...(resolvedExpiry ? { expires_at: resolvedExpiry } : {}),
  };
}

function loadTokens(): CodexTokens | null {
  try {
    const file = authFilePath();
    if (!existsSync(file)) return null;
    const tokens = JSON.parse(readFileSync(file, "utf8")) as CodexTokens;
    if (!tokens.access_token) return null;
    return normalize(tokens);
  } catch {
    return null;
  }
}

function saveTokens(tokens: CodexTokens): void {
  const file = authFilePath();
  mkdirSync(dirname(file), { recursive: true });
  const tempFile = `${file}.${process.pid}.${Date.now()}.tmp`;
  try {
    writeFileSync(tempFile, JSON.stringify(normalize(tokens), null, 2), {
      mode: 0o600,
      encoding: "utf8",
    });
    renameSync(tempFile, file);
  } finally {
    try {
      if (existsSync(tempFile)) unlinkSync(tempFile);
    } catch {
      // Cleanup is best-effort after the atomic rename.
    }
  }
}

function needsRefresh(tokens: CodexTokens): boolean {
  const expiry = expiresAt(tokens);
  if (expiry) {
    return Date.now() / 1000 + REFRESH_WINDOW_SECONDS >= expiry;
  }
  return Boolean(
    tokens.last_refresh &&
      Date.now() / 1000 >= tokens.last_refresh + REFRESH_FALLBACK_SECONDS
  );
}

function assertAccount(
  expected: string | undefined,
  tokens: CodexTokens
): void {
  const actual = accountId(tokens);
  if (expected && expected !== actual) {
    throw new Error(
      "Codex OAuth account changed while the CLI was refreshing credentials. Retry the command."
    );
  }
}

function sanitizeRefreshError(body: string): string {
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

async function requestRefresh(current: CodexTokens): Promise<CodexTokens> {
  if (!current.refresh_token) {
    throw new Error(
      "Codex OAuth access token needs refresh, but no refresh token is available. Run `ccr codex-auth` again."
    );
  }

  const endpoint =
    process.env.CODEX_OAUTH_TOKEN_ENDPOINT ||
    "https://auth.openai.com/oauth/token";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      client_id: CODEX_OAUTH_CLIENT_ID,
      grant_type: "refresh_token",
      refresh_token: current.refresh_token,
    }),
    signal: AbortSignal.timeout(30_000),
  });
  const responseText = await response.text();
  if (!response.ok) {
    throw new Error(
      `Codex OAuth refresh failed (${response.status}): ${sanitizeRefreshError(
        responseText
      )}`
    );
  }

  const data = JSON.parse(responseText);
  if (typeof data.access_token !== "string" || !data.access_token) {
    throw new Error("Codex OAuth refresh response omitted the access token.");
  }

  const now = Date.now() / 1000;
  return normalize({
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

function tryAcquireLock(): number | null {
  const lockFile = `${authFilePath()}.refresh.lock`;
  try {
    return openSync(lockFile, "wx", 0o600);
  } catch (error: any) {
    if (error?.code !== "EEXIST") throw error;
    try {
      if (Date.now() - statSync(lockFile).mtimeMs > LOCK_STALE_MS) {
        unlinkSync(lockFile);
      }
    } catch {
      // The owning process may have just released the lock.
    }
    return null;
  }
}

async function withFileLock<T>(operation: () => Promise<T>): Promise<T> {
  const deadline = Date.now() + LOCK_WAIT_MS;
  let fd: number | null = null;
  while (fd === null) {
    fd = tryAcquireLock();
    if (fd !== null) break;
    if (Date.now() >= deadline) {
      throw new Error(
        "Timed out waiting for another CCR process to refresh Codex OAuth credentials."
      );
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  try {
    return await operation();
  } finally {
    const lockInode = fstatSync(fd).ino;
    closeSync(fd);
    try {
      const lockFile = `${authFilePath()}.refresh.lock`;
      if (statSync(lockFile).ino === lockInode) unlinkSync(lockFile);
    } catch {
      // Another process may have removed or replaced a stale lock.
    }
  }
}

async function refreshTokens(
  initial: CodexTokens,
  force: boolean,
  previousAccessToken: string,
  expectedAccountId: string | undefined
): Promise<CodexTokens> {
  return withFileLock(async () => {
    const current = loadTokens();
    if (!current) {
      throw new Error(
        "Codex OAuth credentials disappeared during refresh. Run `ccr codex-auth` again."
      );
    }
    assertAccount(expectedAccountId, current);
    if (current.access_token !== previousAccessToken) return current;
    if (!force && !needsRefresh(current)) return current;

    const refreshed = await requestRefresh(current);
    assertAccount(expectedAccountId, refreshed);
    const latest = loadTokens();
    if (latest && latest.access_token !== current.access_token) {
      assertAccount(expectedAccountId, latest);
      return latest;
    }
    saveTokens(refreshed);
    return refreshed;
  });
}

async function validOAuthTokens(
  force = false,
  previousAccessToken?: string,
  expectedAccountId?: string
): Promise<CodexTokens> {
  const tokens = loadTokens();
  if (!tokens) {
    throw new Error("Codex OAuth tokens not found. Run `ccr codex-auth` first.");
  }
  if (!force && !needsRefresh(tokens)) return tokens;

  if (!refreshPromise) {
    refreshPromise = refreshTokens(
      tokens,
      force,
      previousAccessToken || tokens.access_token,
      expectedAccountId || accountId(tokens)
    ).finally(() => {
      refreshPromise = null;
    });
  }
  return refreshPromise;
}

async function resolvePat(pat: string): Promise<CliCodexAuth> {
  const authApiBaseUrl = (
    process.env.CODEX_AUTHAPI_BASE_URL ||
    "https://auth.openai.com/api/accounts"
  ).replace(/\/+$/, "");
  const response = await fetch(
    `${authApiBaseUrl}/v1/user-auth-credential/whoami`,
    {
      headers: { Authorization: `Bearer ${pat}` },
      signal: AbortSignal.timeout(15_000),
    }
  );
  if (!response.ok) {
    throw new Error(
      `Codex PAT metadata request failed (${response.status}). Verify that the configured at- token is valid and has Codex access.`
    );
  }

  const data = (await response.json()) as WhoamiResponse;
  if (
    !data.chatgpt_account_id ||
    !data.chatgpt_user_id ||
    !data.chatgpt_plan_type
  ) {
    throw new Error(
      "Codex PAT metadata response is missing required account, user, or plan information."
    );
  }
  return {
    mode: "pat",
    token: pat,
    accountId: data.chatgpt_account_id,
    isFedramp: data.chatgpt_account_is_fedramp === true,
  };
}

export async function resolveCliCodexAuth(
  configuredApiKey: string | undefined
): Promise<CliCodexAuth> {
  const apiKey = configuredApiKey?.trim();
  if (apiKey?.startsWith("at-")) return resolvePat(apiKey);

  const tokens = await validOAuthTokens();
  return {
    mode: "oauth",
    token: tokens.access_token,
    accountId: accountId(tokens),
    isFedramp: isFedramp(tokens),
  };
}

export async function recoverCliCodexOAuth(
  previous: CliCodexAuth
): Promise<CliCodexAuth | null> {
  if (previous.mode !== "oauth") return null;
  const tokens = await validOAuthTokens(
    true,
    previous.token,
    previous.accountId
  );
  return {
    mode: "oauth",
    token: tokens.access_token,
    accountId: accountId(tokens),
    isFedramp: isFedramp(tokens),
  };
}

function osType(): string {
  if (platform() === "darwin") return "Mac OS";
  if (platform() === "linux") return "Linux";
  if (platform() === "win32") return "Windows";
  return platform() || "unknown";
}

function osVersion(): string {
  if (platform() === "darwin") {
    try {
      return execFileSync("sw_vers", ["-productVersion"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      }).trim();
    } catch {
      // Use the runtime OS release below.
    }
  }
  return release() || "unknown";
}

function architecture(): string {
  if (arch() === "x64") return "x86_64";
  return arch() || "unknown";
}

export function buildCliCodexHeaders(
  auth: CliCodexAuth,
  clientVersion: string
): Record<string, string> {
  const headers: Record<string, string> = {
    Authorization: `Bearer ${auth.token}`,
    originator: CODEX_CLIENT_ORIGINATOR,
    "User-Agent": `${CODEX_CLIENT_ORIGINATOR}/${clientVersion} (${osType()} ${osVersion()}; ${architecture()})`,
  };
  if (auth.accountId) headers["ChatGPT-Account-ID"] = auth.accountId;
  if (auth.isFedramp) headers["X-OpenAI-Fedramp"] = "true";
  return headers;
}

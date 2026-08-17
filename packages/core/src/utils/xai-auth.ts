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

const DEFAULT_XAI_AUTH_FILE = join(
  homedir(),
  ".claude-code-router",
  "xai_auth.json"
);
const REFRESH_LOCK_STALE_MS = 2 * 60_000;
const REFRESH_LOCK_WAIT_MS = 35_000;

// Public Grok-CLI OAuth client. Confirmed via xAI's
// /.well-known/openid-configuration (device_authorization_endpoint +
// urn:ietf:params:oauth:grant-type:device_code in grant_types_supported).
const OAUTH_CONFIG = {
  client_id: "b1a00492-073a-47ea-816f-4c329264a828",
  device_authorization_endpoint: "https://auth.x.ai/oauth2/device/code",
  token_endpoint: "https://auth.x.ai/oauth2/token",
  scope: "openid profile email offline_access grok-cli:access api:access",
};

const DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code";
const DEVICE_CODE_DEFAULT_INTERVAL_MS = 5_000;
const DEVICE_CODE_MIN_INTERVAL_MS = 1_000;
const DEVICE_CODE_SLOW_DOWN_INCREMENT_MS = 5_000;
const DEVICE_CODE_DEFAULT_EXPIRES_MS = 5 * 60 * 1000;
const DEVICE_CODE_POLL_SAFETY_MARGIN_MS = 3_000;

// Refresh a little before expiry so a single long-running call doesn't have
// to recover from a mid-flight 401.
const ACCESS_TOKEN_REFRESH_SKEW_SECONDS = 120;

export interface XaiTokens {
  access_token: string;
  refresh_token?: string;
  id_token?: string;
  token_type: string;
  scope?: string;
  expires_at?: number;
  last_refresh?: number;
}

export interface DeviceCodeResponse {
  device_code: string;
  user_code: string;
  verification_uri: string;
  verification_uri_complete?: string;
  expires_in?: number;
  interval?: number;
}

interface DeviceTokenErrorBody {
  error?: string;
  error_description?: string;
}

let refreshPromise: Promise<XaiTokens> | null = null;

function getAuthFilePath(): string {
  return process.env.CCR_XAI_AUTH_FILE || DEFAULT_XAI_AUTH_FILE;
}

function getRefreshLockPath(): string {
  return `${getAuthFilePath()}.refresh.lock`;
}

function authHeaders(): Record<string, string> {
  return {
    "Content-Type": "application/x-www-form-urlencoded",
    Accept: "application/json",
  };
}

/**
 * Decode the `exp` claim out of a JWT access_token without verifying the
 * signature — only used to decide whether to proactively refresh, never to
 * make trust decisions. Returns undefined for opaque tokens (no JWT shape),
 * which conservatively skips the proactive refresh and lets the 401-on-call
 * path drive it instead.
 */
function decodeJwtExpirySeconds(token: string | undefined): number | undefined {
  if (!token) return undefined;
  const parts = token.split(".");
  if (parts.length !== 3 || !parts[1]) return undefined;
  try {
    const payload = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"));
    return typeof payload?.exp === "number" ? payload.exp : undefined;
  } catch {
    return undefined;
  }
}

/**
 * xAI doesn't always return expires_in on refresh (opencode reference
 * comment), so the stored expires_at field is best-effort — the JWT exp
 * claim is the load-bearing check for tokens that lack a fresh stored
 * deadline.
 */
function tokenExpiresAt(tokens: XaiTokens): number | undefined {
  const jwtExpiry = decodeJwtExpirySeconds(tokens.access_token);
  if (Number.isFinite(jwtExpiry)) return jwtExpiry;
  return Number.isFinite(tokens.expires_at) ? tokens.expires_at : undefined;
}

export function loadTokens(): XaiTokens | null {
  try {
    const authFile = getAuthFilePath();
    if (!existsSync(authFile)) return null;
    const tokens = JSON.parse(readFileSync(authFile, "utf8")) as XaiTokens;
    if (!tokens.access_token) return null;
    return tokens;
  } catch {
    return null;
  }
}

export function saveTokens(tokens: XaiTokens): void {
  const authFile = getAuthFilePath();
  mkdirSync(dirname(authFile), { recursive: true });

  const tempFile = `${authFile}.${process.pid}.${Date.now()}.tmp`;
  try {
    writeFileSync(tempFile, JSON.stringify(tokens, null, 2), {
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

export function isTokenExpiring(
  tokens: XaiTokens,
  skewSeconds = ACCESS_TOKEN_REFRESH_SKEW_SECONDS
): boolean {
  const expiresAt = tokenExpiresAt(tokens);
  if (!expiresAt) return false;
  return Date.now() / 1000 + skewSeconds >= expiresAt;
}

function sanitizeTokenEndpointError(body: string): string {
  try {
    const parsed = JSON.parse(body);
    const code = parsed?.error;
    const message = parsed?.error_description;
    return [code, message].filter((value) => typeof value === "string").join(": ");
  } catch {
    return body.slice(0, 240).replace(
      /(access_token|refresh_token|id_token)["']?\s*[:=]\s*["']?[^"',\s}]+/gi,
      "$1=<redacted>"
    );
  }
}

export async function requestDeviceCode(): Promise<DeviceCodeResponse> {
  const response = await fetch(OAUTH_CONFIG.device_authorization_endpoint, {
    method: "POST",
    headers: authHeaders(),
    body: new URLSearchParams({
      client_id: OAUTH_CONFIG.client_id,
      scope: OAUTH_CONFIG.scope,
    }).toString(),
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(
      `xAI device code request failed (${response.status})${
        detail ? `: ${sanitizeTokenEndpointError(detail)}` : ""
      }`
    );
  }
  const json = (await response.json()) as DeviceCodeResponse;
  if (!json.device_code || !json.user_code || !json.verification_uri) {
    throw new Error(
      "xAI device code response is missing device_code / user_code / verification_uri"
    );
  }
  return json;
}

async function defaultSleep(ms: number): Promise<void> {
  await new Promise<void>((resolve) => setTimeout(resolve, ms));
}

/**
 * Normalize a server-supplied seconds value to milliseconds, falling back to
 * the supplied default when the input is missing, non-positive, or not a
 * finite number — defends the polling loop against a misbehaving device-code
 * endpoint (NaN would otherwise slip through and busy-loop via setTimeout(_, NaN)).
 */
function positiveSecondsToMs(value: unknown, defaultMs: number): number {
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : defaultMs;
}

export async function pollDeviceCodeToken(
  device: DeviceCodeResponse,
  options: { sleep?: (ms: number) => Promise<void>; now?: () => number } = {}
): Promise<XaiTokens> {
  const sleep = options.sleep ?? defaultSleep;
  const now = options.now ?? (() => Date.now());
  const expiresInMs = positiveSecondsToMs(device.expires_in, DEVICE_CODE_DEFAULT_EXPIRES_MS);
  const deadline = now() + expiresInMs;
  let intervalMs = Math.max(
    positiveSecondsToMs(device.interval, DEVICE_CODE_DEFAULT_INTERVAL_MS),
    DEVICE_CODE_MIN_INTERVAL_MS
  );

  while (now() < deadline) {
    const response = await fetch(OAUTH_CONFIG.token_endpoint, {
      method: "POST",
      headers: authHeaders(),
      body: new URLSearchParams({
        grant_type: DEVICE_CODE_GRANT_TYPE,
        client_id: OAUTH_CONFIG.client_id,
        device_code: device.device_code,
      }).toString(),
    });
    if (response.ok) {
      const data = (await response.json()) as any;
      const now2 = Date.now() / 1000;
      return {
        access_token: data.access_token,
        refresh_token: data.refresh_token,
        id_token: data.id_token,
        token_type: data.token_type || "Bearer",
        scope: data.scope,
        expires_at:
          typeof data.expires_in === "number" ? now2 + data.expires_in : undefined,
        last_refresh: now2,
      };
    }

    const body = (await response.json().catch(() => ({}))) as DeviceTokenErrorBody;
    const remaining = Math.max(0, deadline - now());
    // RFC 8628 §3.5: authorization_pending = keep polling at the same
    // interval; slow_down = bump the interval by >=5s and keep polling.
    // Anything else is terminal.
    if (body.error === "authorization_pending") {
      await sleep(Math.min(intervalMs + DEVICE_CODE_POLL_SAFETY_MARGIN_MS, remaining));
      continue;
    }
    if (body.error === "slow_down") {
      intervalMs += DEVICE_CODE_SLOW_DOWN_INCREMENT_MS;
      await sleep(Math.min(intervalMs + DEVICE_CODE_POLL_SAFETY_MARGIN_MS, remaining));
      continue;
    }
    if (body.error === "access_denied" || body.error === "authorization_denied") {
      throw new Error("xAI device authorization was denied");
    }
    if (body.error === "expired_token") {
      throw new Error("xAI device code expired - run `ccr xai-auth` again");
    }
    const detail = body.error_description ?? body.error ?? "";
    throw new Error(
      `xAI device token exchange failed (${response.status})${detail ? `: ${detail}` : ""}`
    );
  }
  throw new Error("xAI device authorization timed out - run `ccr xai-auth` again");
}

async function requestRefreshedTokens(current: XaiTokens): Promise<XaiTokens> {
  if (!current.refresh_token) {
    throw new Error(
      "xAI OAuth token cannot be refreshed because no refresh token is available. Run `ccr xai-auth` again."
    );
  }

  const response = await fetch(OAUTH_CONFIG.token_endpoint, {
    method: "POST",
    headers: authHeaders(),
    body: new URLSearchParams({
      grant_type: "refresh_token",
      refresh_token: current.refresh_token,
      client_id: OAUTH_CONFIG.client_id,
    }).toString(),
  });

  const responseText = await response.text();
  if (!response.ok) {
    throw new Error(
      `xAI OAuth token refresh failed (${response.status}): ${sanitizeTokenEndpointError(
        responseText
      )}`
    );
  }

  const data = JSON.parse(responseText);
  if (typeof data.access_token !== "string" || !data.access_token) {
    throw new Error("xAI OAuth token refresh response did not include an access token.");
  }

  const now = Date.now() / 1000;
  return {
    access_token: data.access_token,
    // xAI rotates refresh tokens on every use; fall back to the current one
    // only if the response omitted a replacement.
    refresh_token: data.refresh_token || current.refresh_token,
    id_token: data.id_token || current.id_token,
    token_type: data.token_type || current.token_type || "Bearer",
    scope: data.scope || current.scope,
    expires_at:
      typeof data.expires_in === "number" ? now + data.expires_in : current.expires_at,
    last_refresh: now,
  };
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
      throw new Error("Timed out waiting for another CCR process to refresh xAI OAuth credentials.");
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

async function refreshTokensInternal(initial: XaiTokens): Promise<XaiTokens> {
  return withRefreshFileLock(async () => {
    const current = loadTokens();
    if (!current) {
      throw new Error("xAI OAuth credentials disappeared while refreshing. Run `ccr xai-auth` again.");
    }
    if (current.access_token !== initial.access_token) {
      // Another process already refreshed while we waited for the lock.
      return current;
    }

    const refreshed = await requestRefreshedTokens(current);

    const latest = loadTokens();
    if (latest && latest.access_token !== current.access_token) {
      return latest;
    }

    saveTokens(refreshed);
    return refreshed;
  });
}

export async function refreshTokens(_refreshToken?: string): Promise<XaiTokens> {
  // Same single-flight + file lock as getValidAccessToken. xAI rotates
  // refresh_tokens on every use; a second unlocked refresh with the old
  // token gets invalid_grant. The argument is accepted for call-site
  // compatibility (Claude/Codex) but the lock path always reloads disk.
  return getValidAccessToken({ force: true });
}

export async function getValidAccessToken(
  options: { force?: boolean } = {}
): Promise<XaiTokens> {
  const tokens = loadTokens();
  if (!tokens) {
    throw new Error("No xAI OAuth tokens found. Run `ccr xai-auth` to authenticate.");
  }

  if (!options.force && !isTokenExpiring(tokens)) return tokens;

  if (!refreshPromise) {
    refreshPromise = refreshTokensInternal(tokens).finally(() => {
      refreshPromise = null;
    });
  }
  return refreshPromise;
}

export { getAuthFilePath, OAUTH_CONFIG };

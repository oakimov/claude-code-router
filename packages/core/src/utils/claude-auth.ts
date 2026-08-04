import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { dirname, join } from "path";
import { homedir } from "os";
import { randomBytes } from "crypto";

const DEFAULT_CLAUDE_AUTH_FILE = join(homedir(), ".claude-code-router", "claude_auth.json");
const DEFAULT_CLAUDE_DEVICE_FILE = join(homedir(), ".claude-code-router", "claude_device.json");
const DEVICE_ID_PATTERN = /^[0-9a-f]{64}$/;

function getClaudeAuthFilePath(): string {
  return process.env.CCR_CLAUDE_AUTH_FILE || DEFAULT_CLAUDE_AUTH_FILE;
}

function getClaudeDeviceFilePath(): string {
  return process.env.CCR_CLAUDE_DEVICE_FILE || DEFAULT_CLAUDE_DEVICE_FILE;
}

const OAUTH_CONFIG = {
  client_id: "9d1c250a-e61b-44d9-88ed-5944d1962f5e",
  authorization_endpoint: "https://claude.ai/oauth/authorize",
  token_endpoint: "https://platform.claude.com/v1/oauth/token",
  scope: "user:profile user:inference user:sessions:claude_code user:mcp_servers",
};

export interface ClaudeTokens {
  access_token: string;
  refresh_token?: string;
  id_token?: string;
  token_type: string;
  scope?: string;
  expires_at: number;
  last_refresh?: number;
}

export function loadTokens(): ClaudeTokens | null {
  try {
    const authFile = getClaudeAuthFilePath();
    if (!existsSync(authFile)) return null;
    const data = readFileSync(authFile, "utf-8");
    const tokens = JSON.parse(data);
    if (!tokens.access_token) return null;
    return tokens as ClaudeTokens;
  } catch {
    return null;
  }
}

export function saveTokens(tokens: ClaudeTokens): void {
  const authFile = getClaudeAuthFilePath();
  const dir = dirname(authFile);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(authFile, JSON.stringify(tokens, null, 2), {
    mode: 0o600,
    encoding: "utf-8",
  });
}

export function isTokenExpired(tokens: ClaudeTokens, leewaySeconds = 60): boolean {
  if (!tokens.expires_at) return false;
  return Date.now() / 1000 + leewaySeconds >= tokens.expires_at;
}

export async function refreshTokens(refreshToken: string): Promise<ClaudeTokens> {
  const response = await fetch(OAUTH_CONFIG.token_endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      grant_type: "refresh_token",
      client_id: OAUTH_CONFIG.client_id,
      refresh_token: refreshToken,
    }),
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Token refresh failed (${response.status}): ${text}`);
  }

  const data = await response.json() as any;
  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    id_token: data.id_token,
    token_type: data.token_type || "Bearer",
    scope: data.scope,
    expires_at: data.expires_at || Date.now() / 1000 + (data.expires_in || 3600),
    last_refresh: Date.now() / 1000,
  };
}

export async function getValidAccessToken(): Promise<ClaudeTokens> {
  let tokens = loadTokens();
  if (!tokens) {
    throw new Error(
      "No Claude OAuth tokens found. Run `ccr claude-auth` to authenticate."
    );
  }

  if (isTokenExpired(tokens)) {
    if (!tokens.refresh_token) {
      throw new Error(
        "Claude OAuth token expired and no refresh token available. Run `ccr claude-auth` to re-authenticate."
      );
    }
    tokens = await refreshTokens(tokens.refresh_token);
    saveTokens(tokens);
  }

  return tokens;
}

/**
 * Load the persisted synthesized-client device id, minting and persisting a
 * fresh 64-hex value on first use. Stored alongside the OAuth token file
 * (mode 0600) but kept separate from it — it is not a credential.
 */
export function loadOrCreateDeviceId(): string {
  const deviceFile = getClaudeDeviceFilePath();
  try {
    if (existsSync(deviceFile)) {
      const data = JSON.parse(readFileSync(deviceFile, "utf-8"));
      if (typeof data.device_id === "string" && DEVICE_ID_PATTERN.test(data.device_id)) {
        return data.device_id;
      }
    }
  } catch {
    // Fall through and regenerate.
  }

  const deviceId = randomBytes(32).toString("hex");
  const dir = dirname(deviceFile);
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
  writeFileSync(
    deviceFile,
    JSON.stringify({ device_id: deviceId }, null, 2),
    { mode: 0o600, encoding: "utf-8" }
  );
  return deviceId;
}

export { OAUTH_CONFIG, getClaudeAuthFilePath, getClaudeDeviceFilePath };

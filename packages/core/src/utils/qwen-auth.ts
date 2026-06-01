import { existsSync, readFileSync, writeFileSync, mkdirSync } from "fs";
import { join } from "path";
import { homedir } from "os";

const QWEN_AUTH_FILE = join(homedir(), ".claude-code-router", "qwen_auth.json");
const QWEN_TARGET = "https://qwen.aikit.club";

// Matches qwen-proxy.mjs: hard timeouts on upstream calls so a hung Qwen
// proxy never blocks a request.
const UPSTREAM_TIMEOUT_MS = 10_000;

// Matches qwen-proxy.mjs:14: `expiresAt - 120_000` leeway. Avoids races
// where the upstream rejects a token we still consider fresh.
const EXPIRY_LEEWAY_SECONDS = 120;

interface QwenTokens {
  token: string;
  expiresAt: number | null;
  updatedAt: number;
}

function loadTokens(): QwenTokens | null {
  try {
    if (!existsSync(QWEN_AUTH_FILE)) return null;
    const tokens = JSON.parse(readFileSync(QWEN_AUTH_FILE, "utf-8"));
    if (!tokens.token) return null;
    return tokens as QwenTokens;
  } catch {
    return null;
  }
}

function saveTokens(tokens: QwenTokens): void {
  const dir = join(homedir(), ".claude-code-router");
  if (!existsSync(dir)) {
    mkdirSync(dir, { recursive: true });
  }
  writeFileSync(QWEN_AUTH_FILE, JSON.stringify(tokens, null, 2), {
    mode: 0o600,
    encoding: "utf-8",
  });
}

/**
 * Extract `exp` (ms) from a JWT's payload section, or null if the token
 * is malformed or has no `exp` claim. Matches qwen-proxy.mjs:30-36.
 */
function extractExpFromJwt(token: string): number | null {
  try {
    const payload = JSON.parse(
      Buffer.from(token.split(".")[1], "base64url").toString()
    );
    if (payload.exp) return payload.exp * 1000;
  } catch {
    // Fall through
  }
  return null;
}

/**
 * Ask the upstream Qwen proxy to rotate the token. Returns the new access
 * token, or null if rotation failed (e.g. token revoked upstream).
 */
async function refreshToken(token: string): Promise<string | null> {
  try {
    const res = await fetch(`${QWEN_TARGET}/v1/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.access_token || null;
  } catch {
    return null;
  }
}

/**
 * Load a valid (non-expired) Qwen access token, transparently refreshing it
 * via the upstream rotation endpoint when the token is within 6 hours of
 * expiry. Throws with an actionable message if no token is configured or if
 * the refresh attempt fails.
 */
export async function getValidAccessToken(): Promise<QwenTokens> {
  let tokens = loadTokens();
  if (!tokens) {
    throw new Error(
      "No Qwen token found. Run `ccr qwen-auth` to authenticate."
    );
  }

  if (
    tokens.expiresAt &&
    Date.now() + EXPIRY_LEEWAY_SECONDS * 1000 >= tokens.expiresAt
  ) {
    const rotated = await refreshToken(tokens.token);
    if (!rotated) {
      throw new Error(
        "Qwen token expired and refresh failed. Run `ccr qwen-auth` to re-authenticate."
      );
    }
    tokens = {
      token: rotated,
      expiresAt: extractExpFromJwt(rotated),
      updatedAt: Date.now(),
    };
    saveTokens(tokens);
  }

  return tokens;
}

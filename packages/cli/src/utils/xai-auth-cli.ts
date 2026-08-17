import { join } from "path";
import { homedir } from "os";
import { existsSync, mkdirSync, writeFileSync } from "fs";

const XAI_AUTH_FILE = join(homedir(), ".claude-code-router", "xai_auth.json");

// Public Grok-CLI OAuth client (RFC 8628 device authorization grant).
// See packages/core/src/utils/xai-auth.ts for the server-side counterpart —
// the CLI package doesn't depend on @caeliq/llms, so this duplicates the
// minimal constants and poll loop, matching how codex-cli-auth.ts and
// claude-auth-cli.ts already duplicate their own OAuth constants.
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

interface DeviceCodeResponse {
  device_code: string;
  user_code: string;
  verification_uri: string;
  verification_uri_complete?: string;
  expires_in?: number;
  interval?: number;
}

function authHeaders(): Record<string, string> {
  return {
    "Content-Type": "application/x-www-form-urlencoded",
    Accept: "application/json",
  };
}

async function requestDeviceCode(): Promise<DeviceCodeResponse> {
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
    throw new Error(`xAI device code request failed (${response.status})${detail ? `: ${detail}` : ""}`);
  }
  const json = (await response.json()) as DeviceCodeResponse;
  if (!json.device_code || !json.user_code || !json.verification_uri) {
    throw new Error("xAI device code response is missing device_code / user_code / verification_uri");
  }
  return json;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function positiveSecondsToMs(value: unknown, defaultMs: number): number {
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : defaultMs;
}

async function pollDeviceCodeToken(device: DeviceCodeResponse): Promise<any> {
  const expiresInMs = positiveSecondsToMs(device.expires_in, DEVICE_CODE_DEFAULT_EXPIRES_MS);
  const deadline = Date.now() + expiresInMs;
  let intervalMs = Math.max(
    positiveSecondsToMs(device.interval, DEVICE_CODE_DEFAULT_INTERVAL_MS),
    DEVICE_CODE_MIN_INTERVAL_MS
  );

  while (Date.now() < deadline) {
    const response = await fetch(OAUTH_CONFIG.token_endpoint, {
      method: "POST",
      headers: authHeaders(),
      body: new URLSearchParams({
        grant_type: DEVICE_CODE_GRANT_TYPE,
        client_id: OAUTH_CONFIG.client_id,
        device_code: device.device_code,
      }).toString(),
    });
    if (response.ok) return response.json();

    const body = await response.json().catch(() => ({} as any));
    const remaining = Math.max(0, deadline - Date.now());
    if (body.error === "authorization_pending") {
      process.stdout.write(".");
      await sleep(Math.min(intervalMs + DEVICE_CODE_POLL_SAFETY_MARGIN_MS, remaining));
      continue;
    }
    if (body.error === "slow_down") {
      intervalMs += DEVICE_CODE_SLOW_DOWN_INCREMENT_MS;
      process.stdout.write(".");
      await sleep(Math.min(intervalMs + DEVICE_CODE_POLL_SAFETY_MARGIN_MS, remaining));
      continue;
    }
    if (body.error === "access_denied" || body.error === "authorization_denied") {
      throw new Error("xAI device authorization was denied.");
    }
    if (body.error === "expired_token") {
      throw new Error("xAI device code expired. Run `ccr xai-auth` again.");
    }
    const detail = body.error_description ?? body.error ?? "";
    throw new Error(`xAI device token exchange failed (${response.status})${detail ? `: ${detail}` : ""}`);
  }
  throw new Error("xAI device authorization timed out. Run `ccr xai-auth` again.");
}

function saveTokens(data: any): void {
  const dir = join(homedir(), ".claude-code-router");
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

  const now = Date.now() / 1000;
  const tokens = {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    id_token: data.id_token,
    token_type: data.token_type || "Bearer",
    scope: data.scope,
    expires_at: typeof data.expires_in === "number" ? now + data.expires_in : undefined,
    last_refresh: now,
  };

  writeFileSync(XAI_AUTH_FILE, JSON.stringify(tokens, null, 2), {
    mode: 0o600,
    encoding: "utf-8",
  });
}

function printConfigSnippets(): void {
  console.log("\nAdd to your config (OAuth, backed by the SuperGrok/X Premium+ login above):");
  console.log(`
{
  "name": "xai-subscription",
  "api_base_url": "https://api.x.ai/v1",
  "api_key": "no-key",
  "models": ["grok-4.6", "grok-4.3", "grok-code-fast-1"],
  "transformer": {
    "use": ["xai-auth", "openai-responses"]
  }
}`);
  console.log("\nOr, to use a plain xAI API key instead of OAuth (skips this login entirely):");
  console.log(`
{
  "name": "xai-api-key",
  "api_base_url": "https://api.x.ai/v1",
  "api_key": "xai-... (or $XAI_API_KEY)",
  "models": ["grok-4.6", "grok-4.3", "grok-code-fast-1"],
  "transformer": {
    "use": ["xai-auth", "openai-responses"]
  }
}`);
}

export async function runXaiAuth(): Promise<void> {
  console.log("Requesting a device code from xAI...\n");
  const device = await requestDeviceCode();

  console.log("Open this URL in your browser and approve access:\n");
  console.log(`  ${device.verification_uri_complete ?? device.verification_uri}`);
  if (!device.verification_uri_complete) {
    console.log(`\nThen enter this code when prompted: ${device.user_code}`);
  }
  console.log("\nThis works from any device with a browser — no local callback server, no port forwarding.");
  console.log("Waiting for approval");

  const tokens = await pollDeviceCodeToken(device);
  saveTokens(tokens);

  console.log("\n\nAuthentication successful!");
  if (typeof tokens.expires_in === "number") {
    console.log(`Access token expires in ${Math.round(tokens.expires_in / 60)} minutes (auto-refreshed).`);
  }
  printConfigSnippets();
}

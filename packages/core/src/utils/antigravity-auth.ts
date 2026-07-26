import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  unlinkSync,
  writeFileSync,
} from "fs";
import { dirname, join } from "path";
import { homedir } from "os";
import { randomUUID } from "crypto";

export const ANTIGRAVITY_CLIENT_ID =
  "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
export const ANTIGRAVITY_CLIENT_SECRET =
  "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";
export const ANTIGRAVITY_REDIRECT_URI =
  "http://localhost:51121/oauth-callback";
export const ANTIGRAVITY_SCOPES = [
  "https://www.googleapis.com/auth/cloud-platform",
  "https://www.googleapis.com/auth/userinfo.email",
  "https://www.googleapis.com/auth/userinfo.profile",
  "https://www.googleapis.com/auth/cclog",
  "https://www.googleapis.com/auth/experimentsandconfigs",
] as const;

export const ANTIGRAVITY_ENDPOINT_DAILY =
  "https://daily-cloudcode-pa.sandbox.googleapis.com";
export const ANTIGRAVITY_ENDPOINT_AUTOPUSH =
  "https://autopush-cloudcode-pa.sandbox.googleapis.com";
export const ANTIGRAVITY_ENDPOINT_PROD =
  "https://cloudcode-pa.googleapis.com";

/** generateContent fallback order */
export const ANTIGRAVITY_ENDPOINT_FALLBACKS = [
  ANTIGRAVITY_ENDPOINT_DAILY,
  ANTIGRAVITY_ENDPOINT_AUTOPUSH,
  ANTIGRAVITY_ENDPOINT_PROD,
] as const;

/** loadCodeAssist / fetchAvailableModels prefer prod first */
export const ANTIGRAVITY_LOAD_ENDPOINTS = [
  ANTIGRAVITY_ENDPOINT_PROD,
  ANTIGRAVITY_ENDPOINT_DAILY,
  ANTIGRAVITY_ENDPOINT_AUTOPUSH,
] as const;

export const ANTIGRAVITY_VERSION = "1.18.3";

const DEFAULT_AUTH_FILE = join(
  homedir(),
  ".claude-code-router",
  "antigravity_auth.json"
);
const EXPIRY_LEEWAY_MS = 120_000;
const TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";
const USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v2/userinfo";

export interface AntigravityTokens {
  access_token: string;
  refresh_token: string;
  expires_at: number; // ms epoch
  email?: string;
  project_id?: string;
}

let refreshPromise: Promise<AntigravityTokens> | null = null;
let cachedEndpoint: string | null = null;

export function getAuthFilePath(): string {
  return process.env.CCR_ANTIGRAVITY_AUTH_FILE || DEFAULT_AUTH_FILE;
}

export function getAntigravityHeaders(): Record<string, string> {
  return {
    "User-Agent": `Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Antigravity/${ANTIGRAVITY_VERSION} Chrome/138.0.7204.235 Electron/37.3.1 Safari/537.36`,
    "X-Goog-Api-Client": "google-cloud-sdk vscode_cloudshelleditor/0.1",
    "Client-Metadata": JSON.stringify({
      ideType: "ANTIGRAVITY",
      platform: "PLATFORM_UNSPECIFIED",
      pluginType: "GEMINI",
    }),
  };
}

export function loadTokens(): AntigravityTokens | null {
  try {
    const authFile = getAuthFilePath();
    if (!existsSync(authFile)) return null;
    const tokens = JSON.parse(
      readFileSync(authFile, "utf8")
    ) as AntigravityTokens;
    if (!tokens.access_token || !tokens.refresh_token) return null;
    return tokens;
  } catch {
    return null;
  }
}

export function saveTokens(tokens: AntigravityTokens): void {
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
      // best-effort cleanup
    }
  }
}

function isExpired(tokens: AntigravityTokens): boolean {
  return Date.now() + EXPIRY_LEEWAY_MS >= tokens.expires_at;
}

async function refreshAccessToken(
  tokens: AntigravityTokens
): Promise<AntigravityTokens> {
  const body = new URLSearchParams({
    client_id: ANTIGRAVITY_CLIENT_ID,
    client_secret: ANTIGRAVITY_CLIENT_SECRET,
    refresh_token: tokens.refresh_token,
    grant_type: "refresh_token",
  });

  const res = await fetch(TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Antigravity token refresh failed (${res.status}). Run \`ccr antigravity-auth\` again. ${text.slice(0, 200)}`
    );
  }

  const data = (await res.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  };

  if (!data.access_token) {
    throw new Error(
      "Antigravity token refresh returned no access_token. Run `ccr antigravity-auth` again."
    );
  }

  const next: AntigravityTokens = {
    ...tokens,
    access_token: data.access_token,
    refresh_token: data.refresh_token || tokens.refresh_token,
    expires_at: Date.now() + (data.expires_in || 3600) * 1000,
  };
  saveTokens(next);
  return next;
}

/**
 * Load a valid access token, refreshing with single-flight dedupe when near expiry.
 */
export async function getValidAccessToken(
  options?: { force?: boolean }
): Promise<AntigravityTokens> {
  const tokens = loadTokens();
  if (!tokens) {
    throw new Error(
      "No Antigravity OAuth tokens found. Run `ccr antigravity-auth` to authenticate."
    );
  }

  if (!options?.force && !isExpired(tokens)) {
    return tokens;
  }

  if (!refreshPromise) {
    refreshPromise = refreshAccessToken(tokens).finally(() => {
      refreshPromise = null;
    });
  }
  return refreshPromise;
}

function extractProjectId(payload: any): string | undefined {
  const project = payload?.cloudaicompanionProject;
  if (typeof project === "string" && project.trim()) return project.trim();
  if (project && typeof project.id === "string" && project.id.trim()) {
    return project.id.trim();
  }
  return undefined;
}

function extractTierId(payload: any): string | undefined {
  const fromTier = (tier: unknown): string | undefined => {
    if (!tier) return undefined;
    if (typeof tier === "string" && tier.trim()) return tier.trim();
    if (
      typeof tier === "object" &&
      typeof (tier as { id?: unknown }).id === "string" &&
      (tier as { id: string }).id.trim()
    ) {
      return (tier as { id: string }).id.trim();
    }
    return undefined;
  };

  return (
    fromTier(payload?.paidTier) ||
    fromTier(payload?.currentTier) ||
    fromTier(payload?.allowedTiers?.find((t: any) => t?.isDefault)) ||
    fromTier(payload?.allowedTiers?.[0]) ||
    undefined
  );
}

function codeAssistMetadata(): Record<string, string> {
  // Platform enum is NOT OS names — MACOS/WINDOWS return 400 INVALID_ARGUMENT.
  // Valid: PLATFORM_UNSPECIFIED (or omit). Project id comes from
  // loadCodeAssist.cloudaicompanionProject (string or {id}).
  return {
    ideType: "ANTIGRAVITY",
    platform: "PLATFORM_UNSPECIFIED",
    pluginType: "GEMINI",
  };
}

async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

async function pollOnboardOperation(
  accessToken: string,
  endpoint: string,
  operationName: string
): Promise<string | undefined> {
  const url = operationName.startsWith("http")
    ? operationName
    : `${endpoint}/${operationName.replace(/^\//, "")}`;
  const headers = {
    Authorization: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
    ...getAntigravityHeaders(),
  };

  for (let attempt = 0; attempt < 20; attempt++) {
    try {
      const res = await fetch(url, {
        method: "GET",
        headers,
        signal: AbortSignal.timeout(15_000),
      });
      if (!res.ok) {
        await sleep(2_000);
        continue;
      }
      const op = await res.json();
      if (op?.done === true) {
        return (
          extractProjectId(op?.response) ||
          extractProjectId(op) ||
          undefined
        );
      }
    } catch {
      // keep polling
    }
    await sleep(2_000);
  }
  return undefined;
}

/**
 * Ask Google to provision (or return) the managed Antigravity companion project.
 * Free-tier accounts get one automatically — you do not create it in GCP Console.
 */
async function onboardUser(
  accessToken: string,
  tierId: string,
  endpoint: string
): Promise<string | undefined> {
  try {
    const res = await fetch(`${endpoint}/v1internal:onboardUser`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
        ...getAntigravityHeaders(),
      },
      body: JSON.stringify({
        tierId,
        metadata: codeAssistMetadata(),
      }),
      signal: AbortSignal.timeout(30_000),
    });
    if (!res.ok) return undefined;
    const payload = await res.json();
    const immediate =
      extractProjectId(payload?.response) || extractProjectId(payload);
    if (payload?.done === true || immediate) {
      return immediate;
    }
    if (typeof payload?.name === "string" && payload.name.trim()) {
      return pollOnboardOperation(accessToken, endpoint, payload.name.trim());
    }
    return undefined;
  } catch {
    return undefined;
  }
}

function tierCandidates(payload: any): string[] {
  const primary = extractTierId(payload);
  const defaults = ["free-tier", "FREE"];
  return [...new Set([primary, ...defaults].filter(Boolean) as string[])];
}

async function discoverProjectId(accessToken: string): Promise<string | undefined> {
  const headers = {
    Authorization: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
    ...getAntigravityHeaders(),
  };
  const body = JSON.stringify({ metadata: codeAssistMetadata() });
  let lastPayload: any;

  for (const endpoint of ANTIGRAVITY_LOAD_ENDPOINTS) {
    try {
      const res = await fetch(`${endpoint}/v1internal:loadCodeAssist`, {
        method: "POST",
        headers,
        body,
        signal: AbortSignal.timeout(15_000),
      });
      if (!res.ok) continue;
      const payload = await res.json();
      lastPayload = payload;
      const projectId = extractProjectId(payload);
      if (projectId) return projectId;
    } catch {
      // try next endpoint
    }
  }

  // Google auto-provisions a managed project for free-tier Antigravity accounts.
  for (const tierId of tierCandidates(lastPayload)) {
    for (const endpoint of ANTIGRAVITY_LOAD_ENDPOINTS) {
      const onboarded = await onboardUser(accessToken, tierId, endpoint);
      if (onboarded) return onboarded;
    }
  }

  // Some accounts only surface the managed project after onboardUser.
  for (const endpoint of ANTIGRAVITY_LOAD_ENDPOINTS) {
    try {
      const res = await fetch(`${endpoint}/v1internal:loadCodeAssist`, {
        method: "POST",
        headers,
        body,
        signal: AbortSignal.timeout(15_000),
      });
      if (!res.ok) continue;
      const projectId = extractProjectId(await res.json());
      if (projectId) return projectId;
    } catch {
      // try next
    }
  }

  return undefined;
}

/**
 * Resolve GCP project id: provider config → auth file → loadCodeAssist/onboardUser.
 * Optional — many Antigravity accounts work with OAuth alone; missing project is fine.
 */
export async function resolveProjectId(
  provider: { project_id?: string } | undefined,
  accessToken: string
): Promise<string | undefined> {
  const fromConfig = provider?.project_id?.trim();
  if (fromConfig) return fromConfig;

  const tokens = loadTokens();
  if (tokens?.project_id?.trim()) return tokens.project_id.trim();

  const discovered = await discoverProjectId(accessToken);
  if (discovered) {
    const current = loadTokens();
    if (current) {
      saveTokens({ ...current, project_id: discovered });
    }
    return discovered;
  }

  return undefined;
}

export function getPreferredEndpoint(): string {
  return cachedEndpoint || ANTIGRAVITY_ENDPOINT_DAILY;
}

export function rememberEndpoint(endpoint: string): void {
  cachedEndpoint = endpoint;
}

export function clearPreferredEndpoint(): void {
  cachedEndpoint = null;
}

/**
 * Endpoints this account cannot use at all, e.g. a sandbox host whose API is not
 * enabled on the discovered project. Session-scoped so a cold start re-probes.
 */
const unusableEndpoints = new Set<string>();

/** A 403 SERVICE_DISABLED means the host's API is not enabled for the project. */
export function isEndpointDisabledError(body: string): boolean {
  return (
    body.includes("SERVICE_DISABLED") ||
    body.includes("has not been used in project")
  );
}

export function markEndpointUnusable(endpoint: string): void {
  unusableEndpoints.add(endpoint.replace(/\/$/, ""));
}

export function isEndpointUnusable(endpoint: string): boolean {
  return unusableEndpoints.has(endpoint.replace(/\/$/, ""));
}

export function clearUnusableEndpoints(): void {
  unusableEndpoints.clear();
}

/**
 * Whether a failing response should move on to the next endpoint rather than be
 * reported to the client.
 *
 * 404 and 5xx are the classic "wrong or unhealthy host" signals. 403 belongs
 * here too: the sandbox hosts answer with PERMISSION_DENIED / SERVICE_DISABLED
 * when the account's project is not entitled to that deployment, which is not
 * something the caller can act on — as long as another endpoint (prod is last)
 * is still untried. The final endpoint's error is always surfaced.
 */
export function shouldWalkEndpoint(
  status: number,
  hasMoreEndpoints: boolean
): boolean {
  if (!hasMoreEndpoints) return false;
  return status === 404 || status === 403 || status >= 500;
}

/**
 * Ordered endpoint candidates, skipping hosts already known to be unusable.
 * Never returns an empty list: if every host is marked, re-probe them all.
 */
export function antigravityEndpointCandidates(preferredBase?: string): string[] {
  const preferred = preferredBase?.replace(/\/$/, "") || getPreferredEndpoint();
  const ordered = [...new Set([preferred, ...ANTIGRAVITY_ENDPOINT_FALLBACKS])];
  const usable = ordered.filter((endpoint) => !isEndpointUnusable(endpoint));
  return usable.length ? usable : ordered;
}

export function buildGenerateContentUrl(
  endpoint: string,
  stream: boolean
): string {
  const action = stream
    ? "streamGenerateContent?alt=sse"
    : "generateContent";
  return `${endpoint}/v1internal:${action}`;
}

export function wrapAntigravityRequest(options: {
  project?: string;
  model: string;
  request: Record<string, any>;
}): Record<string, any> {
  return {
    ...(options.project ? { project: options.project } : {}),
    model: options.model,
    request: options.request,
    userAgent: "antigravity",
    requestId: randomUUID(),
  };
}

export async function fetchUserEmail(accessToken: string): Promise<string | undefined> {
  try {
    const res = await fetch(USERINFO_ENDPOINT, {
      headers: { Authorization: `Bearer ${accessToken}` },
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return undefined;
    const data = (await res.json()) as { email?: string };
    return data.email;
  } catch {
    return undefined;
  }
}

export async function exchangeAuthorizationCode(
  code: string,
  codeVerifier: string
): Promise<AntigravityTokens> {
  const body = new URLSearchParams({
    client_id: ANTIGRAVITY_CLIENT_ID,
    client_secret: ANTIGRAVITY_CLIENT_SECRET,
    code,
    grant_type: "authorization_code",
    redirect_uri: ANTIGRAVITY_REDIRECT_URI,
    code_verifier: codeVerifier,
  });

  const res = await fetch(TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Antigravity token exchange failed (${res.status}): ${text.slice(0, 300)}`
    );
  }

  const data = (await res.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  };

  if (!data.access_token || !data.refresh_token) {
    throw new Error(
      "Antigravity token exchange did not return access_token and refresh_token. " +
        "Re-auth with prompt=consent may be required."
    );
  }

  return {
    access_token: data.access_token,
    refresh_token: data.refresh_token,
    expires_at: Date.now() + (data.expires_in || 3600) * 1000,
  };
}

/** Parse RetryInfo.retryDelay like "3.957525076s" into milliseconds. */
export function parseRetryDelayMs(errorBody: string): number | undefined {
  try {
    const json = JSON.parse(errorBody);
    const details = json?.error?.details || json?.details;
    if (!Array.isArray(details)) return undefined;
    for (const detail of details) {
      const delay = detail?.retryDelay || detail?.retry_delay;
      if (typeof delay === "string") {
        const match = delay.match(/^([\d.]+)s$/);
        if (match) return Math.ceil(parseFloat(match[1]) * 1000);
      }
    }
  } catch {
    // ignore
  }
  return undefined;
}

/**
 * List models available to the authenticated Antigravity account/project.
 * POST /v1internal:fetchAvailableModels — Google requires a project for this call.
 */
export async function fetchAvailableModels(
  accessToken: string,
  projectId?: string,
  baseUrl?: string
): Promise<string[]> {
  if (!projectId?.trim()) {
    throw new Error(
      "Antigravity model listing requires a project id (OAuth alone is not enough for fetchAvailableModels). " +
        "Set provider.project_id or run `ccr antigravity-auth --project <id>`."
    );
  }

  const preferred = baseUrl?.replace(/\/$/, "");
  const endpoints = preferred
    ? [preferred, ...ANTIGRAVITY_LOAD_ENDPOINTS]
    : [...ANTIGRAVITY_LOAD_ENDPOINTS];
  const unique = [...new Set(endpoints)];
  const headers = {
    Authorization: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
    ...getAntigravityHeaders(),
  };
  const body = JSON.stringify({ project: projectId.trim() });
  let lastError: Error | undefined;

  for (const endpoint of unique) {
    try {
      const res = await fetch(`${endpoint}/v1internal:fetchAvailableModels`, {
        method: "POST",
        headers,
        body,
        signal: AbortSignal.timeout(30_000),
      });
      if (!res.ok) {
        const text = await res.text().catch(() => "");
        lastError = new Error(
          `fetchAvailableModels ${res.status} at ${endpoint}: ${text.slice(0, 200)}`
        );
        // Try other hosts on auth/permission and transient failures.
        if (
          res.status === 401 ||
          res.status === 403 ||
          res.status === 404 ||
          res.status >= 500
        ) {
          continue;
        }
        throw lastError;
      }
      const data = (await res.json()) as {
        models?: Record<string, unknown> | Array<{ name?: string; id?: string }>;
      };

      if (Array.isArray(data.models)) {
        return data.models
          .map((m) =>
            typeof m === "string"
              ? m
              : typeof m?.name === "string"
                ? m.name
                : typeof m?.id === "string"
                  ? m.id
                  : ""
          )
          .map((s) => s.trim())
          .filter(Boolean);
      }

      if (data.models && typeof data.models === "object") {
        return Object.keys(data.models)
          .map((k) => k.trim())
          .filter(Boolean);
      }

      lastError = new Error(
        `Unexpected fetchAvailableModels response shape at ${endpoint}`
      );
    } catch (error: any) {
      lastError = error instanceof Error ? error : new Error(String(error));
    }
  }

  throw (
    lastError ||
    new Error("fetchAvailableModels failed on all Antigravity endpoints")
  );
}

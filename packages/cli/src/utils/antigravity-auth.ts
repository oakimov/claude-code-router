/**
 * CLI-side Antigravity OAuth helpers for `ccr model get` (and shared token paths).
 * Server/runtime refresh lives in packages/core; this mirrors Codex's CLI auth module.
 */
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

export const ANTIGRAVITY_CLIENT_ID =
  "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com";
export const ANTIGRAVITY_CLIENT_SECRET =
  "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";
export const ANTIGRAVITY_ENDPOINT_DAILY =
  "https://daily-cloudcode-pa.sandbox.googleapis.com";
export const ANTIGRAVITY_ENDPOINT_AUTOPUSH =
  "https://autopush-cloudcode-pa.sandbox.googleapis.com";
export const ANTIGRAVITY_ENDPOINT_PROD =
  "https://cloudcode-pa.googleapis.com";

const AUTH_FILE = join(homedir(), ".claude-code-router", "antigravity_auth.json");
const EXPIRY_LEEWAY_MS = 120_000;
const TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token";

const LOAD_ENDPOINTS = [
  ANTIGRAVITY_ENDPOINT_PROD,
  ANTIGRAVITY_ENDPOINT_DAILY,
  ANTIGRAVITY_ENDPOINT_AUTOPUSH,
] as const;

const ANTIGRAVITY_VERSION = "1.18.3";

function getCliAntigravityHeaders(): Record<string, string> {
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

export interface AntigravityTokens {
  access_token: string;
  refresh_token: string;
  expires_at: number;
  email?: string;
  project_id?: string;
}

let refreshPromise: Promise<AntigravityTokens> | null = null;

function authFilePath(): string {
  return process.env.CCR_ANTIGRAVITY_AUTH_FILE || AUTH_FILE;
}

export function loadAntigravityTokens(): AntigravityTokens | null {
  try {
    const file = authFilePath();
    if (!existsSync(file)) return null;
    const tokens = JSON.parse(readFileSync(file, "utf8")) as AntigravityTokens;
    if (!tokens.access_token || !tokens.refresh_token) return null;
    return tokens;
  } catch {
    return null;
  }
}

export function saveAntigravityTokens(tokens: AntigravityTokens): void {
  const file = authFilePath();
  mkdirSync(dirname(file), { recursive: true });
  const temp = `${file}.${process.pid}.${Date.now()}.tmp`;
  try {
    writeFileSync(temp, JSON.stringify(tokens, null, 2), {
      mode: 0o600,
      encoding: "utf8",
    });
    renameSync(temp, file);
  } finally {
    try {
      if (existsSync(temp)) unlinkSync(temp);
    } catch {
      // ignore
    }
  }
}

async function refresh(tokens: AntigravityTokens): Promise<AntigravityTokens> {
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
    throw new Error(
      `Antigravity token refresh failed (${res.status}). Run \`ccr antigravity-auth\` again.`
    );
  }
  const data = (await res.json()) as {
    access_token?: string;
    refresh_token?: string;
    expires_in?: number;
  };
  if (!data.access_token) {
    throw new Error(
      "Antigravity refresh returned no access_token. Run `ccr antigravity-auth` again."
    );
  }
  const next: AntigravityTokens = {
    ...tokens,
    access_token: data.access_token,
    refresh_token: data.refresh_token || tokens.refresh_token,
    expires_at: Date.now() + (data.expires_in || 3600) * 1000,
  };
  saveAntigravityTokens(next);
  return next;
}

export async function resolveCliAntigravityAuth(): Promise<AntigravityTokens> {
  const tokens = loadAntigravityTokens();
  if (!tokens) {
    throw new Error(
      "No Antigravity OAuth tokens found. Run `ccr antigravity-auth` to authenticate."
    );
  }
  if (Date.now() + EXPIRY_LEEWAY_MS < tokens.expires_at) {
    return tokens;
  }
  if (!refreshPromise) {
    refreshPromise = refresh(tokens).finally(() => {
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
    ...getCliAntigravityHeaders(),
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
        ...getCliAntigravityHeaders(),
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
  return [...new Set([primary, "free-tier", "FREE"].filter(Boolean) as string[])];
}

async function discoverProjectId(accessToken: string): Promise<string | undefined> {
  const body = JSON.stringify({ metadata: codeAssistMetadata() });
  let lastPayload: any;
  const headers = {
    Authorization: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
    ...getCliAntigravityHeaders(),
  };

  for (const endpoint of LOAD_ENDPOINTS) {
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
      const id = extractProjectId(payload);
      if (id) return id;
    } catch {
      // next
    }
  }

  for (const tierId of tierCandidates(lastPayload)) {
    for (const endpoint of LOAD_ENDPOINTS) {
      const onboarded = await onboardUser(accessToken, tierId, endpoint);
      if (onboarded) return onboarded;
    }
  }

  for (const endpoint of LOAD_ENDPOINTS) {
    try {
      const res = await fetch(`${endpoint}/v1internal:loadCodeAssist`, {
        method: "POST",
        headers,
        body,
        signal: AbortSignal.timeout(15_000),
      });
      if (!res.ok) continue;
      const id = extractProjectId(await res.json());
      if (id) return id;
    } catch {
      // next
    }
  }

  return undefined;
}

export async function resolveCliAntigravityProjectId(
  providerProjectId: string | undefined,
  accessToken: string
): Promise<string | undefined> {
  if (providerProjectId?.trim()) return providerProjectId.trim();
  const cached = loadAntigravityTokens()?.project_id?.trim();
  if (cached) return cached;
  const discovered = await discoverProjectId(accessToken);
  if (discovered) {
    const tokens = loadAntigravityTokens();
    if (tokens) saveAntigravityTokens({ ...tokens, project_id: discovered });
    return discovered;
  }
  return undefined;
}

export async function fetchCliAntigravityModels(
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
    ? [preferred, ...LOAD_ENDPOINTS]
    : [...LOAD_ENDPOINTS];
  const unique = [...new Set(endpoints)];
  let lastError: Error | undefined;
  const body = JSON.stringify({ project: projectId.trim() });
  const headers = {
    Authorization: `Bearer ${accessToken}`,
    "Content-Type": "application/json",
    ...getCliAntigravityHeaders(),
  };

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
          `fetchAvailableModels ${res.status}: ${text.slice(0, 200)}`
        );
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
        return Object.keys(data.models).map((k) => k.trim()).filter(Boolean);
      }
      lastError = new Error("Unexpected fetchAvailableModels response shape");
    } catch (error: any) {
      lastError = error instanceof Error ? error : new Error(String(error));
    }
  }

  throw lastError || new Error("fetchAvailableModels failed on all endpoints");
}

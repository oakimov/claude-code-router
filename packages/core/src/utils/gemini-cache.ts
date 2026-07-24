import { createHash } from "crypto";

type CachedContentEntry = {
  name?: string;
  expiresAt: number;
};

const CACHE_CREATE_TIMEOUT_MS = 15_000;
const MAX_CACHED_CONTENT_ENTRIES = 256;
const cachedContents = new Map<string, CachedContentEntry>();
const pendingCachedContents = new Map<string, Promise<string | undefined>>();

function hash(value: unknown): string {
  return createHash("sha256")
    .update(JSON.stringify(value))
    .digest("hex")
    .slice(0, 48);
}

function minimumCacheTokens(model: string): number {
  const normalized = model.toLowerCase();
  if (normalized.includes("gemini-2.5")) return 2048;
  return 4096;
}

function estimatedTokens(value: unknown): number {
  return Math.floor(JSON.stringify(value).length / 4);
}

function cacheablePrefix(body: Record<string, any>): Record<string, any> {
  return Object.fromEntries(
    ["systemInstruction", "tools", "toolConfig"]
      .filter((key) => body[key] !== undefined)
      .map((key) => [key, body[key]])
  );
}

function attachCachedContent(
  body: Record<string, any>,
  name: string
): Record<string, any> {
  const next: Record<string, any> = { ...body, cachedContent: name };
  delete next.systemInstruction;
  delete next.tools;
  delete next.toolConfig;
  return next;
}

function rememberCachedContent(
  key: string,
  entry: CachedContentEntry
): void {
  const now = Date.now();
  for (const [existingKey, existing] of cachedContents) {
    if (existing.expiresAt <= now) cachedContents.delete(existingKey);
  }

  cachedContents.delete(key);
  cachedContents.set(key, entry);
  while (cachedContents.size > MAX_CACHED_CONTENT_ENTRIES) {
    const oldestKey = cachedContents.keys().next().value;
    if (oldestKey === undefined) break;
    cachedContents.delete(oldestKey);
  }
}

export async function attachGeminiCachedContent(options: {
  body: Record<string, any>;
  modelResource: string;
  createUrl: string | URL;
  headers: Record<string, string | undefined>;
  logger?: any;
}): Promise<Record<string, any>> {
  const { body, modelResource, createUrl, headers } =
    options;
  const prefix = cacheablePrefix(body);
  if (
    Object.keys(prefix).length === 0 ||
    estimatedTokens(prefix) < minimumCacheTokens(modelResource)
  ) {
    return body;
  }

  const credentialIdentity = hash(headers);
  const cacheKey = [
    String(createUrl),
    credentialIdentity,
    modelResource,
    hash(prefix),
  ].join(":");
  const now = Date.now();
  const existing = cachedContents.get(cacheKey);
  if (existing && existing.expiresAt > now) {
    return existing.name ? attachCachedContent(body, existing.name) : body;
  }
  if (existing) cachedContents.delete(cacheKey);

  const ttl = 3600;
  const createBody = {
    model: modelResource,
    ...prefix,
    ttl: `${ttl}s`,
  };

  let pending = pendingCachedContents.get(cacheKey);
  if (!pending) {
    pending = (async () => {
      try {
        const response = await fetch(createUrl, {
          method: "POST",
          headers: Object.fromEntries(
            Object.entries({
              "Content-Type": "application/json",
              ...headers,
            }).filter(([, value]) => value !== undefined)
          ) as Record<string, string>,
          body: JSON.stringify(createBody),
          signal: AbortSignal.timeout(CACHE_CREATE_TIMEOUT_MS),
        });

        if (!response.ok) {
          options.logger?.debug?.(
            { status: response.status },
            "Gemini cached-content create failed; using implicit cache"
          );
          rememberCachedContent(cacheKey, {
            expiresAt: Date.now() + 5 * 60 * 1000,
          });
          return undefined;
        }

        const payload: any = await response.json();
        if (!payload?.name) {
          options.logger?.debug?.(
            "Gemini cached-content response had no resource name; using implicit cache"
          );
          rememberCachedContent(cacheKey, {
            expiresAt: Date.now() + 5 * 60 * 1000,
          });
          return undefined;
        }

        rememberCachedContent(cacheKey, {
          name: payload.name,
          // Avoid referencing a resource during its final expiry window.
          expiresAt: Date.now() + (ttl - 30) * 1000,
        });
        return payload.name as string;
      } catch (error: any) {
        options.logger?.debug?.(
          { error: error?.message || String(error) },
          "Gemini cached-content create errored; using implicit cache"
        );
        rememberCachedContent(cacheKey, {
          expiresAt: Date.now() + 5 * 60 * 1000,
        });
        return undefined;
      } finally {
        pendingCachedContents.delete(cacheKey);
      }
    })();
    pendingCachedContents.set(cacheKey, pending);
  }

  const name = await pending;
  return name ? attachCachedContent(body, name) : body;
}

export function clearGeminiCachedContentForTests(): void {
  cachedContents.clear();
  pendingCachedContents.clear();
}

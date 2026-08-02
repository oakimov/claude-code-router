import { randomBytes, timingSafeEqual } from "node:crypto";
import type { FastifyReply, FastifyRequest } from "fastify";

export const UI_SESSION_COOKIE = "ccr_ui_session";
const SESSION_IDLE_TTL_MS = 12 * 60 * 60 * 1000;
const SESSION_ABSOLUTE_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_UI_SESSIONS = 64;

type UiSession = {
  createdAt: number;
  lastSeenAt: number;
};

const sessions = new Map<string, UiSession>();

function pruneSessions(now = Date.now()): void {
  for (const [id, session] of sessions) {
    if (
      now - session.lastSeenAt >= SESSION_IDLE_TTL_MS ||
      now - session.createdAt >= SESSION_ABSOLUTE_TTL_MS
    ) {
      sessions.delete(id);
    }
  }

  while (sessions.size >= MAX_UI_SESSIONS) {
    const oldest = sessions.keys().next().value;
    if (oldest === undefined) break;
    sessions.delete(oldest);
  }
}

export function apiKeysMatch(candidate: unknown, configured: unknown): boolean {
  if (typeof candidate !== "string" || typeof configured !== "string") {
    return false;
  }
  const candidateBytes = Buffer.from(candidate);
  const configuredBytes = Buffer.from(configured);
  if (candidateBytes.length !== configuredBytes.length) return false;
  return timingSafeEqual(candidateBytes, configuredBytes);
}

export function createUiSession(): string {
  pruneSessions();
  const now = Date.now();
  const id = randomBytes(32).toString("base64url");
  sessions.set(id, { createdAt: now, lastSeenAt: now });
  return id;
}

export function hasValidUiSession(request: FastifyRequest): boolean {
  const id = request.cookies?.[UI_SESSION_COOKIE];
  if (!id) return false;

  const now = Date.now();
  const session = sessions.get(id);
  if (!session) return false;
  if (
    now - session.lastSeenAt >= SESSION_IDLE_TTL_MS ||
    now - session.createdAt >= SESSION_ABSOLUTE_TTL_MS
  ) {
    sessions.delete(id);
    return false;
  }

  session.lastSeenAt = now;
  sessions.delete(id);
  sessions.set(id, session);
  return true;
}

export function revokeUiSession(request: FastifyRequest): void {
  const id = request.cookies?.[UI_SESSION_COOKIE];
  if (id) sessions.delete(id);
}

function usesHttps(request: FastifyRequest): boolean {
  const forwardedProto = request.headers["x-forwarded-proto"];
  const firstForwardedProto = Array.isArray(forwardedProto)
    ? forwardedProto[0]
    : forwardedProto?.split(",", 1)[0];
  return firstForwardedProto?.trim().toLowerCase() === "https" || request.protocol === "https";
}

export function setUiSessionCookie(reply: FastifyReply, id: string): void {
  reply.setCookie(UI_SESSION_COOKIE, id, {
    httpOnly: true,
    maxAge: SESSION_ABSOLUTE_TTL_MS / 1000,
    path: "/",
    sameSite: "strict",
    secure: usesHttps(reply.request),
  });
}

export function clearUiSessionCookie(reply: FastifyReply): void {
  reply.clearCookie(UI_SESSION_COOKIE, {
    httpOnly: true,
    path: "/",
    sameSite: "strict",
    secure: usesHttps(reply.request),
  });
}

export function clearUiSessionsForTests(): void {
  sessions.clear();
}

export function uiSessionCountForTests(): number {
  return sessions.size;
}

export const UI_SESSION_LIMIT_FOR_TESTS = MAX_UI_SESSIONS;

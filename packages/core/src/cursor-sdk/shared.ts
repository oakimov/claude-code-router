import { createHash } from "crypto";
import { homedir } from "os";
import { basename, dirname, join } from "path";
import type { UnifiedMessage } from "@/types/llm";

export const CURSOR_SDK_TRANSFORMER_NAME = "cursor-sdk";

export type CursorSdkMode = "bridge" | "plan" | "agent";

export const DEFAULT_CURSOR_MODE: CursorSdkMode = "bridge";

export const CCR_HOME = join(homedir(), ".claude-code-router");
export const CURSOR_SDK_WORKSPACES_ROOT = join(
  CCR_HOME,
  "cursor-sdk-workspaces"
);

export const SESSION_IDLE_TTL_MS = 15 * 60 * 1000;
export const SESSION_LRU_MAX = 32;

/** Orphan scratch workspaces are swept after this long without modification. */
export const ORPHAN_WORKSPACE_TTL_MS = 24 * 60 * 60 * 1000;
/** How often the sweep may run, regardless of eviction tick frequency. */
export const WORKSPACE_SWEEP_INTERVAL_MS = 60 * 60 * 1000;

/** Directory name shape produced by `buildSessionKey` (32 hex chars). */
const SESSION_KEY_DIR = /^[0-9a-f]{32}$/;

/**
 * True only for a directory this module created for a session.
 *
 * Guards every removal: `cursorMode: "agent"` can point the session at a user
 * supplied `cursorCwd`, which must never be swept.
 */
export function isManagedWorkspacePath(
  dir: string,
  root: string = CURSOR_SDK_WORKSPACES_ROOT
): boolean {
  const normalized = dir.replace(/[\\/]+$/, "");
  const parent = dirname(normalized);
  if (parent !== root.replace(/[\\/]+$/, "")) return false;
  return SESSION_KEY_DIR.test(basename(normalized));
}

/** Cursor built-ins we try to deny so Claude Code remains the tool host. */
export const CURSOR_BUILTIN_DENY_LIST = [
  "Shell",
  "Read",
  "Write",
  "Delete",
  "Grep",
  "Glob",
  "Edit",
  "ApplyPatch",
  "Task",
  "SemanticSearch",
  "SemSearch",
  "ReadLints",
  "LS",
  "CreatePlan",
  "UpdateTodos",
  "Await",
  "WebFetch",
  "WebSearch",
  "GenerateImage",
] as const;

export const CUSTOM_USER_TOOLS_SERVER = "custom-user-tools";

export function ensureCcrHomePaths(): string[] {
  return [CCR_HOME, CURSOR_SDK_WORKSPACES_ROOT];
}

/** Flatten Unified message content (string or content parts) into plain text. */
export function contentToText(content: UnifiedMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content);
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object" && "text" in part) {
        return String((part as any).text || "");
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

export function hashSessionFingerprint(parts: string[]): string {
  return createHash("sha256")
    .update(parts.filter(Boolean).join("\n"))
    .digest("hex")
    .slice(0, 32);
}

export function coerceThinkingText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  if (typeof value === "object") {
    const obj = value as Record<string, unknown>;
    if (typeof obj.text === "string") return obj.text;
    if (typeof obj.content === "string") return obj.content;
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export function extractEffort(request: any): string | undefined {
  return (
    request?.output_config?.effort ||
    request?.reasoning?.effort ||
    request?.effort ||
    undefined
  );
}
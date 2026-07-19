import { createHash } from "crypto";
import { homedir } from "os";
import { join } from "path";

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
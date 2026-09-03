import { createHash } from "crypto";
import { existsSync, mkdirSync, writeFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";
import {
  DEFAULT_LOG_BODY_MAX_BYTES,
  sanitizeBodyForLog,
} from "./redact";

/** Wire direction for request/response body and SSE message debug logs. */
export type MessageDebugDirection =
  | "client→ccr"
  | "ccr→provider"
  | "provider→ccr"
  | "ccr→client";

export function isTruthyConfigFlag(value: unknown): boolean {
  return value === true || value === "true" || value === "1";
}

export function resolveLogBodyMaxBytes(configService: {
  get?: (key: string) => unknown;
} | null | undefined): number {
  const raw = configService?.get?.("LOG_REQUEST_BODY_MAX_BYTES");
  return typeof raw === "number" && Number.isFinite(raw) && raw > 0
    ? raw
    : DEFAULT_LOG_BODY_MAX_BYTES;
}

/**
 * Body capture selector (`LOG_REQUEST_BODY_PARTS`): comma-separated top-level
 * body fields (`system,tools`), or `full` for the whole body. Unset/empty
 * disables capture. Unknown names are kept and reported as not-found at write
 * time so typos stay visible.
 */
export type LogBodySelection = "full" | string[];

export function resolveLogBodySelection(configService: {
  get?: (key: string) => unknown;
} | null | undefined): LogBodySelection | undefined {
  const raw = configService?.get?.("LOG_REQUEST_BODY_PARTS");
  if (typeof raw !== "string") return undefined;
  const parts = raw
    .split(",")
    .map((part) => part.trim().toLowerCase())
    .filter(Boolean)
    .filter((part, index, all) => all.indexOf(part) === index);
  if (parts.length === 0) return undefined;
  if (parts.includes("full") || parts.includes("*")) return "full";
  return parts;
}

/** Capture files live next to the rotated server logs. */
export function defaultLogBodyDir(): string {
  return join(homedir(), ".claude-code-router", "logs");
}

function directionSlug(direction: MessageDebugDirection): string {
  switch (direction) {
    case "client→ccr":
      return "client-ccr";
    case "ccr→provider":
      return "ccr-provider";
    case "provider→ccr":
      return "provider-ccr";
    case "ccr→client":
      return "ccr-client";
  }
}

function filenameSegment(value: unknown, fallback: string): string {
  const text = typeof value === "string" && value ? value : fallback;
  // Dots excluded: a ".." segment would traverse out of the bodies dir.
  const clean = text.replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 80);
  return clean || fallback;
}

function sha16Hex(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex").slice(0, 16);
}

/** Directories already created by this process (mkdir once per dir). */
const ensuredBodyDirs = new Set<string>();

export function shouldLogSSEEvents(configService: {
  get?: (key: string) => unknown;
} | null | undefined): boolean {
  if (isTruthyConfigFlag(configService?.get?.("LOG_SSE_EVENTS"))) return true;
  const env = process.env.LOG_SSE_EVENTS || process.env.CCR_LOG_SSE_EVENTS;
  return env === "1" || env === "true";
}

/** Serialize a request/response body for sanitizeBodyForLog. */
export function bodyToLogString(body: unknown): string {
  if (body == null) return "";
  if (typeof body === "string") return body;
  try {
    return JSON.stringify(body);
  } catch {
    return String(body);
  }
}

export type MessageBodyLogOptions = {
  logger: { debug?: (...args: any[]) => void; info?: (...args: any[]) => void };
  direction: MessageDebugDirection;
  /** Prefer debug; info kept for the legacy Anthropic-only inbound path. */
  level?: "debug" | "info";
  reqId?: string | number;
  protocol?: string;
  provider?: string;
  model?: string;
  maxBytes?: number;
  /** Override the historical `type` field when needed. */
  type?: string;
  /**
   * Resolved `LOG_REQUEST_BODY_PARTS` (see resolveLogBodySelection).
   * `"full"` stores the whole body; a list stores one file per named
   * top-level field. The log carries one manifest line per part (with sha256)
   * instead of the body. Undefined keeps the legacy inline whole-body dump.
   */
  selection?: LogBodySelection | undefined;
  /** Code-level directory override (tests); defaults to the logs folder. */
  bodiesDir?: string;
};

/**
 * Opt-in full message-body capture with a stable direction tag so operators
 * can grep client↔CCR and CCR↔provider legs independently.
 */
function itemType(item: unknown): string {
  if (!item || typeof item !== "object") return "unknown";
  const rec = item as Record<string, unknown>;
  if (typeof rec.type === "string" && rec.type) return rec.type;
  if (typeof rec.role === "string" && rec.role) return rec.role;
  return "unknown";
}

function encryptedLen(item: unknown): number {
  if (!item || typeof item !== "object") return 0;
  const value = (item as { encrypted_content?: unknown }).encrypted_content;
  return typeof value === "string" ? value.length : 0;
}

/**
 * Compact keep-wire snapshot for debug logs. Full transcripts stay behind
 * LOG_REQUEST_BODY; this is how keep (Responses/Chat) shows encrypted
 * replay on `ccr→provider` without a 200k-token dump.
 */
export function summarizeKeepWire(body: unknown): Record<string, unknown> {
  if (!body || typeof body !== "object") {
    return { empty: true };
  }
  const rec = body as Record<string, unknown>;
  const inputTypes: Record<string, number> = {};
  const reasoning: Array<Record<string, unknown>> = [];
  let encryptedItems = 0;

  const tally = (item: unknown) => {
    const type = itemType(item);
    inputTypes[type] = (inputTypes[type] || 0) + 1;
    const enc = encryptedLen(item);
    if (enc > 0) encryptedItems += 1;
    if (type === "reasoning" || enc > 0) {
      const entry: Record<string, unknown> = { type, encrypted_len: enc };
      if (item && typeof item === "object") {
        const id = (item as { id?: unknown }).id;
        if (typeof id === "string" && id) entry.id = id;
        const summary = (item as { summary?: unknown }).summary;
        if (Array.isArray(summary)) entry.summary_n = summary.length;
      }
      reasoning.push(entry);
    }
  };

  if (Array.isArray(rec.input)) {
    for (const item of rec.input) tally(item);
  } else if (Array.isArray(rec.messages)) {
    for (const item of rec.messages) {
      tally(item);
      const thinking =
        item && typeof item === "object"
          ? (item as { thinking?: unknown }).thinking
          : undefined;
      if (thinking && typeof thinking === "object") {
        const enc = encryptedLen(thinking);
        if (enc > 0) {
          encryptedItems += 1;
          reasoning.push({ type: "thinking", encrypted_len: enc });
        }
      }
    }
  }

  return {
    ...(rec.store !== undefined ? { store: rec.store } : {}),
    ...(rec.stream !== undefined ? { stream: rec.stream } : {}),
    ...(Array.isArray(rec.include) ? { include: rec.include } : {}),
    ...(typeof rec.prompt_cache_key === "string" && rec.prompt_cache_key
      ? { prompt_cache_key: rec.prompt_cache_key }
      : {}),
    ...(Array.isArray(rec.input) ? { input_n: rec.input.length } : {}),
    ...(Array.isArray(rec.messages) ? { messages_n: rec.messages.length } : {}),
    ...(Object.keys(inputTypes).length ? { input_types: inputTypes } : {}),
    encrypted_content_items: encryptedItems,
    ...(reasoning.length ? { reasoning } : {}),
    ...(Array.isArray(rec.tools) ? { tools_n: rec.tools.length } : {}),
  };
}

export function logKeepWire(
  body: unknown,
  opts: {
    logger?: { debug?: (...args: any[]) => void };
    reqId?: string | number;
    provider?: string;
    model?: string;
  }
): void {
  const write = opts.logger?.debug;
  if (typeof write !== "function") return;
  write.call(opts.logger, {
    type: "keep wire",
    direction: "ccr→provider" as MessageDebugDirection,
    reqId: opts.reqId,
    provider: opts.provider,
    model: opts.model,
    ...summarizeKeepWire(body),
  });
}

export function logMessageBody(
  body: unknown,
  opts: MessageBodyLogOptions
): void {
  const level = opts.level ?? "debug";
  const write = opts.logger?.[level];
  if (typeof write !== "function") return;

  const maxBytes = opts.maxBytes ?? DEFAULT_LOG_BODY_MAX_BYTES;
  if (opts.selection !== undefined) {
    logMessageBodyParts(body, opts, opts.selection, maxBytes, write);
    return;
  }
  write.call(opts.logger, {
    type: opts.type ?? "message body",
    direction: opts.direction,
    reqId: opts.reqId,
    protocol: opts.protocol,
    provider: opts.provider,
    model: opts.model,
    data: sanitizeBodyForLog(bodyToLogString(body), maxBytes),
  });
}

/**
 * Selective capture: one file per requested top-level field plus a manifest
 * line carrying the sha256 of the stored bytes, so components can be compared
 * across turns (`rg '"part":"tools"'` then diff hashes) without opening files.
 * Best-effort: filesystem failures degrade to a single warning line, never to
 * a request-path error.
 */
function logMessageBodyParts(
  body: unknown,
  opts: MessageBodyLogOptions,
  selection: LogBodySelection,
  maxBytes: number,
  write: (...args: any[]) => void
): void {
  const base = {
    direction: opts.direction,
    reqId: opts.reqId,
    protocol: opts.protocol,
    provider: opts.provider,
    model: opts.model,
  };
  const dir = opts.bodiesDir ?? defaultLogBodyDir();
  const slug = directionSlug(opts.direction);
  const req = filenameSegment(opts.reqId, "noreq");
  const names = selection === "full" ? ["full"] : selection;

  for (const part of names) {
    const value =
      part === "full"
        ? { found: true as const, value: body }
        : body !== null && typeof body === "object" &&
            Object.prototype.hasOwnProperty.call(body, part)
          ? {
              found: true as const,
              value: (body as Record<string, unknown>)[part],
            }
          : { found: false as const, value: undefined };
    if (!value.found) {
      write.call(opts.logger, {
        ...base,
        type: "message body part",
        part,
        found: false,
      });
      continue;
    }
    const stored = sanitizeBodyForLog(
      bodyToLogString(value.value),
      maxBytes
    );
    const hash = sha16Hex(stored);
    const safePart = filenameSegment(part, "part");
    const path = join(dir, `${req}.${slug}.${safePart}.${hash}.json`);
    try {
      if (!ensuredBodyDirs.has(dir)) {
        mkdirSync(dir, { recursive: true });
        ensuredBodyDirs.add(dir);
      } else if (!existsSync(dir)) {
        // Removed between requests (log rotation, operator cleanup).
        mkdirSync(dir, { recursive: true });
      }
      writeFileSync(path, stored, "utf-8");
    } catch (error: any) {
      write.call(opts.logger, {
        ...base,
        type: "message body part",
        part,
        sha256: hash,
        bytes: stored.length,
        path,
        error: String(error?.message || error),
      });
      continue;
    }
    write.call(opts.logger, {
      ...base,
      type: "message body part",
      part,
      found: true,
      sha256: hash,
      bytes: stored.length,
      truncated: /…\[truncated \d+ bytes\]$/.test(stored),
      path,
    });
  }
}

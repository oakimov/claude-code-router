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

export function shouldLogRequestBodies(configService: {
  get?: (key: string) => unknown;
} | null | undefined): boolean {
  return isTruthyConfigFlag(configService?.get?.("LOG_REQUEST_BODY"));
}

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
};

/**
 * Opt-in full message-body capture with a stable direction tag so operators
 * can grep client↔CCR and CCR↔provider legs independently.
 */
export function logMessageBody(
  body: unknown,
  opts: MessageBodyLogOptions
): void {
  const level = opts.level ?? "debug";
  const write = opts.logger?.[level];
  if (typeof write !== "function") return;

  const maxBytes = opts.maxBytes ?? DEFAULT_LOG_BODY_MAX_BYTES;
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

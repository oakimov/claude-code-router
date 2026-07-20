import type { FastifyReply, FastifyRequest } from "fastify";

const RETRY_BACKOFF_BASE_MS = 1_000;
const RETRY_BACKOFF_MAX_MS = 30_000;
const RETRY_AFTER_MAX_MS = 60_000;

export const CLIENT_DISCONNECT_REASON = "client disconnected";

export type ClientDisconnectHandle = {
  signal: AbortSignal;
  /** Attach response close/error listeners for the full upstream lifecycle. */
  arm: () => void;
};

export function delay(ms: number, signal?: AbortSignal): Promise<void> {
  if (ms <= 0) return Promise.resolve();
  if (signal?.aborted) {
    return Promise.reject(Object.assign(new Error("Aborted"), { name: "AbortError", code: "ABORT_ERR" }));
  }

  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      if (signal) {
        signal.removeEventListener("abort", onAbort);
      }
      resolve();
    }, ms);

    const onAbort = () => {
      clearTimeout(timer);
      reject(Object.assign(new Error("Aborted"), { name: "AbortError", code: "ABORT_ERR" }));
    };

    if (signal) {
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}

/**
 * AbortSignal that fires when the HTTP client disconnects mid-response.
 * Used to cancel upstream provider fetches and tear down streams.
 *
 * Important: never treat request close/destroyed state as a disconnect after
 * Fastify has parsed the body. A response `close` event is also insufficient by
 * itself: only abort when the response or its socket is actually gone.
 */
export function createClientDisconnectSignal(
  _req: FastifyRequest,
  reply?: FastifyReply
): ClientDisconnectHandle {
  const controller = new AbortController();
  const rawRes = reply?.raw;
  let armed = false;

  const onAbort = (reason: string = CLIENT_DISCONNECT_REASON) => {
    if (!controller.signal.aborted) {
      controller.abort(reason);
    }
  };

  const arm = () => {
    if (armed) return;
    armed = true;

    if (!rawRes) return;
    // Only the response socket matters here. Fastify sets req.raw.destroyed
    // after JSON body parsing while the client is still connected — checking
    // that falsely aborted OpenCode/Zen fetches as "already destroyed".
    if (isRawResponseGone(rawRes)) {
      onAbort(`${CLIENT_DISCONNECT_REASON} (already destroyed)`);
      return;
    }

    rawRes.on("close", () => {
      if (!rawRes.writableEnded && isRawResponseGone(rawRes)) {
        onAbort(`${CLIENT_DISCONNECT_REASON} (response close)`);
      }
    });
    rawRes.on("error", (err: any) => {
      if (
        err?.code === "EPIPE" ||
        err?.code === "ECONNRESET" ||
        err?.code === "ERR_STREAM_PREMATURE_CLOSE" ||
        isClientAbortError(err)
      ) {
        onAbort(`${CLIENT_DISCONNECT_REASON} (response error)`);
      }
    });
  };

  return { signal: controller.signal, arm };
}

export function parseRetryAfterHeaderMs(
  value: string | null | undefined
): number | undefined {
  const trimmed = value?.trim();
  if (!trimmed) return undefined;
  const seconds = Number(trimmed);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return Math.min(seconds * 1000, RETRY_AFTER_MAX_MS);
  }
  const retryAt = Date.parse(trimmed);
  if (!Number.isFinite(retryAt)) return undefined;
  return Math.min(Math.max(0, retryAt - Date.now()), RETRY_AFTER_MAX_MS);
}

export function exponentialRetryBackoffMs(failedAttemptIndex: number): number {
  const exponent = Math.min(10, Math.max(0, failedAttemptIndex));
  return Math.min(RETRY_BACKOFF_MAX_MS, RETRY_BACKOFF_BASE_MS * 2 ** exponent);
}

/** Prefer Retry-After when present; otherwise exponential backoff. */
export function retryDelayAfterFailure(
  failedAttemptIndex: number,
  retryAfterHeader?: string | null
): number {
  const retryAfterMs = parseRetryAfterHeaderMs(retryAfterHeader);
  if (retryAfterMs !== undefined && retryAfterMs > 0) {
    return retryAfterMs;
  }
  return exponentialRetryBackoffMs(failedAttemptIndex);
}

export function selectFallbackModels(
  fallbackConfig: Record<string, unknown> | undefined,
  scenarioType: string
): string[] | undefined {
  if (!fallbackConfig) return undefined;
  const configured =
    scenarioType === "subagent"
      ? fallbackConfig.subagent ?? fallbackConfig.default
      : fallbackConfig[scenarioType];
  return Array.isArray(configured) ? configured : undefined;
}

const NETWORK_ERROR_CODES = new Set([
  "ECONNRESET",
  "ECONNREFUSED",
  "ENOTFOUND",
  "EAI_AGAIN",
  "ETIMEDOUT",
  "EPIPE",
  "UND_ERR_CONNECT_TIMEOUT",
  "UND_ERR_HEADERS_TIMEOUT",
  "UND_ERR_BODY_TIMEOUT",
  "UND_ERR_SOCKET",
  "UND_ERR_CONNECT",
]);

function isTimeoutAbortError(error: unknown): boolean {
  const err = error as any;
  if (!err) return false;
  // DOMException TimeoutError from AbortSignal.timeout() uses numeric code 23.
  if (
    err.name === "TimeoutError" ||
    err.code === "TIMEOUT_ERR" ||
    err.code === 23
  ) {
    return true;
  }
  if (
    err.cause &&
    (err.cause.name === "TimeoutError" ||
      err.cause.code === "TIMEOUT_ERR" ||
      err.cause.code === 23)
  ) {
    return true;
  }
  const message = String(err.message || "");
  return /aborted due to timeout|operation was aborted due to timeout/i.test(
    message
  );
}

export function toClientAbortError(
  reason: unknown = CLIENT_DISCONNECT_REASON
): Error {
  const message =
    typeof reason === "string" && reason
      ? reason
      : reason instanceof Error && reason.message
        ? reason.message
        : CLIENT_DISCONNECT_REASON;
  return Object.assign(new Error(message), {
    name: "AbortError",
    code: "ABORT_ERR",
    reason,
  });
}

export function isClientAbortError(error: unknown): boolean {
  if (error == null || error === false) return false;

  // fetch() + AbortController.abort(string) rejects with the string itself
  // (not an Error). OpenCode false-aborts were logged as HTTP 500 because this
  // helper missed that shape and the errorHandler took the internal path.
  if (typeof error === "string") {
    return (
      error.startsWith(CLIENT_DISCONNECT_REASON) ||
      /client.*(closed|disconnect)|socket hang up/i.test(error)
    );
  }

  const err = error as any;
  // Request timeouts must not be treated as client disconnects (no 499 / skip fallback).
  if (isTimeoutAbortError(err)) return false;

  const reason = err.reason ?? err.cause;
  if (
    typeof reason === "string" &&
    reason.startsWith(CLIENT_DISCONNECT_REASON)
  ) {
    return true;
  }
  if (
    typeof err.message === "string" &&
    err.message.startsWith(CLIENT_DISCONNECT_REASON)
  ) {
    return true;
  }

  if (err.name === "AbortError") return true;
  if (err.code === "ABORT_ERR") return true;
  if (err.code === "ERR_STREAM_PREMATURE_CLOSE") return true;
  const message = String(err.message || "");
  return /client.*(closed|disconnect)|socket hang up/i.test(message);
}

/**
 * True only when the HTTP response socket is actually gone.
 * Prefer this over AbortSignal.aborted / isClientAbortError alone — those can
 * fire during Cursor SDK bootstrap or provider timeouts while the client is
 * still connected and able to receive a JSON body (false 499).
 */
function isRawResponseGone(raw: any): boolean {
  if (!raw || raw.destroyed) return true;
  const socket = raw.socket;
  if (!socket) return false;
  if (socket.destroyed) return true;
  // Both sides closed ⇒ client is gone even if destroyed flag lags.
  return socket.readable === false && socket.writable === false;
}

export function isResponseSocketGone(reply: FastifyReply): boolean {
  return isRawResponseGone(reply.raw);
}

export function isProviderNetworkError(error: unknown): boolean {
  if (!error || isClientAbortError(error)) return false;
  // AbortSignal.timeout / fetch TimeoutError — eligible for fallback, not a client abort.
  if (isTimeoutAbortError(error)) return true;
  const err = error as any;
  if (err.code === "provider_network_error") return true;
  if (typeof err.code === "string" && NETWORK_ERROR_CODES.has(err.code)) {
    return true;
  }
  if (err.cause?.code && NETWORK_ERROR_CODES.has(String(err.cause.code))) {
    return true;
  }
  const message = String(err.message || "");
  return /fetch failed|network|socket|ECONNRESET|ENOTFOUND|ETIMEDOUT|other side closed/i.test(
    message
  );
}

export function isFallbackEligibleError(error: unknown): boolean {
  const err = error as any;
  if (!err || isClientAbortError(err)) return false;
  return (
    err.code === "provider_response_error" ||
    err.code === "provider_network_error" ||
    isProviderNetworkError(err)
  );
}
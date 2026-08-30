/**
 * @cursor/sdk POSTs /auth/exchange_user_api_key on every Agent.create and
 * models.list. Subagent turns mint new session keys, so Agent.create runs
 * again and the SDK does not cache the access token across those calls.
 *
 * Wrap global fetch: coalesce in-flight exchanges per crsr_ key, reuse the
 * access token until a Cursor request with that token returns 401.
 *
 * Critical: never forward a caller's AbortSignal onto the shared exchange.
 * Agent.create often cancels while another waiter still needs the token; if
 * the first caller's signal aborts the shared fetch, every waiter rejects
 * with AbortError and Node surfaces unhandledRejection / late-handled
 * PromiseRejectionHandledWarning.
 */

const EXCHANGE_PATH = "/auth/exchange_user_api_key";

type CachedToken = {
  accessToken: string;
  status: number;
  statusText: string;
  contentType: string;
};

type ExchangeResult = CachedToken | { error: Response };

const tokenByCrsr = new Map<string, CachedToken>();
const crsrByAccessToken = new Map<string, string>();
const inflight = new Map<string, Promise<ExchangeResult>>();

let installed = false;
let outbound: typeof fetch | undefined;
let wrapped: typeof fetch | undefined;

function requestUrl(input: RequestInfo | URL): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  return String((input as Request).url || "");
}

function isExchangeUrl(url: string): boolean {
  try {
    return new URL(url).pathname.endsWith(EXCHANGE_PATH);
  } catch {
    return url.includes("exchange_user_api_key");
  }
}

function isCursorHost(url: string): boolean {
  try {
    const host = new URL(url).hostname.toLowerCase();
    return (
      host === "api2.cursor.sh" ||
      host.endsWith(".cursor.sh") ||
      host.endsWith(".cursor.com")
    );
  } catch {
    return false;
  }
}

function readAuthorization(
  input: RequestInfo | URL,
  init?: RequestInit
): string | undefined {
  const fromInit = init?.headers;
  if (fromInit) {
    const headers = new Headers(fromInit);
    const value = headers.get("authorization") || headers.get("Authorization");
    if (value) return value;
  }
  if (typeof Request !== "undefined" && input instanceof Request) {
    return (
      input.headers.get("authorization") ||
      input.headers.get("Authorization") ||
      undefined
    );
  }
  return undefined;
}

function isAbortError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const candidate = error as { name?: string; code?: string };
  return (
    candidate.name === "AbortError" ||
    candidate.code === "ABORT_ERR" ||
    (typeof DOMException !== "undefined" &&
      error instanceof DOMException &&
      error.name === "AbortError")
  );
}

function abortError(reason?: unknown): Error {
  if (isAbortError(reason)) return reason as Error;
  const message =
    reason instanceof Error
      ? reason.message
      : typeof reason === "string" && reason
        ? reason
        : "This operation was aborted";
  if (typeof DOMException !== "undefined") {
    return new DOMException(message, "AbortError");
  }
  return Object.assign(new Error(message), { name: "AbortError" });
}

function cachedResponse(entry: CachedToken): Response {
  return new Response(JSON.stringify({ accessToken: entry.accessToken }), {
    status: entry.status,
    statusText: entry.statusText,
    headers: { "Content-Type": entry.contentType || "application/json" },
  });
}

function rememberToken(crsrBearer: string, entry: CachedToken): void {
  const previous = tokenByCrsr.get(crsrBearer);
  if (previous) crsrByAccessToken.delete(previous.accessToken);
  tokenByCrsr.set(crsrBearer, entry);
  crsrByAccessToken.set(entry.accessToken, crsrBearer);
}

function forgetCrsrKey(crsrKey: string): void {
  const entry = tokenByCrsr.get(crsrKey);
  if (entry) crsrByAccessToken.delete(entry.accessToken);
  tokenByCrsr.delete(crsrKey);
}

function invalidateByAccessBearer(authorization: string | undefined): void {
  if (!authorization) return;
  const token = authorization.replace(/^Bearer\s+/i, "").trim();
  if (!token) return;
  const crsr = crsrByAccessToken.get(token);
  if (!crsr) return;
  forgetCrsrKey(crsr);
}

/**
 * Drop a cached exchanged token for a dashboard API key (crsr_... or Bearer).
 * Call when Cursor reports auth failure via run status / AuthenticationError —
 * those paths often never surface as an HTTP 401 on the wrapped fetch.
 */
export function invalidateCursorAuthExchange(crsrApiKey: string): void {
  const key = crsrApiKey.replace(/^Bearer\s+/i, "").trim();
  if (!key) return;
  // Cache keys are the Authorization header as the SDK sends it ("Bearer crsr_...").
  // Callers may also pass a bare crsr_ key or (rarely) an access token.
  const bearerForm = `Bearer ${key}`;
  if (tokenByCrsr.has(bearerForm)) {
    forgetCrsrKey(bearerForm);
    return;
  }
  if (tokenByCrsr.has(key)) {
    forgetCrsrKey(key);
    return;
  }
  invalidateByAccessBearer(key);
}

/** Drop AbortSignal so one canceled Agent.create cannot abort a shared exchange. */
function initWithoutSignal(init?: RequestInit): RequestInit | undefined {
  if (!init || init.signal == null) return init;
  const { signal: _signal, ...rest } = init;
  return rest;
}

function requestWithoutSignal(input: RequestInfo | URL): RequestInfo | URL {
  if (typeof Request !== "undefined" && input instanceof Request && input.signal) {
    return new Request(input, { signal: undefined });
  }
  return input;
}

function throwIfAborted(signal: AbortSignal | undefined | null): void {
  if (signal?.aborted) {
    throw abortError(signal.reason);
  }
}

async function awaitUnlessAborted<T>(
  pending: Promise<T>,
  signal: AbortSignal | undefined | null
): Promise<T> {
  if (!signal) return pending;
  throwIfAborted(signal);

  return await new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      cleanup();
      reject(abortError(signal.reason));
    };
    const cleanup = () => {
      signal.removeEventListener("abort", onAbort);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    pending.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      }
    );
  });
}

async function exchangeThroughOutbound(
  crsrBearer: string,
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  const callerSignal = init?.signal;
  throwIfAborted(callerSignal);

  let pending = inflight.get(crsrBearer);
  if (!pending) {
    pending = (async () => {
      const impl = outbound || globalThis.fetch;
      // Never attach a caller AbortSignal to the shared network exchange.
      const response = await impl(
        requestWithoutSignal(input) as any,
        initWithoutSignal(init)
      );
      if (!response.ok) {
        return { error: response };
      }
      const contentType =
        response.headers.get("Content-Type") || "application/json";
      const body = await response.text();
      try {
        const parsed = JSON.parse(body);
        if (typeof parsed?.accessToken === "string" && parsed.accessToken) {
          const entry: CachedToken = {
            accessToken: parsed.accessToken,
            status: response.status,
            statusText: response.statusText,
            contentType,
          };
          rememberToken(crsrBearer, entry);
          return entry;
        }
      } catch {
        return {
          error: new Response(body, {
            status: response.status,
            statusText: response.statusText,
            headers: response.headers,
          }),
        };
      }
      return {
        error: new Response(body, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        }),
      };
    })()
      // Keep the shared promise handled so a canceled waiter rejecting its own
      // race cannot leave the network promise as an unhandledRejection.
      .catch((error) => {
        if (isAbortError(error)) {
          return {
            error: new Response(null, {
              status: 499,
              statusText: "Client Closed Request",
            }),
          } satisfies ExchangeResult;
        }
        throw error;
      })
      .finally(() => {
        if (inflight.get(crsrBearer) === pending) inflight.delete(crsrBearer);
      });
    inflight.set(crsrBearer, pending);
  }

  // Attach a no-op catch so canceled waiters do not mark the shared promise
  // unhandled when they race-reject first.
  void pending.catch(() => undefined);

  const result = await awaitUnlessAborted(pending, callerSignal);
  throwIfAborted(callerSignal);
  if ("error" in result) return result.error.clone();
  return cachedResponse(result);
}

async function wrappedFetch(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<Response> {
  const url = requestUrl(input);
  const authorization = readAuthorization(input, init);
  const impl = outbound || globalThis.fetch;

  if (
    isExchangeUrl(url) &&
    (!init?.method || init.method.toUpperCase() === "POST")
  ) {
    const crsrBearer = authorization || "";
    const cached = crsrBearer ? tokenByCrsr.get(crsrBearer) : undefined;
    if (cached) {
      throwIfAborted(init?.signal);
      return cachedResponse(cached);
    }
    if (crsrBearer) return exchangeThroughOutbound(crsrBearer, input, init);
  }

  const response = await impl(input as any, init);
  if (response.status === 401 && isCursorHost(url)) {
    invalidateByAccessBearer(authorization);
  }
  return response;
}

/** Re-wrap if tests or other code replaced globalThis.fetch. */
export function installCursorAuthExchangeCache(): void {
  if (!wrapped) {
    wrapped = wrappedFetch as typeof fetch;
  }
  if (globalThis.fetch === wrapped) return;
  outbound = globalThis.fetch.bind(globalThis);
  globalThis.fetch = wrapped;
  installed = true;
}

export function __resetCursorAuthExchangeCacheForTests(): void {
  tokenByCrsr.clear();
  crsrByAccessToken.clear();
  inflight.clear();
  if (installed && wrapped && globalThis.fetch === wrapped && outbound) {
    globalThis.fetch = outbound;
  }
  installed = false;
  outbound = undefined;
  wrapped = undefined;
}

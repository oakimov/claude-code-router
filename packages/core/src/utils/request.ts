import { ProxyAgent } from "undici";
import { UnifiedChatRequest } from "../types/llm";
// import { diffHeadersForLog, sanitizeHeadersForLog } from "./redact";
import { createApiError } from "../api/middleware";
import {
  CLIENT_DISCONNECT_REASON,
  isClientAbortError,
  isProviderNetworkError,
  toClientAbortError,
} from "./retry";

function normalizeHostname(hostname: string): string {
  return hostname
    .trim()
    .toLowerCase()
    .replace(/^\[(.*)\]$/, "$1")
    .replace(/\.$/, "");
}

function isPlainIp(value: string): boolean {
  return /^\d+\.\d+\.\d+\.\d+$/.test(value);
}

function ipToInt(ip: string): number | null {
  const parts = ip.split(".");
  if (parts.length !== 4) return null;
  let result = 0;
  for (const part of parts) {
    const num = parseInt(part, 10);
    if (Number.isNaN(num) || num < 0 || num > 255) return null;
    result = (result << 8) | num;
  }
  return result >>> 0;
}

function isInCidr(ip: string, cidr: string): boolean {
  const [network, prefixStr] = cidr.split("/");
  const prefix = parseInt(prefixStr, 10);
  if (Number.isNaN(prefix) || prefix < 0 || prefix > 32) return false;

  const ipInt = ipToInt(ip);
  const netInt = ipToInt(network);
  if (ipInt === null || netInt === null) return false;

  const mask = prefix === 0 ? 0 : (~0 << (32 - prefix)) >>> 0;
  return (ipInt & mask) === (netInt & mask);
}

/** Loopback and NO_PROXY/no_proxy hosts should not go through HTTPS_PROXY. */
export function shouldBypassProxy(target: URL | string): boolean {
  let url: URL;
  try {
    url = typeof target === "string" ? new URL(target) : target;
  } catch {
    // Fail closed to direct (no proxy) on unparseable URLs.
    return true;
  }

  const hostname = normalizeHostname(url.hostname);
  if (
    hostname === "localhost" ||
    hostname === "127.0.0.1" ||
    hostname.startsWith("127.") ||
    hostname === "0.0.0.0" ||
    hostname === "::1" ||
    hostname === "0:0:0:0:0:0:0:1"
  ) {
    return true;
  }

  const noProxy = process.env.NO_PROXY || process.env.no_proxy;
  if (!noProxy) return false;

  const patterns = noProxy
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);

  for (const rawPattern of patterns) {
    const pattern = rawPattern.toLowerCase();
    if (pattern === "*") return true;

    // host:port — match hostname only. Keep IPv6 literals intact (multiple ':').
    const patternHost = stripNoProxyPort(pattern);

    if (patternHost.startsWith(".")) {
      if (
        hostname.endsWith(patternHost) ||
        hostname === patternHost.slice(1)
      ) {
        return true;
      }
      continue;
    }

    if (patternHost.includes("/") && isPlainIp(hostname)) {
      if (isInCidr(hostname, patternHost)) return true;
      continue;
    }

    if (hostname === patternHost) return true;

    // "*.example.com" style
    if (patternHost.startsWith("*.")) {
      const suffix = patternHost.slice(1); // ".example.com"
      if (hostname.endsWith(suffix) || hostname === suffix.slice(1)) {
        return true;
      }
    }
  }

  return false;
}

/** Strip :port from NO_PROXY entries without mangling IPv6 addresses. */
function stripNoProxyPort(pattern: string): string {
  if (pattern.startsWith("[") && pattern.includes("]")) {
    return pattern.slice(1, pattern.indexOf("]"));
  }
  const colonCount = pattern.split(":").length - 1;
  if (colonCount === 1) {
    const [host, port] = pattern.split(":");
    if (/^\d+$/.test(port)) return host;
  }
  return pattern;
}

export async function sendUnifiedRequest(
  url: URL | string,
  request: UnifiedChatRequest,
  config: any,
  _context: any,
  _logger?: any
): Promise<Response> {
  const headers = new Headers({
    "Content-Type": "application/json",
  });
  if (config.headers) {
    Object.entries(config.headers).forEach(([key, value]) => {
      if (value) {
        headers.set(key, value as string);
      }
    });
  }
  const timeoutSignal = AbortSignal.timeout(config.TIMEOUT ?? 60 * 1000 * 60);
  // AbortSignal.any correctly handles already-aborted inputs.
  const combinedSignal: AbortSignal = config.signal
    ? AbortSignal.any([config.signal, timeoutSignal])
    : timeoutSignal;

  const fetchOptions: RequestInit = {
    method: "POST",
    headers: headers,
    body: JSON.stringify(request),
    signal: combinedSignal,
  };

  const requestUrl = typeof url === "string" ? url : url.toString();
  const useProxy =
    Boolean(config.httpsProxy) && !shouldBypassProxy(requestUrl);

  if (useProxy) {
    (fetchOptions as any).dispatcher = new ProxyAgent(
      new URL(config.httpsProxy).toString()
    );
  }

  // const clientHeaders = context?.req?.headers as
  //   | Record<string, unknown>
  //   | undefined;
  // const outboundHeaders = sanitizeHeadersForLog(headers);
  // logger?.debug(
  //   {
  //     reqId: context?.req?.id,
  //     method: fetchOptions.method,
  //     headers: outboundHeaders,
  //     clientHeaders: sanitizeHeadersForLog(clientHeaders),
  //     headerDiff: {
  //       direction: "client -> outbound",
  //       ...diffHeadersForLog(clientHeaders, headers),
  //     },
  //     requestUrl,
  //     useProxy,
  //   },
  //   "final request"
  // );

  // Keep the request execution unchanged below; only verbose header logging is
  // disabled here.

  try {
    return await fetch(requestUrl, fetchOptions);
  } catch (error: any) {
    // AbortSignal.any + abort(string) may reject with a bare string. Normalize
    // so middleware classifies it as 499 instead of a 500 internal error.
    // Do NOT treat timeout aborts (AbortSignal.timeout) as client disconnects.
    if (isClientAbortError(error)) {
      throw typeof error === "string" ? toClientAbortError(error) : error;
    }
    if (
      config.signal?.aborted &&
      isClientAbortError(config.signal.reason ?? CLIENT_DISCONNECT_REASON)
    ) {
      throw toClientAbortError(config.signal.reason ?? error);
    }
    if (isProviderNetworkError(error)) {
      const networkError = createApiError(
        error?.message || "Provider network error",
        502,
        "provider_network_error"
      );
      (networkError as any).cause = error;
      throw networkError;
    }
    throw error;
  }
}

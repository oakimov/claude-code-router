import type { UnifiedChatRequest } from "@/types/llm";
import type { Transformer } from "@/types/transformer";
import {
  ClientProtocolContext,
  ProtocolRouteMatch,
  createClientProtocolContext,
} from "./protocol-endpoints";
import { createApiError } from "@/api/middleware";
import { responsesRequestToUnified } from "../utils/openai.responses.util";

export interface ProtocolAdaptResult {
  /** Cloned client body used as normalization input (never the live req.body). */
  normalizationInput: any;
  context: ClientProtocolContext;
}

/**
 * Adapt path/query fields into a cloned normalization input and build the
 * initial ClientProtocolContext. Does not mutate the caller's body object.
 */
export function adaptClientRequest(
  match: ProtocolRouteMatch,
  rawBody: any,
  _query?: Record<string, unknown>
): ProtocolAdaptResult {
  const body =
    rawBody && typeof rawBody === "object" ? cloneProtocolBody(rawBody) : {};

  let stream = match.stream;
  const originalModel: string | undefined =
    typeof body.model === "string" ? body.model : undefined;

  if (match.protocol === "openai_chat_completions") {
    stream = body.stream === true;
  } else if (match.protocol === "openai_responses") {
    stream = body.stream === true;
  } else if (match.protocol === "anthropic_messages") {
    stream = body.stream === true;
  }

  const context = createClientProtocolContext(match, {
    originalModel,
    stream,
  });

  return { normalizationInput: body, context };
}

/**
 * Normalize client wire → Unified once via the endpoint transformer's
 * transformRequestOut when present. Chat Completions bodies are already
 * Unified-shaped; until Phase 2/3 add full converters, fall back to a
 * lightweight projection sufficient for routing.
 */
export async function normalizeClientToUnified(
  protocol: ClientProtocolContext["protocol"],
  normalizationInput: any,
  endpointTransformer: Transformer,
  context: any
): Promise<UnifiedChatRequest> {
  if (typeof endpointTransformer.transformRequestOut === "function") {
    const out = await endpointTransformer.transformRequestOut(
      normalizationInput,
      context
    );
    // Some transformers return { body, config }; prefer body when present.
    if (out && typeof out === "object" && "body" in (out as any) && (out as any).body) {
      return (out as any).body as UnifiedChatRequest;
    }
    return out as UnifiedChatRequest;
  }

  if (protocol === "openai_chat_completions") {
    return chatBodyAsUnified(normalizationInput);
  }

  if (protocol === "openai_responses") {
    return responsesRequestToUnified(normalizationInput);
  }

  if (protocol === "anthropic_messages") {
    // Anthropic endpoint transformer always defines transformRequestOut;
    // reaching here is unexpected.
    throw createApiError(
      "Anthropic transformRequestOut is required",
      500,
      "transformer_misconfigured"
    );
  }

  throw createApiError(
    `Client protocol '${protocol}' normalization is not implemented yet`,
    501,
    "protocol_not_implemented"
  );
}

function chatBodyAsUnified(body: any): UnifiedChatRequest {
  if (!body || typeof body !== "object") {
    throw createApiError("Invalid Chat Completions body", 400, "invalid_body");
  }
  if (!Array.isArray(body.messages)) {
    throw createApiError(
      "Chat Completions requires messages[]",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }
  if (typeof body.model !== "string" || !body.model) {
    throw createApiError(
      "Chat Completions requires model",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }
  return body as UnifiedChatRequest;
}

/**
 * Provider transformers are allowed to mutate their input. Every primary and
 * fallback attempt therefore needs an independent copy of the normalized body.
 */
export function cloneProtocolBody<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

const PASSTHROUGH_HEADER_DENYLIST = new Set([
  "authorization",
  "proxy-authorization",
  "x-api-key",
  "cookie",
  "set-cookie",
  "host",
  "content-length",
  "content-encoding",
  "accept-encoding",
  "connection",
  "keep-alive",
  "transfer-encoding",
  "upgrade",
  "te",
  "trailer",
  "expect",
  "via",
  "proxy-authenticate",
  "proxy-authentication-info",
  "x-auth-token",
  "openai-secret",
  "x-claude-desktop-no-iap-inject",
]);

const PASSTHROUGH_HEADER_DENY_PREFIXES = [
  "proxy-",
  "x-forwarded-",
  "x-real-ip",
];

/**
 * Preserve all end-to-end application headers for a native client. Only
 * credentials, cookies, CCR/proxy routing metadata and hop-by-hop transport
 * headers are removed; provider authentication is generated independently.
 */
export function sanitizePassthroughHeaders(
  headers: Headers | Record<string, unknown> | undefined
): Record<string, string> {
  const safe: Record<string, string> = {};
  if (!headers) return safe;

  const entries: Array<[string, unknown]> =
    headers instanceof Headers
      ? Array.from(headers.entries())
      : Object.entries(headers);

  for (const [rawName, rawValue] of entries) {
    const name = rawName.toLowerCase();
    if (
      PASSTHROUGH_HEADER_DENYLIST.has(name) ||
      PASSTHROUGH_HEADER_DENY_PREFIXES.some((prefix) => name.startsWith(prefix))
    ) continue;
    const value = Array.isArray(rawValue)
      ? rawValue.filter((item) => item != null).map(String).join(", ")
      : rawValue;
    if (typeof value === "string" && value) {
      safe[name] = value;
    }
  }
  return safe;
}

/**
 * Protocol-aware passthrough: only bypass when the provider speaks the same
 * client protocol (matching transformer name) and the request is same-protocol.
 */
export function shouldBypassTransformersProtocolAware(
  provider: any,
  endpointTransformer: Transformer,
  protocol: ClientProtocolContext["protocol"],
  bodyModel: string | undefined
): boolean {
  const use = provider.transformer?.use;
  if (!Array.isArray(use) || use.length !== 1) {
    return false;
  }
  if (use[0]?.name !== endpointTransformer.name) {
    return false;
  }

  const modelUse = bodyModel
    ? provider.transformer?.[bodyModel]?.use
    : undefined;
  if (Array.isArray(modelUse) && modelUse.length > 0) {
    if (
      modelUse.length !== 1 ||
      modelUse[0]?.name !== endpointTransformer.name
    ) {
      return false;
    }
  }

  // Cross-protocol: never bypass client normalization.
  // Same name means the provider transformer is the endpoint owner.
  const ownerByProtocol: Record<string, string> = {
    anthropic_messages: "Anthropic",
    openai_chat_completions: "OpenAI",
    openai_responses: "openai-responses",
  };
  return endpointTransformer.name === ownerByProtocol[protocol];
}

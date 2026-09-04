import { createApiError } from "@/api/middleware";
import type { FimInboundKind } from "./kinds";
import { V1_FIM_INBOUND_KIND } from "./kinds";
import type { UnifiedFimRequest } from "./types";

/**
 * Inbound → Unified FIM seam.
 * v1: only Codestral/Mistral kind. Future: inboundKind selects an adapter.
 */
export function inboundToUnifiedFim(
  body: unknown,
  inboundKind: FimInboundKind = V1_FIM_INBOUND_KIND
): UnifiedFimRequest {
  if (inboundKind !== "mistral") {
    throw createApiError(
      `FIM inbound kind '${inboundKind}' is not implemented yet`,
      501,
      "fim_inbound_not_implemented"
    );
  }
  return parseMistralCodestralInbound(body);
}

function parseMistralCodestralInbound(body: unknown): UnifiedFimRequest {
  if (!body || typeof body !== "object") {
    throw createApiError("Invalid FIM body", 400, "invalid_body");
  }
  const raw = body as Record<string, unknown>;
  if (typeof raw.prompt !== "string") {
    throw createApiError(
      "FIM requires prompt (string)",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }
  if (typeof raw.model !== "string" || !raw.model.trim()) {
    throw createApiError(
      "FIM requires model",
      400,
      "invalid_body",
      "invalid_request_error"
    );
  }

  const unified: UnifiedFimRequest = {
    model: raw.model.trim(),
    prompt: raw.prompt,
  };

  if (typeof raw.suffix === "string") {
    unified.suffix = raw.suffix;
  }
  if (typeof raw.max_tokens === "number") {
    unified.max_tokens = raw.max_tokens;
  }
  if (typeof raw.temperature === "number") {
    unified.temperature = raw.temperature;
  }
  if (typeof raw.top_p === "number") {
    unified.top_p = raw.top_p;
  }
  if (typeof raw.stream === "boolean") {
    unified.stream = raw.stream;
  }
  if (typeof raw.min_tokens === "number") {
    unified.min_tokens = raw.min_tokens;
  }
  if (typeof raw.random_seed === "number") {
    unified.random_seed = raw.random_seed;
  }
  if (typeof raw.stop === "string" || Array.isArray(raw.stop)) {
    unified.stop = raw.stop as string | string[];
  }

  return unified;
}

/** Clone client body for same-kind passthrough (preserve unknown fields). */
export function cloneFimClientBody(
  body: unknown,
  modelName: string
): Record<string, unknown> {
  const base =
    body && typeof body === "object"
      ? { ...(body as Record<string, unknown>) }
      : {};
  base.model = modelName;
  return base;
}

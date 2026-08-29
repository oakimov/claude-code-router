import type { Transformer } from "@/types/transformer";

export type CompiledTransformerPlan = {
  /** Body/header middleware, then at most one transport owner. */
  request: Transformer[];
  /** Reverse of `request` for onion-style response transforms. */
  response: Transformer[];
  transportOwner?: Transformer;
};

function transformerKey(transformer: Transformer): string | undefined {
  if (typeof transformer.name === "string" && transformer.name.length > 0) {
    return transformer.name;
  }
  const ctor = (transformer as { constructor?: { TransformerName?: string } })
    .constructor;
  if (ctor && typeof ctor.TransformerName === "string") {
    return ctor.TransformerName;
  }
  return undefined;
}

/**
 * Compile provider-level + model-level transformer chains into one plan.
 *
 * - Deduplicates by transformer name (first occurrence wins).
 * - Runs every non-transport transformer before the transport owner.
 * - Rejects configurations that leave more than one distinct transport owner.
 */
export function compileTransformerPlan(
  providerUse: Transformer[] | undefined,
  modelUse: Transformer[] | undefined,
  options?: { skipName?: string }
): CompiledTransformerPlan {
  const skipName = options?.skipName;
  const seen = new Set<string>();
  const body: Transformer[] = [];
  const transportOwners: Transformer[] = [];

  const consider = (transformer: Transformer | null | undefined) => {
    if (!transformer) return;
    const key = transformerKey(transformer);
    if (skipName && key === skipName) return;
    if (key) {
      if (seen.has(key)) return;
      seen.add(key);
    }
    if (transformer.ownsTransport === true) {
      transportOwners.push(transformer);
      return;
    }
    body.push(transformer);
  };

  for (const transformer of providerUse || []) consider(transformer);
  for (const transformer of modelUse || []) consider(transformer);

  if (transportOwners.length > 1) {
    const names = transportOwners
      .map((t) => transformerKey(t) || "<unnamed>")
      .join(", ");
    throw new Error(
      `Ambiguous transformer configuration: multiple transport owners after composition (${names}). Keep exactly one of opencode-headers, antigravity-auth, or cursor-sdk (or other ownsTransport transformers) in the combined provider/model chain.`
    );
  }

  const transportOwner = transportOwners[0];
  const request = transportOwner ? [...body, transportOwner] : body;
  return {
    request,
    response: [...request].reverse(),
    transportOwner,
  };
}

export function isExactProtocolResponsePlan(
  plan: CompiledTransformerPlan,
  endpointTransformer: Transformer,
  clientProtocolOwnerName: string | undefined
): boolean {
  const endpointName = transformerKey(endpointTransformer);
  return Boolean(
    endpointName &&
      endpointName === clientProtocolOwnerName &&
      plan.response.some(
        (transformer) => transformerKey(transformer) === endpointName
      )
  );
}

export function isExactProtocolRequestPlan(
  plan: CompiledTransformerPlan,
  endpointTransformer: Transformer,
  clientProtocolOwnerName: string | undefined
): boolean {
  const endpointName = transformerKey(endpointTransformer);
  return Boolean(
    endpointName &&
      endpointName === clientProtocolOwnerName &&
      plan.request.some(
        (transformer) => transformerKey(transformer) === endpointName
      )
  );
}

const WIRE_SAFE_MIDDLEWARE = new Set([
  "xai-auth",
  "opencode-headers",
  "qwen-auth",
]);

/**
 * v1 allowlist: middleware known to be safe on a kept native wire.
 * `OpenAI` is handled separately — it always runs. `reasoning` is
 * Chat-shaped and only allowed with an OpenAI owner. Everything else
 * (Anthropic/Responses owners) does not run Unified-only middleware.
 */
export function isWireSafeMiddlewareForKeep(
  name: string | undefined,
  ownerName: string | undefined
): boolean {
  if (!name) return false;
  if (name === "reasoning") return ownerName === "OpenAI";
  return WIRE_SAFE_MIDDLEWARE.has(name);
}

export function planContains(
  plan: CompiledTransformerPlan,
  name: string
): boolean {
  return plan.request.some((t) => transformerKey(t) === name);
}

/** Cancel a Response body when a newer transport result replaces it. */
export function cancelReplacedProviderResponse(
  previous: Response | undefined | null,
  next: Response | undefined | null
): void {
  if (!previous || previous === next) return;
  const body = previous.body;
  if (!body) return;
  void body.cancel("replaced by later transport owner").catch(() => {});
}

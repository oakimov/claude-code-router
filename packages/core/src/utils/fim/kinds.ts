/**
 * FIM inbound/outbound family kinds. v1 only implements mistral/codestral
 * inbound; deepseek/qwen inbound adapters are reserved for later.
 */

export type FimInboundKind = "mistral" | "deepseek" | "qwen";
export type FimOutboundFamily = "mistral" | "deepseek" | "qwen";

/** v1 hard-wired inbound kind (Codestral/Mistral-shaped client wire). */
export const V1_FIM_INBOUND_KIND: FimInboundKind = "mistral";

export function isFimProviderTransformerName(name: string | undefined): boolean {
  return typeof name === "string" && name.startsWith("fim.");
}

export function outboundFamilyFromTransformerName(
  name: string | undefined
): FimOutboundFamily | null {
  if (name === "fim.mistral") return "mistral";
  if (name === "fim.deepseek") return "deepseek";
  if (name === "fim.qwen") return "qwen";
  return null;
}

/**
 * Same-kind passthrough: no request or response shape translation —
 * auth/URL only. Kept generic so future DeepSeek→DeepSeek / Qwen→Qwen
 * inbound works the same.
 */
export function shouldFimPassthrough(
  inboundKind: FimInboundKind,
  outboundFamily: FimOutboundFamily
): boolean {
  return inboundKind === outboundFamily;
}

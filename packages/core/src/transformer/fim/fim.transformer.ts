import type { LLMProvider } from "@/types/llm";
import type { Transformer, TransformerContext } from "@/types/transformer";
import {
  V1_FIM_INBOUND_KIND,
  inboundToUnifiedFim,
  type UnifiedFimRequest,
} from "@/utils/fim";

/**
 * Protocol owner for POST /v1/fim/completions.
 * Validates Codestral-shaped inbound → Unified FIM (v1).
 * Client response framing: same-kind passthrough, else encode to inbound wire.
 */
export class FimTransformer implements Transformer {
  static TransformerName = "Fim";
  name = "Fim";
  endPoint = "/v1/fim/completions";
  logger?: any;

  async transformRequestOut(
    request: any,
    context: TransformerContext
  ): Promise<any> {
    const inboundKind =
      (context as any)?.fimInboundKind ?? V1_FIM_INBOUND_KIND;
    const unified = inboundToUnifiedFim(request, inboundKind);
    if (context) {
      (context as any).fimInboundKind = inboundKind;
      (context as any).unifiedFim = unified;
    }
    return unified as UnifiedFimRequest;
  }

  /**
   * Owner does not perform provider outbound; fim.* transformers do.
   * transformResponseIn is a no-op identity for pipeline compatibility.
   */
  async transformResponseIn(response: Response): Promise<Response> {
    return response;
  }
}

/** Type helper — FIM provider transformers accept UnifiedFimRequest. */
export type FimProviderTransformIn = (
  request: UnifiedFimRequest,
  provider: LLMProvider,
  context: TransformerContext
) => Promise<Record<string, any>>;

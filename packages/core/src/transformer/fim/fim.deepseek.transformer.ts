import type { LLMProvider } from "@/types/llm";
import type { Transformer, TransformerContext } from "@/types/transformer";
import {
  bearerAuthHeaders,
  cloneFimClientBody,
  encodeDeepseekFimBody,
  resolveFimDeepseekUrl,
  shouldFimPassthrough,
  type UnifiedFimRequest,
  V1_FIM_INBOUND_KIND,
} from "@/utils/fim";

/**
 * DeepSeek beta completions FIM outbound.
 * Same-kind (future deepseek inbound): auth + URL only.
 * Cross-family from Codestral Unified: prompt+suffix + 4K clamp + non-thinking.
 */
export class FimDeepseekTransformer implements Transformer {
  static TransformerName = "fim.deepseek";
  name = "fim.deepseek";
  logger?: any;

  async transformRequestIn(
    request: UnifiedFimRequest | any,
    provider: LLMProvider,
    context: TransformerContext
  ): Promise<Record<string, any>> {
    const inboundKind =
      (context as any)?.fimInboundKind ?? V1_FIM_INBOUND_KIND;
    const passthrough = shouldFimPassthrough(inboundKind, "deepseek");
    const modelName =
      typeof request?.model === "string" ? request.model : provider.models?.[0];

    const body = passthrough
      ? cloneFimClientBody(
          (context as any)?.fimClientBody ?? request,
          modelName
        )
      : encodeDeepseekFimBody({ ...request, model: modelName });

    return {
      body,
      config: {
        url: resolveFimDeepseekUrl(provider.baseUrl),
        headers: bearerAuthHeaders(provider.apiKey),
        __fimPassthrough: passthrough,
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    return response;
  }
}

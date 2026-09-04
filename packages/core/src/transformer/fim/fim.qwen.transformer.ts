import type { LLMProvider } from "@/types/llm";
import type { Transformer, TransformerContext } from "@/types/transformer";
import {
  bearerAuthHeaders,
  cloneFimClientBody,
  encodeQwenFimBody,
  resolveFimQwenCompletionsUrl,
  shouldFimPassthrough,
  type UnifiedFimRequest,
  V1_FIM_INBOUND_KIND,
} from "@/utils/fim";

/**
 * Qwen Completions FIM (LM Studio + DashScope).
 * Same-kind (future qwen inbound): auth + URL only — do not re-template.
 * Cross-family from Codestral Unified: HF tokens in prompt, no suffix field.
 */
export class FimQwenTransformer implements Transformer {
  static TransformerName = "fim.qwen";
  name = "fim.qwen";
  logger?: any;

  async transformRequestIn(
    request: UnifiedFimRequest | any,
    provider: LLMProvider,
    context: TransformerContext
  ): Promise<Record<string, any>> {
    const inboundKind =
      (context as any)?.fimInboundKind ?? V1_FIM_INBOUND_KIND;
    const passthrough = shouldFimPassthrough(inboundKind, "qwen");
    const modelName =
      typeof request?.model === "string" ? request.model : provider.models?.[0];

    const body = passthrough
      ? cloneFimClientBody(
          (context as any)?.fimClientBody ?? request,
          modelName
        )
      : encodeQwenFimBody({ ...request, model: modelName });

    return {
      body,
      config: {
        url: resolveFimQwenCompletionsUrl(provider.baseUrl),
        headers: bearerAuthHeaders(provider.apiKey),
        __fimPassthrough: passthrough,
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    return response;
  }
}

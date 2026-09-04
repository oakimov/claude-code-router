import type { LLMProvider } from "@/types/llm";
import type { Transformer, TransformerContext } from "@/types/transformer";
import {
  bearerAuthHeaders,
  cloneFimClientBody,
  resolveFimMistralUrl,
  shouldFimPassthrough,
  type UnifiedFimRequest,
  V1_FIM_INBOUND_KIND,
} from "@/utils/fim";

/**
 * Codestral / Mistral native FIM outbound.
 * Same-kind (mistral inbound): body passthrough — auth + URL only.
 */
export class FimMistralTransformer implements Transformer {
  static TransformerName = "fim.mistral";
  name = "fim.mistral";
  logger?: any;

  async transformRequestIn(
    request: UnifiedFimRequest | any,
    provider: LLMProvider,
    context: TransformerContext
  ): Promise<Record<string, any>> {
    const inboundKind =
      (context as any)?.fimInboundKind ?? V1_FIM_INBOUND_KIND;
    const passthrough = shouldFimPassthrough(inboundKind, "mistral");
    const modelName =
      typeof request?.model === "string" ? request.model : provider.models?.[0];

    const body = passthrough
      ? cloneFimClientBody(
          (context as any)?.fimClientBody ?? request,
          modelName
        )
      : {
          model: modelName,
          prompt: request.prompt,
          ...(typeof request.suffix === "string"
            ? { suffix: request.suffix }
            : {}),
          ...(request.max_tokens !== undefined
            ? { max_tokens: request.max_tokens }
            : {}),
          ...(request.temperature !== undefined
            ? { temperature: request.temperature }
            : {}),
          ...(request.top_p !== undefined ? { top_p: request.top_p } : {}),
          ...(request.stream !== undefined ? { stream: request.stream } : {}),
          ...(request.stop !== undefined ? { stop: request.stop } : {}),
          ...(request.min_tokens !== undefined
            ? { min_tokens: request.min_tokens }
            : {}),
          ...(request.random_seed !== undefined
            ? { random_seed: request.random_seed }
            : {}),
        };

    return {
      body,
      config: {
        url: resolveFimMistralUrl(provider.baseUrl),
        headers: bearerAuthHeaders(provider.apiKey),
        __fimPassthrough: passthrough,
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    return response;
  }
}

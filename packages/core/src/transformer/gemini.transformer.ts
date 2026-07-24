import { LLMProvider, UnifiedChatRequest } from "../types/llm";
import { Transformer } from "../types/transformer";
import {
  buildRequestBody,
  transformRequestOut,
  transformResponseOut,
} from "../utils/gemini.util";
import { attachGeminiCachedContent } from "../utils/gemini-cache";

export class GeminiTransformer implements Transformer {
  logger?: any;
  name = "gemini";

  endPoint = "/v1beta/models/:modelAndAction";

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context: any
  ): Promise<Record<string, any>> {
    const model = context?.req?.model || request.model || provider.model || "";
    const body = await attachGeminiCachedContent({
      body: buildRequestBody(request),
      modelResource: model.startsWith("models/") ? model : `models/${model}`,
      createUrl: new URL("../cachedContents", provider.baseUrl),
      headers: {
        "x-goog-api-key": provider.apiKey,
        Authorization: undefined,
      },
      logger: this.logger,
    });
    return {
      body,
      config: {
        url: new URL(
          `./${model}:${request.stream ? "streamGenerateContent?alt=sse" : "generateContent"
          }`,
          provider.baseUrl
        ),
        headers: {
          "x-goog-api-key": provider.apiKey,
          Authorization: undefined,
        },
      },
    };
  }

  transformRequestOut = transformRequestOut;

  async transformResponseOut(response: Response): Promise<Response> {
    return transformResponseOut(response, this.name, this.logger);
  }
}

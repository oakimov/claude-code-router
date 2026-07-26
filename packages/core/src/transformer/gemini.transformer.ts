import { LLMProvider, UnifiedChatRequest } from "../types/llm";
import {
  Transformer,
  TransformerContext,
  TransformerOptions,
} from "../types/transformer";
import {
  buildRequestBody,
  transformRequestOut,
  transformResponseOut,
} from "../utils/gemini.util";
import { attachGeminiCachedContent } from "../utils/gemini-cache";

export class GeminiTransformer implements Transformer {
  static TransformerName = "gemini";

  logger?: any;
  name = "gemini";
  endPoint = "/v1beta/models/:modelAndAction";

  private readonly cachedContent: boolean;
  private readonly thoughtSignatureFallback: "skip" | "none";

  constructor(private readonly options?: TransformerOptions) {
    // Antigravity chains ["gemini", { cachedContent: false }] — it has no
    // cachedContents resource. Both dialects default to the sentinel fallback
    // for unsigned tool replays; "none" opts out.
    this.cachedContent = options?.cachedContent !== false;
    this.thoughtSignatureFallback =
      options?.thoughtSignatureFallback === "none" ? "none" : "skip";
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context: any
  ): Promise<Record<string, any>> {
    const model =
      request.model ||
      (typeof context?.req?.model === "string"
        ? context.req.model
        : Array.isArray(context?.req?.model)
          ? context.req.model.join(",")
          : "") ||
      provider.models?.[0] ||
      "";

    const geminiBody = buildRequestBody(request, {
      thoughtSignatureFallback: this.thoughtSignatureFallback,
      // Scope cached thought signatures to this provider: a signature is only
      // valid at the upstream that minted it.
      signatureScope: provider?.name || this.name,
    });

    const body = this.cachedContent
      ? await attachGeminiCachedContent({
          body: geminiBody,
          modelResource: model.startsWith("models/")
            ? model
            : `models/${model}`,
          createUrl: new URL("../cachedContents", provider.baseUrl),
          headers: {
            "x-goog-api-key": provider.apiKey,
            Authorization: undefined,
          },
          logger: this.logger,
        })
      : geminiBody;

    return {
      body,
      config: {
        url: new URL(
          `./${model}:${
            request.stream
              ? "streamGenerateContent?alt=sse"
              : "generateContent"
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

  async transformRequestOut(request: any): Promise<UnifiedChatRequest> {
    return transformRequestOut(request);
  }

  async transformResponseOut(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    return transformResponseOut(
      response,
      this.name,
      this.logger,
      context?.provider?.name || this.name
    );
  }
}

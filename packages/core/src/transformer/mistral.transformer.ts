import { UnifiedChatRequest } from "../types/llm";
import { Transformer, TransformerContext, LLMProvider } from "../types/transformer";
import {
  buildRequestBody,
  transformRequestOut,
  transformResponseOut,
} from "../utils/mistral.util";

/**
 * Mistral Transformer for Claude Code Router
 *
 * Transforms Anthropic-style requests to Mistral's OpenAI-compatible API format.
 */
export class MistralTransformer implements Transformer {
  name = "mistral";
  logger?: any;

  /**
   * Transform incoming request to Mistral-compatible format
   */
  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    _context: TransformerContext
  ): Promise<Record<string, any>> {
    return {
      body: buildRequestBody(request),
      config: {
        url: new URL("/v1/chat/completions", provider.baseUrl),
        headers: {
          Authorization: `Bearer ${provider.apiKey}`,
        },
      },
    };
  }

  /**
   * Transform Mistral-specific request back to UnifiedChatRequest
   */
  transformRequestOut!: typeof transformRequestOut;

  /**
   * Transform response back — convert Mistral's content-array thinking format
   * to the delta.thinking / delta.content shape expected by @musistudio/llms.
   */
  async transformResponseOut(response: Response, _context: TransformerContext): Promise<Response> {
    try {
      return await transformResponseOut(response, this.name, this.logger);
    } catch (error: any) {
      this.logger?.error({ error: error.message, response }, `Mistral transformResponseOut error:`);
      throw error;
    }
  }
}

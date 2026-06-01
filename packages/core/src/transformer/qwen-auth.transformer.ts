import { Transformer } from "@/types/transformer";
import { getValidAccessToken } from "../utils/qwen-auth";
import { createSSEStreamReader, StreamContext, encodeSSELine } from "../utils/stream";

const QWEN_TARGET = "https://qwen.aikit.club";

// Qwen injects a trailing <details>...</details> metadata block into
// response content. Strip it from both JSON and SSE responses. Matches
// qwen-proxy.mjs:363-364.
const QWEN_META_RE = /(?:\\n|\n)?<details>[\s\S]*?<\/details>\s*/g;

export class QwenAuthTransformer implements Transformer {
  name = "qwen-auth";
  logger?: any;
  // No endPoint property — this transformer is purely an auth/response-shim
  // for use in provider.transformer.use[]. The OpenAI transformer
  // (endPoint = "/v1/chat/completions") registers the actual route.

  private async buildAuthConfig(
    provider: any
  ): Promise<{ url: string; headers: Record<string, string> }> {
    const tokens = await getValidAccessToken();
    const url =
      provider?.api_base_url || provider?.baseUrl || QWEN_TARGET;
    return {
      url,
      headers: { Authorization: `Bearer ${tokens.token}` },
    };
  }

  async transformRequestIn(
    request: any,
    provider: any
  ): Promise<Record<string, any>> {
    return {
      body: request,
      config: await this.buildAuthConfig(provider),
    };
  }

  async auth(_request: any, provider: any): Promise<any> {
    // Passthrough-mode path: identical headers/url, no body change.
    return { config: await this.buildAuthConfig(provider) };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";

    if (!contentType || contentType.includes("text/event-stream")) {
      // Cloudflare sometimes strips Content-Type on SSE — treat missing
      // content-type as a stream and tee it through the strip regex.
      if (!response.body) {
        return response;
      }
      return createSSEStreamReader(
        response,
        (line: string, ctx: StreamContext) => {
          if (!line.trim()) {
            ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
            return;
          }
          // Only the `data:` payload can contain the <details> block.
          // Strip the tag, then re-emit the (possibly shortened) line.
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(5);
            if (dataStr.trim() === "[DONE]") {
              ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
              return;
            }
            const cleaned = dataStr.replace(QWEN_META_RE, "");
            ctx.controller.enqueue(
              encodeSSELine(`data: ${cleaned}`, ctx.encoder)
            );
            return;
          }
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
        }
      );
    }

    if (contentType.includes("application/json")) {
      const text = await response.text();
      const cleaned = text.replace(QWEN_META_RE, "");
      return new Response(cleaned, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    return response;
  }
}

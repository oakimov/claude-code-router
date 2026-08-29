import { UnifiedChatRequest } from "@/types/llm";
import { Transformer } from "@/types/transformer";
import {
  deriveCacheSessionKey,
  stripMessagesCacheControl,
  stripToolsCacheControl,
} from "../utils/cacheControl";
import { extractToolMediaForStringToolApis } from "../utils/tool-content";

const DEFAULT_BRIDGE_URL = "http://127.0.0.1:3457";

/** Prompt API is text-only — collapse leftover image/file parts to notices. */
function flattenMediaPartsToText(messages: any[]): any[] {
  return messages.map((msg) => {
    if (!Array.isArray(msg?.content)) return msg;
    const texts: string[] = [];
    for (const part of msg.content) {
      if (!part || typeof part !== "object") continue;
      if (part.type === "text" && part.text) texts.push(String(part.text));
      else if (part.type === "image_url") texts.push("[Attached image]");
      else if (part.type === "file") {
        texts.push(
          `[Attached file${part.filename ? `: ${part.filename}` : ""}]`
        );
      }
    }
    return { ...msg, content: texts.join("\n") || "" };
  });
}

export class ChromeOnDeviceTransformer implements Transformer {
  name = "chrome-on-device";

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any,
    context?: any
  ): Promise<Record<string, any>> {
    // The model is text-only via Prompt API promptStreaming().
    // Tool definitions are converted to text instructions by the bridge.
    // We just need to route the request to the bridge.

    const bridgeUrl =
      process.env.CHROME_BRIDGE_URL ||
      provider?.baseUrl ||
      DEFAULT_BRIDGE_URL;

    // Ensure streaming in the original request body
    if (context?.req?.body) {
      context.req.body.stream = true;
    }
    const sessionKey = deriveCacheSessionKey(context, request);

    return {
      body: {
        ...request,
        messages: flattenMediaPartsToText(
          extractToolMediaForStringToolApis(
            stripMessagesCacheControl(request.messages)
          )
        ),
        tools: stripToolsCacheControl(request.tools),
        stream: true,
      },
      config: {
        url: `${bridgeUrl}/v1/chat/completions`,
        headers: {
          "Content-Type": "application/json",
          ...(sessionKey ? { "x-ccr-session-id": sessionKey } : {}),
        },
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    const contentType = response.headers.get("Content-Type") || "";

    if (contentType.includes("text/event-stream")) {
      return response;
    }

    if (!response.body) {
      return response;
    }

    // Bridge already emits Anthropic SSE — pass through
    return response;
  }
}

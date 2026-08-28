import { UnifiedChatRequest } from "@/types/llm";
import { Transformer, TransformerOptions } from "../types/transformer";
import { createSSEStreamReader, StreamContext, encodeSSEData, encodeSSELine } from "../utils/stream";
import {
  applyQwenPromptCaching,
  applyRawAnthropicPromptCaching,
  deriveCacheSessionKey,
  stripMessagesCacheControl,
  stripToolsCacheControl,
} from "../utils/cacheControl";
import { applyOpenAIChatCaching } from "../utils/openai.util";
import {
  createReasoningAccumulator,
  accumulateReasoning,
  finalizeReasoning,
  buildThinkingChunk,
  extractReasoningText,
  cleanReasoningFields,
} from "../utils/thinking";
import { HeaderRecord } from "../utils/headers";
import { readHeaderValue } from "../utils/anthropic-client-policy";
import {
  buildSynthesizedIdentityHeaders,
  isClaudeCodeClient,
} from "./claude-auth.transformer";
import { v4 as uuidv4 } from "uuid";

/** Claude Code identity headers forwarded verbatim when the client is genuine CLI. */
const CLAUDE_CODE_IDENTITY_HEADER_NAMES = [
  "user-agent",
  "x-app",
  "x-claude-code-session-id",
  "anthropic-dangerous-direct-browser-access",
  "x-client-request-id",
  "x-stainless-arch",
  "x-stainless-lang",
  "x-stainless-os",
  "x-stainless-package-version",
  "x-stainless-retry-count",
  "x-stainless-runtime",
  "x-stainless-runtime-version",
  "x-stainless-timeout",
] as const;

const DEFAULT_OPENROUTER_HTTP_REFERER =
  "https://github.com/caeliq/claude-code-router";
const DEFAULT_OPENROUTER_TITLE = "Claude Code Router";
const DEFAULT_OPENROUTER_CATEGORIES = "cli-agent";

/** Option keys that belong on the outbound HTTP request, not the JSON body. */
const OPENROUTER_HEADER_OPTION_KEYS = new Set([
  "http-referer",
  "HTTP-Referer",
  "referer",
  "x-title",
  "X-Title",
  "x-openrouter-title",
  "X-OpenRouter-Title",
  "x-openrouter-categories",
  "X-OpenRouter-Categories",
  "user-agent",
  "User-Agent",
]);

function pickOption(
  options: TransformerOptions | undefined,
  ...keys: string[]
): string | undefined {
  if (!options) return undefined;
  for (const key of keys) {
    const value = options[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return undefined;
}

function bodyOptionsFrom(
  options: TransformerOptions | undefined
): Record<string, unknown> {
  if (!options) return {};
  const body: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(options)) {
    if (OPENROUTER_HEADER_OPTION_KEYS.has(key)) continue;
    body[key] = value;
  }
  return body;
}

/**
 * OpenRouter app attribution headers. Some routed upstreams also require a
 * Claude Code–shaped User-Agent; attribution alone is not enough for those.
 * @see https://openrouter.ai/docs/app-attribution
 */
export function buildOpenRouterAttributionHeaders(
  options?: TransformerOptions
): HeaderRecord {
  const httpReferer =
    pickOption(options, "HTTP-Referer", "http-referer", "referer") ||
    process.env.OPENROUTER_HTTP_REFERER?.trim() ||
    DEFAULT_OPENROUTER_HTTP_REFERER;
  const title =
    pickOption(
      options,
      "X-OpenRouter-Title",
      "x-openrouter-title",
      "X-Title",
      "x-title"
    ) ||
    process.env.OPENROUTER_APP_TITLE?.trim() ||
    DEFAULT_OPENROUTER_TITLE;
  const categories =
    pickOption(
      options,
      "X-OpenRouter-Categories",
      "x-openrouter-categories"
    ) ||
    process.env.OPENROUTER_APP_CATEGORIES?.trim() ||
    DEFAULT_OPENROUTER_CATEGORIES;

  return {
    "HTTP-Referer": httpReferer,
    "X-Title": title,
    "X-OpenRouter-Title": title,
    "X-OpenRouter-Categories": categories,
  };
}

/**
 * Claude Code CLI identity headers. Prefer the caller's genuine CLI headers;
 * otherwise synthesize the same profile claude-auth uses for non-CLI clients.
 * Several OpenRouter upstreams reject requests without a claude-cli User-Agent.
 */
export function buildOpenRouterClaudeIdentityHeaders(
  clientHeaders?: Record<string, unknown>,
  options?: TransformerOptions
): HeaderRecord {
  const overrideUa = pickOption(options, "User-Agent", "user-agent");
  const clientUa = readHeaderValue(clientHeaders, "user-agent");

  let identity: HeaderRecord;
  if (isClaudeCodeClient(clientUa)) {
    identity = {};
    for (const name of CLAUDE_CODE_IDENTITY_HEADER_NAMES) {
      const value = readHeaderValue(clientHeaders, name);
      if (value) identity[name] = value;
    }
  } else {
    identity = buildSynthesizedIdentityHeaders();
  }

  if (overrideUa) {
    identity["User-Agent"] = overrideUa;
  }
  return identity;
}

export function buildOpenRouterOutboundHeaders(
  clientHeaders?: Record<string, unknown>,
  options?: TransformerOptions
): HeaderRecord {
  return {
    ...buildOpenRouterClaudeIdentityHeaders(clientHeaders, options),
    ...buildOpenRouterAttributionHeaders(options),
  };
}

export class OpenrouterTransformer implements Transformer {
  static TransformerName = "openrouter";
  logger?: any;

  constructor(private readonly options?: TransformerOptions) {}

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider?: any,
    context?: any
  ): Promise<Record<string, any>> {
    const cacheKey = deriveCacheSessionKey(context, request);

    const normalizedModel = (request.model || "").toLowerCase();
    if (normalizedModel.includes("anthropic/") || normalizedModel.includes("claude")) {
      request = applyRawAnthropicPromptCaching(request);
    } else if (
      normalizedModel.startsWith("openai/") ||
      normalizedModel.startsWith("gpt-")
    ) {
      request = applyOpenAIChatCaching(request, provider, context);
    } else if (
      normalizedModel.includes("qwen") ||
      normalizedModel.includes("alibaba")
    ) {
      request = applyQwenPromptCaching(request);
    } else if (normalizedModel.includes("gemini")) {
      // Gemini caching on OpenRouter requires cache_control breakpoints
      // inserted within message content, in the same ephemeral format as
      // Anthropic (OpenRouter uses only the last breakpoint for Gemini). This
      // is the content-level marker applyQwenPromptCaching emits; a top-level
      // request.cache_control would not be honoured on this endpoint.
      request = applyQwenPromptCaching(request);
    } else {
      request = {
        ...request,
        messages: stripMessagesCacheControl(request.messages),
        tools: stripToolsCacheControl(request.tools),
      };
    }

    if (!request.model.includes("claude")) {
      // Handle non-HTTP image URLs for non-Claude models
      request.messages.forEach((msg) => {
        if (Array.isArray(msg.content)) {
          msg.content.forEach((item: any) => {
            if (item.type === "image_url") {
              if (!item.image_url.url.startsWith("http")) {
                item.image_url.url = `${item.image_url.url}`;
              }
              delete item.media_type;
            }
          });
        }
      });
    } else {
      request.messages.forEach((msg) => {
        if (Array.isArray(msg.content)) {
          msg.content.forEach((item: any) => {
            if (item.type === "image_url") {
              if (!item.image_url.url.startsWith("http")) {
                item.image_url.url = `data:${item.media_type};base64,${item.image_url.url}`;
              }
              delete item.media_type;
            }
          });
        }
      });
    }

    // Body-only options (e.g. OpenRouter `provider` routing). Header options
    // are applied on config.headers below so they are not leaked into JSON.
    Object.assign(request, bodyOptionsFrom(this.options));
    if (cacheKey) {
      (request as any).session_id = cacheKey;
      (request as any).prompt_cache_key = cacheKey;
    }

    const clientHeaders =
      (context?.req?.headers as Record<string, unknown> | undefined) || {};
    const headers = buildOpenRouterOutboundHeaders(clientHeaders, this.options);

    return {
      body: request,
      config: { headers },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    if (response.headers.get("Content-Type")?.includes("application/json")) {
      const jsonResponse = await response.json();
      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    } else if (response.headers.get("Content-Type")?.includes("stream")) {
      if (!response.body) return response;

      const accumulator = createReasoningAccumulator();
      let hasTextContent = false;
      let hasToolCall = false;

      return createSSEStreamReader(response, (line: string, ctx: StreamContext) => {
        if (!line.trim()) {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          return;
        }

        if (!line.startsWith("data: ") || line.trim() === "data: [DONE]") {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
          return;
        }

        try {
          const jsonStr = line.slice(6);
          const data = JSON.parse(jsonStr);

          if (data.usage) {
            this.logger?.debug(
              { usage: data.usage, hasToolCall },
              "usage"
            );
            data.choices[0].finish_reason = hasToolCall
              ? "tool_calls"
              : "stop";
          }

          if (data.choices?.[0]?.finish_reason === "error") {
            ctx.controller.enqueue(
              encodeSSEData(
                JSON.stringify({ error: data.choices?.[0]?.error }),
                ctx.encoder
              )
            );
          }

          if (data.choices?.[0]?.delta?.content && !hasTextContent) {
            hasTextContent = true;
          }

          const delta = data.choices?.[0]?.delta;
          const reasoningText = delta ? extractReasoningText(delta) : null;

          if (reasoningText) {
            accumulateReasoning(accumulator, reasoningText);
            const thinkingChunk = buildThinkingChunk(data, {
              content: reasoningText,
            });
            cleanReasoningFields(thinkingChunk.choices[0].delta);
            ctx.controller.enqueue(encodeSSEData(JSON.stringify(thinkingChunk), ctx.encoder));
            return;
          }

          if (
            delta?.content &&
            accumulator.hasContent &&
            !accumulator.isComplete
          ) {
            const { content, signature } = finalizeReasoning(accumulator);
            const thinkingChunk = buildThinkingChunk(data, {
              content,
              signature,
            });
            cleanReasoningFields(thinkingChunk.choices[0].delta);
            thinkingChunk.choices[0].delta.content = null;
            ctx.controller.enqueue(encodeSSEData(JSON.stringify(thinkingChunk), ctx.encoder));
          }

          if (delta?.reasoning) {
            delete delta.reasoning;
          }

          if (
            delta?.tool_calls?.length &&
            !Number.isNaN(parseInt(delta.tool_calls[0].id, 10))
          ) {
            delta.tool_calls.forEach((tool: any) => {
              tool.id = `call_${uuidv4()}`;
            });
          }

          if (delta?.tool_calls?.length && !hasToolCall) {
            hasToolCall = true;
          }

          if (delta?.tool_calls?.length && hasTextContent) {
            if (typeof data.choices[0].index === "number") {
              data.choices[0].index += 1;
            } else {
              data.choices[0].index = 1;
            }
          }

          ctx.controller.enqueue(encodeSSEData(JSON.stringify(data), ctx.encoder));
        } catch {
          ctx.controller.enqueue(encodeSSELine(line, ctx.encoder));
        }
      });
    }

    return response;
  }
}

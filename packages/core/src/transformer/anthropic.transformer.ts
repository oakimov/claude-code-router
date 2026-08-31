import { ChatCompletion } from "openai/resources";
import {
  LLMProvider,
  MessageContent,
  UnifiedChatRequest,
  UnifiedMessage,
  UnifiedTool,
} from "@/types/llm";
import {
  Transformer,
  TransformerContext,
  TransformerOptions,
} from "@/types/transformer";
import { v4 as uuidv4 } from "uuid";
import { createApiError } from "@/api/middleware";
import { formatBase64 } from "@/utils/image";
import { sanitizeToolCallId } from "@/utils/toolCallId";
import {
  anthropicToolResultToUnified,
  unifiedToolContentToAnthropic,
} from "@/utils/tool-content";
import { buildAnthropicRequestRuntime } from "@/types/turn-intent";
import {
  unprefixClaudeToolNames,
} from "@/utils/claude-billing";
import { createSSEStreamReader } from "@/utils/stream";
import { transformResponseOut as anthropicWireToUnifiedResponse } from "../utils/vertex-claude.util";
import {
  lookupClaudeModelCatalogEntry,
  stripOneMillionContextMarker,
} from "@/utils/claude-model-catalog";
import {
  applyClaudeModelCapabilityAdjustments,
  buildSynthesizedIdentityHeaders,
  buildSynthesizedUserMetadata,
  modelIdForRequestedOneMillionBeta,
  resolveClaudeAuthBetas,
} from "./claude-auth.transformer";
import { buildAnthropicMessagesUrl } from "@/utils/anthropic-url";
import {
  canonicalReasoning,
  isReasoningDisabled,
  normalizeReasoningEffort,
  resolveOutboundReasoningSummary,
  toAnthropicReasoningEffort,
} from "@/utils/reasoning-effort";
// Relative path, not the "@/" alias: esbuild's alias resolver treats the
// trailing ".util" as a file extension and fails to resolve dotted module
// names. Every other importer of this module uses a relative path too.
import {
  anthropicThinkingSignatureFrom,
  canonicalAssistantTurn,
  thinkingFromUnifiedAssistant,
} from "../utils/openai.responses.util";

import { estimateRequestPromptTokens } from "../cursor-sdk/usage";

function toAnthropicCacheUsage(usage: any): Record<string, number> {
  const cached = usage?.prompt_tokens_details?.cached_tokens || 0;
  const written = usage?.prompt_tokens_details?.cache_write_tokens || 0;
  return {
    // Inverse of toOpenAIUsage (vertex-claude.util.ts), where
    // prompt_tokens = input_tokens + cache_read + cache_creation. The three
    // are disjoint in Anthropic semantics, so recover input_tokens by
    // subtracting both cached and written back out.
    input_tokens: Math.max(
      0,
      (usage?.prompt_tokens || 0) - cached - written
    ),
    output_tokens: usage?.completion_tokens || 0,
    cache_creation_input_tokens: written,
    cache_read_input_tokens: cached,
  };
}

function toAnthropicMessageStartUsage(usage: any): Record<string, number> {
  return {
    ...toAnthropicCacheUsage(usage),
    // Native Anthropic puts input/cache on message_start and leaves
    // output_tokens for message_delta. Emitting completion tokens here would
    // double-count clients that sum start+delta.
    output_tokens: 0,
  };
}

function estimateAnthropicMessageStartUsage(
  context?: TransformerContext
): Record<string, number> {
  const estimateSource = (context as any)?.anthropicEstimateRequest as
    | UnifiedChatRequest
    | undefined;
  if (!estimateSource) {
    return {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    };
  }

  try {
    // Claude Code's meter is input+cache_creation+cache_read. An early estimate
    // of the request body is enough to trip autocompact before the final
    // provider usage arrives on message_delta.
    const estimated = estimateRequestPromptTokens(estimateSource);
    return {
      input_tokens: Math.max(0, Math.floor(estimated)),
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    };
  } catch {
    return {
      input_tokens: 0,
      output_tokens: 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    };
  }
}

function resolveMessageStartUsage(
  chunkUsage: any,
  context?: TransformerContext
): Record<string, number> {
  if (chunkUsage) {
    return toAnthropicMessageStartUsage(chunkUsage);
  }
  return estimateAnthropicMessageStartUsage(context);
}

function normalizeAnthropicOutputConfig(
  value: Record<string, any> | undefined,
  reasoningDisabled: boolean
): Record<string, any> | undefined {
  if (!value || typeof value !== "object") return undefined;
  const outputConfig = structuredClone(value);
  if (outputConfig.effort !== undefined) {
    const effort = reasoningDisabled
      ? undefined
      : toAnthropicReasoningEffort(outputConfig.effort);
    if (effort) outputConfig.effort = effort;
    else delete outputConfig.effort;
  }
  return Object.keys(outputConfig).length ? outputConfig : undefined;
}

export class AnthropicTransformer implements Transformer {
  name = "Anthropic";
  endPoint = "/v1/messages";
  private useBearer: boolean;
  logger?: any;

  constructor(private readonly options?: TransformerOptions) {
    this.useBearer = this.options?.UseBearer ?? false;
  }

  async auth(
    request: any,
    provider: LLMProvider,
    context?: TransformerContext
  ): Promise<any> {
    const headers: Record<string, string | undefined> = {};

    if (this.useBearer) {
      headers["Authorization"] = `Bearer ${provider.apiKey}`;
      headers["x-api-key"] = undefined;
    } else {
      headers["x-api-key"] = provider.apiKey;
      headers["Authorization"] = undefined;
    }

    return {
      body: request,
      config: {
        url: buildAnthropicMessagesUrl(provider.baseUrl),
        headers,
      },
    };
  }

  /**
   * Provider-side: Unified → Anthropic Messages wire.
   * Enables cross-protocol clients (Chat/Responses) to reach Anthropic-shaped
   * upstreams. Same-protocol Anthropic clients still use transformRequestOut
   * for client→Unified and may bypass via protocol-aware passthrough.
   */
  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    _context?: TransformerContext
  ): Promise<Record<string, any>> {
    const anthropicBody = AnthropicTransformer.buildAnthropicBody(
      request,
      this.logger,
      {
        ...(_context || {}),
        provider,
      } as TransformerContext
    );

    anthropicBody.model = stripOneMillionContextMarker(anthropicBody.model).modelId;

    const protocolContext = _context?.protocolContext;
    const isThirdPartyAnthropic =
      protocolContext?.anthropicDestinationInScope === true &&
      protocolContext.anthropicClientKind === "other" &&
      protocolContext.anthropicPolicyApplied === true;
    const isApiKeyAnthropic =
      isThirdPartyAnthropic && protocolContext?.anthropicProviderMode === "api_key";

    if (isApiKeyAnthropic) {
      applyClaudeModelCapabilityAdjustments(
        anthropicBody,
        lookupClaudeModelCatalogEntry(anthropicBody.model)
      );
      anthropicBody.metadata = {
        ...(anthropicBody.metadata || {}),
        ...buildSynthesizedUserMetadata(),
      };
    }

    // claude-auth's non-Claude-Code branch stashes catalog-driven capability
    // clamping and synthesized identity metadata here, to run immediately
    // after the body this stage owns building. See TransformerContext.
    _context?.claudeAuthPostBuildHook?.(anthropicBody);

    const url = buildAnthropicMessagesUrl(provider?.baseUrl);

    const headers: Record<string, string | undefined> = {
      "Content-Type": "application/json",
      "anthropic-version": "2023-06-01",
    };

    if (isApiKeyAnthropic) {
      Object.assign(headers, buildSynthesizedIdentityHeaders(), {
        "anthropic-beta": resolveClaudeAuthBetas(
          modelIdForRequestedOneMillionBeta(
            anthropicBody.model,
            protocolContext?.requestedOneMillion
          ),
          {
            includeOAuthBeta: false,
          }
        ),
      });
    }

    // claude-auth owns OAuth bearer auth when it shares this provider's
    // chain; this stage must never introduce or clobber it with a static key.
    const ownsAuth = !provider?.transformer?.use?.some(
      (transformer: Transformer) => transformer?.name === "claude-auth"
    );
    if (ownsAuth) {
      if (this.useBearer) {
        headers["Authorization"] = `Bearer ${provider.apiKey}`;
        headers["x-api-key"] = undefined;
      } else {
        headers["x-api-key"] = provider.apiKey;
        headers["Authorization"] = undefined;
      }
    }

    return {
      body: anthropicBody,
      config: {
        url,
        headers,
      },
    };
  }

  /**
   * Provider-side: Anthropic Messages wire → Unified (Chat Completions shape).
   * Reuses the Vertex Claude Anthropic→Unified converter so streaming and JSON
   * envelopes match existing Anthropic upstream behavior.
   */
  async transformResponseOut(
    response: Response,
    _context?: TransformerContext
  ): Promise<Response> {
    const converted = await anthropicWireToUnifiedResponse(
      response,
      this.name,
      this.logger
    );
    const nameMap = _context?.protocolContext?.claudeAuthToolNameMap as
      | Map<string, string>
      | undefined;
    if (!nameMap?.size || !converted.ok) return converted;

    const contentType = converted.headers.get("Content-Type") || "";
    if (contentType.includes("application/json")) {
      const body = await converted.json();
      unprefixClaudeToolNames(body, nameMap);
      return new Response(JSON.stringify(body), {
        status: converted.status,
        statusText: converted.statusText,
        headers: converted.headers,
      });
    }
    if (contentType.includes("text/event-stream") && converted.body) {
      return createSSEStreamReader(
        converted,
        (line, streamContext) => {
          if (!line.startsWith("data: ") || line.trim() === "data: [DONE]") {
            streamContext.controller.enqueue(
              streamContext.encoder.encode(line + "\n")
            );
            return;
          }
          try {
            const payload = JSON.parse(line.slice(6));
            unprefixClaudeToolNames(payload, nameMap);
            streamContext.controller.enqueue(
              streamContext.encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)
            );
          } catch {
            streamContext.controller.enqueue(
              streamContext.encoder.encode(line + "\n")
            );
          }
        },
        { logger: this.logger }
      );
    }
    return converted;
  }

  async transformRequestOut(
    request: Record<string, any>,
    context?: TransformerContext
  ): Promise<UnifiedChatRequest> {
    if (context) {
      context.unifiedRequest = buildAnthropicRequestRuntime(request);
    }

    const messages: UnifiedMessage[] = [];

    if (request.system) {
      if (typeof request.system === "string") {
        messages.push({
          role: "system",
          content: request.system,
        });
      } else if (Array.isArray(request.system) && request.system.length) {
        const textParts = request.system
          .filter((item: any) => item.type === "text" && item.text)
          .map((item: any) => ({
            type: "text" as const,
            text: item.text,
            cache_control: item.cache_control,
          }));
        messages.push({
          role: "system",
          content: textParts,
        });
      }
    }

    const requestMessages = JSON.parse(JSON.stringify(request.messages || []));

    requestMessages?.forEach((msg: any) => {
      if (msg.role === "user" || msg.role === "assistant") {
        if (typeof msg.content === "string") {
          messages.push({
            role: msg.role,
            content: msg.content,
          });
          return;
        }

        if (Array.isArray(msg.content)) {
          if (msg.role === "user") {
            const toolParts = msg.content.filter(
              (c: any) => c.type === "tool_result" && c.tool_use_id
            );
            if (toolParts.length) {
              toolParts.forEach((tool: any) => {
                const toolMessage: UnifiedMessage = {
                  role: "tool",
                  content: anthropicToolResultToUnified(tool.content),
                  tool_call_id:
                    sanitizeToolCallId(tool.tool_use_id) ?? tool.tool_use_id,
                  cache_control: tool.cache_control,
                };
                messages.push(toolMessage);
              });
            }

            const textAndMediaParts = msg.content.filter(
              (c: any) =>
                (c.type === "text" && c.text) ||
                (c.type === "image" && c.source)
            );
            if (textAndMediaParts.length) {
              messages.push({
                role: "user",
                content: textAndMediaParts.map((part: any) => {
                  if (part?.type === "image") {
                    return {
                      type: "image_url",
                      image_url: {
                        url:
                          part.source?.type === "base64"
                            ? formatBase64(
                                part.source.data,
                                part.source.media_type
                              )
                            : part.source.url,
                      },
                      media_type: part.source.media_type,
                      cache_control: part.cache_control,
                    };
                  }
                  return part;
                }),
              });
            }
          } else if (msg.role === "assistant") {
            const assistantMessage: UnifiedMessage = {
              role: "assistant",
              content: "",
            };
            const textParts = msg.content.filter(
              (c: any) => c.type === "text" && c.text
            );
            if (textParts.length) {
              assistantMessage.content = textParts.some(
                (text: any) => text.cache_control
              )
                ? textParts.map((text: any) => ({
                    type: "text",
                    text: text.text,
                    cache_control: text.cache_control,
                  }))
                : textParts.map((text: any) => text.text).join("\n");
            }

            const toolCallParts = msg.content.filter(
              (c: any) => c.type === "tool_use" && c.id
            );
            if (toolCallParts.length) {
              assistantMessage.tool_calls = toolCallParts.map((tool: any) => {
                return {
                  id: sanitizeToolCallId(tool.id) ?? tool.id,
                  type: "function" as const,
                  function: {
                    name: tool.name,
                    arguments: JSON.stringify(tool.input || {}),
                  },
                  cache_control: tool.cache_control,
                };
              });
            }

            const thinkingPart = msg.content.find(
              (c: any) => c.type === "thinking" && c.signature
            );
            if (thinkingPart) {
              assistantMessage.thinking = {
                content: thinkingPart.thinking,
                signature: thinkingPart.signature,
              };
            }

            messages.push(assistantMessage);
          }
          return;
        }
      }
    });

    const result: UnifiedChatRequest = {
      messages,
      model: request.model,
      max_tokens: request.max_tokens,
      temperature: request.temperature,
      stream: request.stream,
      tools: request.tools?.length
        ? this.convertAnthropicToolsToUnified(request.tools)
        : undefined,
      tool_choice: request.tool_choice,
      ...((request as any).cache_control
        ? { cache_control: (request as any).cache_control }
        : {}),
    };
    const effort = normalizeReasoningEffort(
      request.output_config?.effort ?? request.effort
    );
    if (request.thinking || effort) {
      const thinkingEnabled = request.thinking
        ? request.thinking.type === "enabled" ||
          request.thinking.type === "adaptive"
        : true;
      const thinkingSummary =
        typeof (request.thinking as any)?.summary === "string"
          ? ((request.thinking as any).summary as string).trim().toLowerCase()
          : undefined;
      const preservedSummary =
        thinkingSummary === "auto" ||
        thinkingSummary === "detailed" ||
        thinkingSummary === "concise" ||
        thinkingSummary === "none"
          ? thinkingSummary
          : undefined;
      result.reasoning = {
        // Claude Code sends effort in output_config (observed: `thinking:
        // {type:"adaptive"}` + `output_config: {effort:"high"}` and no budget).
        ...canonicalReasoning(effort, thinkingEnabled),
        // Clients that do send an explicit budget keep control of it; backends
        // on a token-budget dialect (Gemini 2.5, Claude via Antigravity) prefer
        // this over an effort-derived budget.
        ...(typeof request.thinking?.budget_tokens === "number"
          ? { max_tokens: request.thinking.budget_tokens }
          : {}),
        // Preserve Anthropic thinking.summary when present so Responses /
        // REASONING_AUTO_SUMMARY destinations see an explicit preference.
        ...(preservedSummary ? { summary: preservedSummary } : {}),
      };
    }

    // Retain source-only Anthropic parameters before destination routing. They
    // live in request-local protocol context so unrelated providers never
    // serialize implementation fields from the Unified request.
    const protocolContext = context?.protocolContext;
    if (protocolContext?.protocol === "anthropic_messages") {
      protocolContext.anthropicSource = {
        ...(request.metadata
          ? { metadata: JSON.parse(JSON.stringify(request.metadata)) }
          : {}),
        ...(request.thinking
          ? { thinking: JSON.parse(JSON.stringify(request.thinking)) }
          : {}),
        ...(request.output_config
          ? { outputConfig: JSON.parse(JSON.stringify(request.output_config)) }
          : {}),
        ...(request.stop_sequences
          ? { stopSequences: [...request.stop_sequences] }
          : {}),
      };
    }
    // Legacy direct transformer callers may already know the provider chain.
    const usesClaudeAuth = Array.isArray(context?.provider?.transformer?.use) &&
      context.provider.transformer.use.some(
        (t: any) => t?.name === "claude-auth"
      );
    if (usesClaudeAuth) {
      if (request.thinking) result.anthropic_thinking = request.thinking;
      if (request.output_config) result.anthropic_output_config = request.output_config;
      if (request.metadata) result.anthropic_metadata = request.metadata;
      if (request.stop_sequences) {
        result.anthropic_stop_sequences = request.stop_sequences;
      }
    }
    if (request.tool_choice) {
      if (request.tool_choice.type === "tool") {
        result.tool_choice = {
          type: "function",
          function: { name: request.tool_choice.name },
        };
      } else if (request.tool_choice.type === "any") {
        // Unified is Chat Completions: Anthropic "any" is Chat/Responses "required".
        result.tool_choice = "required";
      } else {
        result.tool_choice = request.tool_choice.type;
      }
      if (request.tool_choice.disable_parallel_tool_use === true) {
        result.parallel_tool_calls = false;
      }
    }

    const format = request.output_config?.format;
    if (format?.type === "json_schema" && format.schema && typeof format.schema === "object") {
      // OpenAI Chat Completions requires json_schema.name. Claude Code often
      // omits it on Anthropic output_config.format (title generation, etc.);
      // without a default, OpenAI-compatible upstreams (OpenCode/DeepSeek,
      // NVIDIA, …) 400 on "json_schema.name Field required".
      const schemaName =
        typeof format.name === "string" && format.name.length > 0
          ? format.name
          : "response";
      (result as any).response_format = {
        type: "json_schema",
        json_schema: {
          name: schemaName,
          schema: format.schema,
          ...(format.strict !== undefined ? { strict: format.strict } : {}),
        },
      };
    }

    if (context) {
      // Keep a full Unified request for early message_start usage estimates.
      // context.unifiedRequest is turn-intent only and is too sparse to meter.
      (context as any).anthropicEstimateRequest = result;
    }

    return result;
  }

  async transformResponseIn(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    const isStream = response.headers
      .get("Content-Type")
      ?.includes("text/event-stream");
    if (isStream) {
      if (!response.body) {
        throw new Error("Stream response body is null");
      }
      const convertedStream = await this.convertOpenAIStreamToAnthropic(
        response.body,
        context!
      );
      return new Response(convertedStream, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } else {
      const data = (await response.json()) as any;
      const anthropicResponse = this.convertOpenAIResponseToAnthropic(
        data,
        context!
      );
      return new Response(JSON.stringify(anthropicResponse), {
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  /**
   * Rebuild an Anthropic-format request body from a UnifiedChatRequest.
   * Used by claude-auth.transformRequestIn() to reconstruct the body before
   * sending to Anthropic, preserving all original parameters.
   */
  static buildAnthropicBody(
    request: UnifiedChatRequest,
    logger?: any,
    context?: TransformerContext
  ): Record<string, any> {
    // System prompt: fold every source exactly once and in source order —
    // top-level request.system (set by some transformers) first, then any
    // residual role:"system"/"developer" messages (how transformRequestOut
    // stores it). The normal pipeline consumes those messages earlier, so
    // residual ones only appear for direct callers and must merge, not drop.
    const systemBlocks: any[] = [];
    const appendSystemBlocks = (value: any) => {
      if (typeof value === "string") {
        if (value) systemBlocks.push({ type: "text", text: value });
        return;
      }
      if (!Array.isArray(value)) return;
      for (const part of value) {
        if (part?.type === "text" && part.text) {
          systemBlocks.push({
            type: "text",
            text: part.text,
            ...(part.cache_control ? { cache_control: part.cache_control } : {}),
          });
        }
      }
    };
    if (request.system) {
      appendSystemBlocks(request.system);
    }

    // Messages: convert Unified format back to Anthropic format
    const messages: any[] = [];
    for (const msg of request.messages) {
      if (msg.role === "system" || (msg.role as string) === "developer") {
        appendSystemBlocks(msg.content);
        continue;
      }

      if (msg.role === "tool") {
        // Unified tool messages → Anthropic tool_result blocks, merged into preceding user message
        const toolResult: any = {
          type: "tool_result",
          // Sanitized on both sides of the pair so ids minted before the fix
          // (or by any provider using a non-conforming alphabet) still match.
          tool_use_id:
            sanitizeToolCallId(msg.tool_call_id) ?? msg.tool_call_id,
          content: unifiedToolContentToAnthropic(msg.content),
          ...(msg.cache_control ? { cache_control: msg.cache_control } : {}),
        };
        const last = messages[messages.length - 1];
        if (last?.role === "user" && Array.isArray(last.content)) {
          last.content.push(toolResult);
        } else {
          messages.push({ role: "user", content: [toolResult] });
        }
        continue;
      }

      if (msg.role === "assistant") {
        const content: any[] = [];
        // Fixed order: thinking → text → images → tool_use.
        const turn = canonicalAssistantTurn(msg);
        if (turn.thinking) {
          const signature = anthropicThinkingSignatureFrom(turn.thinking);
          content.push({
            type: "thinking",
            thinking: turn.thinking.content,
            ...(signature ? { signature } : {}),
          });
        }
        for (const text of turn.texts) {
          content.push({
            type: "text",
            text: text.text,
            ...(text.cache_control ? { cache_control: text.cache_control } : {}),
          });
        }
        for (const image of turn.images) {
          const url = image.image_url?.url;
          if (typeof url === "string" && url.startsWith("data:")) {
            const [meta, data] = url.split(",");
            const mediaType = meta.split(":")[1]?.split(";")[0] ?? "image/jpeg";
            content.push({
              type: "image",
              source: { type: "base64", media_type: mediaType, data },
            });
          } else if (typeof url === "string") {
            content.push({
              type: "image",
              source: { type: "url", url },
            });
          }
        }
        if (turn.toolCalls.length) {
          for (const tc of turn.toolCalls) {
            let input: Record<string, any> = {};
            try { input = JSON.parse(tc.function.arguments || "{}"); } catch (e) { (logger?.error ?? console.error)("Failed to parse tool_call arguments for tool '%s': %s", tc.function.name, e); }
            content.push({
              type: "tool_use",
              id: sanitizeToolCallId(tc.id) ?? tc.id,
              name: tc.function.name,
              input,
              ...(tc.cache_control
                ? { cache_control: tc.cache_control }
                : {}),
            });
          }
        }
        if (content.length > 0) messages.push({ role: "assistant", content });
        continue;
      }

      if (msg.role === "user") {
        const content: any[] = [];
        if (typeof msg.content === "string") {
          content.push({ type: "text", text: msg.content });
        } else if (Array.isArray(msg.content)) {
          for (const part of msg.content) {
            if (part.type === "text" && part.text) {
              content.push({ type: "text", text: part.text, ...((part as any).cache_control ? { cache_control: (part as any).cache_control } : {}) });
            } else if (part.type === "image_url" && (part as any).image_url?.url) {
              const url = (part as any).image_url.url;
              if (url.startsWith("data:")) {
                const [meta, data] = url.split(",");
                const mediaType = meta.split(":")[1]?.split(";")[0] ?? "image/jpeg";
                content.push({
                  type: "image",
                  source: { type: "base64", media_type: mediaType, data },
                  ...((part as any).cache_control
                    ? { cache_control: (part as any).cache_control }
                    : {}),
                });
              } else {
                content.push({
                  type: "image",
                  source: { type: "url", url },
                  ...((part as any).cache_control
                    ? { cache_control: (part as any).cache_control }
                    : {}),
                });
              }
            } else if ((part as any).type === "file") {
              const fileBlocks = unifiedToolContentToAnthropic([part]);
              if (Array.isArray(fileBlocks)) content.push(...fileBlocks);
            }
          }
        }
        if (content.length > 0) messages.push({ role: "user", content });
        continue;
      }
    }

    // Tools: convert Unified format back to Anthropic format
    let tools: any[] | undefined;
    if (request.tools?.length) {
      const unsupported = request.tools.find(
        (tool: any) => tool?.type !== "function" || !tool?.function
      );
      if (unsupported) {
        throw createApiError(
          `Tool type '${(unsupported as any)?.type || "unknown"}' cannot be represented by Anthropic Messages`,
          400,
          "unsupported_hosted_tool",
          "invalid_request_error"
        );
      }
      tools = request.tools.map((tool) => ({
        name: tool.function.name,
        description: tool.function.description || "",
        input_schema: tool.function.parameters,
        ...(tool.cache_control ? { cache_control: tool.cache_control } : {}),
      }));
    }

    // Tool choice: convert Unified format back to Anthropic format
    let tool_choice: any | undefined;
    if (request.tool_choice) {
      const tc = request.tool_choice;
      if (tc === "auto") tool_choice = { type: "auto" };
      else if (tc === "required") tool_choice = { type: "any" };
      else if (typeof tc === "string") tool_choice = { type: "tool", name: tc };
      else if (tc.type === "function") tool_choice = { type: "tool", name: tc.function?.name };
    }
    // Responses `parallel_tool_calls: false` is Anthropic's
    // disable_parallel_tool_use. When tools exist and the client did not
    // force a choice, emit auto so the flag has a home.
    if (
      request.parallel_tool_calls === false &&
      request.tool_choice !== "none"
    ) {
      if (!tool_choice && tools?.length) {
        tool_choice = { type: "auto" };
      }
      if (tool_choice && typeof tool_choice === "object") {
        tool_choice.disable_parallel_tool_use = true;
      }
    }

    // Collapse to the legacy string form only for a single plain-text block;
    // anything else stays an Anthropic system block array.
    const system: any =
      systemBlocks.length === 1 && !systemBlocks[0].cache_control
        ? systemBlocks[0].text
        : systemBlocks.length > 0
          ? systemBlocks
          : undefined;

    const body: Record<string, any> = {
      model: request.model,
      max_tokens: request.max_tokens ?? 8192,
      messages,
      stream: request.stream ?? true,
    };

    if (system !== undefined) body.system = system;
    if (request.temperature !== undefined) body.temperature = request.temperature;
    if (request.top_p !== undefined) body.top_p = request.top_p;
    if ((request as any).stop !== undefined) {
      body.stop_sequences = Array.isArray((request as any).stop)
        ? (request as any).stop
        : [(request as any).stop];
    }
    if (request.tool_choice !== "none" && tools?.length) body.tools = tools;
    if (request.tool_choice !== "none" && tool_choice) body.tool_choice = tool_choice;

    const source = context?.protocolContext?.anthropicSource;
    const reasoningDisabled = isReasoningDisabled(
      request.reasoning,
      request.thinking
    );
    // Prefer exact source Anthropic fields when this request originated as
    // Messages. Legacy direct callers can still use the temporary request fields.
    if (reasoningDisabled) {
      body.thinking = { type: "disabled" };
    } else if (source?.thinking) {
      body.thinking = source.thinking;
    } else if (request.anthropic_thinking) {
      body.thinking = request.anthropic_thinking;
    } else if (request.reasoning?.enabled || request.reasoning?.effort) {
      // Prefer adaptive + summarized display when the shared auto-summary /
      // reasoning.summary opt-in asked for readable thinking (Claude Code
      // uses display:"summarized"; "omitted" yields signature-only deltas).
      const summary = resolveOutboundReasoningSummary(
        request,
        (context as any)?.provider
      );
      body.thinking = summary
        ? { type: "adaptive", display: "summarized" }
        : { type: "adaptive" };
    }
    const outputConfigSource = source?.outputConfig
      ? source.outputConfig
      : request.anthropic_output_config
        ? request.anthropic_output_config
        : request.reasoning?.effort
          ? { effort: request.reasoning.effort }
          : undefined;
    const outputConfig =
      normalizeAnthropicOutputConfig(outputConfigSource, reasoningDisabled) ||
      {};
    // Responses text.format arrives as Unified response_format. Anthropic
    // structured output lives on output_config.format — json_schema only.
    // json_object has no Messages equivalent; omit rather than reject.
    const responseFormat = (request as any).response_format;
    if (
      responseFormat?.type === "json_schema" &&
      responseFormat.json_schema?.schema &&
      typeof responseFormat.json_schema.schema === "object"
    ) {
      outputConfig.format = {
        type: "json_schema",
        schema: responseFormat.json_schema.schema,
      };
    }
    if (Object.keys(outputConfig).length) {
      body.output_config = outputConfig;
    }
    if (source?.metadata) {
      body.metadata = source.metadata;
    } else if (request.anthropic_metadata) {
      body.metadata = request.anthropic_metadata;
    }
    if (source?.stopSequences) {
      body.stop_sequences = source.stopSequences;
    } else if (request.anthropic_stop_sequences) {
      body.stop_sequences = request.anthropic_stop_sequences;
    }

    if ((request as any).cache_control) {
      body.cache_control = (request as any).cache_control;
    }

    return body;
  }

  private convertAnthropicToolsToUnified(tools: any[]): UnifiedTool[] {
    return tools.map((tool) => ({
      type: "function",
      function: {
        name: tool.name,
        description: tool.description || "",
        parameters: tool.input_schema,
      },
      ...(tool.cache_control ? { cache_control: tool.cache_control } : {}),
    }));
  }

  private async convertOpenAIStreamToAnthropic(
    openaiStream: ReadableStream,
    context: TransformerContext
  ): Promise<ReadableStream> {
    // Shared with cancel(): client abort must stop the upstream provider stream
    // (e.g. Cursor SDK) or the next request can hang on a live active run.
    let upstreamReader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let markClosed: (() => void) | null = null;

    const readable = new ReadableStream({
      start: async (controller) => {
        const encoder = new TextEncoder();
        const messageId = `msg_${Date.now()}`;
        let stopReasonMessageDelta: null | Record<string, any> = null;
        let model = "unknown";
        let hasStarted = false;
        let hasTextContentStarted = false;
        const hasFinished = false;
        const toolCalls = new Map<number, any>();
        const toolCallIndexToContentBlockIndex = new Map<number, number>();
        let contentChunks = 0;
        let toolCallChunks = 0;
        let isClosed = false;
        markClosed = () => {
          isClosed = true;
        };
        let isThinkingStarted = false;
        let contentIndex = 0;
        let currentContentBlockIndex = -1; // Track the current content block index
        let webSearchRequestCount = 0;

        // Atomic content block index assignment function
        const assignContentBlockIndex = (): number => {
          const currentIndex = contentIndex;
          contentIndex++;
          return currentIndex;
        };

        const safeEnqueue = (data: Uint8Array) => {
          if (!isClosed) {
            try {
              controller.enqueue(data);
            } catch (error) {
              if (
                error instanceof TypeError &&
                error.message.includes("Controller is already closed")
              ) {
                isClosed = true;
              } else {
                throw error;
              }
            }
          }
        };

        const safeClose = () => {
          if (!isClosed) {
            try {
              // Close any remaining open content block
              if (currentContentBlockIndex >= 0) {
                const contentBlockStop = {
                  type: "content_block_stop",
                  index: currentContentBlockIndex,
                };
                safeEnqueue(
                  encoder.encode(
                    `event: content_block_stop\ndata: ${JSON.stringify(
                      contentBlockStop
                    )}\n\n`
                  )
                );
                currentContentBlockIndex = -1;
              }

              if (stopReasonMessageDelta) {
                if (webSearchRequestCount > 0) {
                  stopReasonMessageDelta.usage.server_tool_use = {
                    web_search_requests: webSearchRequestCount,
                  };
                }
                safeEnqueue(
                  encoder.encode(
                    `event: message_delta\ndata: ${JSON.stringify(
                      stopReasonMessageDelta
                    )}\n\n`
                  )
                );
                stopReasonMessageDelta = null;
              } else {
                const fallbackUsage: Record<string, any> = {
                  input_tokens: 0,
                  output_tokens: 0,
                  cache_creation_input_tokens: 0,
                  cache_read_input_tokens: 0,
                };
                if (webSearchRequestCount > 0) {
                  fallbackUsage.server_tool_use = {
                    web_search_requests: webSearchRequestCount,
                  };
                }
                safeEnqueue(
                  encoder.encode(
                    `event: message_delta\ndata: ${JSON.stringify({
                      type: "message_delta",
                      delta: {
                        stop_reason: "end_turn",
                        stop_sequence: null,
                      },
                      usage: fallbackUsage,
                    })}\n\n`
                  )
                );
              }
              const messageStop = {
                type: "message_stop",
              };
              safeEnqueue(
                encoder.encode(
                  `event: message_stop\ndata: ${JSON.stringify(
                    messageStop
                  )}\n\n`
                )
              );
              controller.close();
              isClosed = true;
            } catch (error) {
              if (
                error instanceof TypeError &&
                error.message.includes("Controller is already closed")
              ) {
                isClosed = true;
              } else {
                throw error;
              }
            }
          }
        };

        const safeError = (error: unknown) => {
          if (isClosed) return;

          try {
            if (currentContentBlockIndex >= 0) {
              safeEnqueue(
                encoder.encode(
                  `event: content_block_stop\ndata: ${JSON.stringify({
                    type: "content_block_stop",
                    index: currentContentBlockIndex,
                  })}\n\n`
                )
              );
              currentContentBlockIndex = -1;
            }

            const providerError =
              error && typeof error === "object"
                ? (error as Record<string, unknown>)
                : {};
            const errorType =
              typeof providerError.type === "string"
                ? providerError.type
                : "api_error";
            const message =
              error instanceof Error
                ? error.message
                : typeof providerError.message === "string"
                  ? providerError.message
                  : String(error || "Upstream stream failed");

            safeEnqueue(
              encoder.encode(
                `event: error\ndata: ${JSON.stringify({
                  type: "error",
                  error: {
                    type: errorType,
                    message,
                  },
                })}\n\n`
              )
            );
            controller.close();
            isClosed = true;
          } catch (streamError) {
            try {
              controller.error(streamError);
            } catch (controllerError) {
              this.logger?.error(controllerError);
            }
          }
        };

        let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;

        try {
          reader = openaiStream.getReader();
          upstreamReader = reader;
          const decoder = new TextDecoder();
          let buffer = "";

          while (true) {
            if (isClosed) {
              break;
            }

            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (isClosed || hasFinished) break;

              if (!line.startsWith("data:")) continue;
              const data = line.slice(5).trim();
              // Upstream SSE debug (received data / Original Response) lives in
              // tapUpstreamSSEDebug so exact-wire and conversion paths share one tap.

              if (data === "[DONE]") {
                continue;
              }

              try {
                const chunk = JSON.parse(data);
                if (chunk.error) {
                  const errorMessage = {
                    type: "error",
                    message: {
                      type: "api_error",
                      message: JSON.stringify(chunk.error),
                    },
                  };

                  safeEnqueue(
                    encoder.encode(
                      `event: error\ndata: ${JSON.stringify(errorMessage)}\n\n`
                    )
                  );
                  continue;
                }

                model = chunk.model || model;

                if (!hasStarted && !isClosed && !hasFinished) {
                  hasStarted = true;

                  const messageStart = {
                    type: "message_start",
                    message: {
                      id: messageId,
                      type: "message",
                      role: "assistant",
                      content: [],
                      model: model,
                      stop_reason: null,
                      stop_sequence: null,
                      usage: resolveMessageStartUsage(chunk.usage, context),
                    },
                  };

                  safeEnqueue(
                    encoder.encode(
                      `event: message_start\ndata: ${JSON.stringify(
                        messageStart
                      )}\n\n`
                    )
                  );
                }

                const choice = chunk.choices?.[0];
                if (chunk.usage) {
                  if (!stopReasonMessageDelta) {
                    stopReasonMessageDelta = {
                      type: "message_delta",
                      delta: {
                        stop_reason: "end_turn",
                        stop_sequence: null,
                      },
                      usage: toAnthropicCacheUsage(chunk.usage),
                    };
                  } else {
                    stopReasonMessageDelta.usage =
                      toAnthropicCacheUsage(chunk.usage);
                  }
                }
                if (!choice) {
                  continue;
                }

                const unifiedThinking = thinkingFromUnifiedAssistant(
                  choice?.delta
                );
                if (unifiedThinking && !isClosed && !hasFinished) {
                  const thinkingText =
                    typeof unifiedThinking.content === "string"
                      ? unifiedThinking.content
                      : "";
                  const thinkingSignatureValue = anthropicThinkingSignatureFrom(
                    unifiedThinking
                  );
                  // Replay metadata (encrypted_content / Responses item id)
                  // is not an Anthropic thinking block. Opening one for it —
                  // especially after text has started — produces a trailing
                  // second thinking block and a malformed assistant turn.
                  if (thinkingText || thinkingSignatureValue) {
                  // Close any previous content block if open (e.g. text emitted
                  // before a late signature). Anthropic clients require clean
                  // block boundaries; leaving text open then starting thinking
                  // causes Claude Code to drop the turn.
                  if (currentContentBlockIndex >= 0 && !isThinkingStarted) {
                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: currentContentBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                    hasTextContentStarted = false;
                  }

                  if (!isThinkingStarted) {
                    const thinkingBlockIndex = assignContentBlockIndex();
                    const contentBlockStart = {
                      type: "content_block_start",
                      index: thinkingBlockIndex,
                      content_block: { type: "thinking", thinking: "" },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_start\ndata: ${JSON.stringify(
                          contentBlockStart
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = thinkingBlockIndex;
                    isThinkingStarted = true;
                  }
                  if (thinkingSignatureValue) {
                    const thinkingSignature = {
                      type: "content_block_delta",
                      index: currentContentBlockIndex,
                      delta: {
                        type: "signature_delta",
                        signature: thinkingSignatureValue,
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_delta\ndata: ${JSON.stringify(
                          thinkingSignature
                        )}\n\n`
                      )
                    );
                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: currentContentBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                    isThinkingStarted = false;
                  } else if (thinkingText) {
                    const thinkingChunk = {
                      type: "content_block_delta",
                      index: currentContentBlockIndex,
                      delta: {
                        type: "thinking_delta",
                        thinking: thinkingText,
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_delta\ndata: ${JSON.stringify(
                          thinkingChunk
                        )}\n\n`
                      )
                    );
                  }
                  }
                }

                if (choice?.delta?.content && !isClosed && !hasFinished) {
                  contentChunks++;

                  // Close any previous content block if open and it's not a text content block
                  if (currentContentBlockIndex >= 0) {
                    // Check if current content block is text type
                    const isCurrentTextBlock = hasTextContentStarted;
                    if (!isCurrentTextBlock) {
                      const contentBlockStop = {
                        type: "content_block_stop",
                        index: currentContentBlockIndex,
                      };
                      safeEnqueue(
                        encoder.encode(
                          `event: content_block_stop\ndata: ${JSON.stringify(
                            contentBlockStop
                          )}\n\n`
                        )
                      );
                      currentContentBlockIndex = -1;
                      // The block just closed may have been a thinking block
                      // (reasoning providers interleave thinking → text →
                      // thinking). Clear the flag so a later thinking delta
                      // opens a fresh block instead of emitting thinking_delta
                      // against the text block index.
                      isThinkingStarted = false;
                    }
                  }

                  if (!hasTextContentStarted && !hasFinished) {
                    hasTextContentStarted = true;
                    const textBlockIndex = assignContentBlockIndex();
                    const contentBlockStart = {
                      type: "content_block_start",
                      index: textBlockIndex,
                      content_block: {
                        type: "text",
                        text: "",
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_start\ndata: ${JSON.stringify(
                          contentBlockStart
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = textBlockIndex;
                  }

                  if (!isClosed && !hasFinished) {
                    const anthropicChunk = {
                      type: "content_block_delta",
                      index: currentContentBlockIndex, // Use current content block index
                      delta: {
                        type: "text_delta",
                        text: choice.delta.content,
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_delta\ndata: ${JSON.stringify(
                          anthropicChunk
                        )}\n\n`
                      )
                    );
                  }
                }

                if (
                  choice?.delta?.annotations?.length &&
                  !isClosed &&
                  !hasFinished
                ) {
                  webSearchRequestCount += choice.delta.annotations.length;
                  // Close text content block if open
                  if (currentContentBlockIndex >= 0 && hasTextContentStarted) {
                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: currentContentBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                    hasTextContentStarted = false;
                  }

                  choice?.delta?.annotations.forEach((annotation: any) => {
                    const annotationBlockIndex = assignContentBlockIndex();
                    const contentBlockStart = {
                      type: "content_block_start",
                      index: annotationBlockIndex,
                      content_block: {
                        type: "web_search_tool_result",
                        tool_use_id: `srvtoolu_${uuidv4()}`,
                        content: [
                          {
                            type: "web_search_result",
                            title: annotation.url_citation.title,
                            url: annotation.url_citation.url,
                          },
                        ],
                      },
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_start\ndata: ${JSON.stringify(
                          contentBlockStart
                        )}\n\n`
                      )
                    );

                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: annotationBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                  });
                  // Annotation blocks open and close within this loop, so they
                  // never become "current". Clearing currentContentBlockIndex
                  // here would strand a block this loop did not close — a
                  // tool_use still streaming arguments (the close above is
                  // gated on hasTextContentStarted) is only reachable through
                  // this index, and every later close path requires it >= 0.
                }

                if (choice?.delta?.tool_calls && !isClosed && !hasFinished) {
                  toolCallChunks++;
                  const processedInThisChunk = new Set<number>();

                  for (const toolCall of choice.delta.tool_calls) {
                    if (isClosed) break;
                    const toolCallIndex = toolCall.index ?? 0;
                    if (processedInThisChunk.has(toolCallIndex)) {
                      continue;
                    }
                    processedInThisChunk.add(toolCallIndex);
                    const isUnknownIndex =
                      !toolCallIndexToContentBlockIndex.has(toolCallIndex);

                    if (isUnknownIndex) {
                      // Close any previous content block if open
                      if (currentContentBlockIndex >= 0) {
                        const contentBlockStop = {
                          type: "content_block_stop",
                          index: currentContentBlockIndex,
                        };
                        safeEnqueue(
                          encoder.encode(
                            `event: content_block_stop\ndata: ${JSON.stringify(
                              contentBlockStop
                            )}\n\n`
                          )
                        );
                        currentContentBlockIndex = -1;
                      }
                      // Text/thinking bookkeeping is sticky across block opens.
                      // Leaving hasTextContentStarted true makes a later
                      // delta.content emit text_delta against this tool_use
                      // index — Claude Code then drops the turn with
                      // "Content block is not a text block".
                      hasTextContentStarted = false;
                      isThinkingStarted = false;

                      const newContentBlockIndex = assignContentBlockIndex();
                      toolCallIndexToContentBlockIndex.set(
                        toolCallIndex,
                        newContentBlockIndex
                      );
                      // Last line of defence: an id that leaves here unsanitized
                      // is echoed back by the client on every later request and
                      // poisons the conversation permanently.
                      const toolCallId =
                        sanitizeToolCallId(toolCall.id) ||
                        `call_${Date.now()}_${toolCallIndex}`;
                      const toolCallName =
                        toolCall.function?.name || `tool_${toolCallIndex}`;
                      const contentBlockStart = {
                        type: "content_block_start",
                        index: newContentBlockIndex,
                        content_block: {
                          type: "tool_use",
                          id: toolCallId,
                          name: toolCallName,
                          input: {},
                        },
                      };

                      safeEnqueue(
                        encoder.encode(
                          `event: content_block_start\ndata: ${JSON.stringify(
                            contentBlockStart
                          )}\n\n`
                        )
                      );
                      currentContentBlockIndex = newContentBlockIndex;

                      const toolCallInfo = {
                        id: toolCallId,
                        name: toolCallName,
                        arguments: "",
                        contentBlockIndex: newContentBlockIndex,
                      };
                      toolCalls.set(toolCallIndex, toolCallInfo);
                    } else if (toolCall.id && toolCall.function?.name) {
                      const existingToolCall = toolCalls.get(toolCallIndex)!;
                      const wasTemporary =
                        existingToolCall.id.startsWith("call_") &&
                        existingToolCall.name.startsWith("tool_");

                      if (wasTemporary) {
                        existingToolCall.id = toolCall.id;
                        existingToolCall.name = toolCall.function.name;
                      }
                    }

                    if (
                      toolCall.function?.arguments &&
                      !isClosed &&
                      !hasFinished
                    ) {
                      const blockIndex =
                        toolCallIndexToContentBlockIndex.get(toolCallIndex);
                      if (blockIndex === undefined) {
                        continue;
                      }
                      const currentToolCall = toolCalls.get(toolCallIndex);
                      if (currentToolCall) {
                        currentToolCall.arguments +=
                          toolCall.function.arguments;
                      }

                      try {
                        const anthropicChunk = {
                          type: "content_block_delta",
                          index: blockIndex,
                          delta: {
                            type: "input_json_delta",
                            partial_json: toolCall.function.arguments,
                          },
                        };
                        safeEnqueue(
                          encoder.encode(
                            `event: content_block_delta\ndata: ${JSON.stringify(
                              anthropicChunk
                            )}\n\n`
                          )
                        );
                      } catch {
                        try {
                          const fixedArgument = toolCall.function.arguments
                            .replace(/\p{Cc}/gu, "")
                            .replace(/\\/g, "\\\\")
                            .replace(/"/g, '\\"');

                          const fixedChunk = {
                            type: "content_block_delta",
                            index: blockIndex, // Use the correct content block index
                            delta: {
                              type: "input_json_delta",
                              partial_json: fixedArgument,
                            },
                          };
                          safeEnqueue(
                            encoder.encode(
                              `event: content_block_delta\ndata: ${JSON.stringify(
                                fixedChunk
                              )}\n\n`
                            )
                          );
                        } catch (fixError) {
                          this.logger?.error(fixError);
                        }
                      }
                    }
                  }
                }

                if (choice?.finish_reason && !isClosed && !hasFinished) {
                  if (contentChunks === 0 && toolCallChunks === 0) {
                    this.logger?.warn(
                      "No content in the stream response!"
                    );
                  }

                  // Close any remaining open content block
                  if (currentContentBlockIndex >= 0) {
                    const contentBlockStop = {
                      type: "content_block_stop",
                      index: currentContentBlockIndex,
                    };
                    safeEnqueue(
                      encoder.encode(
                        `event: content_block_stop\ndata: ${JSON.stringify(
                          contentBlockStop
                        )}\n\n`
                      )
                    );
                    currentContentBlockIndex = -1;
                  }

                  if (!isClosed) {
                    const stopReasonMapping: Record<string, string> = {
                      stop: "end_turn",
                      length: "max_tokens",
                      tool_calls: "tool_use",
                      content_filter: "stop_sequence",
                      model_context_window_exceeded:
                        "model_context_window_exceeded",
                    };

                    let anthropicStopReason =
                      stopReasonMapping[choice.finish_reason] || "end_turn";
                    // Safety net: if any tool_use blocks were streamed, never
                    // report end_turn — Claude Code won't continue the tool loop.
                    // Keyed on blocks actually emitted, not on delta.tool_calls
                    // chunks: an empty `tool_calls: []` delta is truthy and would
                    // otherwise claim tool_use with no tool_use block to satisfy.
                    if (
                      toolCallIndexToContentBlockIndex.size > 0 &&
                      anthropicStopReason === "end_turn"
                    ) {
                      anthropicStopReason = "tool_use";
                    }

                    stopReasonMessageDelta = {
                      type: "message_delta",
                      delta: {
                        stop_reason: anthropicStopReason,
                        stop_sequence: null,
                      },
                      usage: chunk.usage
                        ? toAnthropicCacheUsage(chunk.usage)
                        : stopReasonMessageDelta?.usage ?? {
                            input_tokens: 0,
                            output_tokens: 0,
                            cache_creation_input_tokens: 0,
                            cache_read_input_tokens: 0,
                          },
                    };
                  }

                  break;
                }
              } catch (parseError: any) {
                this.logger?.error(
                  `parseError: ${parseError.name} message: ${parseError.message} stack: ${parseError.stack} data: ${data}`
                );
              }
            }
          }
          safeClose();
        } catch (error) {
          safeError(error);
        } finally {
          if (upstreamReader === reader) {
            upstreamReader = null;
          }
          if (reader) {
            try {
              reader.releaseLock();
            } catch (releaseError) {
              this.logger?.error(releaseError);
            }
          }
        }
      },
      cancel: async (reason) => {
        // Stop pumping / enqueueing, then cancel the upstream reader so owns-fetch
        // providers (Cursor SDK) receive ReadableStream.cancel → run.cancel().
        markClosed?.();
        const pending = upstreamReader;
        upstreamReader = null;
        if (pending) {
          // Propagate the teardown promise too. Otherwise Readable.fromWeb()
          // reports cancellation complete while the Cursor SDK run is still
          // active, letting a reconnect race the old agent cleanup.
          await pending.cancel(reason).catch(() => undefined);
        }
        this.logger.debug(
          {
            reqId: context.req.id,
          },
          `cancel stream: ${reason}`
        );
      },
    });

    return readable;
  }

  private convertOpenAIResponseToAnthropic(
    openaiResponse: ChatCompletion,
    context: TransformerContext
  ): any {
    this.logger.debug(
      {
        reqId: context.req.id,
        response: openaiResponse,
      },
      `Original OpenAI response`
    );
    try {
      const choice = openaiResponse.choices[0];
      if (!choice) {
        throw new Error("No choices found in OpenAI response");
      }
      const content: any[] = [];
      // Fixed order: thinking → text → tools (server search, then client tools).
      const unifiedThinking = thinkingFromUnifiedAssistant(choice.message);
      if (unifiedThinking?.content) {
        const signature = anthropicThinkingSignatureFrom(unifiedThinking);
        content.push({
          type: "thinking",
          thinking: unifiedThinking.content,
          ...(signature ? { signature } : {}),
        });
      }
      if (choice.message.content) {
        content.push({
          type: "text",
          text: choice.message.content,
        });
      }
      if (choice.message.annotations) {
        const id = `srvtoolu_${uuidv4()}`;
        content.push({
          type: "server_tool_use",
          id,
          name: "web_search",
          input: {
            query: "",
          },
        });
        content.push({
          type: "web_search_tool_result",
          tool_use_id: id,
          content: choice.message.annotations.map((item) => {
            return {
              type: "web_search_result",
              url: item.url_citation.url,
              title: item.url_citation.title,
            };
          }),
        });
      }
      if (choice.message.tool_calls && choice.message.tool_calls.length > 0) {
        choice.message.tool_calls.forEach((toolCall) => {
          if (!("function" in toolCall) || !toolCall.function) return;
          let parsedInput = {};
          try {
            const argumentsStr = toolCall.function.arguments || "{}";

            if (typeof argumentsStr === "object") {
              parsedInput = argumentsStr;
            } else if (typeof argumentsStr === "string") {
              parsedInput = JSON.parse(argumentsStr);
            }
          } catch {
            parsedInput = { text: toolCall.function.arguments || "" };
          }

          content.push({
            type: "tool_use",
            id: sanitizeToolCallId(toolCall.id) ?? toolCall.id,
            name: toolCall.function.name,
            input: parsedInput,
          });
        });
      }
      const finishReason = String(choice.finish_reason || "");
      const result = {
        id: openaiResponse.id,
        type: "message",
        role: "assistant",
        model: openaiResponse.model,
        content: content,
        stop_reason:
          choice.finish_reason === "stop"
            ? "end_turn"
            : finishReason === "length"
            ? "max_tokens"
            : finishReason === "tool_calls"
            ? "tool_use"
            : finishReason === "content_filter"
            ? "stop_sequence"
            : finishReason === "model_context_window_exceeded"
            ? "model_context_window_exceeded"
            : "end_turn",
        stop_sequence: null,
        usage: {
          ...toAnthropicCacheUsage(openaiResponse.usage),
          ...(choice.message.annotations?.length
            ? {
                server_tool_use: {
                  web_search_requests: choice.message.annotations.length,
                },
              }
            : {}),
        },
      };
      this.logger.debug(
        {
          reqId: context.req.id,
          result,
        },
        `Conversion complete, final Anthropic response`
      );
      return result;
    } catch {
      throw createApiError(
        `Provider error: ${JSON.stringify(openaiResponse)}`,
        500,
        "provider_error"
      );
    }
  }
}

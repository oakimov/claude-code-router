import {
  FastifyInstance,
  FastifyRequest,
  FastifyReply,
} from "fastify";
import { Readable } from "stream";
import { RegisterProviderRequest, LLMProvider } from "@/types/llm";
import { sendUnifiedRequest } from "@/utils/request";
import { createApiError } from "./middleware";
import { readHealthVitals } from "@/utils/health-reporter";
import { version } from "../../package.json";
import { ConfigService } from "@/services/config";
import { ProviderService } from "@/services/provider";
import { TransformerService } from "@/services/transformer";
import { Transformer } from "@/types/transformer";
import {
  // diffHeadersForLog,
  sanitizeErrorForLog,
  // sanitizeHeadersForLog,
  sanitizeUpstreamErrorBody,
  sanitizeUpstreamErrorText,
} from "@/utils/redact";
import {
  CLIENT_DISCONNECT_REASON,
  createClientDisconnectSignal,
  delay,
  isClientAbortError,
  isFallbackEligibleError,
  isResponseSocketGone,
  retryDelayAfterFailure,
  selectFallbackModels,
  toClientAbortError,
} from "@/utils/retry";
import { applyProviderNativeChatCaching } from "../utils/openai.util";
import { sanitizeResponsesWireCallIds } from "../utils/openai.responses.util";
import { applyOpenAIChatReasoning } from "../utils/reasoning-effort";
import { applyRawAnthropicPromptCaching } from "../utils/cacheControl";
import { resolveCachePrefixConversationId } from "../utils/cache-prefix-debug";
import {
  firstSubstantiveUserText,
  isStatuslinePollTurn,
} from "../utils/nested-agent";

function isOpencodeProvider(provider: any): boolean {
  const name = String(provider?.name || "").toLowerCase();
  if (name === "opencode" || name.startsWith("opencode")) return true;
  const base = provider?.baseUrl || provider?.api_base_url || "";
  try {
    const host = new URL(base).hostname.toLowerCase();
    return host === "opencode.ai" || host.endsWith(".opencode.ai");
  } catch {
    return String(base).toLowerCase().includes("opencode.ai");
  }
}
import {
  logOutboundCacheStructure,
  tapClientSSEDebug,
  tapUpstreamSSEDebug,
} from "../utils/sse-debug-tap";
import {
  logKeepWire,
  logMessageBody,
  resolveLogBodyMaxBytes,
  resolveLogBodySelection,
  shouldLogSSEEvents,
} from "../utils/message-debug";
import { withSSEClientKeepalive } from "@/utils/sse/client-keepalive";
import { withChatCompletionsDoneBoundary } from "@/utils/sse/done-boundary";
import { sendWithUnauthorizedAuthRecovery } from "@/utils/auth-recovery";
import {
  canonicalizeOutboundHeaders,
  mergeHeadersCaseInsensitive,
  selectSafeDownstreamHeaders,
} from "@/utils/headers";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";
import { TokenizerService } from "@/services/tokenizer";
import {
  listClientRouteRegistrations,
  matchClientProtocol,
  type ClientProtocol,
} from "@/routing/protocol-endpoints";
import {
  cloneProtocolBody,
  sanitizePassthroughHeaders,
} from "@/routing/protocol-adapter";
import {
  prepareInboundRequest,
  protocolAwareBypass,
  PreparedInboundRequest,
} from "@/routing/inbound-pipeline";
import { handleFimEndpoint } from "@/routing/fim-pipeline";
import {
  compileTransformerPlan,
  cancelReplacedProviderResponse,
  isExactProtocolResponsePlan,
  isExactProtocolRequestPlan,
  isWireSafeMiddlewareForKeep,
  planContains,
} from "@/utils/transformer-plan";
import {
  ensureRequestLatency,
  markLatency,
  attachLatencyMeta,
  emitLatencyRecord,
  tapResponseFirstByte,
} from "@/utils/request-latency";
import {
  applyThirdPartyAnthropicPolicy,
  getAnthropicProviderMode,
  isNativeAnthropicClient,
} from "@/utils/anthropic-client-policy";

function isManualExactProtocolPassthrough(
  provider: any,
  endpointTransformer: Transformer,
  protocolContext: any
): boolean {
  return Boolean(
    protocolContext &&
      provider.transformer?.passthrough === true &&
      provider.transformer?.use?.some(
        (providerTransformer: Transformer) =>
          providerTransformer?.name === endpointTransformer.name
      )
  );
}

function isThirdPartyInScopeAnthropic(protocolContext: any): boolean {
  return Boolean(
    protocolContext?.anthropicDestinationInScope === true &&
      protocolContext?.anthropicClientKind === "other"
  );
}

/**
 * v1 wire-keep predicate: exact-protocol owner in compiled plan, third-party
 * Anthropic never keeps, claude-auth chains excluded for v1.
 * `anthropicNativeWire` and legacy passthrough are still covered by the
 * caller's own booleans; this helper covers the automatic multi-transformer
 * case only.
 */
function computeWireKeepForProvider(
  provider: any,
  endpointTransformer: Transformer,
  protocolContext: any,
  modelName: string | undefined
): { useWire: boolean; plan: any | undefined; isAutoKeep: boolean } {
  if (!protocolContext || isThirdPartyInScopeAnthropic(protocolContext)) {
    return { useWire: false, plan: undefined, isAutoKeep: false };
  }
  const providerUse = provider?.transformer?.use as any[] | undefined;
  const modelUse = (
    modelName ? provider?.transformer?.[modelName]?.use : undefined
  ) as any[] | undefined;
  let plan: any | undefined;
  try {
    // Compile without skipName so the owner stays in plan for the predicate.
    plan = compileTransformerPlan(providerUse, modelUse);
  } catch {
    return { useWire: false, plan: undefined, isAutoKeep: false };
  }
  if (planContains(plan, "claude-auth")) {
    return { useWire: false, plan, isAutoKeep: false };
  }
  const isAutoKeep = isExactProtocolRequestPlan(
    plan,
    endpointTransformer,
    protocolContext.ownerTransformerName
  );
  return { useWire: isAutoKeep, plan, isAutoKeep };
}

/**
 * Single keep/bypass decision for the provider POST.
 * Unified projection still exists for routing; this only chooses the
 * upstream body and whether the protocol owner may rebuild it.
 */
function resolveClientWireKeep(
  provider: any,
  endpointTransformer: Transformer,
  protocolContext: any,
  modelName: string | undefined
): {
  keepClientWire: boolean;
  bypass: boolean;
  isNativeWire: boolean;
  plan: any | undefined;
  isAutoKeep: boolean;
} {
  const isNativeWire = Boolean(protocolContext?.anthropicNativeWire);
  const thirdParty = isThirdPartyInScopeAnthropic(protocolContext);
  const preserveManualWire =
    !thirdParty &&
    isManualExactProtocolPassthrough(
      provider,
      endpointTransformer,
      protocolContext
    );
  const deprecatedWillBypass =
    protocolAwareBypass(
      provider,
      endpointTransformer,
      protocolContext,
      modelName
    ) && !thirdParty;
  const autoKeep = computeWireKeepForProvider(
    provider,
    endpointTransformer,
    protocolContext,
    modelName
  );
  const keepClientWire =
    isNativeWire ||
    preserveManualWire ||
    autoKeep.useWire ||
    deprecatedWillBypass;
  return {
    keepClientWire,
    bypass: isNativeWire || deprecatedWillBypass,
    isNativeWire,
    plan: autoKeep.plan,
    isAutoKeep: autoKeep.isAutoKeep,
  };
}

type RequestTransformKeep = {
  keepClientWire: boolean;
  bypass: boolean;
  wireKeepPlan?: any;
};

// Extend FastifyInstance to include custom services
declare module "fastify" {
  interface FastifyInstance {
    configService: ConfigService;
    providerService: ProviderService;
    transformerService: TransformerService;
    tokenizerService: TokenizerService;
  }

  interface FastifyRequest {
    provider?: string;
  }
}

/**
 * Main handler for transformer endpoints.
 * Canonical lifecycle: normalize client→Unified → route → provider In →
 * upstream → provider Out → client In → format from protocol stream intent.
 */
async function handleTransformerEndpoint(
  req: FastifyRequest,
  reply: FastifyReply,
  fastify: FastifyInstance,
  transformer: any,
  routePath: string
) {
  const disconnect = createClientDisconnectSignal(req, reply);
  const clientSignal = disconnect.signal;
  disconnect.arm();
  const latency = ensureRequestLatency(req as any);
  markLatency(latency, "bodyParsed");

  let prepared: PreparedInboundRequest;
  try {
    prepared = await prepareInboundRequest(
      req,
      reply,
      fastify,
      transformer,
      routePath
    );
    markLatency(latency, "destinationPolicy");
  } catch (error: any) {
    attachLatencyMeta(latency, {
      error: error?.message || String(error),
      cancelled: isClientAbortError(error) || clientSignal.aborted,
    });
    emitLatencyRecord(req.log, latency);
    if (isClientAbortError(error) || clientSignal.aborted) {
      throw typeof error === "string"
        ? toClientAbortError(error)
        : toClientAbortError(clientSignal.reason ?? error);
    }
    throw error;
  }

  const provider = fastify.providerService.getProvider(prepared.providerName);
  if (!provider) {
    attachLatencyMeta(latency, {
      provider: prepared.providerName,
      model: prepared.modelName,
      error: "provider_not_found",
    });
    emitLatencyRecord(req.log, latency);
    throw createApiError(
      `Provider '${prepared.providerName}' not found`,
      404,
      "provider_not_found"
    );
  }

  attachLatencyMeta(latency, {
    protocol: prepared.protocolContext?.protocol,
    provider: prepared.providerName,
    model: prepared.modelName,
    method: req.method,
    url: req.url,
    scenario: (req as any).scenarioType,
    tokenCount: (req as any).tokenCount,
  });

  // Stash prepared state so fallback can reuse Unified + protocol context.
  (req as any)._preparedInbound = prepared;

  try {
    // Native Anthropic clients use the original wire body; legacy exact
    // protocol passthrough uses the adapted wire body. Same-protocol wire
    // keep (multi-transformer with owner in plan) also keeps clientWireBody
    // so prefix + vision bytes stay intact. Third-party Anthropic emulation
    // and all cross-protocol routes use the Unified projection.
    const keep = resolveClientWireKeep(
      provider,
      transformer,
      prepared.protocolContext,
      prepared.modelName
    );
    const isNativeWire = keep.isNativeWire;
    const useWireKeep = keep.keepClientWire;
    const exactWireSource = isNativeWire
      ? prepared.originalBody
      : prepared.clientWireBody ?? prepared.originalBody;
    let pipelineBody = useWireKeep
      ? {
          ...(typeof exactWireSource === "object" && exactWireSource
            ? cloneProtocolBody(exactWireSource)
            : {}),
          model: prepared.modelName,
        }
      : cloneProtocolBody(prepared.unifiedBody);
    (req as any)._wireKeep = useWireKeep;
    (req as any)._autoWireKeep = keep.isAutoKeep;

    recordClientCachePrefix(req, fastify, provider, pipelineBody);
    // Exact-wire keep skips the Responses owner's body rebuild, but call_id is
    // still provider-validated (<=64). Repair only those correlation fields
    // after the client snapshot so cache diagnostics attribute it to the wire.
    if (
      useWireKeep &&
      prepared.protocolContext.protocol === "openai_responses"
    ) {
      pipelineBody = sanitizeResponsesWireCallIds(
        pipelineBody,
        prepared.protocolContext.responsesCallIdMap
      );
    }

    const { requestBody, config, bypass, wireKeep } = await processRequestTransformers(
      pipelineBody,
      provider,
      transformer,
      req.headers,
      {
        req,
        provider,
        signal: clientSignal,
        clientProtocol: prepared.protocolContext.protocol,
        protocolContext: prepared.protocolContext,
        // Client→Unified already ran in prepareInboundRequest (routing only).
        skipClientNormalization: true,
      },
      {
        keepClientWire: useWireKeep,
        bypass: keep.bypass,
        wireKeepPlan: keep.plan,
      }
    );
    markLatency(latency, "requestTransformers");
    attachLatencyMeta(latency, { bypass: bypass || wireKeep, wireKeep });
    if (wireKeep) {
      req.log.debug?.(
        {
          provider: provider?.name,
          protocol: prepared.protocolContext?.protocol,
          model: prepared.modelName,
          pipeline: "wire-keep",
        },
        "same-protocol wire keep (owner skipped, middleware kept)"
      );
    }

    // Fetch start/headers are stamped by sendUnifiedRequest or by ownsTransport
    // transformers (cursor-sdk). Do not mark start here — that would freeze the
    // stamp before the real POST and make upstreamTtftMs include auth/SDK setup.
    const response = await sendRequestToProvider(
      requestBody,
      config,
      provider,
      fastify,
      bypass,
      transformer,
      {
        req,
        signal: clientSignal,
        protocolContext: prepared.protocolContext,
      }
    );

    if (latency.stages.upstreamFetchStart === undefined) {
      markLatency(latency, "upstreamFetchStart");
    }
    if (latency.stages.upstreamHeaders === undefined) {
      markLatency(latency, "upstreamHeaders");
    }
    // Tap the raw provider body before response conversion so upstream TTFT
    // is not delayed by Responses→Unified→Anthropic buffering.
    const timedResponse = tapResponseFirstByte(response, () => {
      markLatency(latency, "upstreamFirstByte");
    });

    const finalResponse = await processResponseTransformers(
      requestBody,
      timedResponse,
      provider,
      transformer,
      bypass,
      {
        req,
        signal: clientSignal,
        protocolContext: prepared.protocolContext,
        skipClientNormalization: bypass,
      }
    );
    markLatency(latency, "responseTransformers");

    return await formatResponse(
      finalResponse,
      reply,
      prepared.originalBody,
      clientSignal,
      prepared.protocolContext.stream,
      prepared.protocolContext.protocol,
      {
        configService: fastify.configService,
        provider: provider?.name,
        model: prepared.modelName,
      }
    );
  } catch (error: any) {
    attachLatencyMeta(latency, {
      error: error?.code || error?.message || String(error),
      cancelled: isClientAbortError(error) || clientSignal.aborted,
    });
    emitLatencyRecord(req.log, latency);
    if (isClientAbortError(error)) {
      throw typeof error === "string" ? toClientAbortError(error) : error;
    }
    if (clientSignal.aborted) {
      throw toClientAbortError(clientSignal.reason ?? error);
    }
    if (isFallbackEligibleError(error)) {
      const fallbackResult = await handleFallback(
        req,
        reply,
        fastify,
        transformer,
        error,
        clientSignal
      );
      if (fallbackResult) {
        return fallbackResult;
      }
      if (clientSignal.aborted) {
        throw toClientAbortError(
          clientSignal.reason ?? CLIENT_DISCONNECT_REASON
        );
      }
    }
    throw error;
  }
}

/**
 * Handle fallback logic when request fails
 * Tries each fallback model in sequence until one succeeds
 */
async function handleFallback(
  req: FastifyRequest,
  reply: FastifyReply,
  fastify: FastifyInstance,
  transformer: any,
  error: any,
  clientSignal?: AbortSignal
): Promise<any> {
  const scenarioType = (req as any).scenarioType || 'default';
  const fallbackConfig = fastify.configService.get<any>('fallback');

  const fallbackList = selectFallbackModels(fallbackConfig, scenarioType);
  if (!fallbackList?.length) {
    return null;
  }

  if (clientSignal?.aborted || isClientAbortError(error)) {
    return null;
  }

  req.log.warn(
    {
      scenarioType,
      fallbackCount: fallbackList.length,
      error: sanitizeErrorForLog(error),
    },
    `Request failed for ${scenarioType}, trying ${fallbackList.length} fallback models`
  );

  let failedAttemptIndex = 0;
  const initialDelayMs = retryDelayAfterFailure(
    failedAttemptIndex,
    error?.headers?.["Retry-After"] ?? error?.headers?.["retry-after"]
  );
  failedAttemptIndex += 1;
  if (initialDelayMs > 0) {
    req.log.info(`Waiting ${initialDelayMs}ms before first fallback attempt`);
    try {
      await delay(initialDelayMs, clientSignal);
    } catch (waitError: any) {
      if (isClientAbortError(waitError) || clientSignal?.aborted) {
        return null;
      }
      throw waitError;
    }
  }

  // Try each fallback model in sequence
  for (let i = 0; i < fallbackList.length; i += 1) {
    if (clientSignal?.aborted) {
      return null;
    }

    const fallbackModel = fallbackList[i];
    try {
      req.log.info(`Trying fallback model: ${fallbackModel}`);

      // Reuse the already-normalized Unified body + protocol context.
      const prepared = (req as any)._preparedInbound as
        | PreparedInboundRequest
        | undefined;
      const [fallbackProvider, ...fallbackModelName] = fallbackModel.split(",");
      const fallbackModelOnly = fallbackModelName.join(",");

      const provider = fastify.providerService.getProvider(fallbackProvider);
      if (!provider) {
        req.log.warn(
          `Fallback provider '${fallbackProvider}' not found, skipping`
        );
        continue;
      }

      const fallbackAnthropicMode = getAnthropicProviderMode(provider);
      const fallbackProtocolContext = prepared?.protocolContext
        ? {
            ...prepared.protocolContext,
            anthropicProviderMode: fallbackAnthropicMode,
            anthropicDestinationInScope:
              fallbackAnthropicMode !== "out_of_scope",
            anthropicNativeWire: Boolean(
              prepared.protocolContext.protocol === "anthropic_messages" &&
                fallbackAnthropicMode !== "out_of_scope" &&
                isNativeAnthropicClient(
                  prepared.protocolContext.anthropicClientKind || "other"
                )
            ),
            anthropicPolicyApplied: false,
            anthropicSystemTransformed: false,
            claudeAuthToolNameMap: new Map(),
          }
        : prepared?.protocolContext;

      const unifiedBody = {
        ...cloneProtocolBody(
          prepared?.prePolicyUnifiedBody ||
            prepared?.unifiedBody ||
            (req as any).unifiedBody ||
            req.body
        ),
        model: fallbackModelOnly,
      };

      if (
        fallbackProtocolContext?.anthropicDestinationInScope &&
        fallbackProtocolContext.anthropicClientKind === "other"
      ) {
        await applyThirdPartyAnthropicPolicy(
          unifiedBody,
          fallbackProtocolContext,
          fastify.configService
        );
      }

      const newReq = {
        ...req,
        provider: fallbackProvider,
        model: fallbackModelOnly,
        body: prepared?.originalBody ?? req.body,
        unifiedBody,
        protocolContext: fallbackProtocolContext,
        clientProtocol: fallbackProtocolContext?.protocol,
        scenarioType,
      };

      const fallbackKeep = resolveClientWireKeep(
        provider,
        transformer,
        fallbackProtocolContext,
        fallbackModelOnly
      );
      const fallbackIsNativeWire = fallbackKeep.isNativeWire;
      const fallbackUseWireKeep = fallbackKeep.keepClientWire;
      const exactWireSource = fallbackIsNativeWire
        ? prepared?.originalBody
        : prepared?.clientWireBody ?? prepared?.originalBody;
      let pipelineBody = fallbackUseWireKeep
        ? {
            ...(typeof exactWireSource === "object" && exactWireSource
              ? cloneProtocolBody(exactWireSource)
              : {}),
            model: fallbackModelOnly,
          }
        : unifiedBody;

      recordClientCachePrefix(newReq, fastify, provider, pipelineBody);
      if (
        fallbackUseWireKeep &&
        fallbackProtocolContext?.protocol === "openai_responses"
      ) {
        pipelineBody = sanitizeResponsesWireCallIds(
          pipelineBody,
          fallbackProtocolContext.responsesCallIdMap
        );
      }

      const { requestBody, config, bypass, wireKeep } = await processRequestTransformers(
        pipelineBody,
        provider,
        transformer,
        req.headers,
        {
          req: newReq,
          provider,
          signal: clientSignal,
          clientProtocol: prepared?.protocolContext?.protocol,
          protocolContext: fallbackProtocolContext,
          skipClientNormalization: true,
        },
        {
          keepClientWire: fallbackUseWireKeep,
          bypass: fallbackKeep.bypass,
          wireKeepPlan: fallbackKeep.plan,
        }
      );
      if (wireKeep) {
        req.log.debug?.(
          {
            provider: provider?.name,
            protocol: fallbackProtocolContext?.protocol,
            model: fallbackModelOnly,
            pipeline: "wire-keep-fallback",
          },
          "same-protocol wire keep on fallback"
        );
      }

      const fallbackLatency = (newReq as any)._latency;
      const response = await sendRequestToProvider(
        requestBody,
        config,
        provider,
        fastify,
        bypass,
        transformer,
        {
          req: newReq,
          signal: clientSignal,
          protocolContext: fallbackProtocolContext,
        }
      );

      if (fallbackLatency?.stages?.upstreamFetchStart === undefined) {
        markLatency(fallbackLatency, "upstreamFetchStart");
      }
      if (fallbackLatency?.stages?.upstreamHeaders === undefined) {
        markLatency(fallbackLatency, "upstreamHeaders");
      }
      const timedResponse = tapResponseFirstByte(response, () => {
        markLatency(fallbackLatency, "upstreamFirstByte");
      });
      const finalResponse = await processResponseTransformers(
        requestBody,
        timedResponse,
        provider,
        transformer,
        bypass,
        {
          req: newReq,
          signal: clientSignal,
          protocolContext: fallbackProtocolContext,
        }
      );

      req.log.info(`Fallback model ${fallbackModel} succeeded`);

      return await formatResponse(
        finalResponse,
        reply,
        prepared?.originalBody ?? req.body,
        clientSignal,
        fallbackProtocolContext?.stream,
        fallbackProtocolContext?.protocol,
        {
          configService: fastify.configService,
          provider: provider?.name,
          model: typeof requestBody?.model === "string" ? requestBody.model : undefined,
        }
      );
    } catch (fallbackError: any) {
      if (isClientAbortError(fallbackError)) {
        throw fallbackError;
      }
      if (clientSignal?.aborted) {
        throw toClientAbortError(
          clientSignal.reason ?? CLIENT_DISCONNECT_REASON
        );
      }

      // A terminal (non-retryable) failure — validation, auth, permissions,
      // model-not-found — goes straight back to the caller: no further
      // fallback models, no Retry-After wait.
      if (!isFallbackEligibleError(fallbackError)) {
        req.log.warn(
          {
            fallbackModel,
            error: sanitizeErrorForLog(fallbackError),
          },
          `Fallback model ${fallbackModel} failed with a terminal error`
        );
        throw fallbackError;
      }

      req.log.warn(
        {
          fallbackModel,
          error: sanitizeErrorForLog(fallbackError),
        },
        `Fallback model ${fallbackModel} failed`
      );

      const hasMore = i < fallbackList.length - 1;
      if (hasMore) {
        const waitMs = retryDelayAfterFailure(
          failedAttemptIndex,
          fallbackError?.headers?.["Retry-After"] ??
            fallbackError?.headers?.["retry-after"]
        );
        failedAttemptIndex += 1;
        if (waitMs > 0) {
          req.log.info(
            `Waiting ${waitMs}ms before next fallback attempt`
          );
          try {
            await delay(waitMs, clientSignal);
          } catch (waitError: any) {
            if (isClientAbortError(waitError) || clientSignal?.aborted) {
              return null;
            }
            throw waitError;
          }
        }
      }
      continue;
    }
  }

  req.log.error(`All fallback models failed for ${scenarioType}`);
  return null;
}

/**
 * Process request transformer chain.
 * Client→Unified already ran in prepareInboundRequest (routing projection).
 * `keep` chooses client wire vs Unified for the upstream POST; the protocol
 * owner does not rebuild a kept Responses/Anthropic body.
 */
async function processRequestTransformers(
  body: any,
  provider: any,
  transformer: any,
  headers: any,
  context: any,
  keep?: RequestTransformKeep
) {
  let requestBody = body;
  let config: any = {};

  const thirdPartyAnthropic = isThirdPartyInScopeAnthropic(
    context?.protocolContext
  );
  const bypass = Boolean(keep?.bypass);
  const wireKeepPlan = keep?.wireKeepPlan;
  // claude-auth stays on the convert/skipName loop even if the caller kept
  // client bytes via passthrough — it is not wire-safe middleware in v1.
  const effectiveWireKeep =
    Boolean(keep?.keepClientWire) &&
    !bypass &&
    !(wireKeepPlan && planContains(wireKeepPlan, "claude-auth"));
  const manualExactPassthrough =
    !thirdPartyAnthropic &&
    isManualExactProtocolPassthrough(
      provider,
      transformer,
      context?.protocolContext
    );

  const skipBodyConversion =
    bypass ||
    effectiveWireKeep ||
    Boolean(keep?.keepClientWire) ||
    (!thirdPartyAnthropic && provider.transformer?.passthrough) ||
    context?.skipClientNormalization === true;

  // Client header copy and Anthropic cache_control injection are Anthropic-wire
  // only. Chat/Responses keep must not reuse this branch: a denylist copy
  // forwards openai-*/stainless headers, and applyRawAnthropicPromptCaching
  // stamps top-level cache_control that Chat Completions / Zen reject or ignore.
  const anthropicWireKeep =
    bypass ||
    (effectiveWireKeep &&
      (context?.protocolContext?.protocol === "anthropic_messages" ||
        (transformer as any)?.name === "Anthropic"));
  if (anthropicWireKeep) {
    config.headers = sanitizePassthroughHeaders(headers);
    if (isOpencodeProvider(provider)) {
      try {
        if (
          requestBody &&
          typeof requestBody === "object" &&
          (requestBody.system !== undefined ||
            Array.isArray(requestBody.messages))
        ) {
          requestBody = applyRawAnthropicPromptCaching(requestBody);
        }
      } catch {
        // never break the request path on caching
      }
    }
  }

  // prompt_cache_key is Responses-owner scoped — only keep when the compiled
  // plan actually contains the Responses owner (either legacy or wire-keep).
  const hasResponsesOwner = (() => {
    if (wireKeepPlan) return planContains(wireKeepPlan, "openai-responses");
    const use = provider.transformer?.use as any[] | undefined;
    return Boolean(
      Array.isArray(use) &&
        use.some((t: any) => t?.name === "openai-responses")
    );
  })();
  if (
    !bypass &&
    !effectiveWireKeep &&
    context?.protocolContext?.protocol === "openai_responses" &&
    !hasResponsesOwner
  ) {
    delete requestBody.prompt_cache_key;
  }

  if (
    !skipBodyConversion &&
    typeof transformer.transformRequestOut === "function"
  ) {
    const transformOut = await transformer.transformRequestOut(
      requestBody,
      context
    );
    if (transformOut.body) {
      requestBody = transformOut.body;
      config = transformOut.config || {};
    } else {
      requestBody = transformOut;
    }
  }

  // Build the plan for the middleware loop.
  // - bypass: skip entire chain (native wire || deprecated)
  // - wire keep: use the already-compiled plan, run allowlisted middleware + OpenAI owner
  // - otherwise: legacy compile with skipName for passthrough shim
  let plan: any | undefined;
  if (bypass) {
    plan = undefined;
  } else if (effectiveWireKeep) {
    plan = wireKeepPlan;
  } else {
    const modelUse = provider.transformer?.[body.model]?.use as
      | Transformer[]
      | undefined;
    const providerUse = provider.transformer?.use as Transformer[] | undefined;
    if (providerUse?.length || modelUse?.length) {
      try {
        plan = compileTransformerPlan(providerUse, modelUse, {
          skipName: manualExactPassthrough ? transformer.name : undefined,
        });
      } catch (error: any) {
        throw createApiError(
          error?.message || "Invalid transformer configuration",
          400,
          "invalid_request_error",
          "invalid_request_error"
        );
      }
    }
  }

  // Wire-keep filtered loop: skip body rebuild for Anthropic/Responses owners,
  // run only wire-safe middleware + always run OpenAI owner (Chat === Unified).
  if (effectiveWireKeep && plan) {
    const ownerName = (transformer as any)?.name as string | undefined;
    for (const chainTransformer of plan.request) {
      if (
        !chainTransformer ||
        typeof chainTransformer.transformRequestIn !== "function"
      ) {
        continue;
      }
      const tName = (chainTransformer as any).name as string | undefined;
      const isOwner = tName === ownerName;
      if (isOwner) {
        if (tName === "Anthropic" || tName === "openai-responses") {
          continue;
        }
        // OpenAI owner: always run (policy + media extract) on kept wire
      } else if (!isWireSafeMiddlewareForKeep(tName, ownerName)) {
        continue;
      }
      const transformIn = await chainTransformer.transformRequestIn(
        requestBody,
        provider,
        context
      );
      if (transformIn.body) {
        requestBody = transformIn.body;
        const nextConfig = transformIn.config || {};
        const previousResponse = config.__providerResponse as
          | Response
          | undefined;
        const nextResponse = nextConfig.__providerResponse as
          | Response
          | undefined;
        cancelReplacedProviderResponse(previousResponse, nextResponse);
        config = {
          ...config,
          ...nextConfig,
          headers: mergeHeadersCaseInsensitive(
            config.headers,
            nextConfig.headers
          ),
        };
      } else {
        requestBody = transformIn;
      }
    }
    // Transport URL/headers for Anthropic/Responses keep were in the owner
    // In body rebuild; now pull them from auth() so fetch has a URL.
    const authOwnerName = (transformer as any)?.name as string | undefined;
    if (
      (authOwnerName === "Anthropic" || authOwnerName === "openai-responses") &&
      typeof transformer.auth === "function"
    ) {
      try {
        const authOut = await transformer.auth(requestBody, provider, context);
        if (authOut?.config?.url && !config.url) config.url = authOut.config.url;
        if (authOut?.config?.headers) {
          config.headers = mergeHeadersCaseInsensitive(
            config.headers,
            authOut.config.headers
          );
          if (authOut.config.__authRecovery) {
            config.__authRecovery = authOut.config.__authRecovery;
          }
        } else if (authOut?.config?.__authRecovery) {
          config.__authRecovery = authOut.config.__authRecovery;
        }
      } catch {
        // auth failure will surface on fetch; don't hide the request path
      }
    }
  } else if (plan) {
    for (const chainTransformer of plan.request) {
      if (
        !chainTransformer ||
        typeof chainTransformer.transformRequestIn !== "function"
      ) {
        continue;
      }
      const transformIn = await chainTransformer.transformRequestIn(
        requestBody,
        provider,
        context
      );
      if (transformIn.body) {
        requestBody = transformIn.body;
        const nextConfig = transformIn.config || {};
        const previousResponse = config.__providerResponse as
          | Response
          | undefined;
        const nextResponse = nextConfig.__providerResponse as
          | Response
          | undefined;
        cancelReplacedProviderResponse(previousResponse, nextResponse);
        config = {
          ...config,
          ...nextConfig,
          headers: mergeHeadersCaseInsensitive(
            config.headers,
            nextConfig.headers
          ),
        };
      } else {
        requestBody = transformIn;
      }
    }
  }

  if (
    !bypass &&
    !effectiveWireKeep &&
    !provider.transformer?.use?.length &&
    Array.isArray(requestBody?.messages)
  ) {
    requestBody = applyOpenAIChatReasoning(requestBody);
    requestBody = applyProviderNativeChatCaching(
      requestBody,
      provider,
      context
    );
  }

  return { requestBody, config, bypass, wireKeep: effectiveWireKeep };
}

/**
 * Snapshot the client-leg cache prefix before the provider transformer chain
 * runs. Held on the request so the wire snapshot can attribute a broken prefix
 * to the client or to our own transformers instead of leaving it ambiguous.
 */
function cachePrefixConversation(req: any, body: any) {
  const nested =
    req?.protocolContext?.nestedAgent === true ||
    req?.protocolContext?.claudeCodeSubagent === true;
  return resolveCachePrefixConversationId({
    sessionId: typeof req?.sessionId === "string" ? req.sessionId : undefined,
    nestedAgent: nested,
    firstUserText: firstSubstantiveUserText(body || {}),
  });
}

function recordClientCachePrefix(
  req: any,
  fastify: FastifyInstance,
  provider: any,
  body: any
): void {
  if (!req) return;
  if (isStatuslinePollTurn(body || {})) {
    // Statusline polls share the session id but are not a real turn. Committing
    // them as the baseline makes the next tool result look like a 25k-token miss.
    req._omitCachePrefix = true;
    return;
  }
  const conversation = cachePrefixConversation(req, body);
  req._cachePrefixConversation = conversation;
  req._cachePrefixClientDiff = logOutboundCacheStructure(body, {
    logger: req.log ?? fastify.log,
    reqId: req.id,
    provider: provider?.name,
    model: body?.model,
    conversationId: conversation.id,
    conversationIdSource: conversation.source,
    stage: "client",
  });
}

/**
 * Send request to LLM provider
 * Handle authentication, build request config, send request and handle errors
 */
async function sendRequestToProvider(
  requestBody: any,
  config: any,
  provider: any,
  fastify: FastifyInstance,
  bypass: boolean,
  transformer: any,
  context: any
) {
  const conversation =
    context?.req?._cachePrefixConversation ??
    cachePrefixConversation(context?.req, requestBody);
  const debugOpts = {
    logger: context?.req?.log ?? fastify.log,
    reqId: context?.req?.id,
    provider: provider?.name,
    model: requestBody?.model,
    conversationId: conversation.id ?? context?.req?.sessionId,
    conversationIdSource: conversation.source,
    commitCachePrefix: context?.req?._omitCachePrefix ? false : undefined,
  };

  const tapProviderResponse = async (response: Response) => {
    if (context?.req?._omitCachePrefix) {
      return response;
    }
    const cursorLifecycle = context?.req?._cursorCacheLifecycle ?? null;
    const cacheDiff = logOutboundCacheStructure(requestBody, {
      ...debugOpts,
      stage: "wire",
      responseStatus: response?.status,
      clientStageDiff: context?.req?._cachePrefixClientDiff,
      outboundBody: requestBody,
      cursorLifecycle,
      cacheAffinity: {
        sessionId: config?.headers?.["session-id"],
        threadId: config?.headers?.["thread-id"],
        clientRequestId: config?.headers?.["x-client-request-id"],
      },
    });
    return tapUpstreamSSEDebug(response, {
      ...debugOpts,
      responseStatus: response?.status,
      clientStageDiff: context?.req?._cachePrefixClientDiff,
      cacheDiff,
      outboundBody: requestBody,
      cursorLifecycle,
      direction: "provider→ccr",
      maxBytes: resolveLogBodyMaxBytes(fastify.configService),
      rawEvents: shouldLogSSEEvents(fastify.configService),
    });
  };

  const logOutboundRequestBody = () => {
    const logger = context?.req?.log ?? fastify.log;
    const model =
      typeof requestBody?.model === "string" ? requestBody.model : undefined;
    if (context?.req?._wireKeep) {
      // Keep has no Anthropic-style inbound `request body` info log, and the
      // global fetch wrapper omits bodies. Digest the
      // kept wire at debug so encrypted replay is greppable without a dump.
      logKeepWire(requestBody, {
        logger,
        reqId: context?.req?.id,
        provider: provider?.name,
        model,
      });
    }
    const selection = resolveLogBodySelection(fastify.configService);
    if (selection === undefined) return;
    logMessageBody(requestBody, {
      logger,
      direction: "ccr→provider",
      reqId: context?.req?.id,
      provider: provider?.name,
      model,
      maxBytes: resolveLogBodyMaxBytes(fastify.configService),
      selection,
    });
  };

  // Allow a transformer to own the full upstream call (non-fetch transports,
  // agent SDKs, etc.) by returning a ready Response via __providerResponse.
  if (config?.__providerResponse) {
    // Body was already posted inside the owning transformer; still record it
    // so part capture covers OpenCode Zen / SDK-owned legs.
    logOutboundRequestBody();
    return tapProviderResponse(config.__providerResponse as Response);
  }

  // Handle authentication in passthrough mode
  const authTransformer =
    provider.transformer?.use?.find(
      (providerTransformer: Transformer) =>
        typeof providerTransformer?.auth === "function"
    ) || transformer;
  if (
    (bypass || provider.transformer?.passthrough) &&
    typeof authTransformer.auth === "function"
  ) {
    const auth = await authTransformer.auth(requestBody, provider, context);
    if (auth.body) {
      requestBody = auth.body;
      let headers = config.headers || {};
      if (auth.config?.headers) {
        headers = mergeHeadersCaseInsensitive(
          headers,
          auth.config.headers
        );
        delete headers.host;
        delete auth.config.headers;
      }
      config = {
        ...config,
        ...auth.config,
        headers,
      };
    } else {
      requestBody = auth;
    }
  }

  // Resolved after auth: a passthrough transformer's auth() is the stage that
  // knows the protocol-correct upstream path (e.g. Anthropic's
  // /v1/messages?beta=true derived from a bare provider origin).
  const url = config.url || new URL(provider.baseUrl);

  // Provider auth is generated independently from client headers. Canonicalize
  // casing and guarantee that only one authentication scheme survives.
  const requestHeaders = canonicalizeOutboundHeaders(
    config?.headers,
    provider.apiKey
  );

  // const clientHeaders = context?.req?.headers as
  //   | Record<string, unknown>
  //   | undefined;
  // context?.req?.log?.debug?.(
  //   {
  //     reqId: context?.req?.id,
  //     provider: provider.name,
  //     outboundHeaders: sanitizeHeadersForLog(requestHeaders),
  //     clientHeaders: sanitizeHeadersForLog(clientHeaders),
  //     headerDiff: {
  //       direction: "client -> outbound",
  //       ...diffHeadersForLog(clientHeaders, requestHeaders),
  //     },
  //   },
  //   "provider request headers diff"
  // );

  // Keep requestHeaders construction intact above; only the verbose header diff
  // logging is disabled here to reduce sensitive metadata in debug logs.


  const {
    __authRecovery,
    ...providerRequestConfig
  } = config || {};
  logOutboundRequestBody();
  const send = (headers: Record<string, string>) =>
    sendUnifiedRequest(
      url,
      requestBody,
      {
        httpsProxy: fastify.configService.getHttpsProxy(),
        ...providerRequestConfig,
        headers: { ...headers },
        signal: context?.signal ?? providerRequestConfig.signal,
      },
      context,
      fastify.log
    );

  const response = await sendWithUnauthorizedAuthRecovery(
    send,
    requestHeaders,
    __authRecovery
  );

  // context?.req?.log?.debug?.(
  //   {
  //     reqId: context?.req?.id,
  //     provider: provider.name,
  //     status: response.status,
  //     headers: sanitizeHeadersForLog(response.headers),
  //   },
  //   "provider response headers"
  // );

  // Keep status handling below active; only raw response header logging is
  // disabled here.

  // Handle request errors. Read the non-2xx body once and throw a structured
  // upstream error: the client-protocol boundary (errorHandler) reshapes the
  // sanitized body into the caller's wire envelope instead of flattening
  // every failure into an unrecognizable provider_response_error string.
  if (!response.ok) {
    const errorText = await response.text();
    const safeErrorText = sanitizeUpstreamErrorText(errorText) || errorText.slice(0, 240);

    let parsedBody: unknown;
    try {
      parsedBody = JSON.parse(errorText);
    } catch {
      parsedBody = undefined;
    }
    const upstreamError =
      parsedBody && typeof parsedBody === "object"
        ? (parsedBody as any).error
        : undefined;

    // Safe response-header map: retry/observability metadata only.
    const safeHeaders = selectSafeDownstreamHeaders(response.headers);
    let headers: Record<string, string> | undefined =
      Object.keys(safeHeaders).length > 0 ? safeHeaders : undefined;

    // Retry-After may also hide inside a Google-RPC RetryInfo body detail.
    if (!headers?.["retry-after"] && response.status === 429) {
      const details =
        (upstreamError as any)?.details || (parsedBody as any)?.details;
      if (Array.isArray(details)) {
        const retryInfo = details.find((d: any) => d['@type'] === 'type.googleapis.com/google.rpc.RetryInfo');
        if (retryInfo?.retryDelay) {
          const seconds = parseInt(retryInfo.retryDelay, 10);
          if (!isNaN(seconds)) {
            headers = { ...(headers || {}), 'retry-after': seconds.toString() };
          }
        }
      }
    }

    const upstreamMessage =
      typeof upstreamError?.message === "string"
        ? upstreamError.message
        : undefined;
    const safeMessage =
      sanitizeUpstreamErrorText(upstreamMessage || errorText) || safeErrorText;

    // Log parsed error details for observability (redacted)
    fastify.log.error(
      {
        status: response.status,
        provider: provider.name,
        model: requestBody.model,
        errorMessage: safeMessage,
        upstreamType:
          typeof upstreamError?.type === "string"
            ? upstreamError.type
            : undefined,
        upstreamCode:
          typeof upstreamError?.code === "string"
            ? upstreamError.code
            : undefined,
      },
      `[provider_response_error] ${provider.name},${requestBody.model}: ${safeMessage}`,
    );

    const error = createApiError(
      `Error from provider(${provider.name},${requestBody.model}: ${response.status}): ${safeMessage}`,
      response.status,
      typeof upstreamError?.code === "string"
        ? upstreamError.code
        : "provider_response_error",
      "api_error",
      headers
    );
    (error as any).upstream = {
      status: response.status,
      statusText: response.statusText,
      headers: headers ?? {},
      body:
        parsedBody !== undefined
          ? sanitizeUpstreamErrorBody(parsedBody)
          : undefined,
      // Log-side provenance only; never serialized to the client.
      provider: provider.name,
      model: requestBody.model,
    };
    throw error;
  }

  return tapProviderResponse(response);
}

/**
 * Process response transformer chain
 * Sequentially execute provider transformers, model-specific transformers, transformer's transformResponseIn
 */
async function processResponseTransformers(
  requestBody: any,
  response: any,
  provider: any,
  transformer: any,
  bypass: boolean,
  context: any
) {
  let finalResponse = response;
  const thirdPartyAnthropic = isThirdPartyInScopeAnthropic(
    context?.protocolContext
  );
  // Raw provider bytes may reach the client only for an exact-protocol bypass,
  // including the legacy multi-transformer passthrough mode when its chain
  // explicitly contains the endpoint owner.
  const skipBodyConversion =
    bypass ||
    (!thirdPartyAnthropic &&
      isManualExactProtocolPassthrough(
        provider,
        transformer,
        context?.protocolContext
      )) ||
    (provider.transformer?.passthrough && !context?.protocolContext);
  const manualExactPassthrough =
    !thirdPartyAnthropic &&
    isManualExactProtocolPassthrough(
      provider,
      transformer,
      context?.protocolContext
    );

  // Response transformers get the provider too: the request-side context
  // already carries it, and provider identity is what scopes per-provider state
  // such as the Gemini thought-signature cache.
  const responseContext = { ...context, provider };

  const modelUse = !bypass
    ? (provider.transformer?.[requestBody.model]?.use as
        | Transformer[]
        | undefined)
    : undefined;
  const providerUse = !bypass
    ? (provider.transformer?.use as Transformer[] | undefined)
    : undefined;
  let exactProtocolResponse = false;

  if (providerUse?.length || modelUse?.length) {
    let plan;
    try {
      plan = compileTransformerPlan(providerUse, modelUse, {
        skipName: manualExactPassthrough ? transformer.name : undefined,
      });
    } catch (error: any) {
      throw createApiError(
        error?.message || "Invalid transformer configuration",
        400,
        "invalid_request_error",
        "invalid_request_error"
      );
    }
    exactProtocolResponse = isExactProtocolResponsePlan(
      plan,
      transformer,
      context?.protocolContext?.ownerTransformerName
    );

    for (const chainTransformer of plan.response) {
      if (
        !chainTransformer ||
        (exactProtocolResponse &&
          chainTransformer.name === transformer.name) ||
        typeof chainTransformer.transformResponseOut !== "function"
      ) {
        continue;
      }
      finalResponse = await chainTransformer.transformResponseOut!(
        finalResponse,
        responseContext
      );
    }
  }

  // Execute transformer's transformResponseIn method
  if (
    !skipBodyConversion &&
    !exactProtocolResponse &&
    transformer.transformResponseIn
  ) {
    finalResponse = await transformer.transformResponseIn(
      finalResponse,
      context
    );
  }

  return finalResponse;
}

/**
 * Format and return response.
 * Stream intent comes from ClientProtocolContext when provided; body.stream
 * remains a fallback for legacy callers.
 */
async function formatResponse(
  response: any,
  reply: FastifyReply,
  body: any,
  clientSignal?: AbortSignal,
  streamIntent?: boolean,
  protocol?: ClientProtocol,
  debug?: {
    configService?: ConfigService;
    provider?: string;
    model?: string;
  }
) {
  // Set HTTP status code
  if (!response.ok) {
    reply.code(response.status);
  }

  // Forward safe upstream observability headers (rate-limit metadata, request
  // ids, retry hints) that transformers preserved on the reshaped Response.
  // Local framing headers (Content-Type, Cache-Control, Connection) are set
  // afterwards, so they always win over anything copied here.
  const downstreamHeaders = selectSafeDownstreamHeaders(response.headers);
  for (const [name, value] of Object.entries(downstreamHeaders)) {
    reply.header(name, value);
  }

  const log = reply.log ?? (reply.request as any)?.log;
  const reqId = (reply.request as any)?.id;
  const rawSSE = shouldLogSSEEvents(debug?.configService);
  const bodySelection = resolveLogBodySelection(debug?.configService);
  const maxBytes = resolveLogBodyMaxBytes(debug?.configService);

  // Handle streaming response
  const isStream =
    typeof streamIntent === "boolean" ? streamIntent : body?.stream === true;
  if (isStream && response.ok) {
    // Convert Web API ReadableStream to Node.js stream for Fastify
    if (response.body && typeof response.body.getReader === "function") {
      // fromWeb() locks response.body; destroy() cancels via that reader.
      // Never call response.body.cancel() afterward — it rejects with
      // ERR_INVALID_STATE ("ReadableStream is locked") as an unhandledRejection.
      //
      // Keepalive comments every 10s of upstream silence so Claude Code's 20s
      // byte-idle spinner ("Waiting for API response · check your network")
      // does not abort-and-retry during Anthropic's ~25–30s ping gaps.
      // Passthrough skips OpenAITransformer.transformResponseIn, so the
      // [DONE]/cost-trailer split has to live here — Chat Completions clients
      // JSON.parse the concatenated payload.
      const framed =
        protocol === "openai_chat_completions"
          ? withChatCompletionsDoneBoundary(response.body)
          : response.body;
      const keptAlive = withSSEClientKeepalive(framed);
      const clientBody = tapClientSSEDebug(keptAlive, {
        logger: log,
        reqId,
        provider: debug?.provider,
        model: debug?.model,
        protocol,
        maxBytes,
        rawEvents: rawSSE,
      });
      const nodeStream = Readable.fromWeb(clientBody as any);
      let cleanedUp = false;
      const latency = (reply.request as any)?._latency;
      const socketGone = isResponseSocketGone(reply);
      let sawFirstByte = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        attachLatencyMeta(latency, {
          cancelled: Boolean(clientSignal?.aborted || socketGone),
        });
        emitLatencyRecord(reply.log ?? (reply.request as any)?.log, latency);
        try {
          if (!nodeStream.destroyed) {
            nodeStream.destroy();
          }
        } catch {
          // ignore
        }
      };

      nodeStream.once("data", () => {
        if (!sawFirstByte) {
          sawFirstByte = true;
          markLatency(latency, "downstreamFirstByte");
        }
      });
      nodeStream.once("end", () => {
        emitLatencyRecord(reply.log ?? (reply.request as any)?.log, latency);
      });

      const detachAbort = () => {
        if (clientSignal) {
          clientSignal.removeEventListener("abort", cleanup);
        }
      };

      // Only 499 when the TCP socket is actually gone. A disconnect signal
      // alone is not proof — Cursor previously got false 499 JSON bodies
      // while still connected ("cancel stream: null" then status 499).
      if (socketGone) {
        cleanup();
        if (!reply.sent && !reply.raw.destroyed) {
          reply.type("application/json");
          return reply.code(499).send({
            error: {
              message: "Client closed request",
              type: "api_error",
              code: "client_aborted",
            },
          });
        }
        return reply;
      }

      if (clientSignal?.aborted) {
        reply.log?.warn?.(
          {
            reason: String((clientSignal as any).reason ?? ""),
            headersSent: reply.raw.headersSent,
            writableEnded: reply.raw.writableEnded,
          },
          "disconnect signal fired but socket still open; continuing SSE"
        );
      }

      reply.header("Content-Type", "text/event-stream");
      reply.header("Cache-Control", "no-cache");
      reply.header("Connection", "keep-alive");

      if (clientSignal) {
        clientSignal.addEventListener("abort", cleanup, { once: true });
      }

      // Client gone / broken pipe while streaming — abort upstream quietly.
      reply.raw.on("error", (err: any) => {
        if (
          err?.code === "EPIPE" ||
          err?.code === "ECONNRESET" ||
          err?.code === "ERR_STREAM_PREMATURE_CLOSE" ||
          isClientAbortError(err)
        ) {
          cleanup();
        }
      });
      reply.raw.on("close", () => {
        if (!reply.raw.writableEnded) {
          cleanup();
        }
      });

      nodeStream.on("error", (err: any) => {
        if (
          err?.code === "ERR_STREAM_PREMATURE_CLOSE" ||
          err?.code === "ABORT_ERR" ||
          isClientAbortError(err)
        ) {
          return;
        }
        reply.log?.error?.(
          sanitizeErrorForLog(err),
          "upstream stream error"
        );
      });

      // After a normal stream end, response close no longer aborts (writableEnded),
      // but drop the listener anyway so cleanup cannot run twice on teardown.
      nodeStream.once("end", detachAbort);
      nodeStream.once("close", detachAbort);

      return reply.send(nodeStream);
    }
    // Web ReadableStream is not a valid Fastify payload for text/event-stream.
    reply.header("Content-Type", "text/event-stream");
    reply.header("Cache-Control", "no-cache");
    reply.header("Connection", "keep-alive");
    if (response.body && typeof (response.body as any).pipe === "function") {
      return reply.send(response.body);
    }
    throw createApiError(
      "Streaming response body is not a readable stream",
      502,
      "provider_response_error"
    );
  } else {
    // Handle regular JSON response (including error responses)
    const json = await response.json();
    if (bodySelection !== undefined || rawSSE) {
      logMessageBody(json, {
        logger: log ?? { debug() {} },
        direction: "ccr→client",
        reqId,
        protocol,
        provider: debug?.provider,
        model: debug?.model,
        maxBytes,
        selection: bodySelection,
      });
    }
    const latency = (reply.request as any)?._latency;
    markLatency(latency, "downstreamFirstByte");
    emitLatencyRecord(reply.log ?? (reply.request as any)?.log, latency);
    return reply.send(json);
  }
}

export const registerApiRoutes = async (
  fastify: FastifyInstance
) => {
  const rateLimitOptions = {
    config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
  };

  // Detect the wire protocol before body parsing and route validation so even
  // malformed JSON and other early failures receive the correct error shape.
  fastify.addHook("onRequest", async (req: FastifyRequest) => {
    const pathname = String(req.url || "").split("?")[0] || "/";
    const match = matchClientProtocol(req.method, pathname);
    (req as any).pathname = pathname;
    if (match) {
      (req as any).protocolMatch = match;
      (req as any).clientProtocol = match.protocol;
    }
  });

  // Health and info endpoints
  fastify.get("/", rateLimitOptions, async () => {
    return { message: "LLMs API", version };
  });

  fastify.get("/health", rateLimitOptions, async () => {
    // The liveness contract (status + timestamp) is fixed. The process owner
    // may attach richer vitals; when it does not, the probe is unchanged.
    const vitals = readHealthVitals();
    return {
      status: "ok",
      timestamp: new Date().toISOString(),
      ...(vitals ? { vitals } : {}),
    };
  });

  // Client protocol routes (canonical + aliases) from the protocol registry.
  // Owner transformers are resolved by name so Vercel cannot claim Chat.
  const allTransformers = fastify.transformerService.getAllTransformers();
  const resolveOwner = (ownerName: string): Transformer | undefined => {
    const entry = allTransformers.get(ownerName);
    if (!entry) return undefined;
    if (typeof entry === "object") return entry as Transformer;
    try {
      const instance = new (entry as any)();
      if (instance && typeof instance === "object") {
        (instance as any).logger = fastify.log;
      }
      return instance as Transformer;
    } catch {
      return undefined;
    }
  };

  const claimedPaths = new Set<string>();
  for (const reg of listClientRouteRegistrations()) {
    const transformer = resolveOwner(reg.ownerTransformerName);
    if (!transformer) {
      fastify.log.warn(
        `protocol route ${reg.path}: owner transformer '${reg.ownerTransformerName}' not found`
      );
      continue;
    }
    for (const routePath of [reg.path, `${reg.path}/`]) {
      if (claimedPaths.has(routePath)) continue;
      claimedPaths.add(routePath);
      const isFim = reg.protocol === "openai_fim_completions";
      fastify.post(
        routePath,
        rateLimitOptions,
        async (req: FastifyRequest, reply: FastifyReply) => {
          if (isFim) {
            return handleFimEndpoint(
              req,
              reply,
              fastify,
              transformer,
              routePath
            );
          }
          return handleTransformerEndpoint(
            req,
            reply,
            fastify,
            transformer,
            routePath
          );
        }
      );
    }
  }

  // `endPoint` remains provider-transformer metadata, but client route
  // registration is intentionally limited to the explicit protocol registry.

  fastify.post(
    "/providers",
    {
      config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
      schema: {
        body: {
          type: "object",
          properties: {
            id: { type: "string" },
            name: { type: "string" },
            type: { type: "string", enum: ["openai", "anthropic"] },
            baseUrl: { type: "string" },
            apiKey: { type: "string" },
            models: { type: "array", items: { type: "string" } },
          },
          required: ["id", "name", "type", "baseUrl", "apiKey", "models"],
        },
      },
    },
    async (
      request: FastifyRequest<{ Body: RegisterProviderRequest }>,
      _reply: FastifyReply
    ) => {
      // Validation
      const { name, baseUrl, apiKey, models } = request.body;

      if (!name?.trim()) {
        throw createApiError(
          "Provider name is required",
          400,
          "invalid_request"
        );
      }

      if (!baseUrl || !isValidUrl(baseUrl)) {
        throw createApiError(
          "Valid base URL is required",
          400,
          "invalid_request"
        );
      }

      if (!apiKey?.trim()) {
        throw createApiError("API key is required", 400, "invalid_request");
      }

      if (!models || !Array.isArray(models) || models.length === 0) {
        throw createApiError(
          "At least one model is required",
          400,
          "invalid_request"
        );
      }

      // Check if provider already exists
      if (fastify.providerService.getProvider(request.body.name)) {
        throw createApiError(
          `Provider with name '${request.body.name}' already exists`,
          400,
          "provider_exists"
        );
      }

      return fastify.providerService.registerProvider(request.body);
    }
  );

  fastify.get("/providers", rateLimitOptions, async () => {
    return fastify.providerService.getProviders();
  });

  fastify.get(
    "/providers/:id",
    {
      config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
      schema: {
        params: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
      },
    },
    async (request: FastifyRequest<{ Params: { id: string } }>) => {
      const provider = fastify.providerService.getProvider(
        request.params.id
      );
      if (!provider) {
        throw createApiError("Provider not found", 404, "provider_not_found");
      }
      return provider;
    }
  );

  fastify.put(
    "/providers/:id",
    {
      config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
      schema: {
        params: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
        body: {
          type: "object",
          properties: {
            name: { type: "string" },
            type: { type: "string", enum: ["openai", "anthropic"] },
            baseUrl: { type: "string" },
            apiKey: { type: "string" },
            models: { type: "array", items: { type: "string" } },
            enabled: { type: "boolean" },
          },
        },
      },
    },
    async (
      request: FastifyRequest<{
        Params: { id: string };
        Body: Partial<LLMProvider>;
      }>,
      _reply
    ) => {
      const provider = fastify.providerService.updateProvider(
        request.params.id,
        request.body
      );
      if (!provider) {
        throw createApiError("Provider not found", 404, "provider_not_found");
      }
      return provider;
    }
  );

  fastify.delete(
    "/providers/:id",
    {
      config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
      schema: {
        params: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
      },
    },
    async (request: FastifyRequest<{ Params: { id: string } }>) => {
      const success = fastify.providerService.deleteProvider(
        request.params.id
      );
      if (!success) {
        throw createApiError("Provider not found", 404, "provider_not_found");
      }
      return { message: "Provider deleted successfully" };
    }
  );

  fastify.patch(
    "/providers/:id/toggle",
    {
      config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
      schema: {
        params: {
          type: "object",
          properties: { id: { type: "string" } },
          required: ["id"],
        },
        body: {
          type: "object",
          properties: { enabled: { type: "boolean" } },
          required: ["enabled"],
        },
      },
    },
    async (
      request: FastifyRequest<{
        Params: { id: string };
        Body: { enabled: boolean };
      }>,
      _reply
    ) => {
      const success = fastify.providerService.toggleProvider(
        request.params.id,
        request.body.enabled
      );
      if (!success) {
        throw createApiError("Provider not found", 404, "provider_not_found");
      }
      return {
        message: `Provider ${request.body.enabled ? "enabled" : "disabled"
          } successfully`,
      };
    }
  );
};

// Helper function
function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

import {
  FastifyInstance,
  FastifyRequest,
  FastifyReply,
} from "fastify";
import { Readable } from "stream";
import { RegisterProviderRequest, LLMProvider } from "@/types/llm";
import { sendUnifiedRequest } from "@/utils/request";
import { createApiError } from "./middleware";
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
import { applyOpenAIChatReasoning } from "../utils/reasoning-effort";
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

  let prepared: PreparedInboundRequest;
  try {
    prepared = await prepareInboundRequest(
      req,
      reply,
      fastify,
      transformer,
      routePath
    );
  } catch (error: any) {
    if (isClientAbortError(error) || clientSignal.aborted) {
      throw typeof error === "string"
        ? toClientAbortError(error)
        : toClientAbortError(clientSignal.reason ?? error);
    }
    throw error;
  }

  const provider = fastify.providerService.getProvider(prepared.providerName);
  if (!provider) {
    throw createApiError(
      `Provider '${prepared.providerName}' not found`,
      404,
      "provider_not_found"
    );
  }

  // Stash prepared state so fallback can reuse Unified + protocol context.
  (req as any)._preparedInbound = prepared;

  try {
    // Native Anthropic clients use the original wire body; legacy exact
    // protocol passthrough uses the adapted wire body. Both receive only the
    // routed model substitution. Third-party Anthropic emulation and all
    // cross-protocol routes use the Unified projection.
    const willBypass = protocolAwareBypass(
      provider,
      transformer,
      prepared.protocolContext,
      prepared.modelName
    ) && !isThirdPartyInScopeAnthropic(prepared.protocolContext);
    const preserveManualWire =
      !isThirdPartyInScopeAnthropic(prepared.protocolContext) &&
      isManualExactProtocolPassthrough(
        provider,
        transformer,
        prepared.protocolContext
      );
    const exactWireSource = prepared.protocolContext.anthropicNativeWire
      ? prepared.originalBody
      : prepared.clientWireBody ?? prepared.originalBody;
    const pipelineBody = prepared.protocolContext.anthropicNativeWire ||
      willBypass || preserveManualWire
      ? {
          ...(typeof exactWireSource === "object" && exactWireSource
            ? cloneProtocolBody(exactWireSource)
            : {}),
          model: prepared.modelName,
        }
      : cloneProtocolBody(prepared.unifiedBody);

    const { requestBody, config, bypass } = await processRequestTransformers(
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
        // Client→Unified already ran in prepareInboundRequest.
        skipClientNormalization: true,
      }
    );

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

    const finalResponse = await processResponseTransformers(
      requestBody,
      response,
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

    return await formatResponse(
      finalResponse,
      reply,
      prepared.originalBody,
      clientSignal,
      prepared.protocolContext.stream
    );
  } catch (error: any) {
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

      const willBypass = protocolAwareBypass(
        provider,
        transformer,
        fallbackProtocolContext,
        fallbackModelOnly
      ) && !isThirdPartyInScopeAnthropic(fallbackProtocolContext);
      const preserveManualWire =
        !isThirdPartyInScopeAnthropic(fallbackProtocolContext) &&
        isManualExactProtocolPassthrough(
          provider,
          transformer,
          fallbackProtocolContext
        );
      const exactWireSource = fallbackProtocolContext?.anthropicNativeWire
        ? prepared?.originalBody
        : prepared?.clientWireBody ?? prepared?.originalBody;
      const pipelineBody = fallbackProtocolContext?.anthropicNativeWire ||
        willBypass || preserveManualWire
        ? {
            ...(typeof exactWireSource === "object" && exactWireSource
              ? cloneProtocolBody(exactWireSource)
              : {}),
            model: fallbackModelOnly,
          }
        : unifiedBody;

      const { requestBody, config, bypass } = await processRequestTransformers(
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
        }
      );

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

      const finalResponse = await processResponseTransformers(
        requestBody,
        response,
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
        fallbackProtocolContext?.stream
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
 * Client→Unified already ran in prepareInboundRequest when
 * skipClientNormalization is set; here we only run provider In chain.
 */
async function processRequestTransformers(
  body: any,
  provider: any,
  transformer: any,
  headers: any,
  context: any
) {
  let requestBody = body;
  let config: any = {};

  const protocolBypass = protocolAwareBypass(
    provider,
    transformer,
    context?.protocolContext,
    body?.model
  );
  const bypass = Boolean(
    context?.protocolContext?.anthropicNativeWire === true ||
      (protocolBypass && !isThirdPartyInScopeAnthropic(context?.protocolContext))
  );
  const thirdPartyAnthropic = isThirdPartyInScopeAnthropic(
    context?.protocolContext
  );
  const manualExactPassthrough =
    !thirdPartyAnthropic &&
    isManualExactProtocolPassthrough(
      provider,
      transformer,
      context?.protocolContext
    );
  const skipBodyConversion =
    bypass ||
    (!thirdPartyAnthropic && provider.transformer?.passthrough) ||
    context?.skipClientNormalization === true;

  if (bypass) {
    config.headers = sanitizePassthroughHeaders(headers);
  }

  if (
    !bypass &&
    context?.protocolContext?.protocol === "openai_responses" &&
    !provider.transformer?.use?.some(
      (providerTransformer: Transformer) =>
        providerTransformer?.name === "openai-responses"
    )
  ) {
    // This key is an opaque hint scoped to the Responses destination that
    // interprets it. It must not cross a protocol/provider boundary.
    delete requestBody.prompt_cache_key;
  }

  // Legacy path: run transformRequestOut only when client normalization was not
  // already performed by prepareInboundRequest.
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

  if (!bypass && provider.transformer?.use?.length) {
    for (const providerTransformer of provider.transformer.use) {
      if (
        manualExactPassthrough &&
        providerTransformer?.name === transformer.name
      ) {
        continue;
      }
      if (
        !providerTransformer ||
        typeof providerTransformer.transformRequestIn !== "function"
      ) {
        continue;
      }
      const transformIn = await providerTransformer.transformRequestIn(
        requestBody,
        provider,
        context
      );
      if (transformIn.body) {
        requestBody = transformIn.body;
        const nextConfig = transformIn.config || {};
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

  if (!bypass && provider.transformer?.[body.model]?.use?.length) {
    for (const modelTransformer of provider.transformer[body.model].use) {
      if (
        manualExactPassthrough &&
        modelTransformer?.name === transformer.name
      ) {
        continue;
      }
      if (
        !modelTransformer ||
        typeof modelTransformer.transformRequestIn !== "function"
      ) {
        continue;
      }
      const transformIn = await modelTransformer.transformRequestIn(
        requestBody,
        provider,
        context
      );
      if (transformIn.body) {
        requestBody = transformIn.body;
        const nextConfig = transformIn.config || {};
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

  return { requestBody, config, bypass };
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
  // Allow a transformer to own the full upstream call (non-fetch transports,
  // agent SDKs, etc.) by returning a ready Response via __providerResponse.
  if (config?.__providerResponse) {
    return config.__providerResponse as Response;
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
  const send = (headers: Record<string, string>) =>
    sendUnifiedRequest(
      url,
      requestBody,
      {
        httpsProxy: fastify.configService.getHttpsProxy(),
        ...providerRequestConfig,
        headers: JSON.parse(JSON.stringify(headers)),
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

  return response;
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

  // Execute provider-level response transformers
  if (!bypass && provider.transformer?.use?.length) {
    for (const providerTransformer of Array.from(
      provider.transformer.use
    ).reverse() as Transformer[]) {
      if (
        manualExactPassthrough &&
        providerTransformer?.name === transformer.name
      ) {
        continue;
      }
      if (
        !providerTransformer ||
        typeof providerTransformer.transformResponseOut !== "function"
      ) {
        continue;
      }
      finalResponse = await providerTransformer.transformResponseOut!(
        finalResponse,
        responseContext
      );
    }
  }

  // Execute model-specific response transformers
  if (!bypass && provider.transformer?.[requestBody.model]?.use?.length) {
    for (const modelTransformer of Array.from(
      provider.transformer[requestBody.model].use
    ).reverse() as Transformer[]) {
      if (
        manualExactPassthrough &&
        modelTransformer?.name === transformer.name
      ) {
        continue;
      }
      if (
        !modelTransformer ||
        typeof modelTransformer.transformResponseOut !== "function"
      ) {
        continue;
      }
      finalResponse = await modelTransformer.transformResponseOut!(
        finalResponse,
        responseContext
      );
    }
  }

  // Execute transformer's transformResponseIn method
  if (!skipBodyConversion && transformer.transformResponseIn) {
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
  streamIntent?: boolean
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

  // Handle streaming response
  const isStream =
    typeof streamIntent === "boolean" ? streamIntent : body?.stream === true;
  if (isStream && response.ok) {
    // Convert Web API ReadableStream to Node.js stream for Fastify
    if (response.body && typeof response.body.getReader === "function") {
      // fromWeb() locks response.body; destroy() cancels via that reader.
      // Never call response.body.cancel() afterward — it rejects with
      // ERR_INVALID_STATE ("ReadableStream is locked") as an unhandledRejection.
      const nodeStream = Readable.fromWeb(response.body as any);
      let cleanedUp = false;

      const cleanup = () => {
        if (cleanedUp) return;
        cleanedUp = true;
        try {
          if (!nodeStream.destroyed) {
            nodeStream.destroy();
          }
        } catch {
          // ignore
        }
      };

      const detachAbort = () => {
        if (clientSignal) {
          clientSignal.removeEventListener("abort", cleanup);
        }
      };

      const socketGone = isResponseSocketGone(reply);

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
    return { status: "ok", timestamp: new Date().toISOString() };
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
      fastify.post(
        routePath,
        rateLimitOptions,
        async (req: FastifyRequest, reply: FastifyReply) => {
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

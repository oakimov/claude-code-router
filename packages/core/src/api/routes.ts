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
import { sendWithUnauthorizedAuthRecovery } from "@/utils/auth-recovery";

// Extend FastifyInstance to include custom services
declare module "fastify" {
  interface FastifyInstance {
    configService: ConfigService;
    providerService: ProviderService;
    transformerService: TransformerService;
  }

  interface FastifyRequest {
    provider?: string;
  }
}

/**
 * Main handler for transformer endpoints
 * Coordinates the entire request processing flow: validate provider, handle request transformers,
 * send request, handle response transformers, format response
 */
async function handleTransformerEndpoint(
  req: FastifyRequest,
  reply: FastifyReply,
  fastify: FastifyInstance,
  transformer: any
) {
  const body = req.body as any;
  const providerName = req.provider!;
  const provider = fastify.providerService.getProvider(providerName);
  const disconnect = createClientDisconnectSignal(req, reply);
  const clientSignal = disconnect.signal;
  disconnect.arm();

  // Validate provider exists
  if (!provider) {
    throw createApiError(
      `Provider '${providerName}' not found`,
      404,
      "provider_not_found"
    );
  }

  // req.log.debug(
  //   {
  //     reqId: req.id,
  //     provider: providerName,
  //     headers: sanitizeHeadersForLog(req.headers as Record<string, unknown>),
  //   },
  //   "client request headers"
  // );

  try {
    // Process request transformer chain
    const { requestBody, config, bypass } = await processRequestTransformers(
      body,
      provider,
      transformer,
      req.headers,
      {
        req,
        provider,
        signal: clientSignal,
      }
    );

    // Send request to LLM provider
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
      }
    );

    // Process response transformer chain
    const finalResponse = await processResponseTransformers(
      requestBody,
      response,
      provider,
      transformer,
      bypass,
      {
        req,
        signal: clientSignal,
      }
    );

    return await formatResponse(finalResponse, reply, body, clientSignal);
  } catch (error: any) {
    // Normalize string / mixed abort shapes from AbortSignal.any + fetch so
    // errorHandler always takes the quiet 499 path (not HTTP 500).
    if (isClientAbortError(error)) {
      throw typeof error === "string" ? toClientAbortError(error) : error;
    }
    if (clientSignal.aborted) {
      throw toClientAbortError(clientSignal.reason ?? error);
    }
    // Fallback on provider response errors and network/fetch failures.
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
      // handleFallback returns null on client abort during backoff — surface an
      // abort error so the handler takes the quiet/499 path instead of the
      // original provider failure.
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

      // Update request with fallback model
      const newBody = { ...(req.body as any) };
      const [fallbackProvider, ...fallbackModelName] = fallbackModel.split(',');
      newBody.model = fallbackModelName.join(',');

      // Create new request object with updated provider and body
      const newReq = {
        ...req,
        provider: fallbackProvider,
        body: newBody,
      };

      const provider = fastify.providerService.getProvider(fallbackProvider);
      if (!provider) {
        req.log.warn(`Fallback provider '${fallbackProvider}' not found, skipping`);
        continue;
      }

      // Process request transformer chain
      const { requestBody, config, bypass } = await processRequestTransformers(
        newBody,
        provider,
        transformer,
        req.headers,
        { req: newReq, provider, signal: clientSignal }
      );

      // Send request to LLM provider
      const response = await sendRequestToProvider(
        requestBody,
        config,
        provider,
        fastify,
        bypass,
        transformer,
        { req: newReq, signal: clientSignal }
      );

      // Process response transformer chain
      const finalResponse = await processResponseTransformers(
        requestBody,
        response,
        provider,
        transformer,
        bypass,
        { req: newReq, signal: clientSignal }
      );

      req.log.info(`Fallback model ${fallbackModel} succeeded`);

      // Format and return response
      return await formatResponse(finalResponse, reply, newBody, clientSignal);
    } catch (fallbackError: any) {
      if (isClientAbortError(fallbackError)) {
        throw fallbackError;
      }
      if (clientSignal?.aborted) {
        throw toClientAbortError(
          clientSignal.reason ?? CLIENT_DISCONNECT_REASON
        );
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
 * Process request transformer chain
 * Sequentially execute transformRequestOut, provider transformers, model-specific transformers
 * Returns processed request body, config, and flag indicating whether to skip transformers
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

  // Check if transformers should be bypassed (passthrough mode)
  const bypass = shouldBypassTransformers(provider, transformer, body);
  const skipBodyConversion = bypass || provider.transformer?.passthrough;

  if (bypass) {
    if (headers instanceof Headers) {
      headers.delete("content-length");
    } else {
      delete headers["content-length"];
    }
    config.headers = headers;
  }

  // Execute transformer's transformRequestOut method
  if (!skipBodyConversion && typeof transformer.transformRequestOut === "function") {
    const transformOut = await transformer.transformRequestOut(requestBody, context);
    if (transformOut.body) {
      requestBody = transformOut.body;
      config = transformOut.config || {};
    } else {
      requestBody = transformOut;
    }
  }

  // Execute provider-level transformers
  if (!bypass && provider.transformer?.use?.length) {
    for (const providerTransformer of provider.transformer.use) {
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
        config = { ...config, ...transformIn.config };
      } else {
        requestBody = transformIn;
      }
    }
  }

  // Execute model-specific transformers
  if (!bypass && provider.transformer?.[body.model]?.use?.length) {
    for (const modelTransformer of provider.transformer[body.model].use) {
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
        config = { ...config, ...transformIn.config };
      } else {
        requestBody = transformIn;
      }
    }
  }

  // Generic OpenAI-compatible providers often have no transformer configured.
  // Translate or remove Anthropic cache markers according to the actual
  // upstream rather than leaking provider-specific fields.
  if (
    !bypass &&
    !provider.transformer?.use?.length &&
    Array.isArray(requestBody?.messages)
  ) {
    requestBody = applyProviderNativeChatCaching(
      requestBody,
      provider,
      context
    );
  }

  return { requestBody, config, bypass };
}

/**
 * Determine if transformers should be bypassed (passthrough mode)
 * Skip other transformers when provider only uses one transformer and it matches the current one
 */
function shouldBypassTransformers(
  provider: any,
  transformer: any,
  body: any
): boolean {
  return (
    provider.transformer?.use?.length === 1 &&
    provider.transformer.use[0].name === transformer.name &&
    (!provider.transformer?.[body.model]?.use.length ||
      (provider.transformer?.[body.model]?.use.length === 1 &&
        provider.transformer?.[body.model]?.use[0].name === transformer.name))
  );
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

  const url = config.url || new URL(provider.baseUrl);

  // Handle authentication in passthrough mode
  if ((bypass || provider.transformer?.passthrough) && typeof transformer.auth === "function") {
    const auth = await transformer.auth(requestBody, provider, context);
    if (auth.body) {
      requestBody = auth.body;
      let headers = config.headers || {};
      if (auth.config?.headers) {
        headers = {
          ...headers,
          ...auth.config.headers,
        };
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

  // Send HTTP request
  // Prepare headers
  const requestHeaders: Record<string, string> = {
    Authorization: `Bearer ${provider.apiKey}`,
    ...(config?.headers || {}),
  };

  // Remove Bearer auth when x-api-key is present
  if (requestHeaders["x-api-key"]) {
    delete requestHeaders.Authorization;
  }

  for (const key in requestHeaders) {
    if (requestHeaders[key] === "undefined") {
      delete requestHeaders[key];
    } else if (
      ["authorization", "Authorization"].includes(key) &&
      requestHeaders[key]?.includes("undefined")
    ) {
      delete requestHeaders[key];
    }
  }

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

  // Handle request errors
  if (!response.ok) {
    const errorText = await response.text();
    const safeErrorText = sanitizeUpstreamErrorText(errorText) || errorText.slice(0, 240);

    let headers: Record<string, string> | undefined = undefined;
    const retryAfter = response.headers.get("retry-after");

    if (retryAfter) {
      headers = { 'Retry-After': retryAfter };
    } else if (response.status === 429) {
      try {
        const errorJson = JSON.parse(errorText);
        const details = errorJson?.error?.details || errorJson?.details;
        if (Array.isArray(details)) {
          const retryInfo = details.find((d: any) => d['@type'] === 'type.googleapis.com/google.rpc.RetryInfo');
          if (retryInfo?.retryDelay) {
            const seconds = parseInt(retryInfo.retryDelay, 10);
            if (!isNaN(seconds)) {
              headers = { 'Retry-After': seconds.toString() };
            }
          }
        }
      } catch {
        // Ignore JSON parse errors
      }
    }

    // Log parsed error details for observability (redacted)
    try {
      const errorJson = JSON.parse(errorText);
      const safeMessage =
        sanitizeUpstreamErrorText(
          String(errorJson?.error?.message || errorText)
        ) || safeErrorText;
      fastify.log.error(
        {
          status: response.status,
          provider: provider.name,
          model: requestBody.model,
          errorMessage: safeMessage,
        },
        `[provider_response_error] ${provider.name},${requestBody.model}: ${safeMessage}`,
      );
    } catch {
      fastify.log.error(
        {
          status: response.status,
          provider: provider.name,
          model: requestBody.model,
          errorText: safeErrorText,
        },
        `[provider_response_error] ${provider.name},${requestBody.model}: ${safeErrorText}`,
      );
    }

    throw createApiError(
      `Error from provider(${provider.name},${requestBody.model}: ${response.status}): ${safeErrorText}`,
      response.status,
      "provider_response_error",
      "api_error",
      headers
    );
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
  const skipBodyConversion = bypass || provider.transformer?.passthrough;

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
 * Format and return response
 * Handle HTTP status codes, format streaming and regular responses
 */
async function formatResponse(
  response: any,
  reply: FastifyReply,
  body: any,
  clientSignal?: AbortSignal
) {
  // Set HTTP status code
  if (!response.ok) {
    reply.code(response.status);
  }

  // Handle streaming response
  const isStream = body.stream === true;
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
  // Health and info endpoints
  fastify.get("/", async () => {
    return { message: "LLMs API", version };
  });

  fastify.get("/health", async () => {
    return { status: "ok", timestamp: new Date().toISOString() };
  });

  const transformersWithEndpoint =
    fastify.transformerService.getTransformersWithEndpoint();

  for (const { transformer } of transformersWithEndpoint) {
    if (transformer.endPoint) {
      fastify.post(
        transformer.endPoint,
        async (req: FastifyRequest, reply: FastifyReply) => {
          return handleTransformerEndpoint(req, reply, fastify, transformer);
        }
      );
    }
  }

  fastify.post(
    "/providers",
    {
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

  fastify.get("/providers", async () => {
    return fastify.providerService.getProviders();
  });

  fastify.get(
    "/providers/:id",
    {
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

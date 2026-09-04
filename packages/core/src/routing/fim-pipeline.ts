import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { Readable } from "stream";
import { createApiError } from "@/api/middleware";
import { sendUnifiedRequest } from "@/utils/request";
import {
  CLIENT_DISCONNECT_REASON,
  createClientDisconnectSignal,
  isClientAbortError,
  isFallbackEligibleError,
  isResponseSocketGone,
  retryDelayAfterFailure,
  selectFallbackModels,
  toClientAbortError,
  delay,
} from "@/utils/retry";
import {
  attachLatencyMeta,
  emitLatencyRecord,
  ensureRequestLatency,
  markLatency,
} from "@/utils/request-latency";
import { withSSEClientKeepalive } from "@/utils/sse/client-keepalive";
import { withChatCompletionsDoneBoundary } from "@/utils/sse/done-boundary";
import {
  canonicalizeOutboundHeaders,
  mergeHeadersCaseInsensitive,
  selectSafeDownstreamHeaders,
} from "@/utils/headers";
import { resolveDestination } from "@/routing/inbound-pipeline";
import {
  createClientProtocolContext,
  matchClientProtocol,
  type ProtocolRouteMatch,
} from "@/routing/protocol-endpoints";
import { normalizeModelSelector } from "@/utils/router";
import {
  V1_FIM_INBOUND_KIND,
  inboundToUnifiedFim,
  isFimProviderTransformerName,
  normalizeFimSseDataPayload,
  encodeFimResponseForInbound,
  outboundFamilyFromTransformerName,
  type UnifiedFimRequest,
  type FimInboundKind,
} from "@/utils/fim";
import type { Transformer } from "@/types/transformer";

function resolveFimRouteModel(
  clientModel: string,
  configService: { get: (key: string) => any }
): { model: string; scenarioType: "fim" } {
  const trimmed =
    typeof clientModel === "string" ? clientModel.trim() : "";
  const normalized = normalizeModelSelector(trimmed) || trimmed;

  if (normalized.includes(",")) {
    return { model: normalized, scenarioType: "fim" };
  }

  const Router = configService.get("Router") || {};
  const destination =
    (typeof Router.fim === "string" && Router.fim.trim()) ||
    (typeof Router.default === "string" && Router.default.trim()) ||
    "";

  if (!destination) {
    throw createApiError(
      "Missing FIM model. Configure Router.fim or send provider,model.",
      400,
      "missing_model",
      "invalid_request_error"
    );
  }

  const destNorm = normalizeModelSelector(destination) || destination;
  if (!destNorm.includes(",")) {
    throw createApiError(
      `Router.fim must be provider,model (got '${destination}').`,
      400,
      "invalid_model",
      "invalid_request_error"
    );
  }
  return { model: destNorm, scenarioType: "fim" };
}

function findFimOutboundTransformer(provider: any): {
  transformer: Transformer;
  family: NonNullable<ReturnType<typeof outboundFamilyFromTransformerName>>;
} {
  const use = provider?.transformer?.use;
  if (!Array.isArray(use) || use.length === 0) {
    throw createApiError(
      `Provider '${provider?.name}' has no fim.* transformer. Configure transformer.use with fim.mistral, fim.deepseek, or fim.qwen.`,
      400,
      "fim_transformer_required",
      "invalid_request_error"
    );
  }

  const nonFim = use.filter(
    (t: Transformer) => !isFimProviderTransformerName(t?.name)
  );
  if (nonFim.length > 0) {
    const names = nonFim.map((t: Transformer) => t?.name || "<unnamed>").join(", ");
    throw createApiError(
      `FIM provider '${provider?.name}' must not stack chat transformers with fim.* (found: ${names}). Use a dedicated FIM provider entry.`,
      400,
      "fim_transformer_conflict",
      "invalid_request_error"
    );
  }

  const fimTransforms = use.filter((t: Transformer) =>
    isFimProviderTransformerName(t?.name)
  );
  if (fimTransforms.length === 0) {
    throw createApiError(
      `Provider '${provider?.name}' has no fim.* transformer.`,
      400,
      "fim_transformer_required",
      "invalid_request_error"
    );
  }

  const primary = fimTransforms[0];
  const family = outboundFamilyFromTransformerName(primary.name);
  if (!family) {
    throw createApiError(
      `Unknown FIM transformer '${primary.name}'.`,
      400,
      "fim_transformer_unknown",
      "invalid_request_error"
    );
  }
  return { transformer: primary, family };
}

async function reshapeFimResponseForClient(
  response: Response,
  inboundKind: FimInboundKind,
  passthrough: boolean
): Promise<Response> {
  // Same-kind: upstream already speaks inbound wire — leave body alone.
  if (!response.ok || passthrough) {
    return response;
  }

  // Cross-family: encode upstream → inbound client wire (v1: mistral).
  const contentType = response.headers.get("Content-Type") || "";
  if (contentType.includes("text/event-stream") && response.body) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let buffer = "";
    const stream = new ReadableStream({
      async pull(controller) {
        const { done, value } = await reader.read();
        if (done) {
          if (buffer.trim().length) {
            const trailing = buffer.startsWith("data: ")
              ? `data: ${normalizeFimSseDataPayload(buffer.slice(6), inboundKind)}\n`
              : buffer;
            controller.enqueue(encoder.encode(trailing));
          }
          controller.close();
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            const normalized = normalizeFimSseDataPayload(data, inboundKind);
            controller.enqueue(encoder.encode(`data: ${normalized}\n`));
          } else {
            controller.enqueue(encoder.encode(`${line}\n`));
          }
        }
      },
      cancel() {
        reader.cancel().catch(() => {});
      },
    });
    return new Response(stream, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }

  const raw = await response.text();
  try {
    const json = JSON.parse(raw);
    const normalized = encodeFimResponseForInbound(json, inboundKind);
    return new Response(JSON.stringify(normalized), {
      status: response.status,
      statusText: response.statusText,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return new Response(raw, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  }
}

async function formatFimResponse(
  response: Response,
  reply: FastifyReply,
  stream: boolean,
  clientSignal?: AbortSignal
) {
  if (!response.ok) {
    reply.code(response.status);
  }

  for (const [name, value] of Object.entries(
    selectSafeDownstreamHeaders(response.headers)
  )) {
    reply.header(name, value);
  }

  if (stream && response.ok && response.body) {
    const framed = withChatCompletionsDoneBoundary(response.body);
    const keptAlive = withSSEClientKeepalive(framed);
    const nodeStream = Readable.fromWeb(keptAlive as any);
    const socketGone = isResponseSocketGone(reply);
    let cleanedUp = false;
    const cleanup = () => {
      if (cleanedUp) return;
      cleanedUp = true;
      try {
        if (!nodeStream.destroyed) nodeStream.destroy();
      } catch {
        // ignore
      }
    };
    if (clientSignal) {
      clientSignal.addEventListener("abort", cleanup, { once: true });
    }
    if (socketGone) {
      cleanup();
      if (!reply.sent) {
        return reply.code(499).send({
          error: {
            message: "Client closed request",
            type: "api_error",
            code: "client_aborted",
          },
        });
      }
      return;
    }
    reply.header("Content-Type", "text/event-stream");
    reply.header("Cache-Control", "no-cache");
    reply.header("Connection", "keep-alive");
    return reply.send(nodeStream);
  }

  const text = await response.text();
  const ct = response.headers.get("Content-Type") || "application/json";
  reply.type(ct);
  try {
    return reply.send(JSON.parse(text));
  } catch {
    return reply.send(text);
  }
}

async function dispatchFimOnce(
  req: FastifyRequest,
  fastify: FastifyInstance,
  provider: any,
  unified: UnifiedFimRequest,
  clientBody: any,
  clientSignal: AbortSignal,
  inboundKind: FimInboundKind = V1_FIM_INBOUND_KIND
): Promise<{ response: Response; passthrough: boolean }> {
  const { transformer } = findFimOutboundTransformer(provider);
  const context = {
    req,
    provider,
    signal: clientSignal,
    fimInboundKind: inboundKind,
    fimClientBody: clientBody,
    unifiedFim: unified,
    clientProtocol: "openai_fim_completions",
  };

  if (typeof transformer.transformRequestIn !== "function") {
    throw createApiError(
      `FIM transformer '${transformer.name}' missing transformRequestIn`,
      500,
      "transformer_misconfigured"
    );
  }

  const out = await transformer.transformRequestIn(
    unified as any,
    provider,
    context as any
  );

  let requestBody = out;
  let config: Record<string, any> = {};
  if (out && typeof out === "object" && "body" in out) {
    requestBody = (out as any).body;
    config = { ...((out as any).config || {}) };
  }

  const passthrough = Boolean(config.__fimPassthrough);
  delete config.__fimPassthrough;

  const url = config.url || provider.baseUrl;
  const headers = canonicalizeOutboundHeaders(
    mergeHeadersCaseInsensitive(
      {
        "Content-Type": "application/json",
      },
      config.headers || {}
    )
  );

  const httpsProxy =
    fastify.configService.get<string>("HTTPS_PROXY") ||
    process.env.HTTPS_PROXY ||
    process.env.https_proxy;

  const response = await sendUnifiedRequest(
    url,
    requestBody as any,
    {
      ...config,
      headers,
      signal: clientSignal,
      httpsProxy,
      TIMEOUT: fastify.configService.get("TIMEOUT"),
    },
    { req }
  );

  if (typeof transformer.transformResponseOut === "function") {
    const shaped = await transformer.transformResponseOut(response, context as any);
    return {
      response: await reshapeFimResponseForClient(
        shaped,
        inboundKind,
        passthrough
      ),
      passthrough,
    };
  }

  return {
    response: await reshapeFimResponseForClient(
      response,
      inboundKind,
      passthrough
    ),
    passthrough,
  };
}

/**
 * Dedicated FIM pipeline — does not call prepareInboundRequest / chat Unified.
 */
export async function handleFimEndpoint(
  req: FastifyRequest,
  reply: FastifyReply,
  fastify: FastifyInstance,
  _ownerTransformer: Transformer,
  routePath: string
) {
  const disconnect = createClientDisconnectSignal(req, reply);
  const clientSignal = disconnect.signal;
  disconnect.arm();
  const latency = ensureRequestLatency(req as any);
  markLatency(latency, "bodyParsed");

  const pathname = (req as any).pathname || routePath;
  const match: ProtocolRouteMatch | null =
    (req as any).protocolMatch ||
    matchClientProtocol(req.method, pathname);

  if (!match || match.protocol !== "openai_fim_completions") {
    throw createApiError(
      "Not a FIM protocol route",
      500,
      "protocol_mismatch"
    );
  }

  const originalBody = req.body;
  const inboundKind: FimInboundKind = V1_FIM_INBOUND_KIND;
  let unified: UnifiedFimRequest;
  try {
    unified = inboundToUnifiedFim(originalBody, inboundKind);
  } catch (error: any) {
    attachLatencyMeta(latency, { error: error?.message || String(error) });
    emitLatencyRecord(req.log, latency);
    throw error;
  }

  const routed = resolveFimRouteModel(
    unified.model,
    fastify.configService
  );
  (req as any).scenarioType = routed.scenarioType;

  const { providerName, modelName } = resolveDestination(
    routed.model,
    "openai_fim_completions"
  );
  unified = { ...unified, model: modelName };

  const protocolContext = createClientProtocolContext(match, {
    originalModel:
      typeof (originalBody as any)?.model === "string"
        ? (originalBody as any).model
        : undefined,
    stream: unified.stream === true,
    scenarioType: "fim",
  });
  (req as any).protocolContext = protocolContext;
  (req as any).clientProtocol = "openai_fim_completions";

  markLatency(latency, "destinationPolicy");

  const tryProviders: Array<{ providerName: string; modelName: string }> = [
    { providerName, modelName },
  ];

  const fallbackConfig = fastify.configService.get<any>("fallback");
  const fallbackList = selectFallbackModels(fallbackConfig, "fim");
  if (Array.isArray(fallbackList)) {
    for (const entry of fallbackList) {
      const norm = normalizeModelSelector(String(entry)) || String(entry);
      if (!norm.includes(",")) continue;
      const [p, ...rest] = norm.split(",");
      const m = rest.join(",");
      if (p && m && !(p === providerName && m === modelName)) {
        tryProviders.push({ providerName: p, modelName: m });
      }
    }
  }

  let lastError: any;
  for (let i = 0; i < tryProviders.length; i++) {
    const dest = tryProviders[i];
    const provider = fastify.providerService.getProvider(dest.providerName);
    if (!provider) {
      lastError = createApiError(
        `Provider '${dest.providerName}' not found`,
        404,
        "provider_not_found"
      );
      continue;
    }

    const attemptUnified: UnifiedFimRequest = {
      ...unified,
      model: dest.modelName,
    };

    attachLatencyMeta(latency, {
      protocol: "openai_fim_completions",
      provider: dest.providerName,
      model: dest.modelName,
      scenario: "fim",
    });

    try {
      const { response } = await dispatchFimOnce(
        req,
        fastify,
        provider,
        attemptUnified,
        originalBody,
        clientSignal,
        inboundKind
      );
      markLatency(latency, "requestTransformers");
      emitLatencyRecord(req.log, latency);
      return await formatFimResponse(
        response,
        reply,
        attemptUnified.stream === true,
        clientSignal
      );
    } catch (error: any) {
      lastError = error;
      if (isClientAbortError(error) || clientSignal.aborted) {
        throw typeof error === "string"
          ? toClientAbortError(error)
          : toClientAbortError(clientSignal.reason ?? error);
      }
      if (
        i < tryProviders.length - 1 &&
        isFallbackEligibleError(error)
      ) {
        await delay(retryDelayAfterFailure(i));
        continue;
      }
      attachLatencyMeta(latency, {
        error: error?.message || String(error),
        cancelled: clientSignal.aborted,
      });
      emitLatencyRecord(req.log, latency);
      throw error;
    }
  }

  attachLatencyMeta(latency, {
    error: lastError?.message || String(lastError),
  });
  emitLatencyRecord(req.log, latency);
  throw lastError || createApiError("FIM request failed", 500, "fim_failed");
}

// Silence unused import warning if CLIENT_DISCONNECT_REASON unused
void CLIENT_DISCONNECT_REASON;

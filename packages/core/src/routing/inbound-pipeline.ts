import { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { createApiError } from "@/api/middleware";
import { Transformer } from "@/types/transformer";
import { UnifiedChatRequest } from "@/types/llm";
import {
  extractAndRemoveClaudeCodeSubagentModelTag,
  inspectClaudeCodeBillingSystemHeader,
  router,
} from "@/utils/router";
import { readFile } from "fs/promises";
import {
  adaptClientRequest,
  normalizeClientToUnified,
  shouldBypassTransformersProtocolAware,
} from "./protocol-adapter";
import {
  ClientProtocolContext,
  matchClientProtocol,
  ProtocolRouteMatch,
} from "./protocol-endpoints";
import { protocolErrorBody } from "./protocol-errors";

export interface PreparedInboundRequest {
  match: ProtocolRouteMatch;
  protocolContext: ClientProtocolContext;
  /** Original client wire body (preserved for Anthropic custom routers). */
  originalBody: any;
  /**
   * Client wire body after protocol adaptation and CCR-only cleanup
   * (subagent tag stripped, REWRITE_SYSTEM_PROMPT applied). This — not
   * originalBody — is what an exact-wire passthrough must send upstream.
   */
  clientWireBody: any;
  /** Normalized Unified body used for routing and provider conversion. */
  unifiedBody: UnifiedChatRequest;
  providerName: string;
  modelName: string;
}

/**
 * Stages 1–7 of the canonical inbound lifecycle:
 * detect → adapt → normalize → route → validate destination.
 */
export async function prepareInboundRequest(
  req: FastifyRequest,
  reply: FastifyReply,
  fastify: FastifyInstance,
  endpointTransformer: Transformer,
  routePath: string
): Promise<PreparedInboundRequest> {
  const pathname =
    (req as any).pathname ||
    String(req.url || routePath).split("?")[0] ||
    routePath;

  const match =
    (req as any).protocolMatch ||
    matchClientProtocol(req.method, pathname) ||
    matchClientProtocol("POST", routePath);

  if (!match) {
    throw createApiError(
      `No inbound protocol match for ${pathname}`,
      404,
      "protocol_not_found"
    );
  }

  // Set protocol identity before adaptation/normalization. Those stages can
  // throw validation errors, and the global error handler needs this value to
  // serialize the client protocol's envelope.
  (req as any).protocolMatch = match;
  (req as any).clientProtocol = match.protocol;

  if (match.ownerTransformerName !== endpointTransformer.name) {
    req.log?.warn?.(
      {
        expected: match.ownerTransformerName,
        actual: endpointTransformer.name,
        path: pathname,
      },
      "endpoint transformer name differs from protocol owner"
    );
  }

  const originalBody = req.body;
  const query = (req.query || {}) as Record<string, unknown>;
  const { normalizationInput, context } = adaptClientRequest(
    match,
    originalBody,
    query
  );

  if (match.protocol === "anthropic_messages") {
    await prepareAnthropicNormalizationInput(
      normalizationInput,
      context,
      fastify.configService
    );
  }

  const transformerContext = {
    req,
    signal: undefined,
    clientProtocol: match.protocol,
    protocolContext: context,
  };

  const unifiedBody = await normalizeClientToUnified(
    match.protocol,
    normalizationInput,
    endpointTransformer,
    transformerContext
  );

  const routingReq = req as any;
  (req as any).protocolContext = context;
  (req as any).unifiedBody = unifiedBody;
  (req as any).originalClientBody = originalBody;

  // Built-in routing consumes canonical Unified. Legacy Anthropic custom routers
  // still observe the immutable original wire body and can read req.unifiedBody.
  const customRouterPath = fastify.configService.get("CUSTOM_ROUTER_PATH");
  const previousBody = routingReq.body;
  routingReq.body =
    match.protocol === "anthropic_messages" && customRouterPath
      ? originalBody
      : unifiedBody;

  try {
    await router(routingReq, reply, {
      configService: fastify.configService,
      tokenizerService: fastify.tokenizerService,
    });
  } finally {
    routingReq.body = previousBody ?? originalBody;
  }

  const routedModel =
    typeof unifiedBody.model === "string"
      ? unifiedBody.model
      : context.originalModel;
  unifiedBody.model = routedModel || "";
  context.scenarioType = routingReq.scenarioType;
  context.originalModel =
    context.originalModel ??
    (typeof (originalBody as any)?.model === "string"
      ? (originalBody as any).model
      : undefined);

  routingReq.unifiedBody = unifiedBody;
  routingReq.protocolContext = context;

  const destination = resolveDestination(unifiedBody.model, match.protocol);
  const provider = fastify.providerService.getProvider(destination.providerName);
  if (!provider) {
    throwProtocolError(
      match.protocol,
      `Provider '${destination.providerName}' not found`,
      404,
      "provider_not_found"
    );
  }

  unifiedBody.model = destination.modelName;
  routingReq.provider = destination.providerName;
  routingReq.model = destination.modelName;
  context.stream =
    context.stream ||
    unifiedBody.stream === true ||
    (originalBody as any)?.stream === true;

  return {
    match,
    protocolContext: context,
    originalBody,
    clientWireBody: normalizationInput,
    unifiedBody,
    providerName: destination.providerName,
    modelName: destination.modelName,
  };
}

async function prepareAnthropicNormalizationInput(
  body: any,
  context: ClientProtocolContext,
  configService: any
): Promise<void> {
  const billing = inspectClaudeCodeBillingSystemHeader(body);
  context.claudeCodeSubagent = billing.isSubagent;
  context.taggedSubagentModel =
    extractAndRemoveClaudeCodeSubagentModelTag(body);
  context.anthropicCacheMode = "preserve";
  const rewritePrompt = configService.get("REWRITE_SYSTEM_PROMPT");
  const system = body?.system;
  if (
    rewritePrompt &&
    Array.isArray(system) &&
    system.length > 1 &&
    typeof system[1]?.text === "string" &&
    system[1].text.includes("<env>")
  ) {
    const prompt = await readFile(rewritePrompt, "utf-8");
    system[1].text = `${prompt}<env>${system[1].text.split("<env>").pop()}`;
  }
}

export function resolveDestination(
  model: string | undefined,
  protocol?: ClientProtocolContext["protocol"]
): { providerName: string; modelName: string } {
  const trimmed = typeof model === "string" ? model.trim() : "";
  if (!trimmed) {
    throwProtocolError(
      protocol,
      "Missing model. Configure Router.default or send provider,model.",
      400,
      "missing_model",
      "invalid_request_error"
    );
  }
  if (!trimmed.includes(",")) {
    throwProtocolError(
      protocol,
      `Unresolved bare model '${trimmed}'. Configure Router.default or send provider,model.`,
      400,
      "unresolved_model",
      "invalid_request_error"
    );
  }
  const [providerName, ...modelParts] = trimmed.split(",");
  const modelName = modelParts.join(",");
  if (!providerName || !modelName) {
    throwProtocolError(
      protocol,
      `Invalid model '${trimmed}'. Expected provider,model.`,
      400,
      "invalid_model",
      "invalid_request_error"
    );
  }
  return { providerName, modelName };
}

export function throwProtocolError(
  protocol: ClientProtocolContext["protocol"] | undefined,
  message: string,
  statusCode: number,
  code: string,
  type: string = "api_error"
): never {
  const shaped = protocolErrorBody(protocol, message, statusCode, code, type);
  const err = createApiError(message, shaped.statusCode, code, type);
  (err as any).protocolBody = shaped.body;
  throw err;
}

export function protocolAwareBypass(
  provider: any,
  transformer: Transformer,
  protocolContext: ClientProtocolContext | undefined,
  modelName: string | undefined
): boolean {
  if (!protocolContext) {
    return (
      provider.transformer?.use?.length === 1 &&
      provider.transformer.use[0].name === transformer.name &&
      (!provider.transformer?.[modelName || ""]?.use?.length ||
        (provider.transformer?.[modelName || ""]?.use.length === 1 &&
          provider.transformer?.[modelName || ""]?.use[0].name ===
            transformer.name))
    );
  }
  return shouldBypassTransformersProtocolAware(
    provider,
    transformer,
    protocolContext.protocol,
    modelName
  );
}

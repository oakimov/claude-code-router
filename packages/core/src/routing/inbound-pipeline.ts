import { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { createApiError } from "@/api/middleware";
import { Transformer } from "@/types/transformer";
import { UnifiedChatRequest } from "@/types/llm";
import {
  extractAndRemoveClaudeCodeSubagentModelTag,
  inspectClaudeCodeBillingSystemHeader,
  router,
} from "@/utils/router";
import {
  applyThirdPartyAnthropicPolicy,
  classifyAnthropicClient,
  getAnthropicProviderMode,
  inspectAnthropicClientFingerprint,
  isNativeAnthropicClient,
} from "@/utils/anthropic-client-policy";
import {
  adaptClientRequest,
  cloneProtocolBody,
  normalizeClientToUnified,
  shouldBypassTransformersProtocolAware,
} from "./protocol-adapter";
import {
  ClientProtocolContext,
  matchClientProtocol,
  ProtocolRouteMatch,
} from "./protocol-endpoints";
import { protocolErrorBody } from "./protocol-errors";
import { stripOneMillionContextMarker } from "@/utils/claude-model-catalog";
import { decodeClaudeModelAlias } from "@caeliq/ccr-shared";
import { applyReasoningAutoSummary } from "@/utils/reasoning-effort";

export interface PreparedInboundRequest {
  match: ProtocolRouteMatch;
  protocolContext: ClientProtocolContext;
  /** Original client wire body (preserved for Anthropic custom routers). */
  originalBody: any;
  /** Client wire body after protocol adaptation and CCR-only cleanup. */
  clientWireBody: any;
  /** Normalized Unified body used for routing and provider conversion. */
  unifiedBody: UnifiedChatRequest;
  /** Unified projection before destination-specific Anthropic emulation. */
  prePolicyUnifiedBody: UnifiedChatRequest;
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
    context.anthropicClientKind = classifyAnthropicClient(
      req.headers as Record<string, unknown>,
      normalizationInput
    );
    req.log?.debug?.(
      {
        anthropicClientKind: context.anthropicClientKind,
        anthropicClientSignals: inspectAnthropicClientFingerprint(
          req.headers as Record<string, unknown>,
          normalizationInput
        ),
      },
      "classified Anthropic client"
    );
    await prepareAnthropicNormalizationInput(
      normalizationInput,
      context
    );
  } else {
    // OpenAI protocols have no native Anthropic wire identity. If routed to an
    // in-scope Anthropic destination they must use the third-party emulation
    // profile, never native pass-through.
    context.anthropicClientKind = "other";
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

  // Opt-in readable thinking for every client → provider direction: stamp
  // Unified reasoning.summary so Responses/Codex/Anthropic/Gemini outbound
  // can request visible thought text when the client only sent effort.
  applyReasoningAutoSummary(
    unifiedBody,
    fastify.configService.get("REASONING_AUTO_SUMMARY")
  );

  unifiedBody.model = resolveConfiguredClaudeModelAlias(
    unifiedBody.model,
    (canonicalId) =>
      fastify.providerService.resolveModelRoute(canonicalId) !== null,
    match.protocol
  ) || "";

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
  unifiedBody.model =
    resolveConfiguredClaudeModelAlias(
      routedModel,
      (canonicalId) =>
        fastify.providerService.resolveModelRoute(canonicalId) !== null,
      match.protocol
    ) || "";
  context.scenarioType = routingReq.scenarioType;
  context.originalModel =
    context.originalModel ??
    (typeof (originalBody as any)?.model === "string"
      ? (originalBody as any).model
      : undefined);

  routingReq.unifiedBody = unifiedBody;
  routingReq.protocolContext = context;

  const destination = resolveDestination(unifiedBody.model, match.protocol);
  const normalizedContextModel = stripOneMillionContextMarker(
    destination.modelName
  );
  destination.modelName = normalizedContextModel.modelId;
  context.requestedOneMillion = normalizedContextModel.requestedOneMillion;
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
  const prePolicyUnifiedBody = cloneProtocolBody(unifiedBody);
  const providerMode = getAnthropicProviderMode(provider);
  context.anthropicProviderMode = providerMode;
  context.anthropicDestinationInScope =
    providerMode !== "out_of_scope";
  context.anthropicNativeWire =
    match.protocol === "anthropic_messages" &&
    context.anthropicDestinationInScope &&
    isNativeAnthropicClient(context.anthropicClientKind || "other");

  if (
    context.anthropicDestinationInScope &&
    context.anthropicClientKind === "other"
  ) {
    await applyThirdPartyAnthropicPolicy(
      unifiedBody,
      context,
      fastify.configService
    );
  }

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
    prePolicyUnifiedBody,
    providerName: destination.providerName,
    modelName: destination.modelName,
  };
}

async function prepareAnthropicNormalizationInput(
  body: any,
  context: ClientProtocolContext
): Promise<void> {
  const billing = inspectClaudeCodeBillingSystemHeader(body);
  context.claudeCodeSubagent = billing.isSubagent;
  context.taggedSubagentModel =
    extractAndRemoveClaudeCodeSubagentModelTag(body);
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

/**
 * Decode a discovery alias before scenario routing. Aliases are constrained to
 * configured canonical routes; ordinary `provider,model` ids are unchanged.
 */
export function resolveConfiguredClaudeModelAlias(
  model: string | undefined,
  isConfigured: (canonicalId: string) => boolean,
  protocol?: ClientProtocolContext["protocol"]
): string | undefined {
  if (typeof model !== "string") return model;
  const decoded = decodeClaudeModelAlias(model);
  if (!decoded) return model;

  const canonicalId = stripOneMillionContextMarker(decoded).modelId;
  if (!isConfigured(canonicalId)) {
    throwProtocolError(
      protocol,
      `Encoded model alias resolves to an unconfigured model '${canonicalId}'.`,
      404,
      "model_not_found",
      "invalid_request_error"
    );
  }
  return decoded;
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

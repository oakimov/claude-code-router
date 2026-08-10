import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import {
  RATE_LIMIT_CONFIG,
  canonicalClaudeModelId,
  encodeClaudeModelAlias,
  modelIdNeedsClaudeAlias,
} from "@caeliq/ccr-shared";
import {
  lookupNativeModelsDevModel,
  ModelsDevCatalogCache,
  type ModelsDevCatalogSource,
  type ModelsDevIndex,
  type ModelsDevModelInfo,
} from "../utils/models-dev";

/**
 * OpenAI-compatible model listing.
 *
 * This is a GET, so it is not part of `PROTOCOL_ROUTE_SPECS` (that table only
 * matches routed LLM POSTs). It is registered here next to the other non-routed
 * endpoints such as `/v1/messages/count_tokens`.
 *
 * Canonical ids that Claude clients would filter out are advertised as
 * reversible `claude-<hex>` aliases. Both forms remain valid inbound ids.
 */

interface ModelEntry {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
  display_name?: string;
  description?: string;
  anthropic_family_tier?: AnthropicFamilyTier;
  context_window?: number;
  max_input_tokens?: number;
  supports_1m?: boolean;
  max_output_tokens?: number;
  effort_levels?: string[];
}

type AnthropicFamilyTier = "sonnet" | "opus" | "haiku" | "fable" | "mythos";
export type ModelIdOutput = "literal" | "masked";

export interface ModelsRouteOptions {
  catalog?: ModelsDevCatalogSource;
}

/**
 * CCR has no real per-model creation time, and a per-boot value would make
 * otherwise identical responses differ between restarts.
 */
const CREATED = 0;

export function resolveModelIdOutput(value: unknown): ModelIdOutput {
  return value === "masked" ? "masked" : "literal";
}

function configuredModelIdOutput(config: any): ModelIdOutput {
  return resolveModelIdOutput(
    config?.initialConfig?.MODEL_ID_OUTPUT ?? config?.MODEL_ID_OUTPUT
  );
}

function advertisedModelId(
  canonicalId: string,
  output: ModelIdOutput
): string {
  return output === "masked" && modelIdNeedsClaudeAlias(canonicalId)
    ? encodeClaudeModelAlias(canonicalId)
    : canonicalId;
}

interface ProviderLike {
  name?: string;
  models?: unknown;
}

function directAnthropicTier(info: ModelsDevModelInfo): AnthropicFamilyTier | undefined {
  if (info.nativeProvider !== "anthropic") return undefined;
  for (const tier of ["sonnet", "opus", "haiku", "fable", "mythos"] as const) {
    if (info.family.includes(tier) || info.id.toLowerCase().includes(tier)) return tier;
  }
  return undefined;
}

/** Map non-Anthropic models into the family tiers Claude Desktop accepts. */
export function inferAnthropicFamilyTier(
  info: ModelsDevModelInfo
): AnthropicFamilyTier | undefined {
  const directTier = directAnthropicTier(info);
  if (directTier) return directTier;
  if (info.id.toLowerCase() === "gpt-5.6-sol") return "fable";
  if (!info.reasoning) return "haiku";
  if (!info.context) return undefined;
  return info.context > 300_000 ? "opus" : "sonnet";
}

function enrichModelEntry(
  entry: ModelEntry,
  modelName: string,
  modelsDev: ModelsDevIndex | null
): ModelEntry {
  const info = lookupNativeModelsDevModel(modelsDev, modelName);
  if (!info) return entry;

  const enriched: ModelEntry = { ...entry };
  if (info.name) enriched.display_name = info.name;
  if (info.description) enriched.description = info.description;
  const familyTier = inferAnthropicFamilyTier(info);
  if (familyTier) enriched.anthropic_family_tier = familyTier;
  if (info.context) {
    enriched.context_window = info.context;
    enriched.max_input_tokens = info.input || info.context;
    enriched.supports_1m = info.context >= 1_000_000;
  }
  if (info.output) enriched.max_output_tokens = info.output;
  if (info.effortLevels.length) enriched.effort_levels = info.effortLevels;
  return enriched;
}

function readProviders(config: any): ProviderLike[] {
  if (Array.isArray(config)) return config as ProviderLike[];
  // `initialConfig.providers` is what the running server was booted with
  // (see packages/server/src/index.ts), so it reflects what the router can
  // actually reach. The other spellings are defensive.
  const candidates = [
    config?.initialConfig?.providers,
    config?.initialConfig?.Providers,
    config?.Providers,
    config?.providers,
  ];
  for (const candidate of candidates) {
    if (Array.isArray(candidate)) return candidate as ProviderLike[];
  }
  return [];
}

export function listModels(
  config: any,
  modelsDev: ModelsDevIndex | null = null,
  modelIdOutput: ModelIdOutput = configuredModelIdOutput(config)
): ModelEntry[] {
  const entries: ModelEntry[] = [];
  const seen = new Set<string>();

  for (const provider of readProviders(config)) {
    const providerName =
      typeof provider?.name === "string" ? provider.name.trim() : "";
    if (!providerName || !Array.isArray(provider.models)) continue;

    for (const model of provider.models) {
      if (typeof model !== "string") continue;
      const modelName = model.trim();
      if (!modelName) continue;

      const canonicalId = `${providerName},${modelName}`;
      if (seen.has(canonicalId)) continue;
      seen.add(canonicalId);

      entries.push(
        enrichModelEntry(
          {
            id: advertisedModelId(canonicalId, modelIdOutput),
            object: "model",
            created: CREATED,
            owned_by: providerName,
          },
          modelName,
          modelsDev
        )
      );
    }
  }

  return entries;
}

export async function registerModelsRoutes(
  app: FastifyInstance,
  config: any,
  options: ModelsRouteOptions = {}
): Promise<void> {
  const rateLimitOptions = {
    config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
  };

  const catalog =
    options.catalog ||
    new ModelsDevCatalogCache({
      onError: (error) => {
        app.log.warn(
          { error: error instanceof Error ? error.message : String(error) },
          "models.dev lookup failed; serving model discovery without refreshed metadata"
        );
      },
    });

  const currentModels = async () => {
    const runtimeProviders = app._server?.providerService?.getProviders();
    const configuredOutput =
      app._server?.configService?.get?.("MODEL_ID_OUTPUT") ??
      config?.initialConfig?.MODEL_ID_OUTPUT ??
      config?.MODEL_ID_OUTPUT;
    return listModels(
      runtimeProviders ?? config,
      await catalog.get(),
      resolveModelIdOutput(configuredOutput)
    );
  };

  const handleList = async (_req: FastifyRequest, reply: FastifyReply) => {
    reply.type("application/json").send({
      object: "list",
      data: await currentModels(),
    });
  };

  app.get("/v1/models", rateLimitOptions, handleList);
  app.get("/models", rateLimitOptions, handleList);

  app.get(
    "/v1/models/:id",
    rateLimitOptions,
    async (req: FastifyRequest, reply: FastifyReply) => {
      const { id } = req.params as { id: string };
      const canonicalRequestedId = canonicalClaudeModelId(id);
      const match = (await currentModels()).find(
        (entry) => canonicalClaudeModelId(entry.id) === canonicalRequestedId
      );

      if (!match) {
        reply.status(404).type("application/json").send({
          error: {
            message: `The model '${id}' does not exist.`,
            type: "invalid_request_error",
            code: "model_not_found",
          },
        });
        return;
      }

      reply.type("application/json").send(match);
    }
  );
}

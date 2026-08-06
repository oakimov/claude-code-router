import { FastifyInstance, FastifyRequest, FastifyReply } from "fastify";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";

/**
 * OpenAI-compatible model listing.
 *
 * This is a GET, so it is not part of `PROTOCOL_ROUTE_SPECS` (that table only
 * matches routed LLM POSTs). It is registered here next to the other non-routed
 * endpoints such as `/v1/messages/count_tokens`.
 *
 * Ids are emitted in CCR's native `provider,model` wire format so a listed id
 * can be sent straight back to `/v1/responses` or `/v1/chat/completions`
 * without any translation.
 */

interface ModelEntry {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

/**
 * CCR has no real per-model creation time, and a per-boot value would make
 * otherwise identical responses differ between restarts.
 */
const CREATED = 0;

interface ProviderLike {
  name?: string;
  models?: unknown;
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

export function listModels(config: any): ModelEntry[] {
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

      const id = `${providerName},${modelName}`;
      if (seen.has(id)) continue;
      seen.add(id);

      entries.push({
        id,
        object: "model",
        created: CREATED,
        owned_by: providerName,
      });
    }
  }

  return entries;
}

export async function registerModelsRoutes(
  app: FastifyInstance,
  config: any
): Promise<void> {
  const rateLimitOptions = {
    config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
  };

  const currentModels = () => {
    const runtimeProviders = app._server?.providerService?.getProviders();
    return listModels(runtimeProviders ?? config);
  };

  const handleList = async (_req: FastifyRequest, reply: FastifyReply) => {
    reply.type("application/json").send({
      object: "list",
      data: currentModels(),
    });
  };

  app.get("/v1/models", rateLimitOptions, handleList);
  app.get("/models", rateLimitOptions, handleList);

  app.get(
    "/v1/models/:id",
    rateLimitOptions,
    async (req: FastifyRequest, reply: FastifyReply) => {
      const { id } = req.params as { id: string };
      const match = currentModels().find((entry) => entry.id === id);

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

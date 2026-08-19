import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";
import {
  getAntigravityAccessToken,
  getClaudeAccessToken,
  getCodexAccessToken,
  getQwenAccessToken,
  getXaiAccessToken,
} from "@caeliq/llms";
import {
  codexPatForProvider,
  findProvider,
  oauthKindForProvider,
  type OAuthKind,
} from "../debug/model";

const rateLimitOptions = {
  config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
};

export async function forceRefreshOAuth(kind: OAuthKind): Promise<void> {
  switch (kind) {
    case "claude-auth":
      await getClaudeAccessToken({ force: true });
      return;
    case "codex":
      await getCodexAccessToken({ force: true });
      return;
    case "qwen-auth":
      await getQwenAccessToken({ force: true });
      return;
    case "antigravity-auth":
      await getAntigravityAccessToken({ force: true });
      return;
    case "xai-auth":
      await getXaiAccessToken({ force: true });
      return;
  }
}

export async function registerOAuthRefreshRoutes(
  app: FastifyInstance,
  getConfig: () => any
): Promise<void> {
  app.post(
    "/api/oauth/refresh",
    rateLimitOptions,
    async (req: FastifyRequest, reply: FastifyReply) => {
      const name = String((req.body as any)?.provider || "").trim();
      if (!name) {
        return reply.status(400).send({ error: "provider is required" });
      }
      const provider = findProvider(getConfig(), name);
      if (!provider) {
        return reply.status(400).send({ error: `Unknown provider "${name}"` });
      }
      const kind = oauthKindForProvider(provider);
      if (!kind) {
        if (codexPatForProvider(provider)) {
          return reply.status(400).send({
            error:
              "Codex PAT credentials cannot be renewed via OAuth. Use CCR mode with the PAT, or switch this provider to Codex OAuth.",
          });
        }
        return reply.status(400).send({
          error: `Provider "${name}" does not use refreshable OAuth`,
        });
      }
      try {
        await forceRefreshOAuth(kind);
        return { success: true, provider: name, kind };
      } catch (error: any) {
        return reply.status(400).send({
          error: error?.message || "OAuth refresh failed",
        });
      }
    }
  );
}

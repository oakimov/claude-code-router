import type { FastifyInstance } from "fastify";
import { RATE_LIMIT_CONFIG, checkForUpdates, performUpdate } from "@caeliq/ccr-shared";
import { version as productVersion } from "../../package.json";

const rateLimitOptions = {
  config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
};

export async function registerUpdateRoutes(app: FastifyInstance): Promise<void> {
  app.get("/api/update/check", rateLimitOptions, async () => {
    return checkForUpdates(productVersion);
  });

  app.post("/api/update/perform", rateLimitOptions, async () => {
    return performUpdate();
  });
}

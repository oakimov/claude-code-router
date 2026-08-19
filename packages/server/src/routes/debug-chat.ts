import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { RATE_LIMIT_CONFIG } from "@caeliq/ccr-shared";
import {
  parseDebugChatBody,
  sendWebResponse,
  streamDebugChat,
} from "../debug/ai-sdk-agent";
import { executeDebugRequest } from "../debug/model";
import type { InboundProtocol } from "../debug/types";

const rateLimitOptions = {
  config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
};

export async function registerDebugChatRoutes(
  app: FastifyInstance,
  getConfig: () => any
): Promise<void> {
  app.post(
    "/api/debug/chat",
    rateLimitOptions,
    async (req: FastifyRequest, reply: FastifyReply) => {
      try {
        const input = parseDebugChatBody(req.body);
        if (!input.provider) {
          return reply.status(400).send({ error: "provider is required" });
        }
        if (!input.model) {
          return reply.status(400).send({ error: "model is required" });
        }
        const controller = new AbortController();
        reply.raw.once("close", () => controller.abort());
        const webResponse = await streamDebugChat(
          input,
          getConfig(),
          controller.signal
        );
        return sendWebResponse(reply, webResponse);
      } catch (error: any) {
        const message = error?.message || "Debug chat failed";
        req.log?.error({ err: error }, "Debug chat failed");
        if (!reply.sent) {
          return reply.status(400).send({
            error: message,
            status: 400,
            headers: {},
            body: JSON.stringify({ error: message }, null, 2),
          });
        }
      }
    }
  );

  app.post(
    "/api/debug/request",
    rateLimitOptions,
    async (req: FastifyRequest, reply: FastifyReply) => {
      try {
        const body = (req.body || {}) as Record<string, any>;
        const provider = String(body.provider || "").trim();
        if (!provider) {
          return reply.status(400).send({ error: "provider is required" });
        }
        const protocolRaw = String(body.protocol || "chat_completions");
        const protocol: InboundProtocol =
          protocolRaw === "messages" || protocolRaw === "responses"
            ? protocolRaw
            : "chat_completions";
        const result = await executeDebugRequest(
          {
            target: body.target === "direct" ? "direct" : "ccr",
            protocol,
            provider,
            headers:
              body.headers && typeof body.headers === "object" && !Array.isArray(body.headers)
                ? body.headers
                : undefined,
            body: body.body ?? {},
          },
          getConfig()
        );
        let responseBody = result.body;
        try {
          responseBody = JSON.stringify(JSON.parse(result.body), null, 2);
        } catch {
          // SSE and other non-JSON bodies stay as received.
        }
        return {
          status: result.status,
          headers: result.headers,
          body: responseBody,
        };
      } catch (error: any) {
        const message = error?.message || "Debug request failed";
        req.log?.error({ err: error }, "Debug request failed");
        if (!reply.sent) {
          return reply.status(400).send({
            error: message,
            status: 400,
            headers: {},
            body: JSON.stringify({ error: message }, null, 2),
          });
        }
      }
    }
  );
}

import { FastifyRequest, FastifyReply } from "fastify";
import { matchClientProtocol, protocolErrorBody } from "@caeliq/llms";
import { apiKeysMatch, hasValidUiSession } from "../auth/ui-session";

function sendAuthError(
  req: FastifyRequest,
  reply: FastifyReply,
  message: string,
  statusCode: number,
  code: string
) {
  const protocol = (req as any).clientProtocol;
  const shaped = protocolErrorBody(
    protocol,
    message,
    statusCode,
    code,
    statusCode === 403 ? "permission_error" : "authentication_error"
  );
  reply.type("application/json");
  reply.status(shaped.statusCode).send(shaped.body);
}

function firstHeader(
  value: string | string[] | undefined
): string {
  if (Array.isArray(value)) return value[0] || "";
  return value || "";
}

export const apiKeyAuth =
  (config: any) =>
    async (req: FastifyRequest, reply: FastifyReply, done: () => void) => {
      // Public endpoints that don't require authentication
      const publicPaths = ["/", "/health", "/api/auth/login", "/callback", "/auth/callback", "/oauth-callback", "/qwen/auth", "/qwen/forget", "/qwen/status"];
      // Match on the path alone — OAuth callbacks arrive as
      // /oauth-callback?code=…&state=… — and keep /oauth-callback an exact match
      // so a lookalike path cannot slip past the API key check.
      const path = req.url.split("?")[0];
      if (
        publicPaths.includes(path) ||
        path.startsWith("/ui") ||
        path.startsWith("/auth") ||
        path.startsWith("/qwen/") ||
        path.startsWith("/callback")
      ) {
        return done();
      }

      // Reject query-string API keys — they leak into URLs and logs.
      const url = new URL(`http://127.0.0.1${req.url}`);
      const hasQueryApiKey =
        url.searchParams.has("key") ||
        url.searchParams.has("api_key") ||
        url.searchParams.has("apiKey");
      if ((req as any).clientProtocol && hasQueryApiKey) {
        sendAuthError(
          req,
          reply,
          "API keys in query strings are not supported",
          400,
          "query_api_key_rejected"
        );
        return;
      }

      // Check if Providers is empty or not configured
      const providers = config.Providers || config.providers || [];
      if (!providers || providers.length === 0) {
        // No providers configured, skip authentication
        return done();
      }

      const apiKey = config.APIKEY;
      if (!apiKey) {
        // If no API key is set, enable CORS for local
        const allowedOrigins = [
          `http://127.0.0.1:${config.PORT || 3456}`,
          `http://localhost:${config.PORT || 3456}`,
        ];
        if (req.headers.origin && !allowedOrigins.includes(req.headers.origin)) {
          sendAuthError(
            req,
            reply,
            "CORS not allowed for this origin",
            403,
            "cors_denied"
          );
          return;
        } else {
          reply.header('Access-Control-Allow-Origin', `http://127.0.0.1:${config.PORT || 3456}`);
          reply.header('Access-Control-Allow-Origin', `http://localhost:${config.PORT || 3456}`);
        }
        return done();
      }

      const authHeaderValue =
        req.headers.authorization ||
        req.headers["x-api-key"];
      const authKey = firstHeader(authHeaderValue as string | string[] | undefined);
      if (authKey) {
        const token = authKey.startsWith("Bearer ")
          ? authKey.slice("Bearer ".length)
          : authKey;
        if (!apiKeysMatch(token, apiKey)) {
          sendAuthError(req, reply, "Invalid API key", 401, "invalid_api_key");
          return;
        }
        done();
        return;
      }

      if (hasValidUiSession(req)) {
        const method = req.method.toUpperCase();
        if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
          const origin = req.headers.origin;
          if (!origin) {
            sendAuthError(
              req,
              reply,
              "Origin required for session request",
              403,
              "origin_required"
            );
            return;
          }
          try {
            if (new URL(origin).host !== req.host) {
              sendAuthError(
                req,
                reply,
                "Cross-origin session request denied",
                403,
                "cors_denied"
              );
              return;
            }
          } catch {
            sendAuthError(
              req,
              reply,
              "Invalid origin for session request",
              403,
              "invalid_origin"
            );
            return;
          }
        }
        done();
        return;
      }

      sendAuthError(req, reply, "APIKEY is missing", 401, "missing_api_key");
    };

/** Detect inbound client protocol before auth replies. */
export function detectClientProtocol(req: FastifyRequest) {
  const pathname =
    (req as any).pathname ||
    String(req.url || "").split("?")[0] ||
    "/";
  const match = matchClientProtocol(req.method, pathname);
  if (match) {
    (req as any).protocolMatch = match;
    (req as any).clientProtocol = match.protocol;
  }
}

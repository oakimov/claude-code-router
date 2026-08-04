import { FastifyRequest, FastifyReply } from "fastify";
import {
  sanitizeErrorForLog,
  sanitizeUpstreamErrorText,
} from "../utils/redact";
import { isClientAbortError } from "../utils/retry";
import { protocolErrorBody } from "../routing/protocol-errors";

export interface ApiError extends Error {
  statusCode?: number;
  code?: string;
  type?: string;
  headers?: Record<string, string>;
}

export function createApiError(
  message: string,
  statusCode: number = 500,
  code: string = "internal_error",
  type: string = "api_error",
  headers?: Record<string, string>
): ApiError {
  const error = new Error(sanitizeUpstreamErrorText(message) || message) as ApiError;
  error.statusCode = statusCode;
  error.code = code;
  error.type = type;
  if (headers) error.headers = headers;
  return error;
}

export async function errorHandler(
  error: ApiError,
  request: FastifyRequest,
  reply: FastifyReply
) {
  // Client disconnects are expected; avoid noisy error responses/logs.
  if (isClientAbortError(error) || reply.raw.destroyed || reply.sent) {
    request.log.debug(sanitizeErrorForLog(error), "request aborted or already closed");
    if (!reply.sent && !reply.raw.destroyed && !reply.raw.headersSent) {
      // Must reset Content-Type: formatResponse may already have set
      // text/event-stream, and Fastify rejects object payloads for that type.
      reply.type("application/json");
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

  request.log.error(sanitizeErrorForLog(error), "request error");

  const statusCode = error.statusCode || 500;
  const message =
    sanitizeUpstreamErrorText(error.message || "Internal error") ||
    "Internal error";
  const code = error.code || "internal_error";
  const type =
    error.type ||
    (statusCode >= 400 && statusCode < 500
      ? "invalid_request_error"
      : "api_error");

  // Prefer an explicit protocol body attached by throwProtocolError; then a
  // structured upstream error, reshaped into the caller's wire envelope;
  // otherwise shape from the detected client protocol (or Anthropic default).
  const protocol = (request as any).clientProtocol;
  let protocolBody: Record<string, unknown> | undefined =
    (error as any).protocolBody;

  const upstreamBody = (error as any).upstream?.body;
  if (!protocolBody && upstreamBody && typeof upstreamBody === "object") {
    const upstreamError = (upstreamBody as any).error;
    if (
      (protocol === "anthropic_messages" || !protocol) &&
      (upstreamBody as any).type === "error" &&
      upstreamError &&
      typeof upstreamError === "object"
    ) {
      // Native Anthropic wire error: keep the envelope so clients recognize
      // e.g. prompt_too_long and the long-context credit rejections.
      protocolBody = {
        type: "error",
        error: {
          type:
            typeof upstreamError.type === "string" ? upstreamError.type : type,
          message:
            typeof upstreamError.message === "string"
              ? upstreamError.message
              : message,
          ...(typeof upstreamError.code === "string"
            ? { code: upstreamError.code }
            : {}),
        },
      };
    } else if (
      (protocol === "openai_chat_completions" ||
        protocol === "openai_responses") &&
      upstreamError &&
      typeof upstreamError === "object"
    ) {
      // OpenAI envelope: retain a meaningful upstream code (e.g.
      // context_length_exceeded) when the provider supplied one.
      protocolBody = {
        error: {
          message:
            typeof upstreamError.message === "string"
              ? upstreamError.message
              : message,
          type:
            typeof upstreamError.type === "string" ? upstreamError.type : type,
          param: upstreamError.param ?? null,
          code:
            typeof upstreamError.code === "string"
              ? upstreamError.code
              : code,
        },
      };
    }
  }

  if (!protocolBody) {
    protocolBody = protocolErrorBody(
      protocol,
      message,
      statusCode,
      code,
      type
    ).body;
  }

  if (error.headers) {
    reply.headers(error.headers);
  }

  // Reset Content-Type to application/json to prevent "invalid payload type" errors
  // when the reply previously had Content-Type set to a non-JSON value (e.g., text/event-stream
  // from a streaming response that failed before the stream was sent).
  reply.type("application/json");

  return reply.code(statusCode).send(protocolBody);
}

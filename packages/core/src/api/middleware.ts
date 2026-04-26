import { FastifyRequest, FastifyReply } from "fastify";

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
  const error = new Error(message) as ApiError;
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
  request.log.error(error);

  const statusCode = error.statusCode || 500;
  const response = {
    error: {
      message: error.message + (error.stack ? "\n" + error.stack : ""),
      type: error.type || "api_error",
      code: error.code || "internal_error",
    },
  };

  if (error.headers) {
    reply.headers(error.headers);
  }

  return reply.code(statusCode).send(response);
}

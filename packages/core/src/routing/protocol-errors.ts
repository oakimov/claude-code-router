import type { ClientProtocol } from "./protocol-endpoints";

export interface ProtocolErrorBody {
  statusCode: number;
  body: Record<string, unknown>;
}

/**
 * Build a client-protocol-shaped error envelope for pre-provider failures
 * (auth, validation, missing Router.default, etc.).
 */
export function protocolErrorBody(
  protocol: ClientProtocol | undefined,
  message: string,
  statusCode: number,
  code: string,
  type: string = "invalid_request_error"
): ProtocolErrorBody {
  switch (protocol) {
    case "openai_chat_completions":
    case "openai_responses":
    case "openai_fim_completions":
      return {
        statusCode,
        body: {
          error: {
            message,
            type,
            param: null,
            code,
          },
        },
      };
    case "anthropic_messages":
    default:
      return {
        statusCode,
        body: {
          error: {
            message,
            type: type === "invalid_request_error" ? "invalid_request_error" : type,
            code,
          },
          type: "error",
        },
      };
  }
}

import type { UnifiedChatRequest } from "@/types/llm";
import type { Transformer, TransformerOptions } from "@/types/transformer";
import { runCursor, type CursorSdkRunnerOptions } from "@/cursor-sdk/runner";
import {
  CURSOR_SDK_TRANSFORMER_NAME,
  DEFAULT_CURSOR_MODE,
  type CursorSdkMode,
} from "@/cursor-sdk/shared";
import {
  stripMessagesCacheControl,
  stripToolsCacheControl,
} from "../utils/cacheControl";

export interface CursorSdkTransformerOptions extends TransformerOptions {
  cursorMode?: CursorSdkMode;
  cursorCwd?: string;
  /** Opt-in only; forced off in Docker. Default false. */
  sandboxEnabled?: boolean;
}

/**
 * Thin CCR transformer that owns the full upstream call via @cursor/sdk.
 * Returns OpenAI-compatible SSE/JSON through config.__providerResponse so
 * AnthropicTransformer can convert to Claude Code's wire format unchanged.
 */
export class CursorSdkTransformer implements Transformer {
  static TransformerName = CURSOR_SDK_TRANSFORMER_NAME;

  name = CURSOR_SDK_TRANSFORMER_NAME;
  logger?: any;

  private options: CursorSdkTransformerOptions;

  constructor(options: CursorSdkTransformerOptions = {}) {
    this.options = options;
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: any,
    context?: any
  ): Promise<Record<string, any>> {
    const cursorMode =
      (this.options.cursorMode as CursorSdkMode) || DEFAULT_CURSOR_MODE;

    const runnerOptions: CursorSdkRunnerOptions = {
      cursorMode,
      cursorCwd: this.options.cursorCwd,
      sandboxEnabled: this.options.sandboxEnabled,
      abortSignal: context?.signal,
      logger: this.logger,
    };

    const nativeRequest = {
      ...request,
      messages: stripMessagesCacheControl(request.messages),
      tools: stripToolsCacheControl(request.tools),
    };
    const response = await runCursor(
      nativeRequest,
      provider,
      context,
      runnerOptions
    );

    // Placeholder URL — never used because __providerResponse short-circuits.
    return {
      body: nativeRequest,
      config: {
        url: provider?.baseUrl || "https://cursor.com",
        headers: {},
        __providerResponse: response,
      },
    };
  }

  async transformResponseOut(response: Response): Promise<Response> {
    // Already OpenAI chat.completion(.chunk) shape for AnthropicTransformer.
    return response;
  }
}

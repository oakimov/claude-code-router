import { LLMProvider, UnifiedChatRequest } from "./llm";
import type { UnifiedRequestRuntime } from "./turn-intent";

export interface TransformerOptions {
  [key: string]: any;
}

interface TransformerWithStaticName {
  new (options?: TransformerOptions): Transformer;
  TransformerName?: string;
}

export type TransformerConstructor = TransformerWithStaticName;

export interface TransformerContext {
  req?: any;
  provider?: any;
  signal?: AbortSignal;
  /** Protocol semantics that must not be serialized into the provider body. */
  unifiedRequest?: UnifiedRequestRuntime;
  /**
   * Set by claude-auth's transformRequestIn (non-Claude-Code branch) so that
   * AnthropicTransformer.transformRequestIn — which owns building the wire
   * body — can apply claude-auth's catalog-driven capability clamping and
   * synthesized user_id metadata immediately after building it. Keeps model
   * capability/identity-synthesis policy owned by claude-auth while
   * AnthropicTransformer remains the sole body-shape/timing owner.
   */
  claudeAuthPostBuildHook?: (anthropicBody: Record<string, any>) => void;
  [key: string]: any;
}

/** Where a request transformer runs in the compiled provider/model plan. */
export type TransformerRequestPhase = "body" | "headers" | "transport";

export type Transformer = {
  transformRequestIn?: (
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context: TransformerContext,
  ) => Promise<Record<string, any>>;
  transformResponseIn?: (response: Response, context?: TransformerContext) => Promise<Response>;

  // Convert request format to generic format
  transformRequestOut?: (request: any, context: TransformerContext) => Promise<UnifiedChatRequest>;
  // Convert response format to generic format
  transformResponseOut?: (response: Response, context: TransformerContext) => Promise<Response>;

  endPoint?: string;
  name?: string;
  /**
   * When true, this transformer performs the upstream call (fetch, SDK, etc.)
   * and returns the Response via config.__providerResponse. The request plan
   * runs at most one such transformer, after every body/header transform.
   */
  ownsTransport?: boolean;
  /** Optional phase hint; ownsTransport implies "transport". */
  requestPhase?: TransformerRequestPhase;
  auth?: (request: any, provider: LLMProvider, context: TransformerContext) => Promise<any>;
  
  // Logger for transformer
  logger?: any;
};

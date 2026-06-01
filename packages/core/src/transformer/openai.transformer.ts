import { Transformer } from "@/types/transformer";

/**
 * Server-side route handler for the OpenAI Chat Completions API.
 *
 * ## How endPoint works
 *
 * At startup, `registerApiRoutes` (see `api/routes.ts`) scans all registered
 * transformers for ones that define `endPoint`. For each, it registers a
 * `POST` route at that path. When a request hits the route,
 * `handleTransformerEndpoint` is invoked with the matching transformer as
 * the "endpoint transformer" — the one responsible for converting between
 * the external wire format and the internal Unified format.
 *
 * ## Why this class has no transform methods
 *
 * The Unified format IS the OpenAI Chat Completions format. The conversion
 * from Anthropic → Unified already happened in
 * `AnthropicTransformer.transformRequestOut()` (which runs first in the
 * pipeline). So by the time the provider chain executes, the body is already
 * in the right shape — no further conversion is needed.
 *
 * When this transformer appears in a provider's `transformer.use[]`,
 * `processRequestTransformers` skips it because it has no
 * `transformRequestIn`. Its role is purely to register the
 * `/v1/chat/completions` route for direct Chat Completions callers.
 *
 * ## Relationship to OpenAIResponsesTransformer
 *
 * `OpenAIResponsesTransformer` (in `openai.responses.transformer.ts`) is the
 * counterpart for the Responses API (`/v1/responses`). Unlike this
 * transformer, it defines `transformRequestIn` / `transformResponseOut`
 * because the Responses API uses a different wire format (e.g. `messages`
 * → `input`, function tools → flat tool definitions). It also uses the
 * shared utilities in `openai.util.ts` (`validateOpenAIToolCalls`,
 * `injectPromptCaching`) to sanitize the Unified body before converting it.
 *
 * ## Full request pipeline (for context)
 *
 *     Client → POST /v1/messages
 *       → AnthropicTransformer.transformRequestOut()        // Anthropic → Unified (OpenAI)
 *       → provider.transformer.use[].transformRequestIn()   // provider middleware
 *       → sendRequestToProvider()                           // HTTP call upstream
 *       → provider.transformer.use[].transformResponseOut() // provider middleware (reversed)
 *       → AnthropicTransformer.transformResponseIn()        // Unified (OpenAI) → Anthropic
 *       → Client
 */
export class OpenAITransformer implements Transformer {
  name = "OpenAI";
  endPoint = "/v1/chat/completions";
}

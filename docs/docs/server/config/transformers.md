---
sidebar_position: 4
---

# Transformers

Transformers are the core mechanism for adapting API differences between LLM providers. They convert requests and responses between different formats, handle authentication, and manage provider-specific features.

## Understanding Transformers

### What is a Transformer?

A transformer is a plugin that can:
- **Own a client protocol** (register a route and convert client wire ↔ Unified)
- **Adapt provider wire** (Unified ↔ provider-specific request/response)
- **Handle authentication** for provider APIs
- **Modify requests** to add or adjust parameters

Most entries in `Providers[].transformer.use` are provider middleware. Four
transformers are **protocol owners** — they register the inbound HTTP routes
and are not listed in `transformer.use` for that purpose:

| Protocol | Owner | Canonical path | Alias(es) |
|----------|-------|----------------|-----------|
| Anthropic Messages | `Anthropic` | `/v1/messages` | — |
| OpenAI Chat Completions | `OpenAI` | `/v1/chat/completions` | `/chat/completions` |
| OpenAI Responses | `openai-responses` | `/v1/responses` | `/responses` |
| FIM Completions | `Fim` | `/v1/fim/completions` | `/fim/completions` |

Chat protocols (Messages, Chat Completions, Responses) share one Unified chat
pipeline. FIM uses a **separate** Unified FIM pipeline and `fim.*` provider
transformers — see [FIM transformers](#fim-transformers) and the
[FIM Completions API](/docs/server/api/fim-completions-api).

### Data Flow (chat protocols)

```
┌──────────────────────┐
│ Incoming client wire │  Messages | Chat Completions | Responses
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Protocol owner: transformRequestOut     │  Client wire → UnifiedChatRequest
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  UnifiedChatRequest                      │  Routing + provider chain input
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  provider.use[]: transformRequestIn      │  Unified → provider wire
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Provider API call                       │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  provider.use[]: transformResponseOut    │  Provider response → Unified-shaped
│  (reversed order)                        │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  Protocol owner: transformResponseIn     │  Unified → same client protocol
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│ Outgoing client wire │  Matches the inbound protocol
└──────────────────────┘
```

Same-protocol / native-wire paths may keep the original client body for
egress (for example native Claude Desktop/CLI to Anthropic). Cross-protocol
routes always normalize through Unified, then rebuild the client response in
the inbound protocol.

### Transformer Interface

All transformers implement the following interface. Method names are relative
to the **provider port**: `*In` faces the upstream; `*Out` faces away from it.
On a **protocol owner**, `transformRequestOut` / `transformResponseIn` are the
client-facing legs (client → Unified → client).

```typescript
interface Transformer {
  // Unified → provider-specific request (provider middleware / egress owner)
  transformRequestIn?: (
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context: TransformerContext
  ) => Promise<Record<string, any>>;

  // Client wire → Unified (protocol owner), or rare provider→Unified helpers
  transformRequestOut?: (
    request: any,
    context: TransformerContext
  ) => Promise<UnifiedChatRequest>;

  // Unified → client wire (protocol owner)
  transformResponseIn?: (
    response: Response,
    context?: TransformerContext
  ) => Promise<Response>;

  // Provider response → Unified-shaped (provider middleware, reversed)
  transformResponseOut?: (
    response: Response,
    context: TransformerContext
  ) => Promise<Response>;

  // Registers a client route when this transformer is a protocol owner
  endPoint?: string;

  // Transformer name (for custom transformers)
  name?: string;

  // Custom authentication handler (optional)
  auth?: (
    request: any,
    provider: LLMProvider,
    context: TransformerContext
  ) => Promise<any>;

  // Logger instance (auto-injected)
  logger?: any;
}
```

### Key Types

#### UnifiedChatRequest

```typescript
interface UnifiedChatRequest {
  messages: UnifiedMessage[];
  model: string;
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  tools?: UnifiedTool[];
  tool_choice?: any;
  reasoning?: {
    effort?: ThinkLevel;  // "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "ultra"
    max_tokens?: number;
    enabled?: boolean;
  };
}
```

CCR normalizes effort across the three **chat** inbound APIs. Chat Completions
`reasoning_effort`, Responses `reasoning.effort`, and Anthropic
`output_config.effort` become the unified `reasoning.effort` field. `none`
sets `reasoning.enabled` to `false`. On output, OpenAI-compatible protocols
retain their native value; Anthropic maps `minimal` to `low` and `ultra` to
`max`, while `none` becomes `thinking: { type: "disabled" }` with no effort.
FIM does not use this chat reasoning field.

#### UnifiedMessage

```typescript
interface UnifiedMessage {
  role: "user" | "assistant" | "system" | "tool";
  content: string | null | MessageContent[];
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
  thinking?: {
    content: string;
    signature?: string;
  };
}
```

## Built-in Transformers

### Protocol owners (client routes)

These register inbound routes. Do **not** put them in `transformer.use` only
to open a client path — the route table already owns that. They *do* appear in
`transformer.use` when the **destination** speaks that protocol (for example
`"use": ["openai-responses", "codex"]` or `"use": ["Anthropic", "claude-auth"]`).

#### Anthropic

Protocol owner for `POST /v1/messages`. Also used as provider egress when the
upstream is Anthropic Messages.

**Client legs:** `transformRequestOut` (Anthropic → Unified),
`transformResponseIn` (Unified → Anthropic).

**Provider legs:** `transformRequestIn` / `transformResponseOut` rebuild and
parse Anthropic wire for destinations that need it.

#### OpenAI

Protocol owner for `POST /v1/chat/completions` (alias `/chat/completions`).
Unified **is** Chat Completions-shaped, so inbound `transformRequestOut`
validates and lightly normalizes. Provider-side `transformRequestIn` applies
OpenAI-native cache policy; `transformResponseIn` shapes Chat client output
(for example `reasoning_content`).

#### openai-responses

Protocol owner for `POST /v1/responses` (alias `/responses`). Converts
Responses `input` / tools ↔ Unified on the client legs, and rebuilds Responses
wire (including encrypted reasoning items) when used as provider egress.
Always pair Codex destinations as `"use": ["openai-responses", "codex"]`.

### anthropic (config examples)

Some docs/examples still show a top-level `transformers` entry named
`"anthropic"`. The registered protocol-owner / provider name is **`Anthropic`**
(see `"use": ["claude-auth", "Anthropic"]`). Prefer provider `transformer.use`
over the legacy top-level array when configuring destinations.

```json
{
  "transformers": [
    {
      "name": "anthropic",
      "providers": ["deepseek", "groq"]
    }
  ]
}
```

### codex

Adapts requests and responses for the Codex (ChatGPT) backend API.

**Features:**
- Must be chained after `openai-responses` (Responses wire owner)
- Supports both OAuth auth (`ccr codex-auth`) and PAT auth when `api_key` starts with `at-`
- Resolves required account headers automatically
- Applies ChatGPT backend constraints (`store: false`, `stream: true`)
- Streaming events are converted back to the **inbound** client protocol (not Anthropic-only)

### claude-auth

Authenticates requests to Anthropic's API using your Claude Pro or Max subscription OAuth token.

**Features:**
- Rebuilds the Anthropic request body from the unified request
- Injects `Authorization: Bearer <token>` using tokens from `~/.claude-code-router/claude_auth.json`
- Always sends the Anthropic `oauth-2025-04-20` beta; preserves and merges any client `anthropic-beta` header, otherwise derives feature betas (thinking / prompt-caching) from the outbound body
- Refreshes expired OAuth access tokens automatically
- Response conversion follows the inbound client protocol via the `Anthropic` owner
- Intended to be used together with `Anthropic` in the provider chain

### antigravity-auth

OAuth + request-envelope middleware for Google's Antigravity gateway (`cloudcode-pa`).

Authenticate with `ccr antigravity-auth` (optional `--manual`, `--project <id>`). Tokens land in `~/.claude-code-router/antigravity_auth.json`. Docker Compose publishes `51121:3456` so Google's redirect to `http://localhost:51121/oauth-callback` reaches the CCR server.

```json
{
  "name": "antigravity",
  "api_base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com",
  "api_key": "oauth",
  "project_id": "$ANTIGRAVITY_PROJECT_ID",
  "models": ["gemini-3-flash", "claude-sonnet-4-6", "claude-opus-4-6-thinking"],
  "transformer": {
    "use": [
      ["gemini", { "cachedContent": false, "thoughtSignatureFallback": "skip" }],
      "antigravity-auth"
    ]
  }
}
```

**Features:**
- Injects Antigravity OAuth bearer tokens and wraps Gemini `generateContent` bodies in the Antigravity request envelope
- Must be chained **after** `gemini` (or compatible Gemini dialect transformer)
- Endpoint fallback across daily → autopush → prod hosts for transport / entitlement failures

**Required Gemini options in the example above:**
- `cachedContent: false` — Antigravity has no Google `cachedContents` resource; leaving the Gemini default (`true`) causes 404s. See [gemini options](#options-cachedcontent-and-thoughtsignaturefallback).
- `thoughtSignatureFallback: "skip"` — explicit form of the default; on a missing tool-call thought signature, stamp Google's `skip_thought_signature_validator` sentinel so Gemini/Antigravity do not 400. Only change to `"none"` if your endpoint rejects that sentinel.

See also: [CLI auth commands](/docs/cli/commands/auth)

### cursor-sdk

Runs Cursor models in-process via `@cursor/sdk` (no HTTP fetch to `api_base_url`).

```json
{
  "name": "cursor",
  "api_base_url": "https://cursor.com",
  "api_key": "$CURSOR_API_KEY",
  "models": ["composer-2"],
  "transformer": {
    "use": [
      [
        "cursor-sdk",
        {
          "cursorMode": "bridge"
        }
      ]
    ]
  }
}
```

**Features:**
- Owns the upstream Agent create/send/stream call and returns OpenAI-compatible SSE/JSON via `__providerResponse`
- Default `cursorMode: "bridge"` — Claude Code hosts tools; Cursor built-ins are denied
- Optional modes: `plan` (text/reasoning only), `agent` (with optional `cursorCwd`)
- Auth: provider `api_key` starting with `crsr_`, or `CURSOR_API_KEY`
- Model list via `ccr model get` using `Cursor.models.list`
- Optional `sandboxEnabled` (opt-in; forced off in Docker)

See also:
- [Codex integration guide](/docs/server/guides/codex)
- [Claude subscription guide](/docs/server/guides/claude-auth)
- [Cursor SDK integration guide](/docs/server/guides/cursor)
- [CLI auth commands](/docs/cli/commands/auth)

### deepseek

Specialized transformer for DeepSeek API:

```json
{
  "transformers": [
    {
      "name": "deepseek",
      "providers": ["deepseek"]
    }
  ]
}
```

**Features:**
- DeepSeek-specific reasoning format
- Handles `reasoning_content` in responses
- Supports thinking budget tokens

### gemini

Transformer for Google Gemini API (also the dialect stage used with Antigravity).

```json
{
  "name": "gemini",
  "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
  "api_key": "$GEMINI_API_KEY",
  "models": ["gemini-3-flash"],
  "transformer": {
    "use": [
      ["gemini", { "cachedContent": true, "thoughtSignatureFallback": "skip" }]
    ]
  }
}
```

**Features:**
- Translates Claude Code `output_config.effort` into Gemini thinking depth: `thinkingLevel` for Gemini 3+ (`low`/`high` on Gemini 3 Pro, plus `medium` on later Pro minors, plus `minimal` on Flash/Lite) or `thinkingBudget` for Gemini 2.5 — never both
- Never rewrites the configured model id (tier-suffixed ids keep talking to the same upstream id)
- Same options below apply to `vertex-gemini`

#### Options: `cachedContent` and `thoughtSignatureFallback`

Pass these as the second element of a `gemini` / `vertex-gemini` entry in `transformer.use`:

```json
["gemini", { "cachedContent": false, "thoughtSignatureFallback": "skip" }]
```

**`cachedContent`** (boolean, **default `true`**)

Controls whether CCR uses Google's separate **`cachedContents` HTTP resource** to store and reuse prompt prefixes (system / tools / history) on the public Gemini API. This is Gemini's *server-side* context cache — **not** Anthropic `cache_control` markers and **not** Claude Code's local prompt cache.

| Value | Behavior | When to use |
| --- | --- | --- |
| `true` (default) | CCR may create/update a `cachedContents` object and send `cachedContent` references on later turns | Public Gemini (`generativelanguage.googleapis.com`) when you want Google-side prefix caching |
| `false` | Never call `cachedContents` | **Required for Antigravity** (that gateway has no `cachedContents` endpoint — leaving `true` yields 404s and wasted retries). Also set `false` for any Gemini-compatible proxy that does not implement the resource |

**`thoughtSignatureFallback`** (`"skip"` \| `"none"`, **default `"skip"`**)

Gemini 3 (and Antigravity) attaches an opaque **`thoughtSignature`** to each `functionCall` part. Claude Code's Anthropic wire format cannot carry that field on `tool_use`, so CCR caches signatures by tool-call id and restores them when the same tools are replayed. When a signature is genuinely missing (cache miss, or the session was re-routed to a different upstream), Gemini returns **400** unless the request includes Google's documented sentinel string `skip_thought_signature_validator` on the **first** `functionCall` of that step.

| Value | Behavior | When to use |
| --- | --- | --- |
| `"skip"` (default) | On a miss, stamp that sentinel on the first function call so the turn can proceed. The name means “use Google's *skip_thought_signature_validator* sentinel,” **not** “skip / disable the fallback.” | Leave this for public Gemini and Antigravity. CCR still prefers real cached signatures when available; the sentinel is a last resort (Google warns repeated use can degrade tool-calling quality) |
| `"none"` | Never stamp the sentinel | Only if your endpoint **rejects** the sentinel (reported on some Vertex deployments). Tool replay without a cached signature will then 400 until a real signature is available again |

### maxtoken

Limits max_tokens in requests:

```json
{
  "transformers": [
    {
      "name": "maxtoken",
      "options": {
        "max_tokens": 8192
      },
      "models": ["deepseek,deepseek-chat"]
    }
  ]
}
```

### customparams

Injects custom parameters into requests:

```json
{
  "transformers": [
    {
      "name": "customparams",
      "options": {
        "include_reasoning": true,
        "custom_header": "value"
      }
    }
  ]
}
```

## FIM transformers

Used only on FIM providers for `POST /v1/fim/completions` (separate pipeline from chat):

| Name | Role |
|------|------|
| `Fim` | Protocol owner (client route) — not listed in `transformer.use` |
| `fim.mistral` | Codestral/Mistral FIM URL + auth; same-kind body passthrough |
| `fim.deepseek` | DeepSeek beta completions FIM |
| `fim.qwen` | Qwen completions FIM (LM Studio / DashScope) |

Outbound request encoding differs per `fim.*`. Responses are encoded to the **inbound** client wire (v1: Codestral/Mistral). Same-kind paths passthrough the response body.

Step-by-step config for **Codestral** and **LM Studio Qwen**: [FIM Completions API](/docs/server/api/fim-completions-api).

Do not stack chat transformers (`mistral`, `deepseek`, …) with `fim.*` on the same provider.

## Creating Custom Transformers

### Simple Transformer: Modifying Requests

The simplest transformers just modify the request before it's sent to the provider.

**Example: Add a custom header to all requests**

```javascript
// custom-header-transformer.js
module.exports = class CustomHeaderTransformer {
  name = 'custom-header';

  constructor(options) {
    this.headerName = options?.headerName || 'X-Custom-Header';
    this.headerValue = options?.headerValue || 'default-value';
  }

  async transformRequestIn(request, provider, context) {
    // Add custom header (will be used by auth method)
    request._customHeaders = {
      [this.headerName]: this.headerValue
    };
    return request;
  }

  async auth(request, provider) {
    const headers = {
      'authorization': `Bearer ${provider.apiKey}`,
      ...request._customHeaders
    };
    return {
      body: request,
      config: { headers }
    };
  }
};
```

**Usage in config:**

```json
{
  "transformers": [
    {
      "name": "custom-header",
      "path": "/path/to/custom-header-transformer.js",
      "options": {
        "headerName": "X-My-Header",
        "headerValue": "my-value"
      }
    }
  ]
}
```

### Intermediate Transformer: Request/Response Conversion

This example shows how to convert between different API formats.

**Example: Mock API format transformer**

```javascript
// mockapi-transformer.js
module.exports = class MockAPITransformer {
  name = 'mockapi';
  endPoint = '/v1/chat';  // Custom endpoint

  // Convert from MockAPI format to unified format
  async transformRequestOut(request, context) {
    const messages = request.conversation.map(msg => ({
      role: msg.sender,
      content: msg.text
    }));

    return {
      messages,
      model: request.model_id,
      max_tokens: request.max_tokens,
      temperature: request.temp
    };
  }

  // Convert from unified format to MockAPI format
  async transformRequestIn(request, provider, context) {
    return {
      model_id: request.model,
      conversation: request.messages.map(msg => ({
        sender: msg.role,
        text: typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)
      })),
      max_tokens: request.max_tokens || 4096,
      temp: request.temperature || 0.7
    };
  }

  // Convert MockAPI response to unified format
  async transformResponseIn(response, context) {
    const data = await response.json();

    const unifiedResponse = {
      id: data.request_id,
      object: 'chat.completion',
      created: data.timestamp,
      model: data.model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: data.reply.text
        },
        finish_reason: data.stop_reason
      }],
      usage: {
        prompt_tokens: data.tokens.input,
        completion_tokens: data.tokens.output,
        total_tokens: data.tokens.input + data.tokens.output
      }
    };

    return new Response(JSON.stringify(unifiedResponse), {
      status: response.status,
      statusText: response.statusText,
      headers: { 'Content-Type': 'application/json' }
    });
  }
};
```

### Advanced Transformer: Streaming Response Processing

This example shows how to handle streaming responses.

**Example: Add custom metadata to streaming responses**

```javascript
// streaming-metadata-transformer.js
module.exports = class StreamingMetadataTransformer {
  name = 'streaming-metadata';

  constructor(options) {
    this.metadata = options?.metadata || {};
    this.logger = null;  // Will be injected by the system
  }

  async transformResponseOut(response, context) {
    const contentType = response.headers.get('Content-Type');

    // Handle streaming response
    if (contentType?.includes('text/event-stream')) {
      return this.transformStream(response, context);
    }

    // Handle non-streaming response
    return response;
  }

  async transformStream(response, context) {
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();

    const transformedStream = new ReadableStream({
      start: async (controller) => {
        const reader = response.body.getReader();
        let buffer = '';

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              if (!line.trim() || !line.startsWith('data: ')) {
                controller.enqueue(encoder.encode(line + '\n'));
                continue;
              }

              const data = line.slice(6).trim();
              if (data === '[DONE]') {
                controller.enqueue(encoder.encode(line + '\n'));
                continue;
              }

              try {
                const chunk = JSON.parse(data);

                // Add custom metadata
                if (chunk.choices && chunk.choices[0]) {
                  chunk.choices[0].metadata = this.metadata;
                }

                // Log for debugging
                this.logger?.debug({
                  chunk,
                  context: context.req.id
                }, 'Transformed streaming chunk');

                const modifiedLine = `data: ${JSON.stringify(chunk)}\n\n`;
                controller.enqueue(encoder.encode(modifiedLine));
              } catch (parseError) {
                // If parsing fails, pass through original line
                controller.enqueue(encoder.encode(line + '\n'));
              }
            }
          }
        } catch (error) {
          this.logger?.error({ error }, 'Stream transformation error');
          controller.error(error);
        } finally {
          controller.close();
          reader.releaseLock();
        }
      }
    });

    return new Response(transformedStream, {
      status: response.status,
      statusText: response.statusText,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
      }
    });
  }
};
```

### Real-World Example: Reasoning Content Transformer

This is based on the actual `reasoning.transformer.ts` from the codebase.

```typescript
// reasoning-transformer.ts
import { Transformer, TransformerOptions } from "@caeliq/llms";

export class ReasoningTransformer implements Transformer {
  static TransformerName = "reasoning";
  enable: boolean;

  constructor(private readonly options?: TransformerOptions) {
    this.enable = this.options?.enable ?? true;
  }

  // Transform request to add reasoning parameters
  async transformRequestIn(request: UnifiedChatRequest): Promise<UnifiedChatRequest> {
    if (!this.enable) {
      request.thinking = {
        type: "disabled",
        budget_tokens: -1,
      };
      request.enable_thinking = false;
      return request;
    }

    if (request.reasoning) {
      request.thinking = {
        type: "enabled",
        budget_tokens: request.reasoning.max_tokens,
      };
      request.enable_thinking = true;
    }
    return request;
  }

  // Transform response to convert reasoning_content to thinking format
  async transformResponseOut(response: Response): Promise<Response> {
    if (!this.enable) return response;

    const contentType = response.headers.get("Content-Type");

    // Handle non-streaming response
    if (contentType?.includes("application/json")) {
      const jsonResponse = await response.json();
      if (jsonResponse.choices[0]?.message.reasoning_content) {
        jsonResponse.thinking = {
          content: jsonResponse.choices[0].message.reasoning_content
        };
      }
      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    // Handle streaming response
    if (contentType?.includes("stream")) {
      // [Streaming transformation code here]
      // See the full implementation in the codebase
    }

    return response;
  }
}
```

## Transformer Registration

### Method 1: Static Name (Class-based)

Use this when creating a transformer in TypeScript/ES6:

```typescript
export class MyTransformer implements Transformer {
  static TransformerName = "my-transformer";

  async transformRequestIn(request: UnifiedChatRequest): Promise<any> {
    // Transformation logic
    return request;
  }
}
```

### Method 2: Instance Name (Instance-based)

Use this for JavaScript transformers:

```javascript
module.exports = class MyTransformer {
  constructor(options) {
    this.name = 'my-transformer';
    this.options = options;
  }

  async transformRequestIn(request, provider, context) {
    // Transformation logic
    return request;
  }
};
```

## Applying Transformers

### Global Application (Provider Level)

Apply to all requests for a provider:

```json
{
  "Providers": [
    {
      "NAME": "deepseek",
      "HOST": "https://api.deepseek.com",
      "APIKEY": "your-api-key",
      "transformers": ["anthropic"]
    }
  ]
}
```

### Model-Specific Application

Apply to specific models only:

```json
{
  "transformers": [
    {
      "name": "maxtoken",
      "options": {
        "max_tokens": 8192
      },
      "models": ["deepseek,deepseek-chat"]
    }
  ]
}
```

Note: The model format is `provider,model` (e.g., `deepseek,deepseek-chat`).

### Global Transformers (All Providers)

Apply transformers to all providers:

```json
{
  "transformers": [
    {
      "name": "custom-logger",
      "path": "/path/to/custom-logger.js"
    }
  ]
}
```

### Passing Options

Some transformers accept configuration options:

```json
{
  "transformers": [
    {
      "name": "maxtoken",
      "options": {
        "max_tokens": 8192
      }
    },
    {
      "name": "customparams",
      "options": {
        "custom_param_1": "value1",
        "custom_param_2": 42
      }
    }
  ]
}
```

## Best Practices

### 1. Immutability

Always create new objects rather than mutating existing ones:

```javascript
// Bad
async transformRequestIn(request) {
  request.max_tokens = 4096;
  return request;
}

// Good
async transformRequestIn(request) {
  return {
    ...request,
    max_tokens: request.max_tokens || 4096
  };
}
```

### 2. Error Handling

Always handle errors gracefully:

```javascript
async transformResponseIn(response) {
  try {
    const data = await response.json();
    // Process data
    return new Response(JSON.stringify(processedData), {
      status: response.status,
      headers: response.headers
    });
  } catch (error) {
    this.logger?.error({ error }, 'Transformation failed');
    // Return original response if transformation fails
    return response;
  }
}
```

### 3. Logging

Use the injected logger for debugging:

```javascript
async transformRequestIn(request, provider, context) {
  this.logger?.debug({
    model: request.model,
    provider: provider.name
  }, 'Transforming request');

  // Your transformation logic

  return modifiedRequest;
}
```

### 4. Stream Handling

When handling streams, always:
- Use a buffer to handle incomplete chunks
- Properly release the reader lock
- Handle errors in the stream
- Close the controller when done

```javascript
const transformedStream = new ReadableStream({
  start: async (controller) => {
    const reader = response.body.getReader();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Process stream...
      }
    } catch (error) {
      controller.error(error);
    } finally {
      controller.close();
      reader.releaseLock();
    }
  }
});
```

### 5. Context Usage

The `context` parameter contains useful information:

```javascript
async transformRequestIn(request, provider, context) {
  // Access request ID
  const requestId = context.req.id;

  // Access original request
  const originalRequest = context.req.original;

  // Your transformation logic
}
```

## Testing Your Transformer

### Manual Testing

1. Add your transformer to the config
2. Start the server: `ccr restart`
3. Check logs: `tail -f ~/.claude-code-router/logs/ccr-*.log`
4. Make a test request
5. Verify the output

### Debug Tips

- Add logging to track transformation steps
- Test with both streaming and non-streaming requests
- Verify error handling with invalid inputs
- Check that original responses are returned on error

## Next Steps

- [Advanced Topics](/docs/server/advanced/custom-router) - Advanced routing customization
- [Agents](/docs/server/config/transformers) - Extending with agents
- [Core Package](/docs/server/intro) - Learn about @caeliq/llms

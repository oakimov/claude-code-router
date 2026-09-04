---
title: Server Introduction
---

# Server Introduction

Claude Code Router Server routes Anthropic Messages, OpenAI Chat Completions,
OpenAI Responses, and FIM Completions requests to different LLM providers. It
provides a complete HTTP API with support for:

- **API Request Routing**: Normalize supported client protocols, select a provider/model, and convert to the provider wire format
- **Authentication & Authorization**: Support API Key authentication
- **Configuration Management**: Dynamic configuration of providers, routing rules, and transformers
- **Web UI**: Built-in management interface
- **Logging System**: Complete request logging with file rotation
- **Documentation**: Built-in Docusaurus documentation site with GitHub Pages deployment

## Architecture Overview

```
┌──────────────────┐     ┌─────────────────────────────┐     ┌──────────────┐
│ Client protocols │────▶│ CCR Server                  │────▶│ LLM Provider │
│ Messages / Chat  │     │  ┌─────────────────────┐    │     │  (OpenAI/    │
│ Responses / FIM  │     │  │ @caeliq/llms        │    │     │   Gemini/etc)│
└──────────────────┘     │  │ (Core Package)       │    │     └──────────────┘
                         │  │ - Request Transform  │    │
                         │  │ - Response Transform │    │
                         │  │ - Auth Handling      │    │
                         │  └─────────────────────┘    │
                         │                             │
                         │  - Routing Logic            │
                         │  - Agent System             │
                         │  - Configuration            │
                         └─────────────────────────────┘
                                │
                                ├─ Web UI
                                ├─ Config API
                                └─ Logs API
```

## Core Package: @caeliq/llms

The server is built on top of **@caeliq/llms**, a universal LLM API transformation library that provides the core request/response transformation capabilities.

### What is @caeliq/llms?

`@caeliq/llms` is a standalone npm package that handles:

- **API Format Conversion**: Transforms between different LLM provider APIs (Anthropic, OpenAI, Gemini, etc.)
- **Request/Response Transformation**: Converts requests and responses to a unified format
- **Authentication Handling**: Manages different authentication methods across providers
- **Streaming Support**: Handles streaming responses from different providers
- **Transformer System**: Provides an extensible architecture for adding new providers

### Key Concepts

#### 1. Unified Request/Response Format

The core package defines a unified format (`UnifiedChatRequest`, `UnifiedChatResponse`) that abstracts away provider-specific differences:

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

#### 2. Transformer Interface

All transformers implement a common interface:

```typescript
interface Transformer {
  transformRequestIn?: (request: UnifiedChatRequest, provider: LLMProvider, context: TransformerContext) => Promise<any>;
  transformRequestOut?: (request: any, context: TransformerContext) => Promise<UnifiedChatRequest>;
  transformResponseIn?: (response: Response, context?: TransformerContext) => Promise<Response>;
  transformResponseOut?: (response: Response, context: TransformerContext) => Promise<Response>;
  endPoint?: string;
  name?: string;
  auth?: (request: any, provider: LLMProvider, context: TransformerContext) => Promise<any>;
}
```

#### 3. Built-in Transformers

The core package includes transformers for:
- **Anthropic** (protocol owner): Anthropic Messages (`/v1/messages`)
- **OpenAI** (protocol owner): Chat Completions (`/v1/chat/completions`)
- **openai-responses** (protocol owner): Responses API (`/v1/responses`)
- **Fim** (protocol owner) + **fim.mistral** / **fim.deepseek** / **fim.qwen**: FIM Completions (`/v1/fim/completions`)
- **gemini**: Google Gemini API format
- **vertex-gemini / vertex-claude**: Google Vertex AI formats
- **deepseek**: DeepSeek API format
- **mistral**: Mistral AI API format
- **groq**: Groq API format
- **cerebras**: Cerebras API format
- **openrouter**: OpenRouter API format
- **codex**: Codex (ChatGPT) backend API
- **claude-auth**: Claude Pro/Max subscription OAuth
- **antigravity-auth**: Google Antigravity (`cloudcode-pa`) OAuth + envelope
- **cursor-sdk**: In-process Cursor Agent via `@cursor/sdk`
- **qwen-auth**: Qwen Chat authentication (JWT-based)
- **chrome-on-device**: Chrome built-in Gemini Nano (local, no API cost)
- **vercel**: Vercel AI SDK format
- **opencode-headers**: OpenCode header injection
- **And more utility transformers**: `cleancache`, `customparams`, `enhancetool`, `forcereasoning`, `maxcompletiontokens`, `maxtoken`, `reasoning`, `sampling`, `streamoptions`, `tooluse`

### Integration with CCR Server

The CCR server integrates `@caeliq/llms` through:

1. **Transformer Service** (`packages/core/src/services/transformer.ts`): Manages transformer registration and instantiation
2. **Provider Configuration**: Maps provider configs to core package's LLMProvider interface
3. **Request Pipeline**: Applies transformers in sequence during request processing
4. **Custom Transformers**: Supports loading external transformer plugins

### Version and Updates

The current version of `@caeliq/llms` is `1.0.68`. It's published as an independent npm package and can be used standalone or as part of CCR Server.

## Core Features

### 1. Request Routing
- Token-count-based intelligent routing
- Project-level routing configuration
- Custom routing functions
- Scenario-based routing (background, think, longContext, webSearch, fim, image, etc.)

### 2. Request Transformation
- Supports API format conversion for multiple LLM providers
- Built-in transformers: Anthropic, DeepSeek, Gemini, OpenRouter, Groq, etc.
- Extensible transformer system

### 3. Agent System
- Plugin-based Agent architecture
- Built-in image processing Agent
- Custom Agent support

### 4. Configuration Management
- JSON5 format configuration file
- Environment variable interpolation
- Hot configuration reload (requires service restart)

### 5. Gateway Model Discovery

`GET /v1/models` and its `/models` alias list every configured model. The default
`MODEL_ID_OUTPUT: "literal"` emits CCR's canonical `provider,model` identifier;
`"masked"` emits reversible `claude-<hex>` aliases for ids that Claude clients
would otherwise filter. Inbound accepts both forms in either mode. The server independently fetches the
[models.dev](https://models.dev) catalog and adds metadata when the model name is
known:

- `display_name` and `description`
- `context_window`, `max_input_tokens`, and `max_output_tokens`
- `effort_levels`
- `anthropic_family_tier` and `supports_1m` for Claude Desktop's gateway picker

CCR's provider prefix is used only for routing. Metadata matching uses the model
name after the comma, and duplicate models.dev entries are resolved to the
model's native provider instead of the configured CCR provider or an arbitrary
reseller. Direct Anthropic models retain their Anthropic family. Other models
map to `haiku` without reasoning support, `sonnet` with reasoning and a context
window up to 300K tokens, and `opus` above 300K; `gpt-5.6-sol` maps to `fable`.

The server keeps its own one-hour in-memory catalog cache and retains the last
good value when refresh fails. A cold models.dev failure is non-fatal and falls
back to the base routing entries. `CCR_MODELSDEV_URL`,
`CCR_MODELSDEV_TIMEOUT`, and `CCR_MODELSDEV_DISABLE` apply independently in the
server process; the CLI does not provide or populate this runtime cache.

## Use Cases

### Scenario 1: Personal Local Service
Run the service locally for personal Claude Code use:

```bash
cd packages/server
docker compose up --build -d
```

### Scenario 2: Team Shared Service
Deploy using Docker Compose to provide shared service for team members:

```yaml
services:
  ccr:
    build:
      context: ../..
      dockerfile: packages/server/Dockerfile
    ports:
      - "3456:3456"
    volumes:
      - ~/.claude-code-router:/root/.claude-code-router
    environment:
      - HOST=0.0.0.0
      - PORT=3456
```

### Scenario 3: Secondary Development
Build custom applications based on exposed APIs:

```bash
GET /api/config
POST /v1/messages
POST /v1/chat/completions
POST /v1/responses
POST /v1/fim/completions
GET /api/logs
```

## Next Steps

- [Docker Deployment Guide](/docs/server/deployment) — Learn how to deploy the service
- [API Reference](/docs/category/api) — View complete API documentation
- [Configuration Guide](/docs/category/server-config) — Understand server configuration options

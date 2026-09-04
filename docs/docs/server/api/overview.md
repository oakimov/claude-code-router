---
title: API Overview
---

# API Overview

Claude Code Router Server provides a complete HTTP API with support for:

- **Messages API**: Anthropic Claude Messages (`POST /v1/messages`)
- **Chat Completions API**: OpenAI Chat Completions (`POST /v1/chat/completions`)
- **Responses API**: OpenAI Responses (`POST /v1/responses`)
- **FIM Completions API**: Fill-in-the-middle (`POST /v1/fim/completions`; separate pipeline)
- **Models API**: OpenAI-compatible listing of routable models
- **Configuration API**: Read and update server configuration
- **Logs API**: View and manage service logs
- **Tools API**: Calculate token counts

All four LLM POST protocols above are first-class inbound routes. Chat
protocols share the Unified chat pipeline; FIM uses its own pipeline. The
response always matches the inbound client protocol.

## Basic Information

**Base URL**: `http://localhost:3456`

**Authentication**: API key via `Authorization: Bearer` or `x-api-key`

```bash
curl -H "x-api-key: your-api-key" http://localhost:3456/api/config
```

## API Endpoints

### Messages

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Send message (compatible with Anthropic API) |
| `/v1/messages/count_tokens` | POST | Count tokens in messages |
| `/v1/chat/completions` | POST | OpenAI Chat Completions (alias: `/chat/completions`) |
| `/v1/responses` | POST | OpenAI Responses (alias: `/responses`) |
| `/v1/fim/completions` | POST | FIM Completions (alias: `/fim/completions`) |
| `/v1/models` | GET | List routable models (alias: `/models`) |
| `/v1/models/{id}` | GET | Single model info |

### Configuration Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Get current configuration |
| `/api/config` | POST | Update configuration |
| `/api/transformers` | GET | Get list of available transformers |

### Log Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/logs/files` | GET | Get list of log files |
| `/api/logs` | GET | Get log content |
| `/api/logs` | DELETE | Clear logs |

### Service Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/restart` | POST | Restart service |
| `/ui` | GET | Web management interface |
| `/ui/` | GET | Web management interface (redirect) |

## Authentication

### API Key Authentication

Add API Key in request header:

```bash
curl -X POST http://localhost:3456/v1/messages \
  -H "x-api-key: your-api-key" \
  -H "content-type: application/json" \
  -d '...'
```

## Streaming Responses

Messages, Chat Completions, Responses, and FIM support Server-Sent Events when
`stream: true`. Each route emits its **inbound** client protocol:

- Messages → Anthropic SSE (`message_start` / `content_block_delta` / …)
- Chat Completions → `chat.completion.chunk` records ending in `[DONE]`
- Responses → ordered `response.*` events ending in `response.completed` / `response.failed`
- FIM → inbound FIM wire (v1: Codestral/Mistral-shaped chunks)

```bash
curl -X POST http://localhost:3456/v1/messages \
  -H "x-api-key: your-api-key" \
  -H "content-type: application/json" \
  -d '{"stream": true, ...}'
```

Example Anthropic Messages stream:

```
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_delta
data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}

event: message_stop
data: {"type":"message_stop"}
```

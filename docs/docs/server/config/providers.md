---
sidebar_position: 2
---

# Providers Configuration

Detailed guide for configuring LLM providers.

## Supported Providers

### DeepSeek

```json
{
  "NAME": "deepseek",
  "HOST": "https://api.deepseek.com",
  "APIKEY": "your-api-key",
  "MODELS": ["deepseek-chat", "deepseek-coder"],
  "transformers": ["anthropic"]
}
```

### Groq

```json
{
  "NAME": "groq",
  "HOST": "https://api.groq.com/openai/v1",
  "APIKEY": "your-api-key",
  "MODELS": ["llama-3.3-70b-versatile"],
  "transformers": ["anthropic"]
}
```

### Gemini

```json
{
  "NAME": "gemini",
  "HOST": "https://generativelanguage.googleapis.com/v1beta",
  "APIKEY": "your-api-key",
  "MODELS": ["gemini-2.0-flash", "gemini-2.5-pro"],
  "transformers": ["gemini"]
}
```

### OpenRouter

```json
{
  "NAME": "openrouter",
  "HOST": "https://openrouter.ai/api/v1",
  "APIKEY": "your-api-key",
  "MODELS": ["anthropic/claude-sonnet-4", "google/gemini-2.5-pro-preview"],
  "transformers": ["openrouter"]
}
```

### Mistral

```json
{
  "NAME": "mistral",
  "HOST": "https://api.mistral.ai/v1",
  "APIKEY": "your-api-key",
  "MODELS": ["mistral-large-latest", "mistral-small-latest"],
  "transformers": ["mistral"]
}
```

### Cerebras

```json
{
  "NAME": "cerebras",
  "HOST": "https://api.cerebras.ai/v1",
  "APIKEY": "your-api-key",
  "MODELS": ["cerebras-gpt"],
  "transformers": ["cerebras"]
}
```

### Codex (ChatGPT)

Requires OAuth authentication via `ccr codex-auth`.

```json
{
  "NAME": "codex",
  "baseUrl": "https://api.githubcopilot.com",
  "apiKey": "$CODEX_ACCESS_TOKEN",
  "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
  "transformer": {
    "use": ["codex"]
  }
}
```

### Qwen Chat

Requires JWT authentication via `ccr qwen-auth`.

```json
{
  "NAME": "qwen",
  "baseUrl": "https://qwen.aikit.club/v1/chat/completions",
  "apiKey": "$QWEN_ACCESS_TOKEN",
  "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
  "transformer": {
    "use": ["qwen-auth", "OpenAI"]
  }
}
```

### Chrome On-Device (Gemini Nano)

Requires the bridge process via `ccr chrome-bridge`.

```json
{
  "NAME": "chrome",
  "baseUrl": "http://127.0.0.1:9229",
  "apiKey": "dummy",
  "models": ["gemini-nano"],
  "transformer": {
    "use": ["chrome-on-device"]
  }
}
```

## Provider Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `NAME` | string | Yes | Unique provider identifier |
| `HOST` | string | Yes | API base URL |
| `APIKEY` | string | Yes | API authentication key |
| `MODELS` | string[] | No | List of available models |
| `transformers` | string[] | No | List of transformers to apply |

## Model Selection

When selecting a model in routing, use the format:

```
{provider-name},{model-name}
```

For example:

```
deepseek,deepseek-chat
codex,gpt-5
chrome,gemini-nano
```

## Next Steps

- [Routing Configuration](/docs/config/routing) — Configure how requests are routed
- [Transformers](/docs/config/transformers) — Apply transformations to requests

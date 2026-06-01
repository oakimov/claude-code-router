---
sidebar_position: 5
---

# Model Discovery

Claude Code Router can **auto-discover available models** from any API provider that exposes a model listing endpoint. This is useful for providers with frequently changing model lists, or for exploring what's available on a new provider.

## Usage

Run the discovery command with a provider name:

```bash
ccr model get <provider-name>
```

The provider must be configured in your `config.json` with at least `baseUrl` and `apiKey`.

## How It Works

1. `ccr model get` sends a request to the provider's model listing endpoint
2. It parses the JSON response using configurable paths to extract model IDs
3. It compares the discovered models against your current config
4. It appends any new models that aren't already listed
5. Your existing configuration is preserved

## Configuration

Model discovery uses conventions from your provider config:

```json
{
  "Providers": [
    {
      "name": "openai",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "$OPENAI_API_KEY",
      "models": ["gpt-4"]
    }
  ]
}
```

Running `ccr model get openai` would discover all available GPT models and append them.

## Supported Providers

Model discovery works with any provider that exposes a `GET /models` endpoint returning a JSON array or object with model IDs. This includes:

- OpenAI
- DeepSeek
- Groq
- OpenRouter
- Custom providers with RESTful model APIs

## Troubleshooting

**No models found**: The provider may not expose a model listing endpoint, or the response format differs from what's expected. Check the provider's API documentation for the correct model listing path.

**Duplicate models**: The tool only appends models not already in your config — duplicates are automatically skipped.

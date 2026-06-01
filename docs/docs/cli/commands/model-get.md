---
sidebar_position: 3
---

# ccr model get

Discover available models from a provider non-interactively.

## Usage

```bash
ccr model get <provider-name>
```

## Description

Fetches the list of available models from the specified provider's API, parses the response using configurable paths, and appends any new models to your local configuration. Existing models are preserved — only missing models are added.

The provider must already be configured in your `config.json` with at least `baseUrl` and `apiKey` set.

## Examples

### Discover OpenAI models

```bash
ccr model get openai
```

### Discover DeepSeek models

```bash
ccr model get deepseek
```

### Discover Groq models

```bash
ccr model get groq
```

---
sidebar_position: 1
---

# Codex (ChatGPT) Integration

Claude Code Router supports routing requests through the **Codex backend API**, which powers GitHub Copilot's ChatGPT models (GPT-5 series). This requires OAuth authentication via your GitHub account.

## How It Works

1. `ccr codex-auth` opens a browser for GitHub Copilot sign-in
2. The OAuth flow returns an access token stored in `~/.claude-code-router/codex-token.json`
3. The `codex` provider transformer uses this token to authenticate API requests
4. Requests are sent to `https://api.githubcopilot.com` using the Responses API format

## Prerequisites

- An active [GitHub Copilot](https://github.com/features/copilot) subscription
- Claude Code Router running (Docker Compose or local)

## Setup

### 1. Authenticate

Run the OAuth flow:

```bash
ccr codex-auth
```

This opens a browser. Sign in with your GitHub account and authorize the application. The token is saved automatically.

To verify the token is valid:

```bash
ccr codex-auth --check
```

### 2. Configure Provider

Add the Codex provider to your `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "codex",
      "baseUrl": "https://api.githubcopilot.com",
      "apiKey": "$CODEX_ACCESS_TOKEN",
      "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
      "transformer": {
        "use": ["codex"]
      }
    }
  ],
  "Router": {
    "default": "codex,gpt-5"
  }
}
```

### 3. Restart

```bash
docker compose restart ccr
```

## Features

- **SSE streaming** — Full streaming support for real-time responses
- **Reasoning/thinking content** — Supports models with reasoning capabilities
- **Tool calls** — Function calling with multiple tools
- **Web search** — Built-in web search via `{ type: "web_search" }`
- **Image handling** — Vision support for image inputs

## Usage

Use Codex as your default model or route specific scenarios:

```json
{
  "Router": {
    "default": "codex,gpt-5",
    "webSearch": "codex,gpt-5-high",
    "think": "codex,gpt-5-high",
    "background": "codex,gpt-5-mini"
  }
}
```

## Model Reference

| Model | Description |
|-------|-------------|
| `gpt-5` | Standard GPT-5 model |
| `gpt-5-high` | High-performance variant (reasoning tasks) |
| `gpt-5-mini` | Lightweight variant (background tasks) |

## Troubleshooting

**Token expired or invalid**: Re-run `ccr codex-auth` to refresh the token.

**Provider not found**: Ensure the provider name in your config matches `body.model` (e.g., `codex,gpt-5`).

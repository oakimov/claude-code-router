---
sidebar_position: 1
---

# Codex (ChatGPT) Integration

Claude Code Router can use a **ChatGPT/Codex subscription** to route Claude Code requests through OpenAI's models. The Codex backend powers OpenAI's ChatGPT product and this integration lets you leverage that subscription with Claude Code.

Authentication uses OpenAI OAuth — a ChatGPT Plus or Pro subscription is required.

## How It Works

1. `ccr codex-auth` prints an authorization URL and starts a local callback server on port 1455
2. You open the URL in your browser and sign into your OpenAI / ChatGPT account
3. OpenAI redirects to `http://localhost:1455/auth/callback`, where the CCR server exchanges the authorization code for tokens (PKCE flow)
4. Tokens are saved to `~/.claude-code-router/codex_auth.json`
5. You return to the terminal and press Enter — the CLI confirms the tokens were saved
6. The `codex` transformer reads the access token and uses it to authenticate API requests
7. When the token nears expiry, it's refreshed automatically using the refresh token

## Prerequisites

- A [ChatGPT Plus or Pro](https://chat.openai.com) subscription
- Claude Code Router running (Docker Compose or local)

## Setup

### 1. Authenticate

Run the OAuth flow:

```bash
ccr codex-auth
```

The CLI prints an authorization URL. Open it in your browser, sign in with your OpenAI / ChatGPT account, and authorize the application. After the browser shows "Authentication Successful", return to your terminal and press Enter. The tokens are saved automatically.

### 2. Configure Provider

Add the Codex provider to your `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "codex",
      "baseUrl": "https://chatgpt.com/backend-api/codex",
      "apiKey": "oauth_dummy_key",
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

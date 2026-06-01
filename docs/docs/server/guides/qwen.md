---
sidebar_position: 2
---

# Qwen Chat Integration

Claude Code Router supports routing through **Qwen Chat** (通义千问) via the `qwen.aikit.club` API. It uses JWT-based authentication where you paste a token copied from the `chat.qwen.ai` web application.

## How It Works

1. `ccr qwen-auth` prompts you to paste a JWT token from `chat.qwen.ai`
2. The token is stored in `~/.claude-code-router/qwen-token.json`
3. Automatic token rotation — expired tokens are detected and you'll be prompted for a fresh one
4. Qwen's trailing `<details>...</details>` metadata block is stripped from responses automatically

## Prerequisites

- Access to the Qwen Chat API via `qwen.aikit.club`
- A Qwen Chat account

## Setup

### 1. Obtain a JWT Token

1. Open [chat.qwen.ai](https://chat.qwen.ai) in your browser
2. Open Developer Tools (`F12` or `Cmd+Option+I`)
3. Go to **Application** → **Local Storage** → `https://chat.qwen.ai`
4. Find the key that contains the `access_token` or `token` value
5. Copy the full JWT token string

### 2. Authenticate

Run the authentication command:

```bash
ccr qwen-auth
```

You'll be prompted to paste the JWT token. The CLI stores it securely and can refresh it automatically when needed.

### 3. Configure Provider

Add the Qwen provider to your `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "qwen",
      "baseUrl": "https://qwen.aikit.club/v1/chat/completions",
      "apiKey": "$QWEN_ACCESS_TOKEN",
      "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
      "transformer": {
        "use": ["qwen-auth", "OpenAI"]
      }
    }
  ],
  "Router": {
    "default": "qwen,qwen-max"
  }
}
```

The `qwen-auth` transformer handles:
- Adding the `Authorization: Bearer` header with your token
- Stripping the `<details>...</details>` metadata block from responses

The `OpenAI` transformer registers the `/v1/chat/completions` endpoint — Qwen uses standard Chat Completions format.

### 4. Restart

```bash
docker compose restart ccr
```

## Model Reference

| Model | Description |
|-------|-------------|
| `qwen-max` | Flagship model, best quality |
| `qwen-plus` | Balanced performance and cost |
| `qwen-turbo` | Fast, cost-effective for simple tasks |

## Troubleshooting

**Invalid token**: The JWT token may have expired. Re-run `ccr qwen-auth` and paste a fresh token from `chat.qwen.ai`.

**Trailing metadata in responses**: The `<details>...</details>` block should be stripped automatically. If you see it in raw responses, the `qwen-auth` transformer may not be active — check that it's in your provider's `transformer.use` array.

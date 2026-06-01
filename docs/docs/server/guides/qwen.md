---
sidebar_position: 2
---

# Qwen Chat Integration

Claude Code Router supports routing through **Qwen Chat** (通义千问) via the `qwen.aikit.club` API. It uses JWT-based authentication where you paste a token copied from the `chat.qwen.ai` web application.

## How It Works

1. The CCR server hosts an auth page at `http://localhost:3456/qwen/auth`
2. The page offers a bookmarklet and a manual paste form for obtaining the JWT token
3. Once submitted, the token is validated against Qwen's API and saved to `~/.claude-code-router/qwen_auth.json`
4. Automatic token rotation — expired tokens are detected on the next request and you'll be prompted to refresh
4. Qwen's trailing `<details>...</details>` metadata block is stripped from responses automatically

## Prerequisites

- Access to the Qwen Chat API via `qwen.aikit.club`
- A Qwen Chat account

## Setup

### 1. Authenticate via Browser

The CCR server hosts an auth page at `http://localhost:3456/qwen/auth`. Open it in your browser:

```
http://localhost:3456/qwen/auth
```

The page offers two methods:

**Option 1 — Bookmarklet (recommended):**
1. On the auth page, drag the **"Get Qwen Token"** button to your bookmarks bar
2. Open [chat.qwen.ai](https://chat.qwen.ai) in another tab and sign in
3. Click the bookmarklet on the Qwen page — it reads the token from localStorage and sends it back to the auth page automatically

**Option 2 — Manual paste:**
1. Open [chat.qwen.ai](https://chat.qwen.ai), sign in, and open DevTools (`F12`)
2. In the Console tab, run: `copy(localStorage.getItem('token'))`
3. Go back to the auth page and paste the token into the form

The token is validated against Qwen's API and saved to `~/.claude-code-router/qwen-auth.json`.

### 3. Configure Provider

Add the Qwen provider to your `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "qwen",
      "baseUrl": "https://qwen.aikit.club/v1/chat/completions",
      "apiKey": "oauth_dummy_key",
      "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
      "transformer": {
        "use": ["qwen-auth", "reasoning", "OpenAI"]
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

The `reasoning` transformer captures and replays Qwen's reasoning content across multi-turn conversations, maintaining coherence.

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

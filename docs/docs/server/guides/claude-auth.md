---
sidebar_position: 2
---

# Claude Subscription Integration

Claude Code Router can route requests through your **existing Claude subscription** (Pro or Max) using OAuth authentication. This lets you leverage your Claude.ai subscription directly — no separate API key needed.

## How It Works

1. `ccr claude-auth` generates a PKCE challenge and prints an authorization URL from `claude.ai`
2. You open the URL in your browser and sign into your Claude account
3. Claude redirects to `http://localhost:1455/callback`, where the CCR server exchanges the authorization code for tokens
4. Tokens are saved to `~/.claude-code-router/claude_auth.json`
5. You return to the terminal and press Enter — the CLI confirms the tokens were saved
6. The `claude-auth` transformer reads the access token and injects it as a `Bearer` token on every request
7. When the token nears expiry, it's refreshed automatically using the refresh token

## Prerequisites

- A [Claude Pro or Max](https://claude.ai) subscription
- Claude Code Router running (Docker Compose or local)

## Setup

### 1. Authenticate

Run the OAuth flow:

```bash
ccr claude-auth
```

The CLI prints an authorization URL. Open it in your browser, sign into your Claude account, and authorize the application. After the browser shows "Authentication Successful", return to your terminal and press Enter. The tokens are saved automatically.

### 2. Configure Provider

Add the provider to your `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "claude-subscription",
      "api_base_url": "https://api.anthropic.com",
      "api_key": "no-key",
      "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
      "transformer": {
        "use": ["claude-auth", "Anthropic"]
      }
    }
  ],
  "Router": {
    "default": "claude-subscription,claude-sonnet-4-6"
  }
}
```

### 3. Restart

```bash
docker compose restart ccr
```

## Running with Docker

The OAuth callback uses port `1455`, which is already mapped to the CCR server in `docker-compose.yml` (`"1455:3456"`). When running in Docker:

```bash
docker exec -it claude-code-router ccr claude-auth
```

The CLI prints a URL to open in your host browser. After signing in, the browser redirects to `http://localhost:1455/callback`, which Docker forwards to the container. Tokens persist across container restarts via the volume-mounted `./ccr-config` directory.

## Transformer Chain

Two transformers are required:

- `claude-auth` — converts the request from Unified (OpenAI) format to Anthropic format, injects `Authorization: Bearer <token>` (loading/refreshing the token from `~/.claude-code-router/claude_auth.json`), and converts the Anthropic SSE response back to Unified format
- `Anthropic` — registers the `POST /v1/messages` route; it has no body conversion of its own in the provider chain so it acts as a no-op endpoint stub

### OAuth beta header

Anthropic requires the `oauth-2025-04-20` beta for Claude subscription / Claude Code OAuth Bearer auth. The `claude-auth` transformer always includes `anthropic-beta: oauth-2025-04-20` and merges it with any feature betas (for example thinking / prompt-caching) when those features are used. Requests also use `?beta=true` on the Anthropic Messages URL.

You do not need to configure this beta yourself — it is applied automatically when the provider uses `claude-auth`.

## Token Storage

Tokens are stored in `~/.claude-code-router/claude_auth.json` (mode 0600):

```json
{
  "access_token": "sk-ant-oat01-...",
  "refresh_token": "...",
  "token_type": "Bearer",
  "scope": "user:profile user:inference user:sessions:claude_code user:mcp_servers",
  "expires_at": 1760000000,
  "last_refresh": 1759996400
}
```

## Troubleshooting

**Token expired or invalid**: Re-run `ccr claude-auth` to re-authenticate.

**"Redirect URI not supported"**: Ensure you are using `localhost` (not `127.0.0.1`) in the browser and that the CCR server is running on port 1455.

**Provider not found**: Ensure the provider name in your config matches the model string (e.g., `claude-subscription,claude-sonnet-4-6`).
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

### Outbound headers

The `claude-auth` transformer builds the outbound request headers as follows. "Always set" means the header is present on every request regardless of the client; "conditionally set" means it is only forwarded when the client included it.

| Header | Condition | Value |
|---|---|---|
| `Authorization` | Always set | `Bearer <access_token>` from `claude_auth.json`; refreshed automatically on expiry |
| `Content-Type` | Always set | `application/json` |
| `anthropic-beta` | Always set | See [Beta header logic](#anthropic-beta-header-logic) below |
| `anthropic-version` | Always set | Forwarded from client if present; falls back to `2023-06-01` |
| `User-Agent` | Always set | Forwarded from client if present; falls back to `claude-cli/2.1.195 (external, cli)` |
| `x-app` | Conditionally set | Forwarded verbatim (e.g. `cli`) |
| `x-claude-code-session-id` | Conditionally set | Forwarded verbatim; used by Anthropic for session attribution |
| `anthropic-dangerous-direct-browser-access` | Conditionally set | Forwarded verbatim when set by the client |

Hop-by-hop headers (`connection`, `host`, `accept-encoding`, `content-length`) and SDK-internal headers (`x-stainless-*`) sent by the client are **not** forwarded.

#### `anthropic-beta` header logic

`oauth-2025-04-20` is always included — Anthropic requires it for subscription OAuth Bearer auth. The rest of the value depends on which client is sending the request:

**Claude Code / Anthropic-native clients** (client sends `anthropic-beta`):  
The client's beta tokens are preserved as-is and `oauth-2025-04-20` is appended if not already present. No tokens are added or removed. Example — client sends:

```
claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,effort-2025-11-24
```

Outbound:

```
claude-code-20250219,interleaved-thinking-2025-05-14,context-management-2025-06-27,effort-2025-11-24,oauth-2025-04-20
```

**OpenAI-compatible clients** (client does not send `anthropic-beta`):  
Feature betas are derived from the rebuilt Anthropic request body:
- If the request uses extended thinking → `interleaved-thinking-2025-05-14,effort-2025-11-24,prompt-caching-scope-2026-01-05` are added
- If any message or tool carries a `cache_control` block → same set is added
- Otherwise → only `oauth-2025-04-20` is sent

The URL also receives `?beta=true`. You do not need to configure any of this — it is applied automatically.

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
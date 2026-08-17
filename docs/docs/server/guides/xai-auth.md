---
sidebar_position: 2
---

# xAI Grok Integration

Claude Code Router can route requests to **xAI's Grok models** two ways: through your existing **SuperGrok or X Premium+ subscription** via OAuth (no `XAI_API_KEY` needed), or through a plain xAI API key. Both use the same `xai-auth` transformer.

:::warning Consult provider Terms & Conditions

This integration authenticates against a third-party service using your own account. Before using it, review the provider's Terms & Conditions — access may be limited by subscription tier, region, or the provider's service terms, and using client credentials outside the client they were issued for may violate those terms. You use this functionality at your own risk; CCR provides it for interoperability only and does not guarantee continued access to any third-party service.

See [DISCLAIMER.md](https://github.com/oakimov/claude-code-router/blob/main/DISCLAIMER.md) for the project's interoperability statement.
:::

## How It Works

Unlike Codex, Claude, and Antigravity — which use a browser redirect + PKCE flow requiring the CCR server to host an OAuth callback — xAI uses an **RFC 8628 device authorization grant**:

1. `ccr xai-auth` requests a device code from `auth.x.ai`
2. It prints a verification URL (and a short code, if the URL doesn't embed it)
3. You open the URL in **any browser on any device** and approve access — no local callback server, no port mapping, works over SSH/Docker/headless out of the box
4. The CLI polls `auth.x.ai` in the background until you approve, then saves tokens to `~/.claude-code-router/xai_auth.json`
5. The `xai-auth` transformer reads the access token and injects it as a `Bearer` token on requests
6. When the token nears expiry, it's refreshed automatically using the (rotating) refresh token

If you'd rather not use OAuth, set `api_key` to a literal `xai-...` key (or a `$XAI_API_KEY`/`${XAI_API_KEY}` reference) instead — the `xai-auth` transformer detects it and skips OAuth entirely.

## Prerequisites

- Either a [SuperGrok](https://x.ai/grok) or X Premium+ subscription (for OAuth), **or** an xAI API key (for the API-key path)
- Claude Code Router running (Docker Compose or local)

## Setup

### 1. Authenticate (OAuth path)

```bash
ccr xai-auth
```

The CLI prints a verification URL. Open it in a browser (any device — this doesn't have to be the machine running CCR), approve access, and the CLI confirms once the tokens are saved. Skip this step entirely if you're using the API-key path.

### 2. Configure Provider

Add one of the following to your `~/.claude-code-router/config.json`, depending on which auth path you're using:

```json title="OAuth (SuperGrok / X Premium+)"
{
  "Providers": [
    {
      "name": "xai-subscription",
      "api_base_url": "https://api.x.ai/v1",
      "api_key": "no-key",
      "models": ["grok-4.6", "grok-4.3", "grok-code-fast-1"],
      "transformer": {
        "use": ["xai-auth", "openai-responses"]
      }
    }
  ],
  "Router": {
    "default": "xai-subscription,grok-4.6"
  }
}
```

```json title="Plain API key"
{
  "Providers": [
    {
      "name": "xai-api-key",
      "api_base_url": "https://api.x.ai/v1",
      "api_key": "xai-... (or $XAI_API_KEY)",
      "models": ["grok-4.6", "grok-4.3", "grok-code-fast-1"],
      "transformer": {
        "use": ["xai-auth", "openai-responses"]
      }
    }
  ],
  "Router": {
    "default": "xai-api-key,grok-4.6"
  }
}
```

Both configs use the same transformer chain — `xai-auth` resolves whichever credential is available (a literal/env-referenced `xai-...` key first, otherwise the stored OAuth token) and lets `openai-responses` own the wire format.

### 3. Restart

```bash
docker compose restart ccr
```

## Transformer Chain

Two transformers are required, in this order:

- `xai-auth` — resolves the credential (PAT-first, then OAuth) and injects it as a `Bearer` token. Owns 401 recovery via the pipeline's generic `__authRecovery` hook.
- `openai-responses` — owns the `POST /v1/responses` route and builds the Responses API wire body. xAI exposes both `/v1/chat/completions` (legacy) and `/v1/responses` (current default) — this integration targets the Responses API, matching what xAI's own quickstart and `@ai-sdk/xai` lead with.

## Model Discovery

`ccr model get <provider>` works for both auth modes — it resolves the same credential precedence as the transformer (literal/env `xai-...` key first, otherwise the stored OAuth token from `xai_auth.json`) and lists live models from `https://api.x.ai/v1/models`.

## Token Storage

OAuth tokens are stored in `~/.claude-code-router/xai_auth.json` (mode 0600):

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "Bearer",
  "scope": "openid profile email offline_access grok-cli:access api:access",
  "expires_at": 1760000000,
  "last_refresh": 1759996400
}
```

xAI doesn't always return an explicit expiry on refresh, so the access token's own JWT `exp` claim is checked as a fallback whenever `expires_at` is missing or stale.

## Troubleshooting

**Token expired or invalid**: Re-run `ccr xai-auth` to re-authenticate.

**"xAI device authorization was denied" / "timed out"**: The device-code approval window is time-limited. Re-run `ccr xai-auth` and approve promptly.

**Provider not found**: Ensure the provider name in your config matches the model string (e.g., `xai-subscription,grok-4.6`).

**OAuth login succeeds but requests get HTTP 403**: xAI has been observed gating OAuth API access to specific SuperGrok tiers even with an active subscription. Switch to the API-key path (`export XAI_API_KEY=xai-...` and use the plain-API-key config above) if this happens.

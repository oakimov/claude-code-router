---
sidebar_position: 1
---

# Codex (ChatGPT) Integration

Claude Code Router can use a **ChatGPT/Codex subscription** to route Claude Code requests through OpenAI's models. The Codex backend powers OpenAI's ChatGPT product and this integration lets you leverage that subscription with Claude Code.

Codex supports **two authentication modes**:

- **OAuth** via `ccr codex-auth` — recommended when you want CCR to manage OpenAI tokens for you
- **PAT** (Personal Access Token) via `api_key: "at-..."` — recommended when you already have a Codex-compatible PAT

A ChatGPT Plus or Pro subscription is still required.

## Authentication Modes

### OAuth via `ccr codex-auth`

1. `ccr codex-auth` prints an authorization URL and starts a local callback server on port 1455
2. You open the URL in your browser and sign into your OpenAI / ChatGPT account
3. OpenAI redirects to `http://localhost:1455/auth/callback`, where the CCR server exchanges the authorization code for tokens (PKCE flow)
4. Tokens are saved to `~/.claude-code-router/codex_auth.json`
5. You return to the terminal and press Enter — the CLI confirms the tokens were saved
6. The `codex` transformer reads the access token and uses it to authenticate API requests
7. The CLI and server independently refresh the token five minutes before expiry

The ID token supplies the selected `chatgpt_account_id` and FedRAMP state.
CCR sends those routing headers for both inference and model discovery. Token
updates are atomic and coordinated with a filesystem lock because the CLI and
server are separate processes. If an inference request receives a 401, the
server reloads or refreshes the same OAuth account and retries once.

### PAT via `api_key`

If the provider `api_key` starts with `at-`, the Codex transformer treats it as a Personal Access Token instead of using OAuth tokens.

1. You place the PAT directly in the provider `api_key` field
2. Before backend use, CCR calls OpenAI's whoami endpoint to resolve account,
   user, plan, and FedRAMP metadata
3. The server deduplicates concurrent lookups and briefly caches the result
4. Runtime and model-discovery requests send the PAT plus the resolved account
   and FedRAMP headers

If `api_key` is missing, is just a placeholder, or does not start with `at-`,
CCR selects the OAuth token flow from
`~/.claude-code-router/codex_auth.json`. An `at-` value always remains in PAT
mode: a revoked or invalid PAT fails directly and does not silently use OAuth.

## Prerequisites

- A [ChatGPT Plus or Pro](https://chat.openai.com) subscription
- Claude Code Router running (Docker Compose or local)

## Setup

### Option A: OAuth setup

#### 1. Authenticate

Run the OAuth flow:

```bash
ccr codex-auth
```

The CLI prints an authorization URL. Open it in your browser, sign in with your OpenAI / ChatGPT account, and authorize the application. After the browser shows "Authentication Successful", return to your terminal and press Enter. The tokens are saved automatically.

#### 2. Configure Provider

Add the Codex provider to your `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
    {
      "name": "codex",
      "api_base_url": "https://chatgpt.com/backend-api/codex",
      "api_key": "oauth_dummy_key",
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

### Option B: PAT setup

If you already have a Codex-compatible PAT, you can skip `ccr codex-auth` and place the token directly in `api_key`.

```json
{
  "Providers": [
    {
      "name": "codex",
      "api_base_url": "https://chatgpt.com/backend-api/codex",
      "api_key": "at-your-personal-access-token",
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

PAT detection is intentionally simple: if `api_key` starts with `at-`, CCR uses PAT auth. Otherwise it falls back to OAuth.

### Final step: Restart

```bash
docker compose restart ccr
```

## Authentication Fallback Order

The Codex transformer uses this order:

1. If `api_key` starts with `at-` → use PAT auth
2. Otherwise → use OAuth tokens from `~/.claude-code-router/codex_auth.json`
3. If neither is available → authentication fails

Use OAuth when you want browser-based sign-in and automatic token refresh. Use PAT when you want explicit static credentials in the provider config.

## Running with Docker

The OAuth callback uses port `1455`, which is mapped to the CCR server port in `docker-compose.yml` (`"1455:3456"`). When running in Docker and using OAuth:

```bash
docker exec -it claude-code-router ccr codex-auth
```

The CLI prints a URL to open in your host browser. After signing in, the browser redirects to `http://localhost:1455/auth/callback`, which Docker forwards to the container. Tokens persist across container restarts via the volume-mounted `./ccr-config` directory.

PAT auth does not require the browser flow, but it still uses the same provider configuration inside the container.

## Provider Configuration Notes

- Use `api_base_url`, not `baseUrl`, in `config.json`
- Use `api_key`, not `apiKey`, in `config.json`
- The `api_key` value may be either:
  - `oauth_dummy_key` (or another placeholder) for OAuth mode
  - a real PAT starting with `at-` for PAT mode
- The provider still uses the `codex` transformer in both modes
- `ccr model get codex` works with either auth mode

## Transformer Behavior

The `codex` transformer:

- converts the unified request into the ChatGPT backend format
- authenticates using either OAuth tokens or a PAT
- resolves and sends `ChatGPT-Account-ID` automatically
- adds `X-OpenAI-Fedramp: true` when required by the authenticated account
- converts streaming Responses-style events back into Claude Code-compatible output

## When to use `ccr codex-auth`

Run `ccr codex-auth` when:

- you want OAuth instead of a PAT
- your OAuth tokens expired or were revoked
- you removed a PAT from config and want to fall back to OAuth again

You do **not** need `ccr codex-auth` when `api_key` already contains a valid PAT starting with `at-`.

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

**OAuth token expired or invalid**: Re-run `ccr codex-auth` to refresh the token.

**PAT rejected**: Ensure `api_key` contains the full PAT and that it starts with `at-`.

**Provider not found**: Ensure the provider name in your config matches `body.model` (e.g., `codex,gpt-5`).

**Wrong config fields**: Use `api_base_url` and `api_key` in `config.json`, not `baseUrl` / `apiKey`.

**Unexpected OAuth fallback**: If PAT mode did not activate, verify that `api_key` begins with `at-` after trimming whitespace.

**No auth available**: Configure either a PAT in `api_key` or OAuth tokens via `ccr codex-auth`.

## Related Docs

- [CLI auth commands](/docs/cli/commands/auth)
- [Claude subscription guide](/docs/server/guides/claude-auth)
- [Providers configuration](/docs/server/config/providers)
- [Transformers configuration](/docs/server/config/transformers)

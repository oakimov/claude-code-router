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

Two transformers are required, in this order:

- `claude-auth` — loads/refreshes the OAuth access token, classifies the calling client, and builds the identity/billing headers described below. It hands off body and URL construction to `Anthropic`.
- `Anthropic` — owns the `POST /v1/messages` client route and builds the actual Anthropic Messages wire body (`transformRequestIn`) and converts the SSE/JSON response back to Unified (`transformResponseOut`). It detects that `claude-auth` is present earlier in the same provider's chain and skips setting its own `Authorization`/`x-api-key` headers, so `claude-auth`'s Bearer token is never overwritten.

Requests can reach this chain from any inbound protocol CCR accepts — Anthropic Messages (`/v1/messages`), OpenAI Chat Completions (`/v1/chat/completions`), or OpenAI Responses (`/v1/responses`). Every inbound request is normalized to the internal Unified format before routing, so a non-Anthropic client (e.g. an OpenAI-shaped tool) can be routed to a `claude-auth` provider exactly like an Anthropic-shaped one; it is simply treated as a non-Claude-Code client (see below).

### Client classification

`claude-auth` decides how to build the outbound request based on a single check: does the inbound `User-Agent` start with `claude-cli/`? (`isClaudeCodeClient()` in `claude-auth.transformer.ts`.) This produces two branches:

- **Real Claude Code client** — the request already looks like genuine Claude Code traffic. `claude-auth` forwards the client's own identity headers, system blocks, and `anthropic-beta` value untouched (merging in only the OAuth beta), because re-deriving them would risk *diverging* from what Claude Code actually sent.
- **Any other client** (OpenAI SDKs, Anthropic SDKs, custom tools, Cursor, etc.) — the request does not look like Claude Code, so `claude-auth` synthesizes the billing block, system identity, identity headers, and `anthropic-beta` value that Claude Code itself would send for an equivalent request, so Anthropic sees the same shape of traffic regardless of which client CCR is fronting.

### Outbound headers

| Header | Real Claude Code client | Other clients |
|---|---|---|
| `Authorization` | `Bearer <access_token>` from `claude_auth.json`, refreshed automatically on expiry | same |
| `Content-Type` | `application/json` (set by `Anthropic`) | same |
| `anthropic-version` | `2023-06-01` (set by `Anthropic`) | same |
| `anthropic-beta` | Client's own value merged with `oauth-2025-04-20` | Synthesized from the model capability catalog — see below |
| `User-Agent` | Forwarded verbatim | `ANTHROPIC_USER_AGENT` env override, else `claude-cli/${CC_VERSION} (external, cli)` |
| `x-app` | Forwarded verbatim | `cli` |
| `x-claude-code-session-id` | Forwarded verbatim | Synthesized UUID, cached per process |
| `x-client-request-id` | Forwarded verbatim | Synthesized UUID, fresh per request |
| `anthropic-dangerous-direct-browser-access` | Forwarded verbatim | `true` |
| `x-stainless-arch` / `-lang` / `-os` / `-package-version` / `-retry-count` / `-runtime` / `-runtime-version` / `-timeout` | Forwarded verbatim | Synthesized from the current process (arch/OS/Node version) plus a fixed Anthropic SDK package version |

Hop-by-hop headers (`connection`, `host`, `accept-encoding`, `content-length`) sent by the client are never forwarded. The billing marker (`x-anthropic-billing-header`) is **not** an HTTP header — see [Billing and identity system blocks](#billing-and-identity-system-blocks).

#### `anthropic-beta` header logic

`oauth-2025-04-20` is always included — Anthropic requires it for subscription OAuth Bearer auth.

**Real Claude Code client:** the client's own `anthropic-beta` tokens are preserved as-is and `oauth-2025-04-20` is appended if not already present (case-insensitive dedupe). Nothing else is added or removed.

**Other clients:** the value is built from the [model capability catalog](#model-capability-catalog), mirroring what Claude Code itself sends for that model:

- `claude-code-20250219` and `oauth-2025-04-20` — always
- `context-1m-2025-08-07` — only when the requested model id carries the `[1m]` suffix (see [1M context](#1m-context))
- `interleaved-thinking-2025-05-14`, `thinking-token-count-2026-05-13` — only for models the catalog marks as supporting extended thinking
- `context-management-2025-06-27` — only for models with the `context_management` capability
- `prompt-caching-scope-2026-01-05` — always
- `mid-conversation-system-2026-04-07` — only for models with the `mid_conv_system` capability
- `advanced-tool-use-2025-11-20` — always
- `effort-2025-11-24` — only for models with the `effort` capability (an explicit per-model denylist, not a name-prefix match — e.g. `claude-sonnet-4-5` and `claude-haiku-4-5` are excluded but `claude-sonnet-4-6` is included)
- `fallback-credit-2026-06-01` — only for models with the `fast_mode` capability

Setting `ANTHROPIC_BETA_FLAGS` replaces this entire synthesized list wholesale (it does not merge). The URL also receives `?beta=true` in both branches. You do not need to configure any of this — it is applied automatically.

### Billing and identity system blocks

For **other clients**, `claude-auth` builds the Anthropic `system` array down to exactly two entries, matching what Claude Code itself sends:

1. A billing marker text block: `x-anthropic-billing-header: cc_version=${CC_VERSION}.${suffix}; cc_entrypoint=${CC_ENTRYPOINT}; cch=${sessionCch};` — despite its name this travels as `system[0]` text, **not** an HTTP header. `suffix` is a 3-hex-char digest derived from the first user message's text and the CLI version; `cch` is a random 5-hex-char value generated once per process (not derived from request content). Neither carries `cache_control`.
2. The identity text block: `You are Claude Code, Anthropic's official CLI for Claude.` (`system[1]`), preserving `cache_control` on whatever the caller's own first system entry carried if it started with (but wasn't identical to) this string.

Anthropic's OAuth billing validator inspects `system[]` content past the identity block and rejects requests carrying a foreign harness prompt there with an "out of extra usage" 400 — the calling client's own system prompt makes the traffic self-evidently non-Claude-Code even with correct headers. `claude-auth` works around this the same way third-party Claude Code OAuth shims do (e.g. `opencode-claude-auth`): whatever the caller supplied beyond `system[1]` is relocated into the first user message (prepended, in order) rather than left in `system[]`. The content still reaches the model unchanged, just as part of the first user turn. If there is no user message to attach it to, nothing is relocated — the caller's system content stays in `system[]` rather than being dropped.

For **other clients**, tool names are also rewritten to Claude Code's OAuth spelling before the body is built: `bash` becomes `mcp_Bash`, `read` becomes `mcp_Read`, and existing `mcp_...` names are left unchanged. Tool definitions, historical assistant `tool_calls`, and a forced `tool_choice` are kept consistent. The response path removes the synthetic prefix using a request-local name map, so the caller still receives its original tool names in both JSON and streaming responses.

For a **real Claude Code client**, its own system blocks — including its own billing marker and identity string — are forwarded exactly as sent; `claude-auth` does not touch, remove, or relocate them.

### Model capability catalog

`claude-model-catalog.ts` holds a per-model table (context window, native-1M support, max output tokens, default effort, and a `capabilities` list such as `effort`, `context_management`, `mid_conv_system`, `fast_mode`, `adaptive_thinking`) that drives both the beta synthesis above and a post-build adjustment pass (`applyClaudeModelCapabilityAdjustments`): stripping an unsupported `effort` field from `thinking`/`output_config`, shaping the `thinking` block (`adaptive` vs `enabled`), and clamping `max_tokens` to the model's known upper bound. This replaces per-model conditionals with a single table lookup, keyed by a normalized model id (CCR's `provider,` prefix, the `[1m]` marker, and Anthropic's `-YYYYMMDD` date suffix are all stripped before lookup).

### 1M context

Claude Code strips the `[1m]` marker from the wire `model` field and adds the `context-1m-2025-08-07` beta only when the requested model id carries that marker; native-1M models get the larger window without any beta. CCR follows the identical rule for both client branches — the marker is never used to reject, downgrade, or reroute a request.

### Prompt caching: exact vs. normalized

When a request reaches `Anthropic` through the same-protocol, same-destination path (exact Anthropic passthrough), the caller's own `cache_control` placement is preserved as sent — `context.protocolContext.anthropicCacheMode === "preserve"` skips automatic cache rewriting. Cross-protocol projection (e.g. an OpenAI-shaped request routed to an Anthropic destination) instead runs `applyRawAnthropicPromptCaching` to insert reasonable cache breakpoints, since the source protocol carries no `cache_control` concept of its own. This only affects *where* cache markers are placed — it has no bearing on the client-classification/header logic above.

### Auth recovery

`claude-auth` returns an `__authRecovery` hook that runs on a 401: it reloads `claude_auth.json` in case another process (e.g. a concurrent `ccr claude-auth` re-login) rotated the token externally, and only refreshes-and-saves if it didn't. It never falls through to an unauthenticated retry.

### Usage parity, not usage suppression

CCR's goal is to make requests **indistinguishable from genuine Claude Code traffic**, not to minimize account usage. Concretely:

- No `/count_tokens` preflight is added to the normal Messages path — Claude Code doesn't preflight either.
- No local 200,000-token cap is imposed. If the account permits extra usage and the client sends a larger request, CCR sends it as-is; if Anthropic rejects it, the upstream error is preserved unchanged.
- `[1m]` and native-1M models behave exactly as described above — never suppressed to save usage.
- `Router.longContext` remains an independent operator routing feature; this integration does not force an API-key long-context lane or rewrite that routing choice.
- Response headers such as `anthropic-ratelimit-unified-overage-in-use` are preserved through conversion and logged at debug level as usage observability — they are not treated as a warning condition, since the same direct Claude Code request would use overage too.

### Environment overrides

| Variable | Effect |
|---|---|
| `ANTHROPIC_CLI_VERSION` | Overrides the `CC_VERSION` used in the billing marker and the synthesized `User-Agent` (default `2.1.220`) |
| `CLAUDE_CODE_ENTRYPOINT` | Overrides the `cc_entrypoint` value in the billing marker and the synthesized `User-Agent` (default `cli`) |
| `ANTHROPIC_USER_AGENT` | Overrides the synthesized `User-Agent` header outright (non-Claude-Code branch only; a real Claude Code client's own `User-Agent` is always forwarded verbatim) |
| `ANTHROPIC_BETA_FLAGS` | Replaces the entire synthesized `anthropic-beta` value (non-Claude-Code branch only) |

No `claudeAuth.*` config flags exist for any of this — behavior is derived automatically from the request.

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

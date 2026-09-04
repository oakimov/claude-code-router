---
sidebar_position: 2
---

# Claude Subscription Integration

Claude Code Router can route requests through your **existing Claude subscription** (Pro or Max) using OAuth authentication. This lets you leverage your Claude.ai subscription directly — no separate API key needed.

:::warning Consult provider Terms & Conditions

This integration authenticates against a third-party service using your own account. Before using it, review the provider's Terms & Conditions — access may be limited by subscription tier, region, or the provider's service terms, and using client credentials outside the client they were issued for may violate those terms. You use this functionality at your own risk; CCR provides it for interoperability only and does not guarantee continued access to any third-party service.

See [DISCLAIMER.md](https://github.com/oakimov/claude-code-router/blob/main/DISCLAIMER.md) for the project's interoperability statement.
:::

## How It Works

1. `ccr claude-auth` generates a PKCE challenge and prints an authorization URL from `claude.ai`
2. You open the URL in your browser and sign into your Claude account
3. Claude redirects to `http://localhost:1455/callback`, where the CCR server exchanges the authorization code for tokens
4. Tokens are saved to `~/.claude-code-router/claude_auth.json`
5. You return to the terminal and press Enter — the CLI confirms the tokens were saved
6. The `claude-auth` transformer reads the access token and injects it as a `Bearer` token on OAuth provider requests
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

For subscription OAuth, two transformers are required, in this order:

- `claude-auth` — loads/refreshes the OAuth access token and owns OAuth authentication/recovery. The shared Anthropic client policy handles client classification and third-party emulation.
- `Anthropic` — owns the `POST /v1/messages` client route and builds the actual Anthropic Messages wire body (`transformRequestIn`) and converts the SSE/JSON response back to Unified (`transformResponseOut`). It detects that `claude-auth` is present earlier in the same provider's chain and skips setting its own `Authorization`/`x-api-key` headers, so `claude-auth`'s Bearer token is never overwritten.

The same shared policy also applies to a direct `Anthropic` provider configured with API-key auth. It does not apply to arbitrary Anthropic-compatible gateways.

Requests can reach either in-scope configuration from any **chat** inbound
protocol CCR accepts — Anthropic Messages (`/v1/messages`), OpenAI Chat
Completions (`/v1/chat/completions`), or OpenAI Responses (`/v1/responses`).
Every chat inbound request is normalized to the internal Unified format for
routing, while native Anthropic Desktop/CLI requests retain their original wire
body for egress. FIM (`/v1/fim/completions`) is a separate pipeline and does not
use `claude-auth`.

### Client classification

CCR classifies Anthropic Messages requests using a client fingerprint before normalization:

- **Claude Desktop** — recognizes either the top-bar transport (`anthropic-desktop-topbar: 1` plus the Anthropic JS SDK/header fingerprint) or the current 3P Agent SDK transport (a complete native CLI-shaped fingerprint whose UA contains `claude-desktop`/`claude-desktop-3p` and `agent-sdk/<version>`). Desktop 3P currently bundles its own Claude Code and SDK versions; CCR preserves those versions rather than replacing them with its emulation profile.
- **Claude Code CLI** — requires the `claude-cli/<version>` UA, `x-app: cli`, session/Stainless headers, and the native billing/identity body shape, without a Desktop Agent SDK entrypoint. A UA alone is not enough.
- **Other** — incomplete/unknown fingerprints and all OpenAI Chat/Responses clients. This path receives the versioned Claude Code emulation when routed to an in-scope Anthropic provider.

Native Desktop and CLI requests use raw Anthropic request/response pass-through. CCR changes only the routed model, provider URL, provider credential, and transport-managed headers. The classifier is a compatibility fingerprint, not authentication.

### Outbound headers

| Header | Native Desktop/CLI | Other clients |
|---|---|---|
| `Authorization` / `x-api-key` | Provider-owned credential: API key for direct `Anthropic`, OAuth Bearer for `claude-auth` | Provider-owned credential for the selected profile |
| `Content-Type` | `application/json` (set by `Anthropic`) | same |
| `anthropic-version` | `2023-06-01` (set by `Anthropic`) | same |
| `anthropic-beta` | Preserved; OAuth additionally ensures `oauth-2025-04-20` | Synthesized from the 2.1.226 profile; API-key mode does not add the OAuth beta |
| Application headers | Forwarded verbatim, including Desktop custom headers | Synthesized as the 2.1.226 CLI profile |

Hop-by-hop headers (`connection`, `host`, `accept-encoding`, `content-length`) sent by the client are never forwarded. The billing marker (`x-anthropic-billing-header`) is **not** an HTTP header — see [Billing and identity system blocks](#billing-and-identity-system-blocks).

#### `anthropic-beta` header logic

`oauth-2025-04-20` is required for subscription OAuth Bearer auth and is included only on the OAuth variant.

**Native Desktop/CLI:** application headers and body shapes are preserved. OAuth adds the required `oauth-2025-04-20` token; API-key auth does not invent OAuth headers.

**Other clients:** the value is built from the frozen Claude Code 2.1.226 profile and model capability catalog. OAuth mode includes `oauth-2025-04-20`; API-key mode does not:

- `claude-code-20250219` — ordinary non-Haiku profiles (the current CLI omits it for ordinary Haiku requests)
- `oauth-2025-04-20` — OAuth mode only
- `context-1m-2025-08-07` — only when the requested model id carries the `[1m]` suffix (see [1M context](#1m-context))
- `interleaved-thinking-2025-05-14`, `thinking-token-count-2026-05-13` — only for models the catalog marks as supporting extended thinking
- `context-management-2025-06-27` — only for models with the `context_management` capability
- `prompt-caching-scope-2026-01-05` — always
- `mid-conversation-system-2026-04-07` — only for models with the `mid_conv_system` capability
- `advanced-tool-use-2025-11-20` — only for an explicit tool-search request; not added to the ordinary emulation profile
- `effort-2025-11-24` and `fallback-credit-2026-06-01` — request/feature-gated optional betas; not added unconditionally by the emulation profile

Setting `ANTHROPIC_BETAS` appends comma-separated custom beta tokens to the synthesized capability list; OAuth still ensures its required `oauth-2025-04-20` token. The URL also receives `?beta=true` in both branches. You do not need to configure any of this — it is applied automatically.

Beta tokens are emitted only in the `anthropic-beta` HTTP header. CCR does not put a `betas` field in the Messages JSON body; that name is an Anthropic SDK option which the SDK itself removes before sending the request.

### Billing and identity system blocks

For **other clients routed to an in-scope Anthropic provider**, CCR builds the Anthropic `system` array down to the Claude Code 2.1.226 profile:

1. A billing marker text block: `x-anthropic-billing-header: cc_version=${CC_VERSION}.${suffix}; cc_entrypoint=unknown; cch=00000;` for the first-party Anthropic profile — despite its name this travels as `system[0]` text, **not** an HTTP header. `suffix` is a 3-hex-char digest derived from the first user message's text and the CLI version. The current `2.1.226` profile does not use the older random `cch` behavior. Neither carries `cache_control`.
2. The identity text block: `You are Claude Code, Anthropic's official CLI for Claude.` (`system[1]`). The emulation then applies the selected cache profile to this cacheable block; caller-authored cache markers are not allowed to override the pinned profile.

Anthropic's OAuth billing validator inspects `system[]` content past the identity block and rejects requests carrying a foreign harness prompt there with an "out of extra usage" 400 — the calling client's own system prompt makes the traffic self-evidently non-Claude-Code even with correct headers. CCR's third-party Anthropic emulation mirrors the technique used by Claude Code OAuth clients (e.g. `opencode-claude-auth`): whatever the caller supplied beyond `system[1]` is relocated into the first user message (prepended, in order) rather than left in `system[]`. The content still reaches the model unchanged, just as part of the first user turn. If there is no user message to attach it to, nothing is relocated — the caller's system content stays in `system[]` rather than being dropped.

For **other clients**, tool names are also rewritten to Claude Code's OAuth spelling before the body is built: `bash` becomes `mcp_Bash`, `read` becomes `mcp_Read`, and existing `mcp_...` names are left unchanged. Tool definitions, historical assistant `tool_calls`, and a forced `tool_choice` are kept consistent. The response path restores the caller's original tool names via a request-local name map in both JSON and streaming responses.

For **native Desktop and CLI**, their own system blocks — including billing, identity and opaque fields — are forwarded exactly as sent. No system prompt or cache transformation occurs.

### Model capability catalog

`claude-model-catalog.ts` holds a per-model table (context window, native-1M support, max output tokens, default effort, and a `capabilities` list such as `effort`, `context_management`, `mid_conv_system`, `fast_mode`, `adaptive_thinking`) that drives both the beta synthesis above and a post-build adjustment pass (`applyClaudeModelCapabilityAdjustments`): stripping an unsupported `effort` field from `thinking`/`output_config`, shaping the `thinking` block (`adaptive` vs `enabled`), and clamping `max_tokens` to the model's known upper bound. This replaces per-model conditionals with a single table lookup, keyed by a normalized model id (CCR's `provider,` prefix, the `[1m]` marker, and Anthropic's `-YYYYMMDD` date suffix are all stripped before lookup).

### 1M context

Claude Code strips the `[1m]` marker from the wire `model` field and adds the `context-1m-2025-08-07` beta only when the requested model id carries that marker; native-1M models get the larger window without any beta. CCR follows the identical rule for both client branches — the marker is never used to reject, downgrade, or reroute a request.

### Prompt caching: native vs. emulated

Native Desktop and CLI cache markers are preserved exactly, including feature-gated TTL/scope choices. Current Desktop 3P conversations run through Desktop's bundled Agent SDK and can therefore author Claude Code-shaped system/message breakpoints; observed requests use 5-minute ephemeral entries. CCR neither adds markers to a marker-less native request nor changes markers that are present. Only the `other` path authors cache fields, using the 2.1.226 profile: the billing block remains unmarked, cacheable system blocks receive the profile's cache control, and one message tail receives the final breakpoint. There is no universal CCR cache normalizer.

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
| `ANTHROPIC_CLI_VERSION` | Overrides the emulation profile's billing marker and synthesized `User-Agent` (default `2.1.226`) |
| `CLAUDE_CODE_ENTRYPOINT` | Overrides the `cc_entrypoint` value in the billing marker and the synthesized `User-Agent` (billing default `unknown`; User-Agent default `cli`) |
| `ANTHROPIC_USER_AGENT` | Overrides the synthesized `User-Agent` header outright (non-Claude-Code branch only; a real Claude Code client's own `User-Agent` is always forwarded verbatim) |
| `ANTHROPIC_CUSTOM_HEADERS` | Adds newline-delimited custom application headers to the synthesized CLI profile; credentials and transport headers are ignored |
| `ANTHROPIC_BETAS` | Appends comma-separated custom betas to the synthesized `anthropic-beta` value (non-Claude-Code branch only) |

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

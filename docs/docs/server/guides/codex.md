---
sidebar_position: 1
---

# Codex (ChatGPT) Integration

Claude Code Router can use a **ChatGPT/Codex subscription** to route Claude Code requests through OpenAI's models. The Codex backend powers OpenAI's ChatGPT product and this integration lets you leverage that subscription with Claude Code.

:::warning Consult provider Terms & Conditions

This integration authenticates against a third-party service using your own account. Before using it, review the provider's Terms & Conditions — access may be limited by subscription tier, region, or the provider's service terms, and using client credentials outside the client they were issued for may violate those terms. You use this functionality at your own risk; CCR provides it for interoperability only and does not guarantee continued access to any third-party service.

See [DISCLAIMER.md](https://github.com/oakimov/claude-code-router/blob/main/DISCLAIMER.md) for the project's interoperability statement.
:::

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
6. The `codex` transformer (paired with `openai-responses`) reads the access token and uses it to authenticate API requests. `openai-responses` owns the Responses wire; `codex` supplies ChatGPT auth, headers, `store: false`, and `stream: true`.
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

Add the Codex provider to your `~/.claude-code-router/config.json`. Always pair `codex` with `openai-responses`: the latter owns the Responses API body (including encrypted reasoning items); `codex` only authenticates and applies ChatGPT backend constraints (`store: false`, `stream: true`, no `role: system` in `input`).

```json
{
  "Providers": [
    {
      "name": "codex",
      "api_base_url": "https://chatgpt.com/backend-api/codex",
      "api_key": "oauth_dummy_key",
      "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
      "transformer": {
        "use": ["openai-responses", "codex"]
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
        "use": ["openai-responses", "codex"]
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
- Model discovery sends the current Codex CLI `client_version` because the ChatGPT backend can gate newly released Codex model slugs by client version. CCR defaults to the latest stable version known at release time; override it with `codex_client_version` on the provider or `CCR_CODEX_CLIENT_VERSION` when testing a newer Codex CLI rollout. Runtime Codex requests are handled by the core Codex transformer, which presents the Codex CLI request version and identity headers without depending on CCR's CLI package.
- The Codex CLI version CCR identifies as (currently `0.159.1`) is set by the `CODEX_CLI_VERSION` environment variable on the CCR server. It sets the runtime `User-Agent` and `client_version` and the model-discovery default. Keep it current with the latest stable `@openai/codex` release so newly gated slugs (such as `gpt-6.1-sol`) appear in discovery, and at or above upstream's per-model minimum (`gpt-6.1-sol` / Astra ≥ `0.153.0`, GPT-6 Sol/Luna ≥ `0.155.0`).

## Transformer Behavior

The `codex` transformer:

- converts the unified request into the ChatGPT backend format
- authenticates using either OAuth tokens or a PAT
- resolves and sends `ChatGPT-Account-ID` automatically
- adds `X-OpenAI-Fedramp: true` when required by the authenticated account
- converts streaming Responses-style events back into the **inbound** client protocol

## Model Handling

CCR sends each OpenAI model what Codex CLI sends it. The per-model data is
imported from Codex's bundled catalog (`codex-rs/models-manager/models.json`)
into `packages/core/src/utils/codex-model-catalog.ts`; refresh it when Codex
adds or retires models.

- **Reasoning effort** is limited to the model's supported levels. `none` /
  `minimal` move up to the lowest level and anything above the top level
  moves down to it. `ultra` is a Codex picker alias and is never sent: it
  becomes the model's multi-agent effort (`xhigh` for `gpt-6-astra` /
  `gpt-6.1-sol`), else `max`, else the highest level below `ultra`. Applies to
  every Responses destination (`openai-responses`) and to same-protocol
  Codex wire-keep. Models outside the catalog pass through unchanged; GPT-6
  slugs newer than the table use the GPT-6 family's levels.
- **Responses Lite** (Codex provider only; models with `use_responses_lite`):
  the request carries `x-openai-internal-codex-responses-lite: true`, tools
  move from `tools` into a leading `{type: "additional_tools", role:
  "developer"}` input item (function and custom tools grouped in the
  `functions` namespace), `parallel_tool_calls` is `false`, `reasoning.context`
  is `all_turns`, and input images carry no `detail`.
- **Verbosity** is sent as `text.verbosity`. The client's value wins, then the
  provider's `verbosity`, then the value implied by `REASONING_AUTO_SUMMARY`
  (`detailed` → `high`, `auto` → `medium`, `concise` → `low`) for catalog
  models that support verbosity.
- **Fast / priority tier** is never selected; Responses clients cannot send
  `service_tier`.

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

## Stream Bootstrap Buffering

The ChatGPT backend can smuggle overload/quota rejections *inside* an
HTTP 200 SSE stream (after the handshake frames) instead of returning a
retryable status. Enable opt-in buffering so those fail over transparently:

```json
{
  "name": "codex",
  "api_base_url": "https://chatgpt.com/backend-api/codex",
  "api_key": "oauth_dummy_key",
  "models": ["gpt-5"],
  "transformer": {
    "use": ["openai-responses", ["codex", { "streamBootstrapBuffering": true }]]
  }
}
```

Held frames (handshake, `*.added`, heartbeats) stay uncommitted until
output, a terminal event, or a configured budget releases the buffer.
A capacity error in a `response.failed` / `error` event cancels the rejected
upstream stream and throws a fallback-eligible error *before* downstream
headers commit. Classification follows Codex CLI: the exact error `code`
(plus `type` for plan and quota errors), never message text.

| Upstream error | CCR error |
| --- | --- |
| `server_is_overloaded` | 503 `server_overloaded` |
| `rate_limit_exceeded`, `slow_down`, `flex_unavailable` | 429 `rate_limit_exceeded`, with `Retry-After` from the event (`error.headers`, else "try again in Ns") |
| `insufficient_quota`, `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded`, `usage_not_included`; type `usage_limit_reached` / `usage_not_included` / `insufficient_quota` | 429 `usage_limit_reached` |

Every other failure (`context_length_exceeded`, `invalid_prompt`, policy
codes, unknown codes) reaches the client unchanged, even when its message
says "try again": another model would fail the same way. Error-like words in
generated text or tool arguments are not signals either. A transport failure while buffering
propagates to the normal error/fallback path instead of returning a successful
truncated stream. See
[Transformers → codex](/docs/server/config/transformers#codex) for budgets.
Pair with a `continue-and-cooldown`
[request-scoped-error](/docs/server/config/routing#request-scoped-errors)
rule on `usage_limit_reached` so the exhausted model is skipped briefly.

## Multi-Agent Compatibility

Two opt-in top-level flags for Codex multi-agent (delegation) traffic:

```json
{
  "orphanDelegationCompatibility": true,
  "optimizeMultiAgentV2": true
}
```

- `orphanDelegationCompatibility` — `function_call_output` /
  `custom_tool_call_output` items without a non-empty string `call_id` become
  user messages instead of a 400. Only applies when the request carries
  `X-Openai-Subagent: collab_spawn`. Outputs with a non-empty string id are left
  correlated as tool results; absence of a matching local call alone does
  not enable this repair.
- `optimizeMultiAgentV2` — role-less `agent_message` input items become user
  messages instead of a 400. Text/image/file content parts and their
  validation are preserved. Role-bearing items keep their existing message
  projection regardless of the flag.

Both repairs apply to Unified conversion and Responses wire keep, including
fallback attempts, without rebuilding unrelated images, files, reasoning
ciphertext, or cache fields. Compatibility user messages arriving inside a
tool-call group are deferred until its pending tool results have been emitted.
These top-level flags apply to the main CCR namespace; preset namespaces
currently carry only their provider/router configuration and do not inherit them.

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

Codex catalog models (see [Model Handling](#model-handling)). Discover what
your account can use with `ccr model get codex`.

| Model | Reasoning levels | Responses Lite |
|-------|------------------|----------------|
| `gpt-6.1-sol`, `gpt-6-astra` | `low`–`max` (`ultra` → `xhigh`) | yes |
| `gpt-6-sol`, `gpt-5.6-sol`, `gpt-5.6-terra` | `low`–`max` (`ultra` → `max`) | yes |
| `gpt-6-luna`, `gpt-5.6-luna`, `codex-auto-review` | `low`–`max` | yes |
| `gpt-5.5` | `low`–`xhigh` | no |

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

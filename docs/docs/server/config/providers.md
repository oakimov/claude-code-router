---
sidebar_position: 2
---

# Providers Configuration

Detailed guide for configuring LLM providers.

## Provider Schema

In this fork, provider entries in `config.json` use the following fields:

```json
{
  "name": "provider-name",
  "api_base_url": "https://example.com/v1/chat/completions",
  "api_key": "$PROVIDER_API_KEY",
  "models": ["model-1", "model-2"],
  "transformer": {
    "use": ["OpenAI"]
  }
}
```

## Provider Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Unique provider identifier |
| `api_base_url` | string | Yes | Provider API base URL or endpoint |
| `api_key` | string | Yes | API key, placeholder, or subscription auth token marker depending on provider |
| `models` | string[] | No | List of available models |
| `transformer.use` | string[] | No | Transformers applied to the provider |

## Supported Providers

### DeepSeek

```json
{
  "name": "deepseek",
  "api_base_url": "https://api.deepseek.com/chat/completions",
  "api_key": "$DEEPSEEK_API_KEY",
  "models": ["deepseek-chat", "deepseek-reasoner"],
  "transformer": {
    "use": ["deepseek"]
  }
}
```

### Groq

```json
{
  "name": "groq",
  "api_base_url": "https://api.groq.com/openai/v1/chat/completions",
  "api_key": "$GROQ_API_KEY",
  "models": ["llama-3.3-70b-versatile"],
  "transformer": {
    "use": ["OpenAI"]
  }
}
```

### Gemini

```json
{
  "name": "gemini",
  "api_base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
  "api_key": "$GEMINI_API_KEY",
  "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
  "transformer": {
    "use": ["gemini"]
  }
}
```

### OpenRouter

```json
{
  "name": "openrouter",
  "api_base_url": "https://openrouter.ai/api/v1/chat/completions",
  "api_key": "$OPENROUTER_API_KEY",
  "models": ["anthropic/claude-sonnet-4", "google/gemini-2.5-pro-preview"],
  "transformer": {
    "use": ["openrouter"]
  }
}
```

### Mistral

```json
{
  "name": "mistral",
  "api_base_url": "https://api.mistral.ai/v1/chat/completions",
  "api_key": "$MISTRAL_API_KEY",
  "models": ["mistral-large-latest", "mistral-small-latest"],
  "transformer": {
    "use": ["mistral"]
  }
}
```

### Codex (ChatGPT)

Codex supports **two auth modes**:

- **OAuth** via `ccr codex-auth`
- **PAT** via `api_key` starting with `at-`

#### Codex with OAuth

```json
{
  "name": "codex",
  "api_base_url": "https://chatgpt.com/backend-api/codex",
  "api_key": "oauth_dummy_key",
  "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
  "transformer": {
    "use": ["openai-responses", "codex"]
  }
}
```

#### Codex with PAT

```json
{
  "name": "codex",
  "api_base_url": "https://chatgpt.com/backend-api/codex",
  "api_key": "at-your-personal-access-token",
  "models": ["gpt-5", "gpt-5-high", "gpt-5-mini"],
  "transformer": {
    "use": ["openai-responses", "codex"]
  }
}
```

If `api_key` starts with `at-`, CCR uses PAT auth. Otherwise it falls back to OAuth tokens from `~/.claude-code-router/codex_auth.json`.

### Cursor (SDK)

Routes through Cursor models via `@cursor/sdk`. Auth uses a dashboard key starting with `crsr_`, or the `CURSOR_API_KEY` environment variable.

```json
{
  "name": "cursor",
  "api_base_url": "https://cursor.com",
  "api_key": "$CURSOR_API_KEY",
  "models": ["composer-2", "claude-opus-4-8", "gpt-5.4"],
  "transformer": {
    "use": [
      [
        "cursor-sdk",
        {
          "cursorMode": "bridge"
        }
      ]
    ]
  }
}
```

Discover models with `ccr model get cursor`. See the [Cursor SDK guide](/docs/server/guides/cursor).

### Claude Subscription

Claude subscription auth uses OAuth via `ccr claude-auth` and requires the `claude-auth` + `Anthropic` transformer chain.

```json
{
  "name": "claude-subscription",
  "api_base_url": "https://api.anthropic.com",
  "api_key": "no-key",
  "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
  "transformer": {
    "use": ["claude-auth", "Anthropic"]
  }
}
```

### Antigravity

Routes through Google's Antigravity gateway with OAuth via `ccr antigravity-auth`. Chain `gemini` **before** `antigravity-auth`.

```json
{
  "name": "antigravity",
  "api_base_url": "https://daily-cloudcode-pa.sandbox.googleapis.com",
  "api_key": "oauth",
  "project_id": "$ANTIGRAVITY_PROJECT_ID",
  "models": [
    "gemini-3-pro-high",
    "gemini-3-flash",
    "claude-sonnet-4-6",
    "claude-opus-4-6-thinking"
  ],
  "transformer": {
    "use": [
      ["gemini", { "cachedContent": false, "thoughtSignatureFallback": "skip" }],
      "antigravity-auth"
    ]
  }
}
```

- **`cachedContent: false`** — required. Antigravity has no `cachedContents` resource; the Gemini default (`true`) causes 404s.
- **`thoughtSignatureFallback: "skip"`** — keep the default. On a missing tool-call thought signature, stamp Google's `skip_thought_signature_validator` sentinel so the gateway does not 400. Only set `"none"` if the endpoint rejects that sentinel.

Full option reference: [Transformers → gemini](/docs/server/config/transformers#options-cachedcontent-and-thoughtsignaturefallback). See also [CLI auth commands](/docs/cli/commands/auth).

### Qwen Chat

Requires JWT authentication via `ccr qwen-auth`.

```json
{
  "name": "qwen",
  "api_base_url": "https://qwen.aikit.club/v1/chat/completions",
  "api_key": "qwen-placeholder",
  "models": ["qwen-max", "qwen-plus", "qwen-turbo"],
  "transformer": {
    "use": ["qwen-auth", "reasoning", "OpenAI"]
  }
}
```

### Chrome On-Device (Gemini Nano)

Requires the bridge process via `ccr chrome-bridge`.

```json
{
  "name": "chrome-nano",
  "api_base_url": "http://127.0.0.1:3457",
  "api_key": "placeholder",
  "models": ["gemini-nano"],
  "transformer": {
    "use": ["chrome-on-device", "tooluse"]
  }
}
```

### OpenCode Zen

`opencode-headers` presents requests as the OpenCode CLI (session, request and
project headers) and owns the upstream call to Zen. Keep it provider-wide,
after the protocol owner.

```json
{
  "name": "opencode",
  "api_base_url": "https://opencode.ai/zen/v1/responses",
  "api_key": "$OPENCODE_API_KEY",
  "models": ["muse-spark-1.3-contributor-free"],
  "transformer": {
    "use": ["openai-responses", "opencode-headers"]
  }
}
```

**Session and retries.** Each conversation gets a persisted `x-opencode-session`
(stable across restarts), keyed by the client's own session id when it sends
one (Claude Code metadata, or headers such as `x-session-id`) and by a
first-message fingerprint otherwise. Zen routes by that session. Transient
failures (408/409/425/429/5xx, network errors) retry on the same session;
Zen's deterministic bad-bucket errors (`401 No provider available`,
`400 … Upstream request failed`) re-roll the session. When retries run out,
bad-bucket errors surface as `503`, so CCR's normal fallback can try another
model.

**Free-tier models** (model id ending in `-free` on an `opencode.ai/zen/`
endpoint) pass Zen's free-tier check only when requests look like the OpenCode
client. For those models CCR:

- always streams upstream. A non-streaming client still receives a JSON
  response, collected from the stream.
- adds function tools named exactly `read` and `shell` when the client lacks
  them. Each stub clones a matching client tool's schema (`Read` / `read_file`,
  `Bash` / `pwsh` / `exec`, or a `run_code`-style tool), and calls to a stub
  are renamed back to that client tool in the response. A stub with no
  counterpart tells the model not to call it.
- sets `prompt_cache_key` to the Zen session id (Responses wire). Zen stalls
  with `response.incomplete` when given a foreign key.
- asks for `reasoning.summary: "detailed"` when the client reasons without
  stating a summary preference.
- fails over stalled streams: CCR holds the response until Zen sends real
  output. It returns `504` (fallback-eligible) if no complete event arrives
  within 30 s or no output within 60 s. A reasoning item that streams nothing
  may stay silent for up to 5 minutes. After output starts, 60 s without data
  (5 minutes inside reasoning) ends the stream with an error.

Paid Zen models are sent unchanged.

## Model Selection

When selecting a model in routing, use the format:

```
{provider-name},{model-name}
```

For example:

```
deepseek,deepseek-chat
codex,gpt-5
cursor,composer-2
claude-subscription,claude-sonnet-4-6
chrome-nano,gemini-nano
```

## FIM providers

FIM uses a **separate** endpoint (`POST /v1/fim/completions`) and dedicated `fim.*` transformers. Prefer a second provider entry next to your chat provider (same API key). Full walkthroughs for **Codestral** and **local Qwen via LM Studio**: [FIM Completions API](/docs/server/api/fim-completions-api).

| Transformer | Typical upstream | Notes |
|-------------|------------------|-------|
| `fim.mistral` | Codestral `/v1/fim/completions` | Same-kind request/response passthrough when inbound is mistral |
| `fim.qwen` | LM Studio or DashScope `/v1/completions` + FIM tokens | Cross-family: response encoded to **inbound** wire (v1 → Codestral shape) |
| `fim.deepseek` | DeepSeek `/beta/completions` | Cross-family: same inbound-wire encode |

Client response follows inbound kind, not a forever-fixed Codestral shape.

## Auth Notes by Provider

- **Standard API providers** usually use a normal API key in `api_key`
- **Codex** can use either OAuth (`ccr codex-auth`) or a PAT in `api_key`
- **Cursor** uses a `crsr_` key in `api_key` or `CURSOR_API_KEY` (no OAuth CLI)
- **Claude subscription** uses OAuth tokens managed by `ccr claude-auth`; `api_key` is just a placeholder marker
- **Qwen** uses a JWT managed by `ccr qwen-auth`
- **Chrome on-device** uses a local bridge, so `api_key` is only a placeholder

## Related Docs

- [Codex integration guide](/docs/server/guides/codex)
- [Cursor SDK integration guide](/docs/server/guides/cursor)
- [Claude subscription guide](/docs/server/guides/claude-auth)
- [CLI auth commands](/docs/cli/commands/auth)
- [Transformers configuration](/docs/server/config/transformers)
- [Routing configuration](/docs/server/config/routing)

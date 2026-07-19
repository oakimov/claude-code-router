---
sidebar_position: 2
---

# Cursor SDK Integration

Claude Code Router can route Claude Code requests through **Cursor** models via the official `@cursor/sdk` package. Unlike HTTP providers, the `cursor-sdk` transformer owns the full upstream call in-process and returns OpenAI-compatible SSE/JSON that AnthropicTransformer converts back for Claude Code.

The default mode is **bridge**: Cursor chooses what to do next, but **Claude Code remains the tool host**. Cursor built-in tools are denied in the isolated workspace; host tools are exposed to the SDK as custom MCP tools (`custom-user-tools`).

## Prerequisites

- A Cursor account with an API key starting with `crsr_` (from the Cursor dashboard)
- Claude Code Router running (Docker Compose or local)
- **Node.js ≥ 22.13.0** when running from source or publishing packages (`@cursor/sdk` requires this engine)

## Authentication

Cursor auth does **not** use a browser OAuth CLI command. Resolve order:

1. Provider `api_key` that starts with `crsr_` (concrete key, not an unresolved `$…` / `${…}` placeholder)
2. Otherwise `CURSOR_API_KEY` from the environment

Recommended patterns:

```json
"api_key": "crsr_your_key_here"
```

or keep the secret out of the file:

```json
"api_key": "$CURSOR_API_KEY"
```

and export / inject the env var (Docker Compose already passes `CURSOR_API_KEY` into the container when set).

## Setup

### 1. Configure Provider

Add a Cursor provider to `~/.claude-code-router/config.json`:

```json
{
  "Providers": [
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
  ],
  "Router": {
    "default": "cursor,composer-2"
  }
}
```

Notes:

- Use `api_base_url` / `api_key` in `config.json` (not `baseUrl` / `apiKey`)
- `api_base_url` is a placeholder for provider identity; the SDK call does not use HTTP fetch to that URL
- Discover the live model list with `ccr model get cursor` (see below)

### 2. Restart

```bash
docker compose restart ccr
# or
ccr restart
```

## Modes

Pass options on the transformer entry: `["cursor-sdk", { … }]`.

| Option | Default | Description |
|--------|---------|-------------|
| `cursorMode` | `"bridge"` | `bridge` — Claude Code hosts tools; Cursor builtins denied. `plan` — text/reasoning only (no tool execution). `agent` — Cursor agent mode; optional `cursorCwd` for local agent cwd. |
| `cursorCwd` | (session workspace) | Used when `cursorMode` is `agent` to set the SDK local cwd. |
| `sandboxEnabled` | `false` | Opt-in Cursor local sandbox. Forced off in Docker / unsupported hosts. Can also enable with `CCR_CURSOR_SANDBOX=1` on a supported desktop host. |

### Bridge mode (recommended for Claude Code)

1. CCR starts / resumes an in-process Cursor agent session
2. Host tools from the Claude Code request are registered as SDK custom tools
3. When Cursor wants a tool, CCR parks the call and streams OpenAI-style `tool_calls` back to Claude Code
4. Claude Code runs the tool and posts results; CCR resolves the parked promises and continues the stream
5. Deny-hooks in the isolated workspace block Cursor built-ins so filesystem/shell stay with Claude Code

Isolated workspaces live under:

```text
~/.claude-code-router/cursor-sdk-workspaces/
```

### Plan / agent modes

- **plan** — planning/chat assistant; do not execute tools
- **agent** — Cursor agent with its own local tooling semantics; prefer bridge when you want Claude Code to own tools

## Model Discovery

Cursor models are listed through `@cursor/sdk` (not a REST `/models` URL):

```bash
ccr model get cursor
```

CCR detects a Cursor provider when the provider name is `cursor` or when `transformer.use` includes `cursor-sdk`. Auth for discovery uses the same `crsr_` / `CURSOR_API_KEY` rules as the server.

After syncing models into `config.json`, restart so the running server picks up the list.

## Sessions

Cursor conversations are **stateful in-process**:

- Session key from `x-ccr-cursor-session` header, Claude `metadata.user_id` (`…_session_…`), or a hash of model + system/first user text
- LRU cap of **32** sessions; idle TTL **15 minutes**
- In-flight sessions (live stream, running run, or parked tools) are not idle-evicted
- If the stream dies mid-turn (disconnect / cancel), the next request uses a slim follow-up prompt when the agent session already has history

## Running with Docker

`@cursor/sdk` ships platform-native packages and is installed into the runtime image separately from the pnpm workspace (version taken from `packages/server/package.json`).

Ensure the container receives the key:

```yaml
environment:
  - CURSOR_API_KEY=${CURSOR_API_KEY}
```

Local sandboxing is disabled inside Docker even if requested.

## Transformer Behavior

The `cursor-sdk` transformer:

- runs `@cursor/sdk` Agent create/send/stream in-process
- returns a ready `Response` via `__providerResponse` (skips HTTP `fetch` to the provider URL)
- emits OpenAI chat.completion / chat.completion.chunk SSE for AnthropicTransformer
- supports streaming and non-streaming Claude Code requests
- maps effort / reasoning fields onto SDK model selection when available

## Usage

```json
{
  "Router": {
    "default": "cursor,composer-2",
    "think": "cursor,claude-opus-4-8",
    "background": "cursor,claude-haiku-4-5"
  }
}
```

## Troubleshooting

**Cursor API key not found**: Set `Providers[].api_key` to a key starting with `crsr_`, or export `CURSOR_API_KEY`. Placeholders like `$CURSOR_API_KEY` only work when the env var is actually set.

**Wrong key prefix**: Cursor dashboard keys start with `crsr_`, not `sk-`.

**Node engine errors**: Local install / publish requires Node **≥ 22.13.0**.

**No models from `ccr model get cursor`**: Confirm auth and that the provider uses `cursor-sdk`. Restart after writing models.

**Tools run inside Cursor instead of Claude Code**: Use `cursorMode: "bridge"` (default) and do not enable unsupported sandbox options that change hosting assumptions.

**Session disposed / stream dropped under load**: Sessions are capped (32) and idle-evicted after 15 minutes when not in flight. Prefer stable session headers for long conversations.

## Related Docs

- [Providers configuration](/docs/server/config/providers)
- [Transformers configuration](/docs/server/config/transformers)
- [Model discovery](/docs/server/guides/model-discovery)
- [`ccr model get`](/docs/cli/commands/model-get)

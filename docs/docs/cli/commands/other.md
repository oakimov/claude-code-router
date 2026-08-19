---
sidebar_position: 4
---

# Other Commands

Additional CLI commands for managing Claude Code Router.

## ccr stop

Stop the running server.

```bash
ccr stop
```

## ccr restart

Restart the server.

```bash
ccr restart
```

## ccr code

Execute a claude command through the router.

```bash
ccr code [args...]
```

## ccr ui

Open the Web UI in a browser.

```bash
ccr ui
```

The UI includes a **Debug agent** playground (`/debug`):

- **CCR vs Direct** — CCR loops through the local router (`provider,model`). Direct calls the provider `api_base_url` with the bare model name. In both modes, API keys, `$ENV` placeholders, and OAuth tokens are taken from CCR config on the server — they are never required in the browser.
- **Inbound protocol** (CCR) — Chat Completions (`/v1/chat/completions`), Messages (`/v1/messages`), or Responses (`/v1/responses`).
- **System prompt, reasoning effort, and tools** — Instructions opens with the default Debug agent system prompt. Load a system prompt (text), tools (JSON array or object), or user prompt (text) from a file. The user message is sent and cleared; system prompt and tools stay for the browser session. Switch models to continue the same conversation (messages, reasoning, and tool calls). Optional CCR effort: `none` / `minimal` / `low` / `medium` / `high` / `xhigh` / `max` / `ultra`. Tool calls are **stubbed** (inspect-only; user JSON is never executed).
- **Token usage** — each assistant message shows a collapsed token total that expands to reads, writes, cached reads, and cached writes.
- **Renew OAuth** — force-refresh for `claude-auth`, Codex OAuth (not PAT), `qwen-auth`, `antigravity-auth`, and `xai-auth`. Tokens are never returned to the browser.

The Chat tab streams through `POST /api/debug/chat`. The Body tab prerenders the same turn as JSON for the selected inbound endpoint (Chat Completions, Messages, or Responses). You can edit that JSON and Send it to the request URL. The response pane shows the latest raw body, response headers, and HTTP status. Drag the rule between the request and response panes to resize them. Copy cURL copies the full command with `PLACEHOLDER` for authorization keys.

## ccr activate

Output shell environment variables for integration with external tools.

```bash
ccr activate
```

## Global Options

These options can be used with any command:

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help |
| `-v, --version` | Show version number |
| `--config <path>` | Path to configuration file |
| `--verbose` | Enable verbose output |

## Examples

### Stop the server

```bash
ccr stop
```

### Restart with custom config

```bash
ccr restart --config /path/to/config.json
```

### Open Web UI

```bash
ccr ui
```

## Related Documentation

- [Getting Started](/docs/cli/intro) - Introduction to Claude Code Router
- [Configuration](/docs/server/config/basic) - Configuration guide

---
title: Basic Configuration
---

# Basic Configuration

CLI uses the same configuration file as Server: `~/.claude-code-router/config.json`

## Configuration Methods

You can configure Claude Code Router in two ways:

### Option 1: Edit Configuration File Directly

Edit `~/.claude-code-router/config.json` with your favorite editor:

```bash
nano ~/.claude-code-router/config.json
```

### Option 2: Use Web UI

Open the web interface and configure visually:

```bash
ccr ui
```

## Restart After Configuration Changes

After modifying the configuration file or making changes through the Web UI, you must restart the service:

```bash
ccr restart
```

Or restart directly through the Web UI.

## Configuration File Location

```bash
~/.claude-code-router/config.json
```

## Minimal Configuration Example

```json5
{
  // API key (optional, used to protect service)
  "APIKEY": "your-api-key-here",

  // LLM providers
  "Providers": [
    {
      "name": "openai",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "$OPENAI_API_KEY",
      "models": ["gpt-4", "gpt-3.5-turbo"]
    }
  ],

  // Default routing
  "Router": {
    "default": "openai,gpt-4"
  }
}
```

## Environment Variables

Configuration supports environment variable interpolation:

```json5
{
  "Providers": [
    {
      "apiKey": "$OPENAI_API_KEY"  // Read from environment variable
    }
  ]
}
```

Set in `.bashrc` or `.zshrc`:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Common Configuration Options

### HOST and PORT

```json5
{
  "HOST": "127.0.0.1",  // Listen address
  "PORT": 3456          // Listen port
}
```

### Logging Configuration

```json5
{
  "LOG": true,          // Enable logging
  "LOG_LEVEL": "info"   // Log level
}
```

#### Capturing request and response messages

Two independent opt-in flags cover every wire direction. Both require
`LOG_LEVEL: "debug"` (or lower) so the records are emitted.

| Flag | What it logs |
|---|---|
| `LOG_REQUEST_BODY` | Full request bodies on **client→CCR** (all protocols) and **CCR→provider**, plus non-stream JSON on **CCR→client** |
| `LOG_SSE_EVENTS` | Every SSE event on **provider→CCR** and **CCR→client** |

Each record carries a `direction` field (`client→ccr`, `ccr→provider`,
`provider→ccr`, `ccr→client`) so you can grep one leg at a time.

```json5
{
  "LOG_LEVEL": "debug",
  "LOG_REQUEST_BODY": true,           // Off by default
  "LOG_REQUEST_BODY_MAX_BYTES": 32768, // Per-body / per-event cap
  "LOG_SSE_EVENTS": true              // Off by default
}
```

:::warning
This writes full conversation content — system prompts, user messages, tool
results, and streaming deltas — to `~/.claude-code-router/logs/`. Credentials
are redacted (`Authorization`, `api_key`, `access_token`, `sk-…` style keys),
and Responses `encrypted_content` blobs are replaced with
`[redacted-encrypted]`, but the prompts and readable reasoning summaries
themselves are not. Log files rotate at 50 MB with 3 kept. Enable it for a
debugging window, then turn it back off.
:::

Without `LOG_REQUEST_BODY`, Anthropic Messages clients still get the legacy
`type: "request body"` info log on inbound only. Responses and Chat Completions
inbound bodies are silent unless the flag is on.

Useful greps:

```bash
# What OpenCode / Claude Code sent CCR
rg '"direction":"client→ccr"' ~/.claude-code-router/logs/ccr.log

# What CCR posted upstream after transformers
rg '"direction":"ccr→provider"' ~/.claude-code-router/logs/ccr.log

# Raw Zen / provider SSE (includes reasoning_summary_text)
rg '"direction":"provider→ccr"' ~/.claude-code-router/logs/ccr.log

# What the client actually received after response transformers
rg '"direction":"ccr→client"' ~/.claude-code-router/logs/ccr.log
```

#### Health heartbeat

Everything above goes to the rotating log file, so a foreground `ccr start` or
`docker compose logs -f ccr` shows nothing between requests. The health
heartbeat is the one thing written to stdout: a periodic snapshot of process
health and routing activity since the previous report. It is on by default and
reports every 10 minutes.

```json5
{
  "HEARTBEAT_INTERVAL_MS": 600000  // Default 600000 (10 min); 0 disables
}
```

`CCR_HEARTBEAT_INTERVAL_MS` in the environment does the same thing and is
useful for containers.

```
[ccr:health] uptime 3h 12m · pid 41231 · node v22.23.2
[ccr:health] memory rss 412.0 MB (+18.2 MB) · heap 180.4 MB/240.0 MB · external 22.1 MB · system 19.9 GB/32.0 GB used
[ccr:health] load 2.41 / 1.98 / 1.75 (10 cpus) · proc cpu 0.37 cores · event loop mean 0.9 ms, p99 18.4 ms
[ccr:health] sessions 2 running · 7 active in the last 10m
[ccr:health] requests 2 in flight (oldest 41s) · 128 completed in 10m · 3 failed (2.3%) · p50 1.9s · p95 12.4s
[ccr:health] upstream openrouter 90 ok / 1 failed · claude 38 ok / 2 failed
[ccr:health] cache 82.3% prompt-cache hit · 145.0k cached / 176.3k prompt tokens · 3.4k written
```

How to read it:

- **memory** — `rss` with its change since the previous report. A steady climb
  across reports is the leak signal; a single spike is usually one large
  request. Under a cgroup limit the last field reads `container` instead of
  `system`.
- **load / proc cpu** — machine load average and this process's own CPU time
  over the window, expressed in cores. `proc cpu` near `1.00` means the single
  Node thread is saturated.
- **event loop** — how long callbacks waited beyond their scheduled time. This
  is the streaming-stall signal: memory and load look fine while a blocking
  logger or transformer holds the loop, and a p99 in the tens of milliseconds
  is what a client experiences as a stalled response.
- **sessions** — Claude Code sessions with a request in flight right now, and
  how many were seen at all during the window.
- **requests** — in-flight count with the age of the oldest. A single
  long-lived entry that survives several reports is a hung upstream stream.
  `failed` counts responses with a 4xx/5xx status.
- **upstream** — the same counts split per provider, so a degrading provider is
  visible without reading the log file.
- **cache** — prompt-cache hit rate for the window, the always-on companion to
  the `cache outcome` records that `LOG_LEVEL: "debug"` writes per request.
  Only routed LLM requests are counted; UI and status polling are excluded.

#### Health state file and endpoint

Each report is also written to `~/.claude-code-router/health.json`. The file
holds the **current state only** — no history and no accumulated snapshots — so
it stays small and never needs pruning. It is written to a temp file and
renamed, so a reader never sees a half-written document.

```json
{
  "version": 1,
  "pid": 41231,
  "node": "v22.23.2",
  "updatedAt": 1755500000000,
  "intervalMs": 600000,
  "current": { "memory": {}, "load": {}, "sessions": {}, "requests": {}, "cache": {} }
}
```

The same payload is served as the `vitals` field of the existing liveness probe:

```bash
curl -s http://127.0.0.1:3456/health
```

```json
{ "status": "ok", "timestamp": "2026-08-18T09:00:00.000Z", "vitals": { "...": "as above" } }
```

`/health` requires no authentication, and `vitals` is a live read rather than
the last written report, so it is never up to one interval stale. On a server
without the heartbeat (or with `HEARTBEAT_INTERVAL_MS: 0`) the field is simply
absent and the probe keeps its original `status` + `timestamp` contract.

#### Status bar in the Web UI

`ccr ui` shows a compact bar above every page fed by `/health` — never by the
file on disk. It polls on the server's own cadence (`intervalMs`, so every
10 minutes by default) and colours each metric:

| Metric | Yellow | Red |
| --- | --- | --- |
| Memory (system/container used) | 75% | 90% |
| Process CPU (cores) | 0.50 | 0.80 |
| Event loop p99 | 50 ms | 200 ms |
| Failed requests | 2% | 10% |

The overall dot is the worst of those four. Sessions and cache hit rate are
shown for context but do not drive the colour — a low hit rate costs money, it
does not mean the proxy is unhealthy. The bar hides itself entirely when the
server reports no `vitals`.

### Routing Configuration

```json5
{
  "Router": {
    "default": "openai,gpt-4",
    "background": "openai,gpt-3.5-turbo",
    "think": "openai,gpt-4",
    "longContext": "anthropic,claude-3-opus"
  }
}
```

## Configuration Validation

Configuration file is automatically validated. Common errors:

- **Missing Providers**: Must configure at least one provider
- **Missing API Key**: If Providers are configured, must provide API Key
- **Model doesn't exist**: Ensure model is in provider's models list

## Configuration Backup

Configuration is automatically backed up on each update:

```
~/.claude-code-router/config.backup.{timestamp}.json
```

## Apply Configuration Changes

After modifying the configuration file or making changes through the Web UI, restart the service:

```bash
ccr restart
```

Or restart directly through the Web UI by clicking the "Save and Restart" button.

## View Current Configuration

```bash
# View via API
curl http://localhost:3456/api/config

# Or view configuration file
cat ~/.claude-code-router/config.json
```

## Example Configurations

### OpenAI

```json5
{
  "Providers": [
    {
      "name": "openai",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "$OPENAI_API_KEY",
      "models": ["gpt-4", "gpt-3.5-turbo"]
    }
  ],
  "Router": {
    "default": "openai,gpt-4"
  }
}
```

### Anthropic

```json5
{
  "Providers": [
    {
      "name": "anthropic",
      "baseUrl": "https://api.anthropic.com/v1",
      "apiKey": "$ANTHROPIC_API_KEY",
      "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
    }
  ],
  "Router": {
    "default": "anthropic,claude-3-5-sonnet-20241022"
  }
}
```

### Multiple Providers

```json5
{
  "Providers": [
    {
      "name": "openai",
      "baseUrl": "https://api.openai.com/v1",
      "apiKey": "$OPENAI_API_KEY",
      "models": ["gpt-4", "gpt-3.5-turbo"]
    },
    {
      "name": "anthropic",
      "baseUrl": "https://api.anthropic.com/v1",
      "apiKey": "$ANTHROPIC_API_KEY",
      "models": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229"]
    }
  ],
  "Router": {
    "default": "openai,gpt-4",
    "think": "anthropic,claude-3-5-sonnet-20241022",
    "background": "openai,gpt-3.5-turbo"
  }
}
```

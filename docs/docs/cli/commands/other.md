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

Open the Web UI in your browser.

```bash
ccr ui
```

## ccr activate

Output shell environment variables for integration with external tools.

```bash
ccr activate
```

## ccr codex-auth

Authenticate with the Codex (ChatGPT) API via OAuth. Opens a browser for GitHub Copilot sign-in and saves the access token.

```bash
ccr codex-auth
```

## ccr qwen-auth

Authenticate with the Qwen Chat API. Prompts you to paste a JWT token copied from `chat.qwen.ai` localStorage, saves it, and provides automatic token rotation.

```bash
ccr qwen-auth
```

## ccr chrome-bridge

Start the Chrome On-Device model bridge for Gemini Nano. Must be run on the host (not inside Docker). Connects to Chrome's Prompt API via CDP.

```bash
ccr chrome-bridge
```

## ccr model get

Discover available models from a provider non-interactively. Fetches remote models, parses custom JSON structures, and appends missing models to your config.

```bash
ccr model get <provider-name>
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

- [Getting Started](/docs/intro) - Introduction to Claude Code Router
- [Configuration](/docs/config/basic) - Configuration guide

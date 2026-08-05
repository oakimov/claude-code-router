---
title: Getting Started
---

# Getting Started

Claude Code Router is a proxy that routes Claude Code requests through your configured LLM providers.

## Installation

Run the published Docker image:

```bash
mkdir -p ~/.claude-code-router
docker run -d --name ccr \
  -p 3456:3456 \
  -v ~/.claude-code-router:/root/.claude-code-router \
  ghcr.io/oakimov/claude-code-router:latest
```

The router will be available at `http://localhost:3456`. Set `"HOST": "0.0.0.0"` in your `config.json` so the port mapping can reach the server.

## Configuration

Before using Claude Code Router, you need to configure your providers. You can either:

1. **Edit configuration file directly**: Edit `~/.claude-code-router/config.json` (mounted into the container)
2. **Use Web UI**: Open `http://localhost:3456/ui/` to configure visually

After making configuration changes, restart the service:

```bash
docker restart ccr
```

## Using Claude Code

Once configured, set the environment variables and run Claude Code:

```bash
export ANTHROPIC_BASE_URL="http://localhost:3456/v1"
export ANTHROPIC_API_KEY="dummy"
claude
```

Your requests will be routed through the router to your configured provider.

## Service Management

```bash
docker start ccr      # Start the router
docker stop ccr       # Stop the router
docker restart ccr    # Restart the router
docker logs -f ccr    # View logs
```

## Web UI

Open `http://localhost:3456/ui/` in your browser to manage configuration and monitor the service.

## Configuration File

The configuration file is located at `~/.claude-code-router/config.json` and is mounted into the container at `/root/.claude-code-router/config.json`.

## Next Steps

- [Installation Guide](/docs/cli/installation) — Detailed installation instructions
- [Quick Start](/docs/cli/quick-start) — Get started in 5 minutes
- [Configuration Guide](/docs/category/cli-config) — Configuration file details
- [Integration Guides](/docs/category/integration-guides) — Set up provider-specific features

---
title: Getting Started
---

# Getting Started

Claude Code Router is a proxy that routes Claude Code requests through your configured LLM providers.

## Installation

Clone the repo and start the service with Docker Compose:

```bash
git clone https://github.com/oakimov/claude-code-router.git
cd claude-code-router/packages/server
docker compose up --build -d
```

The router will be available at `http://localhost:3456`.

## Configuration

Before using Claude Code Router, you need to configure your providers. You can either:

1. **Edit configuration file directly**: Edit `packages/server/ccr-config/config.json` (mounted into the container)
2. **Use Web UI**: Open `http://localhost:3456/ui/` to configure visually

After making configuration changes, restart the service:

```bash
docker compose restart ccr
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
docker compose up --build -d    # Start the router
docker compose down             # Stop the router
docker compose restart ccr      # Restart the router
docker compose logs -f ccr      # View logs
docker compose ps               # Check status
```

## Web UI

Open `http://localhost:3456/ui/` in your browser to manage configuration and monitor the service.

## Configuration File

The configuration file is located at `packages/server/ccr-config/config.json` and is mounted into the container at `/root/.claude-code-router/config.json`.

## Next Steps

- [Installation Guide](/docs/cli/installation) — Detailed installation instructions
- [Quick Start](/docs/cli/quick-start) — Get started in 5 minutes
- [Configuration Guide](/docs/category/cli-config) — Configuration file details
- [Integration Guides](/docs/category/integration-guides) — Set up provider-specific features

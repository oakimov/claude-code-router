---
sidebar_position: 3
---

# Quick Start

Get up and running with Claude Code Router in 5 minutes.

## 1. Start the Router

Run the published Docker image:

```bash
mkdir -p ~/.claude-code-router
docker run -d --name ccr \
  -p 3456:3456 \
  -v ~/.claude-code-router:/root/.claude-code-router \
  ghcr.io/oakimov/claude-code-router:latest
```

The router will start on `http://localhost:3456`. Make sure your `config.json` sets `"HOST": "0.0.0.0"` so the port mapping can reach the server.

## 2. Configure the Router

Before using Claude Code Router, you need to configure your LLM providers. Edit the configuration at `~/.claude-code-router/config.json` (mounted into the container):

```json5
{
  "HOST": "0.0.0.0",
  "PORT": 3456,
  "Providers": [
    {
      "name": "my-provider",
      "baseUrl": "https://api.example.com/v1",
      "apiKey": "$YOUR_API_KEY",
      "models": ["model-name"]
    }
  ],
  "Router": {
    "default": "my-provider,model-name"
  }
}
```

After editing the config, restart the service:

```bash
docker restart ccr
```

You can also use the Web UI at `http://localhost:3456/ui/` to configure providers visually.

## 3. Use Claude Code

Now you can use Claude Code with your configured provider. Set the required environment variables and run Claude Code directly:

```bash
export ANTHROPIC_BASE_URL="http://localhost:3456/v1"
export ANTHROPIC_API_KEY="dummy"
claude
```

Your requests will be routed through Claude Code Router to your configured provider.

## Restart After Configuration Changes

If you modify the configuration file or make changes through the Web UI, restart the service:

```bash
docker restart ccr
```

## What's Next?

- [Basic Configuration](/docs/cli/config/basic) — Learn about configuration options
- [Routing](/docs/server/config/routing) — Configure smart routing rules
- [Integration Guides](/docs/category/integration-guides) — Set up provider-specific features

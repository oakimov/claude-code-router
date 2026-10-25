---
sidebar_position: 2
---

# Installation

Run Claude Code Router with the published Docker image.

## Prerequisites

- **Docker**
- An API key from your preferred LLM provider

**Bun is not required for the Docker install** — the image ships its own Bun runtime (pinned to Bun 1.4.x). OpenCode free-tier models require Bun's TLS fingerprint; Node undici is rejected upstream.

You need **Bun ≥ 1.4.0** locally to install the CLI, run from source, or use the [Chrome On-Device bridge](/docs/server/guides/chrome-on-device). The floor is enforced by the `engines.bun` field of every published package:

| Package | Minimum Bun |
| --- | --- |
| `@caeliq/claude-code-router` (CLI) | ≥ 1.4.0 |
| `@caeliq/llms` (core) | ≥ 1.4.0 |
| `@caeliq/ccr-shared` | ≥ 1.4.0 |

CI and Docker use Bun end-to-end (`oven-sh/setup-bun`, `oven/bun:1.4.4-alpine`). Node is not part of the toolchain.

## Install with Docker

Run the published image, mounting a config directory:

```bash
mkdir -p ~/.claude-code-router
docker run -d --name ccr \
  -p 3456:3456 \
  -v ~/.claude-code-router:/root/.claude-code-router \
  ghcr.io/oakimov/claude-code-router:latest
```

The router will start on `http://localhost:3456`. Set `"HOST": "0.0.0.0"` in your `config.json` so the port mapping can reach the server.

To view logs:

```bash
docker logs -f ccr
```

To stop the service:

```bash
docker stop ccr && docker rm ccr
```

> **Note**: Building from the repository instead? The [Server Deployment](/docs/server/deployment) guide covers the `docker-compose.yml` in `packages/server`, which builds the image locally from source.

## Next Steps

Once installed, proceed to [Quick Start](/docs/cli/quick-start) to configure and start using the router.

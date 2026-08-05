---
sidebar_position: 2
---

# Installation

Run Claude Code Router with the published Docker image.

## Prerequisites

- **Docker**
- An API key from your preferred LLM provider

**Node.js is not required for the Docker install** — the image ships its own runtime (currently Node 22 LTS).

You only need Node.js locally to install the CLI from npm, run from source, or use the [Chrome On-Device bridge](/docs/server/guides/chrome-on-device). In those cases the minimum is **Node.js ≥ 22.19.0**, enforced by the `engines` field of every published package:

| Package | Minimum Node |
| --- | --- |
| `@caeliq/claude-code-router` (CLI) | ≥ 22.19.0 |
| `@caeliq/llms` (core) | ≥ 22.19.0 |
| `@caeliq/ccr-shared` | ≥ 22.19.0 |

The floor comes from `undici`, the HTTP client used for provider requests. `22.19.0` is a Node 22 **LTS** release; any newer Node 22 or 24 also works. Installing on an older runtime fails with an `EBADENGINE` warning and then errors at runtime.

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

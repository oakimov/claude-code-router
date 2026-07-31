---
sidebar_position: 2
---

# Installation

Run Claude Code Router using Docker Compose.

## Prerequisites

- **Docker** and **Docker Compose**
- An API key from your preferred LLM provider

**Node.js is not required for the Docker install** — the image ships its own runtime (currently Node 22 LTS).

You only need Node.js locally to install the CLI from npm, run from source, or use the [Chrome On-Device bridge](/docs/server/guides/chrome-on-device). In those cases the minimum is **Node.js ≥ 22.19.0**, enforced by the `engines` field of every published package:

| Package | Minimum Node |
| --- | --- |
| `@caeliq/claude-code-router` (CLI) | ≥ 22.19.0 |
| `@caeliq/llms` (core) | ≥ 22.19.0 |
| `@caeliq/ccr-shared` | ≥ 22.19.0 |

The floor comes from `undici`, the HTTP client used for provider requests. `22.19.0` is a Node 22 **LTS** release; any newer Node 22 or 24 also works. Installing on an older runtime fails with an `EBADENGINE` warning and then errors at runtime.

## Install with Docker Compose

Clone the repository and start the service using the provided Compose file:

```bash
git clone https://github.com/oakimov/claude-code-router.git
cd claude-code-router/packages/server
docker compose up --build -d
```

The router will start on `http://localhost:3456`.

To view logs:

```bash
docker compose logs -f ccr
```

To stop the service:

```bash
docker compose down
```

## Next Steps

Once installed, proceed to [Quick Start](/docs/cli/quick-start) to configure and start using the router.

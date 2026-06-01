---
sidebar_position: 2
---

# Installation

Run Claude Code Router using Docker Compose.

## Prerequisites

- **Docker** and **Docker Compose**
- An API key from your preferred LLM provider

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

Once installed, proceed to [Quick Start](/docs/quick-start) to configure and start using the router.

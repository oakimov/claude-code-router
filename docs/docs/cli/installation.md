---
sidebar_position: 2
---

# Installation

You can run Claude Code Router using Docker Compose (recommended) or via a package manager.

## Prerequisites

- **Docker** and **Docker Compose** (for Docker method)
- **Node.js**: >= 18.0.0 (for npm/pnpm method)
- An API key from your preferred LLM provider

## Install with Docker Compose (Recommended)

Clone the repository and start the service using the provided Compose file:

```bash
git clone https://github.com/oakimov/claude-code-router.git
cd claude-code-router
cd packages/server
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

## Install via Package Manager (Alternative)

If you prefer to run directly without Docker, install globally using your preferred package manager.

### Install via npm

```bash
npm install -g @musistudio/claude-code-router
```

### Install via pnpm

```bash
pnpm add -g @musistudio/claude-code-router
```

### Install via Yarn

```bash
yarn global add @musistudio/claude-code-router
```

## Verify Installation (Package Manager)

After installation, verify that `ccr` is available:

```bash
ccr --version
```

You should see the version number displayed.

## Next Steps

Once installed, proceed to [Quick Start](/docs/quick-start) to configure and start using the router.

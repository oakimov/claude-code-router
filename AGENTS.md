# AGENTS.md

This file provides guidance to coding agents (Cursor, Claude Code, Codex, and others) when working with code in this repository.

## Project Overview

Claude Code Router is a tool that routes Claude Code requests to different LLM providers. It uses a Monorepo architecture with these packages:

- **cli** (`@caeliq/claude-code-router`): Command-line tool providing the `ccr` command
- **server** (`@caeliq/ccr-server`): Core server handling API routing and transformations
- **shared** (`@caeliq/ccr-shared`): Shared constants, utilities, and preset management
- **ui** (`@caeliq/ccr-ui`): Web management interface (React + Vite)
- **core** (`@caeliq/llms`): Universal LLM API transformation / routing framework

### Core Objectives
- **Model Versatility**: Enable Claude Code to leverage the best model for a specific task (e.g., high-reasoning models for Plan Mode vs. lightweight models for background tasks).
- **Cost & Performance Optimization**: Route requests based on complexity, token counts, or provider performance.
- **Provider Abstraction**: Create a unified interface that abstracts the differences between various LLM provider APIs.

### Getting Started
The fastest way to start and verify the build is using Docker Compose:
1. **Launch with Docker**: `cd packages/server && docker compose up --build -d`
2. **Setup configuration**: `ccr preset install <source>` or manually edit the configuration file.
3. **Verify**: Use `ccr code` to execute a command or open the UI via `ccr ui`.

*Alternative (Local Development)*: If you prefer to run locally without Docker, run `pnpm install` first, then use the `pnpm dev:*` commands.

## Knowledge Base
For critical lessons learned regarding LLM provider integrations (e.g., DeepSeek reasoning replay, Mistral thinking formats, Gemini streaming issues, Gemini Nano constraints), refer to `tasks/lessons.md`. This file contains the "hard-won" knowledge required to avoid common provider-specific pitfalls.

For tracked improvements, planned features, and known issues (especially for the Chrome On-Device bridge), refer to `tasks/TODO.md`.

## Build Commands

### Primary: Build and run via Docker Compose (Recommended for verification)
This is the preferred method to verify the build and deployment configuration.
```bash
cd packages/server
docker compose up --build -d
```

Useful follow-up commands:
```bash
docker compose logs -f ccr
docker compose restart ccr
docker compose down
```

Notes:
- The compose file is `packages/server/docker-compose.yml`.
- It builds from the repo root using `packages/server/Dockerfile`.
- Runtime config is mounted from `packages/server/ccr-config` to `/root/.claude-code-router` in the container.
- The proxy listens on `http://localhost:3456`.
- After editing `packages/server/ccr-config/config.json`, restart the `ccr` service.

### Secondary: Local Build (pnpm)
Use these commands for local development and iterative coding.

#### Build all packages
```bash
pnpm build
```

#### Build individual packages
```bash
pnpm build:cli      # Build CLI
pnpm build:server   # Build Server
pnpm build:ui       # Build UI
```

#### Development mode
```bash
pnpm dev:cli        # Develop CLI (tsx)
pnpm dev:server     # Develop Server (tsx)
pnpm dev:ui         # Develop UI (Vite)
```

### Verification

Run these locally before tagging a release. `npm-publish.yml` runs the same
checks on the tag, but a failure there means the version is already burned —
catch it here instead.

```bash
pnpm typecheck            # tsc --noEmit across all packages
pnpm lint                 # eslint across all packages
pnpm test                 # all hermetic tests
pnpm test:chrome-bridge   # additionally runs the Chrome bridge test
```

`pnpm test` executes every `packages/*/src/tests/*.ts` file via
`scripts/run-tests.js`. Each test is a standalone tsx entry point that exits
non-zero on failure — there is no test framework. Add a new test by dropping a
file in that directory; it is picked up automatically.

Tests run from their owning package directory so core's `@/*` path alias
resolves; invoking `npx tsx packages/core/src/tests/foo.ts` from the repo root
fails for that reason.

`ccr-anthropic-flow.test.ts` is an end-to-end check of the
Claude Code → CCR → Gemini Nano bridge → CCR round trip. It needs a running CCR
server plus `ccr chrome-bridge` against a Chrome with Gemini Nano available, so
it is opt-in via `--chrome-bridge` and never runs in CI. Register any other test
with operator-provided dependencies in `OPT_IN_ONLY` in the runner.

### Publish
```bash
pnpm release        # Build and publish all packages
```

Full trusted-publishing details: `docs/PUBLISHING.md`.

### Version bump (required before tagging a release)

When asked to bump the version for a release, update **all** of the places below in one change. Do not ask which files to touch — follow this checklist. See recent bumps: `2a8509b` (`chore: bump packages to 2.0.1 / 1.0.55`), `3df5edb` (feature + bump to `2.0.2` / `@caeliq/llms@1.0.56`).

There are **two version lines**:

| Line | Current role | Packages |
|---|---|---|
| **Product** (`2.0.x`) | CLI / product release; git tag `v2.0.x` matches this | root, `@caeliq/claude-code-router` (cli), `@caeliq/ccr-shared`, `@caeliq/ccr-server`, `@caeliq/ccr-ui` |
| **Core** (`1.0.x`) | Published as `@caeliq/llms`; independent semver | `packages/core` only |

**Always bump together (keep product versions identical):**
1. `package.json` (repo root)
2. `packages/cli/package.json` → `@caeliq/claude-code-router`
3. `packages/shared/package.json` → `@caeliq/ccr-shared` (release script publishes shared every run; reusing a published version fails)
4. `packages/server/package.json` → `@caeliq/ccr-server` (not published to npm; ships inside the CLI bundle, but version stays in sync)
5. `packages/ui/package.json` → `@caeliq/ccr-ui`

**Bump when core changed (usual for any release that touches `packages/core`):**
6. `packages/core/package.json` → `@caeliq/llms` (own `1.0.x` line; bump patch/minor independently of product)

**Docs that must reflect the new versions:**
7. `docs/PUBLISHING.md` — update the published-version table at the top, and any example `vX.Y.Z` / `git tag` snippets so they match the new CLI version

**Do not change for a routine bump:**
- `docs/package.json` (`0.0.0`)
- Preset / example `"version"` fields (`examples/*`, preset docs)
- Historical blog posts under `docs/blog/` and `docs/i18n/**/docusaurus-plugin-content-blog/`
- Inter-package deps that use `workspace:*` (rewritten to `^<shared version>` at publish time)
- Stale pinned install examples like `@caeliq/claude-code-router@1.0.8` in README GitHub Actions samples, unless the user explicitly asks to refresh those examples

**Package review (`/package-review`):**

Before finishing a version bump, treat dependency hygiene by semver size of the **product** line (`X.Y.Z`):

| Bump | Action |
|---|---|
| **Patch** (`Z`) | Optional — run only if the release touched deps, lockfile, or overrides |
| **Minor** (`Y`) | **Suggest** running `.claude/skills/package-review` (audit, dedupe, consolidate versions) before tagging; proceed if the user declines |
| **Major** (`X`) | **Always execute** `/package-review` (or follow that skill end-to-end) before tagging; do not skip |

The same rule applies when bumping core's independent `1.0.x` line by its own minor/major. Skill path: `.claude/skills/package-review/SKILL.md`.

**Procedure:**
1. Read current versions from the `package.json` files above.
2. Apply the requested bump (default: product patch +1; core patch +1 if core changed or if prior releases always shipped both).
3. Update `docs/PUBLISHING.md` to match.
4. For minor bumps, suggest `/package-review`; for major bumps, run it before continuing.
5. Verify product versions match across root/cli/shared/server/ui, and confirm core is the intended `1.0.x`:
   ```bash
   node -p "require('./package.json').version"
   node -p "require('./packages/cli/package.json').version"
   node -p "require('./packages/shared/package.json').version"
   node -p "require('./packages/server/package.json').version"
   node -p "require('./packages/ui/package.json').version"
   node -p "require('./packages/core/package.json').version"
   ```
6. Commit message style from history: `chore: bump packages to X.Y.Z / A.B.C for …` or include the bump in the feature commit (`… (vX.Y.Z)` / `Bump packages to X.Y.Z / @caeliq/llms@A.B.C`).
7. Tagging / publish (only when the user asks): push `main`, then `git tag vX.Y.Z` matching the **CLI** version and `git push github vX.Y.Z`. CI reads versions from `package.json`, not from the tag alone.

## Core Architecture

### 1. Routing System

The routing logic is handled by the core framework in the `@caeliq/llms` package. It determines which model a request should be sent to:

- **Default routing**: Uses `Router.default` configuration
- **Project-level routing**: Checks `~/.claude/projects/<project-id>/claude-code-router.json`
- **Custom routing**: Loads custom JavaScript router function via `CUSTOM_ROUTER_PATH`
- **Built-in scenario routing**:
  - `background`: Background tasks (typically lightweight models)
  - `think`: Thinking-intensive tasks (Plan Mode)
  - `longContext`: Long context (exceeds `longContextThreshold` tokens)
  - `webSearch`: Web search tasks
  - `fim`: Fill-in-the-middle (`POST /v1/fim/completions`; separate pipeline)
  - `image`: Image-related tasks

Token calculation uses `tiktoken` (cl100k_base) to estimate request size.

### 2. Transformer System

> **⚠️ CRITICAL — Read this before working with transformers.**
>
> **Inbound client protocols** (see `routing/protocol-endpoints.ts`):
> - `anthropic_messages` → `POST /v1/messages` (owner: `Anthropic`)
> - `openai_chat_completions` → `POST /v1/chat/completions` (owner: `OpenAI`; alias `/chat/completions`)
> - `openai_responses` → `POST /v1/responses` (owner: `openai-responses`; alias `/responses`)
> - `openai_fim_completions` → `POST /v1/fim/completions` (owner: `Fim`; alias `/fim/completions`; **separate** FIM pipeline)
>
> Chat protocols share one lifecycle (`prepareInboundRequest` in
> `routing/inbound-pipeline.ts` → `api/routes.ts`). Example Anthropic inbound:
>
>     Client → POST /v1/messages
>       → AnthropicTransformer.transformRequestOut()        // Anthropic → Unified (OpenAI Chat Completions)
>       → provider.transformer.use[].transformRequestIn()   // provider middleware
>       → sendRequestToProvider()                           // HTTP call upstream
>       → provider.transformer.use[].transformResponseOut() // provider middleware (reversed)
>       → AnthropicTransformer.transformResponseIn()        // Unified (OpenAI) → Anthropic
>       → Client
>
> Chat Completions and Responses follow the same shape with their own owners
> (`OpenAI` / `openai-responses`) on the client legs. The response is always
> encoded for the **inbound** protocol, not Anthropic-only.
>
> **The Unified chat format IS the OpenAI Chat Completions format.**
> `AnthropicTransformer.transformRequestOut()` converts an Anthropic body into
> Unified. Chat Completions inbound validates an already-Unified-shaped body.
> Responses inbound projects `input`/tools onto Unified. By the time the
> provider chain runs, chat bodies are Unified — no further conversion is needed
> for providers that accept Chat Completions.
>
> **`OpenAITransformer`** (`openai.transformer.ts`) owns `/v1/chat/completions`,
> validates inbound Chat bodies via `transformRequestOut`, applies provider-side
> cache policy in `transformRequestIn`, and shapes Chat client responses in
> `transformResponseIn`.
>
> **`OpenAIResponsesTransformer`** (`openai.responses.transformer.ts`) owns
> `/v1/responses` on the client side and, when used as provider egress, converts
> Unified → Responses wire (`messages` → `input`, function tools → flat tool
> definitions, `web_search` → `{ type: "web_search" }`, etc.). Shared utilities
> live in `openai.util.ts` / `openai.responses.util.ts`.
>
> **FIM** does not use the chat Unified pipeline. See `routing/fim-pipeline.ts`
> and `transformer/fim/`.

The project uses transformers to adapt to different provider API differences:

- Built-in transformers: `anthropic`, `deepseek`, `gemini`, `openrouter`, `groq`, `maxtoken`, `tooluse`, `reasoning`, `enhancetool`, etc.
- Custom transformers: Load external plugins via `transformers` array in `config.json`

Transformer configuration supports:
- Global application (provider level)
- Model-specific application
- Option passing (e.g., `max_tokens` parameter for `maxtoken`)

### 3. Agent System (packages/server/src/agents/)

Agents are pluggable feature modules that can:
- Detect whether to handle a request (`shouldHandle`)
- Modify requests (`reqHandler`)
- Provide custom tools (`tools`)

Built-in agents:
- **imageAgent**: Handles image-related tasks

Agent tool call flow:
1. Detect and mark agents in `preHandler` hook
2. Add agent tools to the request
3. Intercept tool call events in `onSend` hook
4. Execute agent tool and initiate new LLM request
5. Stream results back

### 4. SSE Stream Processing

The server uses custom Transform streams to handle Server-Sent Events:
- `SSEParserTransform`: Parses SSE text stream into event objects
- `SSESerializerTransform`: Serializes event objects into SSE text stream
- `rewriteStream`: Intercepts and modifies stream data (for agent tool calls)

### 5. Configuration Management

Configuration file location: `~/.claude-code-router/config.json`

**Critical Rule**: When editing `config.json` (or provider configs), **never resolve environment variables**. Keep placeholders like `$OPENAI_API_KEY` or `${GEMINI_API_KEY}` exactly as they are. Do not replace them with actual secret values in the file.

Key features:
- Supports environment variable interpolation (`$VAR_NAME` or `${VAR_NAME}`)
- JSON5 format (supports comments)
- Automatic backups (keeps last 3 backups)
- Hot reload requires service restart (`ccr restart`)

- **`HOST`** (optional): You can set the host address for the server. If `APIKEY` is not set, the host will be forced to `127.0.0.1` for security reasons. Example: `"HOST": "0.0.0.0"`.

### 6. Logging System

Two separate logging systems:

**Server-level logs** (pino):
- Location: `~/.claude-code-router/logs/ccr-*.log`
- Content: HTTP requests, API calls, server events
- Configuration: `LOG_LEVEL` (fatal/error/warn/info/debug/trace)

**Application-level logs**:
- Location: `~/.claude-code-router/claude-code-router.log`
- Content: Routing decisions, business logic events

## CLI Commands

Commands can be run locally or inside the Docker container using:
`docker exec -it <container_id> ccr <command>`

```bash
ccr start         # Start server
ccr stop          # Stop server
ccr restart       # Restart server
ccr status        # Show status
ccr code          # Execute claude command
ccr model         # Interactive model selection and configuration
ccr preset        # Manage presets (export, install, list, info, delete)
ccr activate      # Output shell environment variables (for integration)
ccr ui            # Open Web UI
ccr statusline    # Integrated statusline (reads JSON from stdin)
ccr codex-auth    # Authenticate with Codex API via OAuth
ccr codex-config  # Publish CCR models to Codex (config.toml + model catalog)
ccr claude-auth   # Authenticate with Claude Pro/Max subscription via OAuth
ccr qwen-auth     # Authenticate with Qwen Chat (JWT from localStorage)
ccr antigravity-auth  # Authenticate with Google Antigravity via OAuth
ccr chrome-bridge # Start Chrome on-device model bridge (Gemini Nano) — must run on host
```

### Preset Commands

If running in Docker, use: `docker exec -it <container_id> ccr preset <subcommand>`

```bash
ccr preset export <name>      # Export current configuration as a preset
ccr preset install <source>   # Install a preset from file, URL, or name
ccr preset list               # List all installed presets
ccr preset info <name>        # Show preset information
ccr preset delete <name>      # Delete a preset
```

## Subagent Routing

Use special tags in subagent prompts to specify models:
```
<CCR-SUBAGENT-MODEL>provider,model</CCR-SUBAGENT-MODEL>
Please help me analyze this code...
```

## Preset System

The preset system allows users to save, share, and reuse configurations easily.

### Preset Structure

Presets are stored in `~/.claude-code-router/presets/<preset-name>/manifest.json`

Each preset contains:
- **Metadata**: name, version, description, author, keywords, etc.
- **Configuration**: Providers, Router, transformers, and other settings
- **Dynamic Schema** (optional): Input fields for collecting required information during installation
- **Required Inputs** (optional): Fields that need to be filled during installation (e.g., API keys)

### Core Functions

Located in `packages/shared/src/preset/`:

- **export.ts**: Export current configuration as a preset directory
  - `exportPreset(presetName, config, options)`: Creates preset directory with manifest.json
  - Automatically sanitizes sensitive data (api_key fields become `{{field}}` placeholders)

- **install.ts**: Install and manage presets
  - `extractPreset(sourceZip, targetDir)`: Extract preset from ZIP file to target directory
  - `loadPreset(source)`: Load preset from directory
  - `listPresets()`: List all installed presets
  - `isPresetInstalled(presetName)`: Check if preset is installed
  - `validatePreset(preset)`: Validate preset structure

- **merge.ts**: Merge preset configuration with existing config
  - Handles conflicts using different strategies (ask, overwrite, merge, skip)

- **sensitiveFields.ts**: Identify and sanitize sensitive fields
  - Detects api_key, password, secret fields automatically
  - Replaces sensitive values with environment variable placeholders

### Preset File Format

**manifest.json** (in preset directory):
```json
{
  "name": "my-preset",
  "version": "1.0.0",
  "description": "My configuration",
  "author": "Author Name",
  "keywords": ["openai", "production"],
  "Providers": [...],
  "Router": {...},
  "schema": [
    {
      "id": "apiKey",
      "type": "password",
      "label": "OpenAI API Key",
      "prompt": "Enter your OpenAI API key"
    }
  ]
}
```

### CLI Integration

The CLI layer (`packages/cli/src/utils/preset/`) handles:
- User interaction and prompts
- File operations
- Display formatting

Key files:
- `commands.ts`: Command handlers for `ccr preset` subcommands
- `export.ts`: CLI wrapper for export functionality
- `install.ts`: CLI wrapper for install functionality

## Dependencies

```
cli → server → shared
server → @caeliq/llms (core routing and transformation logic)
ui (standalone frontend application)
```

## Development Notes

1. **Node.js version**: Floor is **>= 22.19.0**, declared in `engines.node` of every
   workspace package (root, core, cli, shared, server, ui, docs) — keep them identical.
   The floor is set by `undici` (`>=22.19.0`); `@cursor/sdk` needs `>=22.13`. Build-only
   deps ask for more (`react-router@8` wants `>=22.22.0`, `puppeteer-core` `>=22.12.0`)
   but never reach an installed user: the UI ships as a prebuilt `index.html` and
   `puppeteer-core` is a lazily-required devDependency.
   Runtime pins are separate and track the newest Node 22 LTS patch (`22.23.2` in the
   Dockerfile, `node-version: "22"` in CI). Bump the pin freely; only raise the floor
   when a runtime dependency actually demands it. esbuild `--target=node22` in all four
   build scripts must stay in sync with the floor.
2. **Package manager**: Uses pnpm (monorepo depends on workspace protocol)
3. **TypeScript**: All packages use TypeScript, but UI package is ESM module
4. **Build tools**:
   - cli/server/shared: esbuild
   - ui: Vite + TypeScript
5. **@caeliq/llms**: In-monorepo core package (`packages/core`) providing the server framework and transformer functionality; published independently to npm. Type declarations are generated into `packages/core/dist/*.d.ts` (barrel `dist/index.d.ts`) during the core build; `packages/server` imports them via the `@caeliq/llms` package entry (no manual `packages/server/src/types.d.ts` shim).
6. **Code comments**: All comments in code MUST be written in English
7. **Documentation**: When implementing new features, add documentation to the docs project instead of creating standalone md files

## Configuration Example Locations

- Main configuration example: Complete example in README.md
- Custom router example: `custom-router.example.js`

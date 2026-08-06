---
sidebar_position: 4
---

# ccr codex-config

Publish CCR-routed models to Codex so they appear in its native model picker.

## Usage

```bash
ccr codex-config [options]
```

## Description

Codex does not build its picker from `GET /v1/models` — it reads a local catalog file named by `model_catalog_json` **at startup**. So this command writes two things:

1. A Codex-format model catalog at `~/.claude-code-router/codex/models.json`
2. Managed blocks in `<codex-home>/config.toml` pointing Codex at CCR and at that catalog

Model metadata (friendly name, context window, reasoning effort levels, modalities) is enriched from the [models.dev](https://models.dev) catalog, fetched live on each run. A models.dev failure is non-fatal — the command warns and falls back to conservative defaults.

## Options

| Option | Default | Description |
|---|---|---|
| `--providers <a,b,c>` | all configured | Comma-separated CCR provider names |
| `--models <glob>` | `*` | Comma-separated wildcard patterns (`*`, `?`); matched against the model id and the full `provider,model` |
| `--base-url <url>` | `http://127.0.0.1:<PORT>/v1` | CCR base URL written into Codex's config |
| `--codex-home <dir>` | `$CODEX_HOME`, else `~/.codex` | Codex home directory |
| `--dry-run` | off | Print the resulting config.toml, write nothing |
| `--force` | off | Overwrite a user-owned `model_provider` / `model_catalog_json` |
| `--no-codex-probe` | off | Skip cloning a template from `codex debug models` |

## Examples

```bash
# Everything CCR is configured for
ccr codex-config

# One provider, reasoning models only
ccr codex-config --providers openai --models 'gpt-*,o3'

# Preview without touching any file
ccr codex-config --providers gemini --dry-run
```

## What gets written

```toml
# BEGIN ccr-managed
model_provider = "ccr"
model_catalog_json = "/Users/you/.claude-code-router/codex/models.json"
# END ccr-managed

# BEGIN ccr-provider-managed
[model_providers.ccr]
name = "Claude Code Router"
base_url = "http://127.0.0.1:3456/v1"
env_key = "CCR_API_KEY"
wire_api = "responses"
requires_openai_auth = false
# END ccr-provider-managed
```

The `env_key` line is emitted only when CCR has `APIKEY` configured. In that
case, export the matching value before launching Codex:

```bash
export CCR_API_KEY=your-router-api-key
```

Only the delimited regions are rewritten, so re-running is idempotent and the rest of your Codex configuration is left untouched. Root keys are inserted before the first `[table]` header, as TOML requires. `config.toml` is backed up to `config.toml.bak` before any change.

If `model_provider` or `model_catalog_json` already exists **outside** a managed block, the command refuses rather than hijacking an existing setup. Re-run with `--force` to replace it.

## Template capture

Codex's catalog schema is undocumented and version-dependent; real entries carry fields (such as `base_instructions`) a synthesized entry would omit. When a `codex` binary is on `PATH`, the command clones a real entry from `codex debug models` and overrides the routing fields. Otherwise it synthesizes an entry and warns that the picker may reject it. Use `--no-codex-probe` to always synthesize.

The catalog contains only CCR-routed models and **replaces** Codex's native model catalog in the picker — native GPT models do not appear alongside it. This is intentional: the command is for routing through CCR, and it never merges in or probes the native catalog for signed-in/signed-out state.

## After running

Codex loads the catalog only at startup:

```text
Fully quit Codex and reopen it.
```

Then pick a CCR model from the normal model picker. Requests flow over the Responses API (`wire_api = "responses"`) to `/v1/responses`.

## Environment

| Variable | Default | Description |
|---|---|---|
| `CODEX_HOME` | `~/.codex` | Codex home directory |
| `CCR_MODELSDEV_URL` | `https://models.dev/api.json` | Metadata catalog URL |
| `CCR_MODELSDEV_TIMEOUT` | `10000` | Fetch timeout in ms |
| `CCR_MODELSDEV_DISABLE` | unset | Set to `1` to skip models.dev entirely |

## See also

- [Models API](/docs/server/api/models-api) — the HTTP listing for SDK clients
- [Codex integration guide](/docs/server/guides/codex) — routing Claude Code *to* Codex

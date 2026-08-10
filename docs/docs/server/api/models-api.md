---
title: Models API
---

# Models API

CCR exposes an OpenAI-compatible model listing so SDK clients and tools can discover what the router can reach:

- `GET /v1/models`
- `GET /models`
- `GET /v1/models/{id}`

## Model IDs

`MODEL_ID_OUTPUT` controls the representation returned by the API:

- `"literal"` (default) emits CCR's native `provider,model` IDs.
- `"masked"` emits IDs that do not begin with `claude` or `anthropic` as
  `claude-<lowercase UTF-8 hex>` so Claude clients do not filter them out.

Both forms can be sent to `/v1/responses`, `/v1/chat/completions`, or
`/v1/messages` in either mode. For example, these resolve identically:

```text
codex,gpt-5.6-sol
claude-636f6465782c6770742d352e362d736f6c
```

```bash
curl -H "x-api-key: your-router-api-key" http://localhost:3456/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "deepseek,deepseek-chat",
      "object": "model",
      "created": 0,
      "owned_by": "deepseek"
    }
  ]
}
```

`owned_by` is the CCR provider name. `created` is always `0` — CCR has no per-model creation time, and a per-boot value would make otherwise identical responses differ between restarts.

## Single model

```bash
curl -H "x-api-key: your-router-api-key" \
  "http://localhost:3456/v1/models/deepseek,deepseek-chat"
```

URL-encode the id when it contains `/`. For example,
`openrouter,anthropic/claude-3.5-sonnet` becomes
`openrouter,anthropic%2Fclaude-3.5-sonnet` in the path.

An unknown id returns `404` with an OpenAI-shaped error (`code: "model_not_found"`).
Single-model lookup accepts either representation and returns the ID format selected by
`MODEL_ID_OUTPUT`.

## Behavior

The list is built from the `Providers` entries the running server was started with — one entry per provider/model pair, de-duplicated. Providers with no `models` array are skipped, and a configuration with no providers returns an empty `data` array rather than an error.

Because the list reflects the running server, restart after editing `config.json`:

```bash
ccr restart
```

## Authentication

The listing is protected by the same API key check as the rest of the surface. Send `Authorization: Bearer <key>` or `x-api-key: <key>` when `APIKEY` is configured.

## Codex

Codex does not populate its model picker from this endpoint — it reads a local catalog file at startup. Use [`ccr codex-config`](/docs/cli/commands/codex-config) to generate that catalog and point Codex at CCR.

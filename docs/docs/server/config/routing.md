---
sidebar_position: 3
---

# Routing Configuration

Configure how requests are routed to different models. The same `Router`
rules apply after Unified normalization for every **chat** inbound protocol
(Messages, Chat Completions, Responses). FIM uses `Router.fim` on its own
pipeline — see [FIM](#fim-fill-in-the-middle).

## Default Routing

Set the default model for all requests:

```json
{
  "Router": {
    "default": "deepseek,deepseek-chat"
  }
}
```

## Built-in Scenarios

### Background Tasks

Route background tasks to a lightweight model:

```json
{
  "Router": {
    "background": "groq,llama-3.3-70b-versatile"
  }
}
```

### Thinking Mode (Plan Mode)

Route thinking-intensive tasks to a more capable model:

```json
{
  "Router": {
    "think": "deepseek,deepseek-chat"
  }
}
```

### Long Context

Route requests with long context:

```json
{
  "Router": {
    "longContextThreshold": 100000,
    "longContext": "gemini,gemini-1.5-pro"
  }
}
```

### Web Search

Route web search tasks:

```json
{
  "Router": {
    "webSearch": "deepseek,deepseek-chat"
  }
}
```

### FIM (fill-in-the-middle)

Route `POST /v1/fim/completions` to a FIM-capable provider (must use a `fim.*` transformer).

**Codestral:**

```json
{
  "Router": {
    "fim": "codestral-fim,codestral-latest"
  }
}
```

**Local Qwen (LM Studio):**

```json
{
  "Router": {
    "fim": "lmstudio-qwen-fim,qwen/qwen2.5-coder-14b"
  }
}
```

Bare models on the FIM endpoint resolve via `Router.fim`, then `Router.default`. Provider JSON and client setup for both cases: [FIM Completions API](../api/fim-completions-api.md).

### Image Tasks

Route image-related tasks:

```json
{
  "Router": {
    "image": "gemini,gemini-1.5-pro"
  }
}
```

## Fallback

When a request fails, you can configure a list of backup models. The system will try each model in sequence until one succeeds:

### Basic Configuration

```json
{
  "Router": {
    "default": "deepseek,deepseek-chat",
    "background": "ollama,qwen2.5-coder:latest",
    "think": "deepseek,deepseek-reasoner",
    "longContext": "openrouter,google/gemini-2.5-pro-preview",
    "longContextThreshold": 60000,
    "webSearch": "gemini,gemini-2.5-flash"
  },
  "fallback": {
    "default": [
      "aihubmix,Z/glm-4.5",
      "openrouter,anthropic/claude-sonnet-4"
    ],
    "background": [
      "ollama,qwen2.5-coder:latest"
    ],
    "think": [
      "openrouter,anthropic/claude-3.7-sonnet:thinking"
    ],
    "longContext": [
      "modelscope,Qwen/Qwen3-Coder-480B-A35B-Instruct"
    ],
    "webSearch": [
      "openrouter,anthropic/claude-sonnet-4"
    ],
    "fim": [
      "codestral-fim,codestral-latest"
    ],
    "subagent": [
      "openrouter,anthropic/claude-sonnet-4"
    ]
  }
}
```

### How It Works

1. **Trigger**: When a model request fails for a routing scenario. Eligible failures include provider HTTP error responses and provider network/transport errors (for example connection reset or fetch failures). Client disconnects / aborts do **not** trigger fallback.
2. **Backoff**: Before the first fallback attempt (and between later attempts), CCR waits using the upstream `Retry-After` header when present; otherwise it uses exponential backoff.
3. **Auto-switch**: The system checks the fallback configuration for that scenario.
4. **Sequential retry**: Tries each backup model in order. If the client disconnects during a wait or attempt, remaining fallbacks are cancelled.
5. **Success**: Once a model responds successfully, returns immediately.
6. **All failed**: If all backup models fail, returns the original error.

### Configuration Details

- **Format**: Each backup model format is `provider,model`
- **Validation**: Backup models must exist in the `Providers` configuration
- **Flexibility**: Different scenarios can have different fallback lists
- **Optional**: If a scenario doesn't need fallback, omit it or use an empty array
- **Subagents**: `fallback.subagent` overrides fallback for subagent-routed requests. If omitted, subagents inherit `fallback.default` for backward compatibility.
- **Abort-aware**: Closing the client mid-request cancels fallback waits and further attempts

### Request-Scoped Errors

By default CCR decides fallback eligibility from the HTTP status alone
(408/409/425/429/5xx + network errors). `request_scoped_errors` refines that
with body-pattern rules — per provider, plus a top-level global list.
Provider rules are evaluated first; the first match wins.

```json
{
  "request_scoped_errors": [
    { "status": 400, "match": ["maximum_context_length"], "action": "stop" },
    { "status": 400, "match": ["server_is_overloaded"], "action": "continue" }
  ],
  "Providers": [
    {
      "name": "codex",
      "api_base_url": "https://chatgpt.com/backend-api/codex",
      "api_key": "oauth_dummy_key",
      "models": ["gpt-5"],
      "transformer": { "use": ["openai-responses", "codex"] },
      "request_scoped_errors": [
        { "status": 429, "match": ["usage_limit_reached"], "action": "continue-and-cooldown" }
      ]
    }
  ]
}
```

- **Match**: `status` (optional) must equal the error status; `match`
  substrings (case-insensitive) and `match_regex` patterns hit against the
  raw upstream error text (first 16,384 characters) plus a specific error
  code. When raw text is available, it replaces CCR's sanitized message and
  `Error from provider(...)` prefix for matching. Otherwise the error message
  and captured upstream body are used. CCR's generic `api_error` type and
  catch-all `provider_response_error` code are not added to matching text.
  A rule with no body patterns matches on status alone. Raw text is retained
  for matching only; clients and logs still see sanitized error details.
- **Actions**: `stop` (terminal — no fallback), `continue` (force fallback
  even when the status alone would not qualify), plus `stop-and-cooldown` /
  `continue-and-cooldown`, which additionally cool the `provider,model` down
  (default 60 s, overridable per rule with `cooldown_seconds`, minimum 1).
  The cooldown applies to fallback candidates only: the fallback loop skips
  a cooled-down model and spends no Retry-After/backoff wait on it, while a
  model routed as the primary is still attempted. A trailing `[1m]` marker
  does not change the cooled-down identity.
- **Validation**: a rule with an unknown key (for example `matches`), an
  invalid `action`, a non-integer `status`, a pattern list that is not an
  array of non-empty strings, an invalid regex or a `cooldown_seconds` below
  1 is ignored, and CCR logs a warning naming the rule index and the reason.
- **Key spellings**: `request_scoped_errors` / `requestScopedErrors` /
  `request-scoped-errors` work on both providers and the top level;
  provider updates through any spelling replace the stored canonical list.
  `match_regex` / `matchRegex` /
  `match-regex` and `cooldown_seconds` / `cooldownSeconds` are equivalent.

### Use Cases

#### Scenario 1: Primary Model Quota Exhausted

```json
{
  "Router": {
    "default": "openrouter,anthropic/claude-sonnet-4"
  },
  "fallback": {
    "default": [
      "deepseek,deepseek-chat",
      "aihubmix,Z/glm-4.5"
    ]
  }
}
```

Automatically switches to backup models when the primary model quota is exhausted.

#### Scenario 2: Service Reliability

```json
{
  "Router": {
    "background": "volcengine,deepseek-v3-250324"
  },
  "fallback": {
    "background": [
      "modelscope,Qwen/Qwen3-Coder-480B-A35B-Instruct",
      "dashscope,qwen3-coder-plus"
    ]
  }
}
```

Automatically switches to other providers when the primary service fails.

### Log Monitoring

The system logs detailed fallback process:

```
[warn] Request failed for default, trying 2 fallback models
[info] Waiting 2000ms before first fallback attempt
[info] Trying fallback model: aihubmix,Z/glm-4.5
[warn] Fallback model aihubmix,Z/glm-4.5 failed: API rate limit exceeded
[info] Waiting 4000ms before next fallback attempt
[info] Trying fallback model: openrouter,anthropic/claude-sonnet-4
[info] Fallback model openrouter,anthropic/claude-sonnet-4 succeeded
```

Upstream failure details in logs and client error payloads are privacy-sanitized (hosts, IPs, bearer tokens, and API key material are redacted).

### Important Notes

1. **Cost consideration**: Backup models may incur different costs, configure appropriately
2. **Performance differences**: Different models may have varying response speeds and quality
3. **Quota management**: Ensure backup models have sufficient quotas
4. **Testing**: Regularly test the availability of backup models
5. **Retry-After**: When providers return `Retry-After`, fallback waits honor that delay before trying the next model

## Project-Level Routing

Configure routing per project in `~/.claude/projects/<project-id>/claude-code-router.json`:

```json
{
  "Router": {
    "default": "groq,llama-3.3-70b-versatile"
  }
}
```

Project-level configuration takes precedence over global configuration.

## Custom Router

Create a custom JavaScript router function:

1. Create a router file (e.g., `custom-router.js`):

```javascript
module.exports = function(config, context) {
  // Analyze the request context
  const { scenario, projectId, tokenCount } = context;

  // Custom routing logic
  if (scenario === 'background') {
    return 'groq,llama-3.3-70b-versatile';
  }

  if (tokenCount > 100000) {
    return 'gemini,gemini-1.5-pro';
  }

  // Default
  return 'deepseek,deepseek-chat';
};
```

2. Set the `CUSTOM_ROUTER_PATH` environment variable:

```bash
export CUSTOM_ROUTER_PATH="/path/to/custom-router.js"
```

## Token Counting

The router uses `tiktoken` (cl100k_base) to estimate request token count. This is used for:

- Determining if a request exceeds `longContextThreshold`
- Custom routing logic based on token count

## Subagent Routing

Claude Code subagent requests can be routed in three ways (highest priority first):

1. **Explicit tag** in system or message text (the tag is stripped before the upstream request):

```
<CCR-SUBAGENT-MODEL>provider,model</CCR-SUBAGENT-MODEL>
Please help me analyze this code...
```

`provider/model` is also accepted and normalized to `provider,model`.

2. **Environment variable** for Claude Code subagent turns (when Claude Code marks the request as a subagent, for example via its billing helper header):

```bash
export CLAUDE_CODE_SUBAGENT_MODEL="provider,model"
```

3. If neither tag nor env model is set, normal Router rules apply.

## Next Steps

- [Transformers](/docs/server/config/transformers) - Apply transformations to requests
- [Custom Router](/docs/server/advanced/custom-router) - Advanced custom routing
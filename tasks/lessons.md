# Project Lessons Learned

## Git & Workflow
- **Cherry-picking for Integration**: When merging a feature branch from a divergent fork into a base repo with critical fixes, cherry-picking specific commits is preferred over `git merge` to avoid regression and maintain a linear history.

## LLM Provider Integration (Mistral)
- **Parameter Mapping**: Mistral requires `reasoning_effort` (low, medium, high) instead of a `reasoning` object. 
- **Model ID Wildcards**: Use `.startsWith()` for model families (e.g., `mistral-small-`) to support multiple versions of the same model without manual list updates.
- **Effort Heuristics**: Mapping `max_tokens` to effort levels (low < 1k, medium < 5k, high > 5k) is a reliable way to translate unified reasoning requests to provider-specific efforts.
- **Nested Thinking Format**: Some Mistral models return thinking as an array of blocks within the `content` field (`delta.content = [{ type: "thinking", thinking: [...] }]`). Naive string concatenation of these objects results in `[object Object]` in the UI.
- **Aggressive Serialization**: To prevent `[object Object]` outputs, always verify if thinking content is a string or an object. Use a fallback to `.text` property or `JSON.stringify()` when dealing with provider-specific thinking blocks.

## Architecture
- **Thin Transformer Pattern**: For maintainability, keep transformer classes as thin wrappers that handle high-level provider config and delegate all data transformation logic to a dedicated utility file (e.g., `gemini.util.ts`, `mistral.util.ts`).

## Infrastructure & Deployment
- **Docker Compose Builds**: Always include the `build` block with `context` and `dockerfile` when using local images to prevent Docker from attempting to pull the image from a remote registry.
- **Port Consistency**: Cross-verify `docker-compose` port mappings with the application's `config.json` (e.g., port `3456`) to ensure connectivity.
- **Persistence**: Use volume mounts for the entire configuration root (e.g., `/root/.claude-code-router`) to preserve both settings and logs across container restarts.

## Development & Tooling
- **Dependency Scoping**: `pino` is provided by Fastify in the server package but is not a direct dependency of the `core` package. Direct imports of `pino` in `core` will cause build failures; use the passed-in `logger` instance or native `fs` for separate log files.
- **Response Cloning**: When implementing background logging for responses, use `response.clone()` to avoid consuming the original stream, which would otherwise prevent the transformer from processing the output.

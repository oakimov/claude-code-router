# Provider Bridges: Future Optimizations

This document tracks potential improvements and known unknowns for the provider bridges. Most of it concerns `chrome-device-bridge.ts` (Gemini Nano); Cursor SDK items are collected at the end.

## 📖 Core Concepts & Mechanics

### Whitespace Stalling (`MAX_WS_STALL`)
Gemini Nano can enter deterministic loops when emitting highly structured content (e.g., deeply indented code), producing an endless stream of whitespace.
- **Handling**: The bridge monitors `nonWsChars`. If `stallChars` exceeds `MAX_WS_STALL` (currently 1000) without any non-whitespace content, the bridge calls `controller.abort()` to kill the session.
- **Recovery**: This triggers a `truncated: true` signal to the server, which initiates a fallback retry without `responseConstraint` and with a dynamic temperature increase.

## 🛠 Potential Optimizations

### 1. Assistant History Management
- [x] Implement selective inclusion of assistant messages in `buildTurnPrompt` to preserve conversation memory.
- **Done**: Assistant messages are included on the first turn (`processedMsgCount === 0`) so the model sees its own prior tool calls when a conversation starts with pre-existing history. Subsequent turns skip them since the Prompt API session already has them in context.

### 2. Time-Based Stall Detection
- [ ] Implement hybrid stall detection (time + char count) to prevent premature aborts on indented files.
- **Problem**: The current `MAX_WS_STALL` is based purely on the number of whitespace characters (1000). Highly indented files (e.g., deeply nested JSON or Python) might trigger this abort prematurely.
- **Potential Fix**: Implement a hybrid stall detection mechanism that combines the character count with a time-based check (e.g., if no non-whitespace content is produced for 15-20 seconds, regardless of the char count).

### 3. Dynamic Top-K Scaling
- [ ] Implement Top-K increase during fallback retries to break deterministic loops.
- **Problem**: We currently only increase temperature during fallback retries. In some deterministic loops, the model may be stuck between a few high-probability tokens.
- **Potential Fix**: Increase `DEFAULT_TOPK` (e.g., from 40 to 60) during the fallback retry (alongside the temperature increase) to provide the model with a wider selection of tokens to break the loop.

## ✅ Current Status (Baseline)
- [x] Operational Override for hallucination prevention.
- [x] Dynamic temperature scaling for stall recovery.
- [x] JSON-robust extraction (`extractJson`).
- [x] Persistent session management with parameter clamping.
- [x] Tool Result labeling and result-checking guidelines.

## 🖱 Cursor SDK Bridge

### 1. When does Cursor read workspace `AGENTS.md`? — resolved
- [x] Answer: **once per session**, when the agent's rules service is constructed.
- **Evidence**: `@cursor/sdk` `LocalCursorRulesService` sets `this._rules = this.load(...)` in its constructor and only re-reads through an *optional* file watcher (`.cursor/rules/**/*.mdc`, `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.cursorignore`). Confirmed empirically: a canary line appended to a live session's `AGENTS.md` was not visible to the model on the next turn, so no watcher is active in the headless local runtime.
- **Consequence**: `refreshWorkspaceGuidance` serves the next agent created against that directory, not the live session. That is fine — a live turn already receives changed host facts through the prompt head/tail, and the in-memory `session.hostEnv` update (used by the scratch-path correction) happens regardless. Documented at the function.

### 2. Model-specific grounding strength — closed, not worth doing
- [x] Decision: keep grounding uniform across models; rely on `scratchPathViolations` for monitoring.
- **Measurement**: guidance + tail reminder is 2,002 chars ≈ **501 tokens**. Real Claude Code turns on a live `glm-5.2` session measured 39.8k–53.5k prompt tokens, so the cost is **~1%**, and it sits at the head of the prompt where Cursor's cache absorbs it (one turn reported 583,872 of 651,111 raw prompt tokens as cached).
- **Why not adaptive**: escalating only after a violation means shipping weaker defaults and waiting for a user-visible failure to trigger the fix, in exchange for ~1%.

## 🗂 Core: prompt-cache policy

### 1. Declarative cache injection points
- [ ] Lift breakpoint / `prompt_cache_key` placement into a single config-driven policy (LiteLLM-style `cache_control_injection_points`), then have provider transformers only transport markers.
- **Why**: Today OpenAI, Anthropic, OpenRouter, OpenCode, Vercel, etc. each invent placement in their transformers. Correct for Zen today (98–99%+ hits), but hard to audit and easy for providers to diverge.
- **Not a latency/hit-rate win** for the current OpenCode → Zen path; do this when multi-provider cache consistency becomes painful.
- **Touch**: `packages/core/src/utils/cacheControl.ts`, `openai.util.ts`, provider transformers that call `applyRawAnthropicPromptCaching` / `injectPromptCaching` / OpenRouter Gemini content markers.
- **Reference**: LiteLLM `integrations/anthropic_cache_control_hook.py`; pipeline notes in the CCR vs LiteLLM comparison.

## 🔀 Core: same-protocol wire keep (request side)

- [x] Same-protocol keep: client owner in the compiled plan (`openai-responses` / `Anthropic` / `OpenAI`) → send `clientWireBody`, skip that owner's body rebuild, still run allowlisted middleware (`opencode-headers`, `xai-auth`, `qwen-auth`). Unified projection remains for routing only.
- [x] One keep decision (`resolveClientWireKeep`) shared by primary and fallback; `processRequestTransformers` does not recompute it.
- [x] Tests: Responses + `opencode-headers`; Responses + `xai-auth`; Anthropic → same Zen chain still converts. `claude-auth` and `codex` are converting owners, not keep middleware.

**Not in keep:** `use: ["codex"]` (Codex dialect), `claude-auth` (v1), or stacking `openai-responses` with `codex`.

## 🔒 Temporary security overrides (`pnpm-workspace.yaml`)

These pins either clear product high-severity advisories or consolidate compatible transitive versions that cannot be fixed by upgrading our direct deps alone. Remove each override once upstream ships a clean, deduplicated tree.

Every override is scoped to an exact `parent@version>child` dependency edge, so a
bare package name is never force-resolved across the whole graph. When an entry
stops matching after an upstream bump, pnpm reports it as unused — that is the
signal to delete it, not to widen the selector.

### 1. `@connectrpc/connect-node@1.7.0>undici` → `^8.9.0`
- [ ] Drop when `@cursor/sdk` no longer pulls vulnerable `undici@5.x`.
- **Why**: `@cursor/sdk` → `@connectrpc/connect-node@1.x` declares `undici: ^5.28.4`. Connect-node's only undici API is `Headers`, and it only uses that polyfill on Node `<18` (dead on our Node `>=22.19`). The package must still resolve at module load time, so point this edge at the same maintained undici 8 range that core already uses rather than installing a second major.
- **Compatibility**: undici 8 still exports `Headers` from its package root; the complete Cursor SDK test suite passes with connect-node resolving to undici 8.
- **Scope**: the connect-node edge only; pnpm deduplicates it with `@caeliq/llms`'s direct `undici@^8.9.0`.
- **Exit**: Cursor ships SDK on `@connectrpc/connect-node@2.x` (no undici dep) or connect-node 1.x raises its undici range; then delete the override and re-audit.

### 1b. `@ai-sdk/provider-utils@4.0.46>undici` → `^8.9.0`
- [ ] Drop when `@ai-sdk/provider-utils` raises its `undici` range to `^8` (or removes the declared dep).
- **Why**: `@ai-sdk/provider-utils@4.0.46` declares `undici: ^6.28.0`, creating a second undici major alongside core's direct `undici@^8.9.0`. It never imports undici at runtime — its fetch path uses `globalThis.fetch` / `safe-node-fetch` (verified: zero `require("undici")` in the compiled bundle; the only `undici` string is a code comment). The declared-major collision is collapsed onto core's maintained 8 line.
- **Compatibility**: undici 8 still exports the `Headers`/`fetch`/`Response` surface; provider-utils does not consume it, so the AI SDK test suite passes with the package resolving to undici 8. `pnpm why undici` now shows a single `undici@8.10.0`.
- **Scope**: the provider-utils edge only; pnpm deduplicates it with `@caeliq/llms`'s direct `undici@^8.9.0`.
- **Exit**: provider-utils widens its range to `^8` (or drops the dep); then delete the override and re-audit.

### 2. Compatible transitive consolidation
- [ ] Drop each edge when its parent reaches the selected child naturally.
- `@pnpm/network.ca-file@1.0.2>graceful-fs` → `4.2.11`: one-patch update adds `EBUSY` retry handling without changing the API.
- `sitemap@7.1.3>@types/node` → `26.4.0` and `p-retry@4.6.2>@types/retry` → `0.12.2`: type-only dependencies; workspace typecheck passes against the unified definitions.
- `serve-handler@6.1.7>bytes` → `3.1.2` and `accepts@1.3.8>negotiator` → `0.6.4`: same-major bugfix/minor releases with their existing APIs preserved.
- `readable-stream@2.3.8>safe-buffer` and `string_decoder@1.1.1>safe-buffer` → `5.2.1`: same-major Buffer compatibility release; both consumers pass the runtime/API smoke suite.
- `@google/genai@2>google-auth-library` → `^11.0.2`: `@google/genai` pins auth 10 while core depends on 11, and that single split was duplicating `google-auth-library`, `gcp-metadata` and `google-logging-utils` at once. Fixing it at the source removes all three pairs, which is why the two former `google-auth-library@*>google-logging-utils` edges are gone rather than retargeted. **Compatibility**: auth 10.9.1 and 11.0.2 ship a byte-identical `build/src`; the major exists only to raise `engines.node` to `>=22` (already our floor) and to take `gcp-metadata` 9 / `google-logging-utils` 2, whose `.d.ts` are likewise unchanged. `@google/genai` uses only `GoogleAuth` with `.getClient()`, `.getRequestHeaders()` and `.request()`, all present in 11; loading `@google/genai@2.17.1` from its real `.pnpm` path resolves auth `11.0.2` and constructs `GoogleGenAI` successfully. **Exit**: drop when `@google/genai` widens its range to `^11`.

### 3. Docs / UI toolchain bridges
- [ ] Drop each when its parent ships a range that reaches the maintained child.
- `copy-webpack-plugin@11>serialize-javascript` and `css-minimizer-webpack-plugin@5>serialize-javascript` → `^7.0.7`: Docusaurus 3.10.2 is current but its Webpack plugins still pin `serialize-javascript@6`.
- `monaco-editor@0.56.0>dompurify` → `^3.4.13`: Monaco pins DOMPurify 3.4.8.
- `sockjs@0.3.24>uuid` → `^11.1.0` and `minimatch@3.1.5>brace-expansion` → `^5.0.8`: deprecated / unpatched transitive children of current parents.
- `gaxios@7>node-fetch` → `npm:node-fetch-native@^1.6.7`: gaxios still requests `node-fetch@3`, whose deprecated chain is unnecessary on Node 22+ (native fetch). The selector deliberately stays on the **major**: it was pinned to `gaxios@7.3.0`, and a routine patch bump to 7.3.1 silently unmatched it, letting `node-fetch@3` → `fetch-blob` → `node-domexception` back in. pnpm does not report a stalled selector as unused, so the only symptom was a generic "1 deprecated subdependency" warning. Prefer major-scoped parents for any child that a patch bump can re-admit.
- **Note**: `react-router` was migrated to `8.x` directly in `packages/ui`, so it no longer needs an override.
- **Maintenance note (2026-08-25)**: the `postcss>nanoid` floor override tracks the parent version resolved in the lockfile (`postcss@8.5.26>nanoid` now; was `8.5.25`). PostCSS already declares `nanoid: ^3.3.17`, so this edge is a lockfile guard, not a forced upgrade — refresh the selector whenever postcss moves.
- **Exit**: re-run `pnpm audit` after each upstream bump and delete entries pnpm reports as unused.

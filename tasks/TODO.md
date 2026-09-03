# Provider Bridges: Future Optimizations

This document tracks potential improvements and known unknowns for the provider bridges. Most of it concerns `chrome-device-bridge.ts` (Gemini Nano).

## 📖 Core Concepts & Mechanics

### Whitespace Stalling (`MAX_WS_STALL`)
Gemini Nano can enter deterministic loops when emitting highly structured content (e.g., deeply indented code), producing an endless stream of whitespace.
- **Handling**: The bridge monitors `nonWsChars`. If `stallChars` exceeds `MAX_WS_STALL` (currently 1000) without any non-whitespace content, the bridge calls `controller.abort()` to kill the session.
- **Recovery**: This triggers a `truncated: true` signal to the server, which initiates a fallback retry without `responseConstraint` and with a dynamic temperature increase.

## 🛠 Potential Optimizations

### 1. Time-Based Stall Detection
- [ ] Implement hybrid stall detection (time + char count) to prevent premature aborts on indented files.
- **Problem**: The current `MAX_WS_STALL` is based purely on the number of whitespace characters (1000). Highly indented files (e.g., deeply nested JSON or Python) might trigger this abort prematurely.
- **Potential Fix**: Implement a hybrid stall detection mechanism that combines the character count with a time-based check (e.g., if no non-whitespace content is produced for 15-20 seconds, regardless of the char count).

### 2. Dynamic Top-K Scaling
- [ ] Implement Top-K increase during fallback retries to break deterministic loops.
- **Problem**: We currently only increase temperature during fallback retries. In some deterministic loops, the model may be stuck between a few high-probability tokens.
- **Potential Fix**: Increase `DEFAULT_TOPK` (e.g., from 40 to 60) during the fallback retry (alongside the temperature increase) to provide the model with a wider selection of tokens to break the loop.

## 🗂 Core: prompt-cache policy

### 1. Declarative cache injection points
- [ ] Lift breakpoint / `prompt_cache_key` placement into a single config-driven policy (LiteLLM-style `cache_control_injection_points`), then have provider transformers only transport markers.
- **Why**: Today OpenAI, Anthropic, OpenRouter, OpenCode, Vercel, etc. each invent placement in their transformers. Correct for Zen today (98–99%+ hits), but hard to audit and easy for providers to diverge.
- **Not a latency/hit-rate win** for the current OpenCode → Zen path; do this when multi-provider cache consistency becomes painful.
- **Touch**: `packages/core/src/utils/cacheControl.ts`, `openai.util.ts`, provider transformers that call `applyRawAnthropicPromptCaching` / `injectPromptCaching` / OpenRouter Gemini content markers.
- **Reference**: LiteLLM `integrations/anthropic_cache_control_hook.py`; pipeline notes in the CCR vs LiteLLM comparison.

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

### 4. qs / fast-uri security floors (added 2026-09-03, `pnpm audit` clean)
- [ ] Drop each when its parent ships a range that reaches the patched child.
- `express@4.22.2>qs` and `body-parser@1.20.6>qs` → `^6.16.0`: both pin `qs ~6.15.1`, unreachable to the patched 6.16.0 (GHSA-4mjr-xmp4-gh2g array-limit bypass, GHSA-x5fp-wj9c-mxmx isBuffer DoS). express 5 / body-parser 2 already admit `^6`, so only the two tilde-pinned edges need floors. **Compatibility**: qs 6.16 preserves the parse/stringify API and keeps the same two deps; smoke-tested `qs.parse` from the real `.pnpm` path.
- `ajv@8.20.0>fast-uri` → `^3.1.7`, `@fastify/ajv-compiler@4.0.6>fast-uri` and `fast-json-stringify@7.0.1>fast-uri` → `^4.1.4`: clears the fast-uri host-confusion/SSRF advisories on both majors (`<3.1.6`, `<4.1.3`). Parents declare `^3`/`^4` but pnpm retains locked pins across plain installs (even `pnpm update --depth Infinity` reported "up to date"), so the floors force re-resolution. Smoke-tested `parse` on both lines from their real `.pnpm` paths.
- **Rejected**: `csso@5.0.5>css-tree ^2.3.1` was added and then reverted the same day — csso forks css-tree with a syntax definition written against the 2.2 node schema, and 2.3 throws `Missed 'structure' field in 'String' node type definition` at load. css-tree stays dual (2.2.1 for csso, 2.3.1 for svgo) until csso supports 2.3; mdn-data stays dual with it.

### 5. Same-minor consolidations (added 2026-09-03)
- [ ] Drop each when its parent reaches the selected child naturally.
- `@tailwindcss/node@4.3.3>lightningcss` → `^1.33.0`: exact pin `1.32.0` while the rest of the tree is on 1.33.0; unifies all 11 platform binaries. Smoke-tested native `transform` at 1.33.0 and `@tailwindcss/node` `compile` load.
- `@tailwindcss/typography@0.5.20>postcss-selector-parser` → `^6.1.4`: exact pin `6.0.10`; postcss-calc already resolves 6.1.4. The 7.1.5 line (csstools) stays — intentional major split.
- `@docsearch/react@4.7.0>@algolia/autocomplete-core` → `^1.19.9`: exact pin `1.19.2` while docusaurus theme-search-algolia resolves 1.19.9 (even docsearch 5.x still pins 1.19.2, so no upgrade path exists). One edge collapses all three pairs (core, shared, plugin-insights) since core's `^1.19` children follow. Smoke-tested `createAutocomplete` and the docsearch ESM bundle from their real `.pnpm` paths.

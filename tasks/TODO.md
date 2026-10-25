# Provider Bridges: Future Optimizations

This document tracks potential improvements and known unknowns for the provider bridges. Most of it concerns `chrome-device-bridge.ts` (Gemini Nano).

## Adversarial review of uncommitted changes (2026-10-18)

- [x] Trace changed request handling, stream conversion, dependency overrides, and container configuration.
- [x] Reproduce confirmed failures, fix their causes, and add regression coverage.
- [x] Run workspace checks and record the review results and validation limits.

Review results:
- Fixed schema relaxation through surviving and indirect local references, prototype-name tool alias corruption, and empty/stale authentication headers. Regressions cover accepted JSON Schema values and both Chat/Responses tool names.
- Excluded the runtime configuration volume from Docker build contexts and made the gate-stream regression entry point fail unless its asynchronous checks complete.
- Passed typecheck, lint, 122/122 hermetic test entry points (4 operator-dependent tests skipped), workspace/docs builds, Compose configuration validation, and a separate Node 22 Alpine image build plus offline runtime-import smoke test. All 50 override selectors match the lockfile; dedupe check passes.
- Audit: one high `braces` advisory ([GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm)), with no published patched version, on docs/UI/root build-tool paths. Live provider acceptance and cache behavior were not rerun; the running service was not replaced.

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

## ⚡ Fast mode (Codex + Cursor)

### 1. Design client-selectable fast mode
- [ ] Design one CCR-wide way for a client to opt into a provider's fast / priority tier, then implement it for Codex and Cursor.
- **Today**: fast is never selected. Codex: Responses inbound rejects `service_tier` (`utils/openai.responses.util.ts`), CCR adds none, and `ccr codex-config` publishes `service_tiers: []`. Cursor: `cursor-sdk/model-selection.ts` always picks `fast=false`.
- **Upstream shapes**: Codex catalog `service_tiers` (`priority`, "Fast"; `gpt-6-sol` / `gpt-6-luna` default to it) sent as `service_tier`; Cursor catalog `fast` parameter on parameterized models (Grok 4.7).
- **Open questions**: how a client asks for it per protocol (Responses/Chat `service_tier`, an Anthropic equivalent, a model suffix), whether configs may set a per-provider/per-route default, usage/cost reporting for the higher tier, and how the Cursor session remint rule treats a fast toggle.

## 🔒 Temporary security overrides (`package.json` `overrides` / `bun.lock`)

These pins either clear product high-severity advisories or consolidate compatible transitive versions that cannot be fixed by upgrading our direct deps alone. Remove each override once upstream ships a clean, deduplicated tree.

Every override is scoped to an exact `parent@version>child` dependency edge, so a
bare package name is never force-resolved across the whole graph. Retarget or
delete selectors that no longer match after an upstream bump — do not widen them.
(Toolchain is Bun workspaces; historical notes below may still say `pnpm`.)

### 1b. `@ai-sdk/provider-utils@4.0.54>undici` → `^8.9.0`
- [ ] Drop when `@ai-sdk/provider-utils` raises its `undici` range to `^8` (or removes the declared dep).
- **Why**: `@ai-sdk/provider-utils@4.0.54` declares `undici: ^6.28.0`. CCR itself no longer depends on undici (Bun `fetch` + `proxy` option), but the AI SDK transitive still pulls it. Collapse that edge onto undici 8 so the graph does not keep a vulnerable/legacy 6 line.
- **Maintenance note (2026-09-24)**: selector was `4.0.46`, retargeted to `4.0.54` when the AI SDK 3.0.122 / ai 6.0.290 bump moved provider-utils. Exact-version selectors go stale on every provider-utils bump — retarget, never widen.
- **Compatibility**: undici 8 still exports the `Headers`/`fetch`/`Response` surface; provider-utils does not consume it at runtime (fetch path uses `globalThis.fetch`).
- **Scope**: the provider-utils edge only (`package.json` overrides).
- **Exit**: provider-utils widens its range to `^8` (or drops the dep); then delete the override and re-audit.

### 2. Compatible transitive consolidation
- [ ] Drop each edge when its parent reaches the selected child naturally.
- `@pnpm/network.ca-file@1.0.2>graceful-fs` → `4.2.11`: one-patch update adds `EBUSY` retry handling without changing the API.
- `sitemap@7.1.3>@types/node` → `26.6.5` and `p-retry@4.6.2>@types/retry` → `0.12.2`: type-only dependencies; workspace typecheck passes against the unified definitions. (Retargeted `26.4.1` → `26.6.5` in the 2026-10-17 review — see §8.)
- `postcss-colormin@6.1.0>colord` and `postcss-minify-gradients@6.0.3>colord` → `^2.10.0`: both declare `colord ^2.9.x`; the minor fixes the oversized-color-string advisory. Two edges because two parents pin the stale lockfile entry.
- `@svgr/plugin-svgo@8.1.0>svgo` and `postcss-svgo@6.0.3>svgo` → `^3.3.5`: both declare `svgo ^3`; 3.3.5 patches both removeScripts advisories (4.x is a major split — left alone).
- `serve-handler@6.1.7>bytes` → `3.1.2` and `accepts@1.3.8>negotiator` → `0.6.4`: same-major bugfix/minor releases with their existing APIs preserved.
- `readable-stream@2.3.8>safe-buffer` and `string_decoder@1.1.1>safe-buffer` → `5.2.1`: same-major Buffer compatibility release; both consumers pass the runtime/API smoke suite.
- `@google/genai@2>google-auth-library` → `^11.0.2`: `@google/genai` pins auth 10 while core depends on 11, and that single split was duplicating `google-auth-library`, `gcp-metadata` and `google-logging-utils` at once. Fixing it at the source removes all three pairs, which is why the two former `google-auth-library@*>google-logging-utils` edges are gone rather than retargeted. **Compatibility**: auth 10.9.1 and 11.0.2 ship a byte-identical `build/src`; the major exists only to raise `engines.node` to `>=22` (already our floor) and to take `gcp-metadata` 9 / `google-logging-utils` 2, whose `.d.ts` are likewise unchanged. `@google/genai` uses only `GoogleAuth` with `.getClient()`, `.getRequestHeaders()` and `.request()`, all present in 11; loading `@google/genai@2.17.1` from its real `.pnpm` path resolves auth `11.0.2` and constructs `GoogleGenAI` successfully. **Exit**: drop when `@google/genai` widens its range to `^11`.

### 3. Docs / UI toolchain bridges
- [ ] Drop each when its parent ships a range that reaches the maintained child.
- `copy-webpack-plugin@11>serialize-javascript` and `css-minimizer-webpack-plugin@5>serialize-javascript` → `^7.0.7`: Docusaurus 3.10.2 is current but its Webpack plugins still pin `serialize-javascript@6`.
- `monaco-editor@0.56.0>dompurify` → `^3.4.13`: Monaco pins DOMPurify 3.4.8.
- `sockjs@0.3.24>uuid` → `^11.1.0` and `minimatch@3.1.5>brace-expansion` → `^5.0.8`: deprecated / unpatched transitive children of current parents.
- `gaxios@8>node-fetch` → `npm:node-fetch-native@^1.6.7`: gaxios still requests `node-fetch@3`, whose deprecated chain is unnecessary on Node 22+ (native fetch). The selector deliberately stays on the **major**: it was pinned to `gaxios@7.3.0`, and a routine patch bump to 7.3.1 silently unmatched it, letting `node-fetch@3` → `fetch-blob` → `node-domexception` back in. pnpm does not report a stalled selector as unused, so the only symptom was a generic "1 deprecated subdependency" warning. Prefer major-scoped parents for any child that a patch bump can re-admit. (Parent moved 7 → 8 in the 2026-10-17 update wave; the major-scoped selector followed it — see §8.)
- **Note**: `react-router` was migrated to `8.x` directly in `packages/ui`, so it no longer needs an override.
- **Maintenance note (2026-08-25 → removed 2026-10-17)**: the `postcss>nanoid` floor override tracked the resolved parent (`postcss@8.5.26>nanoid`; was `8.5.25`). Deleted during the 2026-10-17 review: postcss 8.5.30 declares `nanoid: ^3.3.20` itself, so the lockfile guard became unnecessary (see §8).
- **Exit**: re-run `bun audit` and `.claude/skills/package-review/scripts/check-overrides.js` after each upstream bump; retarget or delete `STALE` selectors.

### 4. qs / fast-uri security floors (added 2026-09-03)
- [ ] Drop each when its parent ships a range that reaches the patched child.
- `express@4.22.3>qs` and `body-parser@1.20.9>qs` → `^6.16.0`: both pin `qs ~6.15.1`, unreachable to the patched 6.16.0 (GHSA-4mjr-xmp4-gh2g array-limit bypass, GHSA-x5fp-wj9c-mxmx isBuffer DoS). express 5 / body-parser 2 already admit `^6`, so only the two tilde-pinned edges need floors. **Compatibility**: qs 6.16 preserves the parse/stringify API. (Parents retargeted `4.22.2` → `4.22.3` / `1.20.6` → `1.20.8` in the 2026-10-17 wave; `body-parser` → `1.20.9` in the 2026-10-25 Bun review — see §9.)
- `ajv@8.20.0>fast-uri` → `^3.1.8`, `@fastify/ajv-compiler@4.0.7>fast-uri` and `fast-json-stringify@7.2.0>fast-uri` → `^4.1.5`: clears the fast-uri host-confusion/SSRF advisories on both majors (`<3.1.6`, `<4.1.3`). Floors force re-resolution past locked pins. (`fast-json-stringify` retargeted `7.1.0` → `7.2.0` in the 2026-10-25 Bun review — see §9.)
- **Rejected**: `csso@5.0.5>css-tree ^2.3.1` was added and then reverted the same day — csso forks css-tree with a syntax definition written against the 2.2 node schema, and 2.3 throws `Missed 'structure' field in 'String' node type definition` at load. css-tree stays dual (2.2.1 for csso, 2.3.1 for svgo) until csso supports 2.3; mdn-data stays dual with it. **Re-confirmed 2026-10-17**: re-added during the review, failed identically at `require('csso')`, re-reverted the same day — do not re-attempt until csso ships a 2.3-compatible syntax definition.

### 5. Same-minor consolidations (added 2026-09-03)
- [ ] Drop each when its parent reaches the selected child naturally.
- `@tailwindcss/node@4.3.3>lightningcss` → `^1.33.0`: exact pin `1.32.0` while the rest of the tree is on 1.33.0; unifies all 11 platform binaries. Smoke-tested native `transform` at 1.33.0 and `@tailwindcss/node` `compile` load.
- `@tailwindcss/typography@0.5.20>postcss-selector-parser` → `^6.1.4`: exact pin `6.0.10`; postcss-calc already resolves 6.1.4. Retargeted to `^7.1.6` during the 2026-10-06 package review (see §7): typography only uses `parser()` / `astSync` / `selector()` / `prepend` / `toString`, all intact in v7 (7.0.0 changed insert-during-iteration only). Verified by a warning-clean `build:docs`.
- `@docsearch/react@4.7.1>@algolia/autocomplete-core` → `^1.19.9`: exact pin `1.19.2` while docusaurus theme-search-algolia resolves 1.19.9 (even docsearch 5.x still pins 1.19.2, so no upgrade path exists). One edge collapses all three pairs (core, shared, plugin-insights) since core's `^1.19` children follow. Smoke-tested `createAutocomplete` and the docsearch ESM bundle from their real `.pnpm` paths. (Parent retargeted `4.7.0` → `4.7.1` in the 2026-10-17 wave — see §8.)

### 6. Direct-dependency currency + residual advisories (reviewed 2026-09-24)
- `adm-zip` `^0.6.0` → `^0.6.1` in cli/server/shared: patch release clears both advisories (symlink overwrite + uncompressed-size DoS). Direct dep, no override needed.
- `@cursor/sdk` `^1.0.30` → `^1.0.32` in root/cli/core/server: patch line, purely additive (new `SDKToolAnnotations` / `SteerAckOutcome` exports, optional `steer()` / `systemPrompt`). Tarball-diffed 1.0.30 vs 1.0.32, no signature changes; `@connectrpc/connect-node` stays 1.7.0 so the undici edge still matches.
- **Residual (accepted, docs-only, major-split — needs opt-in to force)**: `joi@17.13.4` (2× low, fix = joi 18), `js-yaml@4.3.1` (1× high, fix = js-yaml 5). Both live only under `docs>@docusaurus/*` (build-time toolchain, never shipped to users); forcing the majors would break docusaurus 3.10.2's declared ranges. Revisit when docusaurus widens its ranges.
- **AI SDK / ai / zod same-major rollup (2026-09-24)**: `ai` 6.0.258→6.0.290, `@ai-sdk/anthropic` 3.0.111→3.0.122, `@ai-sdk/openai` 3.0.97→3.0.117 (cli+server exact pins), `zod` 4.4.3→4.6.5 (cli exact; server `^4.4.3` follows), `@ai-sdk/react` 3.0.261→3.0.293 via `pnpm --filter @caeliq/ccr-ui update` (pairs ai 6.0.290 + provider-utils 4.0.54; needed a `minimumReleaseAgeExclude` entry for `3.0.293`). Collapses `ai`, `@ai-sdk/gateway`, `@ai-sdk/provider-utils` to one version each and keeps the §1b undici edge matching. AI SDK 3.x peer range (`zod ^3.25.76 || ^4.1.8`) admits zod 4.6.5. Full suite (typecheck/lint/test/build/docs) green after the bump.
- **Left alone (intentional dual majors)**: `@jsonjoy/*` 1.x vs 17.x, `@sindresorhus/is` 4/5, `@types/express` 4/5 (+ serve-static/send), `@types/unist` 2/3, `css-tree` 2.2/2.3 (see §4 rejection). (`postcss-selector-parser` 6/7 was unified to 7.1.6 in the 2026-10-06 review, see §7.)

### 7. Package review 2026-10-06 (full scope: security + strong + high-risk majors)
- Result: `pnpm audit` 38 advisories (3 critical / 13 high / 16 moderate / 6 low) → **1 high** (`braces@3.0.3`, no patched release exists upstream — accepted residual, docs-build-time only via `micromatch`). `pnpm dedupe --check` passes. 132 true multi-version packages inventoried; intentional duals (express 4/5 + `@types/*` pairs, webpack/docusaurus toolchains, `@jsonjoy/*` 1/17, `commander`, `zod` 3 vs 4 v3-leg) left alone.
- Stale-pin refresh (no override changes needed — parent ranges already admitted the patched releases): `undici` 8.10.1→8.11.2, `fastify` 5.12.1→5.12.5, `proxy-addr` 2.0.7→2.0.8, `brace-expansion` 5.0.9→5.0.12, `dompurify` 3.4.13→3.4.16, `ip-address` 10.5.0→10.7.3, `compression`→1.8.2, `http-cache-semantics`→4.3.0, `@fastify/busboy`→3.2.2, `source-map-js`→1.2.2. `zod` 4.4.3→4.6.5 (`@anthropic-ai/sdk`, `@modelcontextprotocol/sdk`, `openai@7.5.0`, `zod-to-json-schema` all declare open `^4` ranges); v3 leg (`@cursor/sdk`, `chromium-bidi`) stays.
- Floor raises: `ajv@8.20.0>fast-uri` `^3.1.7`→`^3.1.8`, `@fastify/ajv-compiler` + `fast-json-stringify` `^4.1.4`→`^4.1.5` (new host-confusion/mailto advisories on both lines).
- New exact-edge overrides (compat evidence in `pnpm-workspace.yaml` comments):
  - `'@docusaurus/core@3.10.2>tinypool'` → `^2.2.0` (critical RCE advisories; docusaurus declares `^1.0.2`). Verified against tinypool 2.2.0 types (default export, all used options, `run`/`destroy`) and a `build:docs` run with `DOCUSAURUS_SSG_WORKER_THREAD_COUNT=2` forcing pool construction. Drop when docusaurus supports tinypool 2.
  - `'mermaid@11.16.1>katex'` → `^0.19.0` (mermaid pins `^0.16.45`; only uses stable `render`/`renderToString`).
  - Six `postcss-selector-parser` → `^7.1.6` edges (`postcss-calc`, `postcss-minify-selectors`, `postcss-merge-rules`, `postcss-unique-selectors`, `stylehacks`, `postcss-discard-unused`) plus the `@tailwindcss/typography` retarget: all declare `^6` and use only `parser(cb)` / `processSync` / `walk` / `astSync` / constructors, all present in v7 (7.0.0 changed insert-during-iteration only). The remaining preset-env/csstools legs moved via their own `^7`-admitting ranges on refresh — `postcss-selector-parser` is now single-version (7.1.6).
- `@cursor/sdk` `^1.0.32` → `^1.0.36` in root/cli/core/server: patch line, purely additive (new optional executor-override / `sessionId` / `metadata` / `subagentInherit` fields; tarball-diffed .d.ts, no signature changes). Still pulled `@connectrpc/connect-node@1.7.0` at the time; cursor 1.0.37 (2026-10-17) dropped it, so the undici edge was deleted — see §8 and §1b.
- Deferred (noted, not forced): `ai` 7.x / `@ai-sdk/*` 4.x majors, `mermaid` 12, `monaco-editor` 0.57, `@google/genai` 2.18→2.27, `@modelcontextprotocol/sdk` 1.30→1.32, `openai` 7.5→7.28 (in-range but 23 minors — bump deliberately for a dedicated review), same-major tight-pin pairs (`mime-db`, `mime-types` 2.x, `ms`, `range-parser`, `cookie-signature`, `iconv-lite`, `source-map`, `babel-plugin-polyfill-corejs3` — parents pin exact/tilde ranges; override value does not justify the compat work).
- Registry privacy test (`cursor-sdk.inbound-session.ts` › `testRegistryStoresHashesAndSkipsUnchangedWrites`): previously scanned the raw file for `4711`, which collided with `updatedAt` timestamps. Now walks parsed JSON string fields only.

### 8. Package review 2026-10-17 (full scope: security + dedupe + currency)
- Result: `pnpm audit` 3 advisories (1 critical `shell-quote`, 2 high — `braces` and the MCP SDK / `hono` path) → **1 high** (`braces`; no patched release exists upstream — re-confirmed, accepted residual, reached only at build time via `micromatch`). `pnpm dedupe --check` passes; selector scan: 50 `parent@version>child` selectors, 0 stale.
- Advisories cleared:
  - `shell-quote` `^1.10.0` → `^1.12.0` in cli/server (direct dep; critical advisory fixed).
  - `@google/genai` `^2.18.0` → `^2.30.0` in core: the optional `@modelcontextprotocol/sdk` peer is gone from the graph, removing the MCP SDK / `hono` advisory path entirely. The now-dead `@modelcontextprotocol/sdk@1.30.0>hono` override (§2) was deleted.
- In-range direct-dependency update wave (`pnpm update -r`; manifests + lockfile): `fastify` 5.12.7, `@fastify/*` (cookie, cors, multipart, rate-limit 11.2.1, static), `undici` 8.12.2, `openai` 7.34.0 (range held `^7`), `vite` 8.3.4, `react`/`react-dom` 19.3.1, radix minors, `puppeteer-core` 25.14.0, `@cursor/sdk` 1.0.37, eslint/TS toolchain, `@types/node` 26.6.5. (First attempt hit a transient `ERR_PNPM_NO_MATCHING_VERSION` for `@ai-sdk/provider-utils@4.0.60`; immediate retry succeeded.)
- New exact-edge consolidations (7, single-version results): `serve-handler@6.1.7` × {content-disposition 0.5.4, mime-types 2.1.35, range-parser 1.3.0}, `express@4.22.3>range-parser` and `send@0.19.2>range-parser` → `range-parser` single 1.3.0; `mime-types@2.1.35>mime-db` → single 1.54.0; `debug@2.6.9>ms` → single 2.1.3. Parents exact-pin stale children; consumers use only the preserved parse/serialize/lookup APIs; each edge smoke-tested by loading the consumer from its real `.pnpm` path.
  - **Rejected (re-confirmed)**: `csso@5.0.5>css-tree ^2.3.1` re-added and re-reverted the same day — `require('csso')` throws `Missed 'structure' field in 'String' node type definition` (see §4). `css-tree` stays dual (2.2.1 csso / 2.3.1 svgo), `mdn-data` dual with it.
- Retargeted selectors after the wave: `gaxios@7` → `gaxios@8.3.1` (major-scoped `node-fetch-native` alias kept), `express@4.22.2`/`body-parser@1.20.6` → `4.22.3`/`1.20.8` (qs floors), `@fastify/ajv-compiler@4.0.6` → `4.0.7` and `fast-json-stringify@7.0.1` → `7.1.0` (fast-uri floors), `@docsearch/react@4.7.0` → `4.7.1`, `sitemap@7.1.3>@types/node` `26.4.1` → `26.6.5`.
- Dead overrides removed: `@connectrpc/connect-node@1.7.0>undici` (§1 — cursor SDK 1.0.37 no longer pulls connect-node), `@modelcontextprotocol/sdk@1.30.0>hono` (§2 — SDK left the graph), `mermaid@11.16.1>katex` (§7 — streamdown 2.7 dropped mermaid), `postcss@8.5.26>nanoid` (§3 — postcss 8.5.30 declares `nanoid ^3.3.20`).
- AI SDK / zod exact alignment: the update wave turned `@caeliq/ccr-ui`'s caret ranges into a second family instance (`ai` 6.0.308, `@ai-sdk/gateway` 3.0.216, `provider-utils` 4.0.60, `@ai-sdk/react` 3.0.311) alongside the cli/server exact pins. Fixed by pinning ui **exact** `"ai": "6.0.290"` / `"@ai-sdk/react": "3.0.293"` (3.0.293 exact-depends on ai 6.0.290 + provider-utils 4.0.54). `zod` aligned: cli exact `4.6.7`, server `^4.6.7`; the stale 4.6.5 peer context refreshed with `pnpm update zod -r --depth 9999`. Singles now: ai 6.0.290, gateway 3.0.200, provider 3.0.17, provider-utils 4.0.54, react 3.0.293, zod 4.x = 4.6.7 (v3 leg via `@cursor/sdk` stays).
- `@types/node` 26.6.5 renamed `node:perf_hooks` `IntervalHistogram` → `ELDHistogram`; `packages/server/src/utils/health-heartbeat.ts` adapted (type-only). Lesson recorded in `lessons.md`.
- Held out-of-range majors (dedicated reviews later): `ai` 7 / `@ai-sdk/anthropic` + `@ai-sdk/openai` + `@ai-sdk/react` 4, `typescript` 7, `dotenv` 18, `@anthropic-ai/sdk` 0.135, `monaco-editor` 0.57, `@docsearch/react` 5, `@huggingface/tokenizers` 0.2, `@types/adm-zip` 0.6 (deprecated).
- `minimumReleaseAgeExclude` still names several superseded versions (`@cursor/sdk*@1.0.36`, `body-parser@1.20.6`, `ai@6.0.258`, …). Inert: no base `minimumReleaseAge` is configured; left alone.
- Verification: `pnpm install --frozen-lockfile` ✓; `pnpm typecheck` ✓; `pnpm lint` ✓; `pnpm test` 122/122 (4 opt-in skipped) ✓; `pnpm build` ✓; `pnpm build:docs` (en + zh-CN) ✓; `git diff --check` ✓; `pnpm dedupe --check` ✓; `pnpm audit` → 1 accepted residual.
- Toolchain: `pnpm` 12.6.0 → 12.14.0 (`packageManager` field + global install). The lockfile gained pnpm's self-managed `packageManagerDependencies` document (`@pnpm/exe.*`). The 12.14 dedupe normalized a redundant `ajv-formats@3.0.1(ajv@8.20.0)` peer-suffix row onto the suffix-less `3.0.1` (same resolved version). `node_modules` was rebuilt from scratch (`--frozen-lockfile`) afterwards, clearing 624 superseded virtual-store dirs left by the update waves; the fresh-install path (what CI/Docker run) re-passed the full battery above.

### 9. Package review 2026-10-25 (Bun workspaces)
- Toolchain migrated to Bun (`packageManager: bun@1.4.4`, lockfile `bun.lock`, overrides in root `package.json`). Skill path: `.claude/skills/package-review`.
- Result: `bun audit` → **0 advisories** (1515 packages). `bun dedupe --check` clean. Override checker: 46 exact-ok, 4 major-scoped, 0 stale (after retargets).
- Multi-version inventory: 100 packages with >1 version; only **6 same-major** duals — all left alone:
  - `css-tree` 2.2.1 / 2.3.1 + `mdn-data` 2.0.28 / 2.0.30 (csso rejection, §4)
  - `source-map` 0.6.1 / 0.7.6, `iconv-lite` 0.4.24 / 0.7.3, `babel-plugin-polyfill-corejs3` 0.13 / 0.14 (prior deferred)
  - `@napi-rs/wasm-runtime` 1.0.7 / 1.2.5 (rspack optional WASM bindings exact-pin different lines)
- Retargeted stale selectors after Bun install resolved newer parents:
  - `body-parser@1.20.8>qs` → `body-parser@1.20.9>qs`
  - `fast-json-stringify@7.1.0>fast-uri` → `fast-json-stringify@7.2.0>fast-uri`
- Direct-dep currency (`bun outdated`): only `typescript` 6 → 7 latest; held (major).
- Verification: `bun install --frozen-lockfile` ✓; `bun run typecheck` ✓; `bun run lint` ✓; `bun run test` 123/123 (4 opt-in skipped) ✓; `bun run build` ✓; `bun run build:docs` (en + zh-CN) ✓; `bun audit` ✓; `bun dedupe --check` ✓; override checker 0 stale ✓.

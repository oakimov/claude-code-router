---
name: package-review
description: Review and harden the Bun monorepo dependency graph — security audit (bun audit), dedupe (bun dedupe), consolidate duplicate versions with exact-edge overrides in package.json, minimize package versions, and keep packages on latest compatible supported releases. Use whenever the user asks to review packages, audit dependencies, dedupe, consolidate versions, reduce duplicate packages, fix advisories, clean up bun.lock, or run /package-review before a minor/major version bump.
---

# Package Review (Bun)

Audit and tighten this monorepo's dependency graph under **Bun workspaces**. Goal: as few package versions as safely possible, zero known vulnerabilities (or documented residual risk), and only supported/maintained releases.

Do not commit unless the user explicitly asks.

## Toolchain facts (do not invent pnpm/npm workflows)

| Concern | Bun location |
|---|---|
| Package manager | `packageManager: "bun@1.4.4"`, `engines.bun: ">=1.4.0"` |
| Workspaces | root `package.json` → `workspaces: ["packages/*", "docs"]` |
| Lockfile | `bun.lock` (JSONC) — **not** `pnpm-lock.yaml` / `package-lock.json` |
| Overrides | root `package.json` → `overrides` (flat `parent@version>child`) |
| Lifecycle scripts | root `trustedDependencies` (`esbuild`, `@swc/core`, …); `bun pm trust` if blocked |
| Install | `bun install` / `bun install --frozen-lockfile` |
| Audit | `bun audit` / `bun audit --json` / `bun audit fix` |
| Dedupe | `bun dedupe` / `bun dedupe --check` |
| Why installed | `bun pm why <pkg>` |
| Tree | `bun pm ls --all` |

Never run `pnpm`, `npm install`, or `npm ci` in this repo (including inside `docs/` or a workspace package). That bypasses root overrides and can rewrite the wrong lockfile.

Historical notes in `tasks/TODO.md` / `tasks/lessons.md` may still say `pnpm` — treat commands there as Bun equivalents when acting.

## Standing rules

1. **Prefer latest compatible** — upgrade; never downgrade to shrink a diff. Hold back only when a concrete incompatibility cannot reasonably be resolved, and explain that blocker.
2. **Exact-edge overrides only** — every `overrides` entry must be `parent@version>child` (or major-scoped `parent@N>child` when a patch bump would silently unmatch). Never a bare package name that force-resolves across the whole graph.
3. **Workspace install only** — always from the repo root with `bun.lock`.
4. **Document temporary bridges** — every new override gets rationale + exit condition in `tasks/TODO.md` under Temporary security overrides.
5. **Delete stale overrides** — Bun does **not** report unused overrides. After every update wave, run the override checker below and remove or retarget `STALE` rows. Do not widen a selector to hide staleness.

## Bundled helpers

Run from the repository root:

```bash
# Multi-version inventory from bun.lock
bun .claude/skills/package-review/scripts/list-multi-versions.js

# Stale / bare / major-scoped override selectors vs bun.lock
bun .claude/skills/package-review/scripts/check-overrides.js
```

## Workflow

Complete phases in order. Ask before applying high-risk consolidations unless the user already said to fix everything / consolidate / do strong candidates.

### Phase 1 — Inventory

```bash
bun audit
bun dedupe --check
bun .claude/skills/package-review/scripts/list-multi-versions.js
bun .claude/skills/package-review/scripts/check-overrides.js
```

Also read:

- root `package.json` → `overrides`, `trustedDependencies`, `packageManager`, `engines`
- `tasks/TODO.md` → Temporary security overrides

Useful probes:

```bash
bun pm why undici
bun pm ls undici
bun pm ls --all | head   # huge; prefer why/ls for named pkgs
```

Report:

- advisory count / severities (`bun audit --json` if needed)
- whether `bun dedupe --check` is clean
- multi-version package list
- stale override selectors
- candidate buckets (below)

### Phase 2 — Classify duplicates

| Bucket | Criteria | Action |
|---|---|---|
| **Security bridge** | Advisory on older version; parent cannot reach patched child | Exact-edge override to patched/maintained release |
| **Strong candidate** | Same major (or type-only); newer API is a strict superset / compatible; parent usage verified | Exact-edge override after smoke check |
| **Higher risk** | Major split (`zod` 3/4, `picomatch` 2/4, `google-auth-library` 10/11, …) | Investigate separately; do not force unless user opts in |
| **Leave alone** | Intentional dual majors still required, or unrelated name collisions | Document why |

`bun dedupe` only collapses overlaps already allowed by declared semver ranges onto versions already in the lockfile. It does **not** prove an override is impossible — strong candidates often need exact-edge overrides beyond what dedupe can do.

### Phase 3 — Security first

1. Prefer upgrading direct dependencies in workspace `package.json` files.
2. For transitive holes the parent cannot reach, add a scoped override:
   ```json
   "@ai-sdk/provider-utils@4.0.54>undici": "^8.9.0"
   ```
3. Prefer replacing deprecated chains with runtime-native equivalents when safe (example: `gaxios@8>node-fetch` → `npm:node-fetch-native@^1.6.7`). Bun understands the `npm:` alias in overrides.
4. Optional autofix when ranges already admit a safe version:
   ```bash
   bun audit fix --dry-run
   bun audit fix
   ```
   Use `bun audit fix --latest` only when the user accepts rewriting declared ranges.
5. Re-run `bun audit` until clean (or only accepted residual risk remains, documented in `tasks/TODO.md`).

### Phase 4 — Dedupe and consolidate

1. Run `bun dedupe` and keep the `bun.lock` change if it reduces versions.
2. For each **strong candidate**:
   - Identify the exact parent→child edge (`bun pm why <child>`).
   - Diff old vs new (`bun pm diff <pkg>@old new` or registry changelog).
   - Confirm parents only use preserved APIs (grep under `node_modules/<parent>` / Bun's install layout).
   - Add the exact-edge override in root `package.json` `overrides`.
   - Regenerate and verify the frozen path:
     ```bash
     bun install
     bun install --frozen-lockfile
     ```
3. Confirm the target package now has **one** live version (`list-multi-versions.js` / `bun pm ls <pkg>`).
4. Smoke-test by loading the **consumer** entry point (not only the child):
   ```bash
   bun -e "require('<consumer-package>')"
   ```
   Resolving cleanly is not enough — some forks load then throw.

### Phase 5 — Direct dependency currency

For workspace manifests:

- Stay within `engines.bun` (`>=1.4.0`). Docker/CI pin is `1.4.4` (`oven/bun:1.4.4-alpine`, `oven-sh/setup-bun`).
- Align shared runtime pins across workspaces (`ai` / `@ai-sdk/*` / `zod` exact families; `@cursor/sdk`).
- Prefer `bun update` / targeted `bun update <pkg>` from the root. After caret-range updates in one workspace, check for a second family instance next to exact pins in another.
- Do not downgrade. Raise `engines.bun` only when a runtime dependency truly requires it.

### Phase 6 — Full verification

After any lockfile or override change:

```bash
bun install --frozen-lockfile
bun pm trust esbuild @swc/core   # if postinstalls were blocked
bun run typecheck
bun run lint
bun run test                     # Chrome bridge: bun run test:chrome-bridge
bun run build
bun run build:docs               # must stay warning-clean
bun audit
bun dedupe --check
bun .claude/skills/package-review/scripts/check-overrides.js
git diff --check
```

All must pass before declaring the review done.

Fresh-tree check when the incremental `node_modules` looks suspicious:

```bash
rm -rf node_modules packages/*/node_modules docs/node_modules
bun install --frozen-lockfile
```

### Phase 7 — Document

Update `tasks/TODO.md` Temporary security overrides (`package.json` `overrides` / `bun.lock`):

- Why the edge exists
- Compatibility evidence (`bun pm why`, smoke require, advisory IDs)
- Exit condition ("drop when parent reaches child naturally", …)

Remove entries the override checker marks `STALE`.

## Override authoring checklist

**Good** (exact edge):

```json
"@ai-sdk/provider-utils@4.0.54>undici": "^8.9.0"
```

**Good** (major-scoped when patch bumps would unmatch):

```json
"gaxios@8>node-fetch": "npm:node-fetch-native@^1.6.7"
```

**Bad** (graph-wide force):

```json
"undici": "^8.9.0"
```

Notes:

- Selectors embed `parent@version`. In-range parent bumps go stale — retarget, never widen to a bare name.
- Bun rewrites overrides into a nested form inside `bun.lock`; author and review them as flat `parent@version>child` in `package.json`.
- Local `file:` overrides are allowed for workspace patches (example: `@docusaurus/mdx-loader@…>image-size` → `file:./packages/docusaurus-image-size`).

## Report shape

When finishing a review, summarize:

1. Audit before → after (severities; accepted residuals)
2. Dedupe / multi-version count before → after
3. Overrides added / retargeted / removed
4. Verification commands run and outcomes
5. Follow-ups left in `tasks/TODO.md`

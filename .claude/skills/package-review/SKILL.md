---
name: package-review
description: Review and harden the pnpm monorepo dependency graph — security audit, dedupe, consolidate duplicate versions with exact-edge overrides, minimize package versions, and keep packages on latest compatible supported releases. Use when the user asks to review packages, audit dependencies, dedupe, consolidate versions, reduce duplicate packages, fix advisories, or clean up the lockfile.
---

# Package Review

Audit and tighten this monorepo's dependency graph. Goal: as few package versions as safely possible, zero known vulnerabilities, and only supported/maintained releases.

Do not commit unless the user explicitly asks.

## Standing rules

1. **Prefer latest compatible** — upgrade; never downgrade to shrink a diff. Hold back only when a concrete incompatibility cannot reasonably be resolved, and explain that blocker.
2. **Exact-edge overrides only** — every `pnpm.overrides` entry must be `parent@version>child`, never a bare package name that force-resolves across the whole graph.
3. **Workspace install only** — install from the repo root with pnpm and the lockfile. Never `npm install` inside a workspace package (especially `docs/`); that bypasses overrides and reintroduces advisories.
4. **Document temporary bridges** — every new override gets rationale + exit condition in `tasks/TODO.md` under Temporary security overrides.
5. **Delete unused overrides** — when pnpm reports an override as unused after an upstream bump, remove it; do not widen the selector.

## Workflow

Run from the repository root. Complete phases in order. Ask before applying high-risk consolidations unless the user already said to fix everything / consolidate / do strong candidates.

### Phase 1 — Inventory

```bash
pnpm audit
pnpm dedupe --check
```

List every package name that resolves to more than one version in the live graph:

```bash
node - <<'NODE'
const fs = require('fs');
const lock = fs.readFileSync('pnpm-lock.yaml', 'utf8');
const versions = new Map();
for (const line of lock.split('\n')) {
  // importers/packages keys look like:  /name@version( or :
  const m = line.match(/^\s{2}(?:'\/|\/)((?:@[^/]+\/)?[^@/]+)@([^(':\s]+)/);
  if (!m) continue;
  const [, name, ver] = m;
  if (!versions.has(name)) versions.set(name, new Set());
  versions.get(name).add(ver);
}
[...versions.entries()]
  .filter(([, vs]) => vs.size > 1)
  .sort((a, b) => b[1].size - a[1].size || a[0].localeCompare(b[0]))
  .forEach(([name, vs]) => console.log(`${name}: ${[...vs].sort().join(', ')}`));
NODE
```

Also read current overrides in `pnpm-workspace.yaml` and the tracked bridges in `tasks/TODO.md`.

Report:

- advisory count / severities
- whether `pnpm dedupe --check` passes
- multi-version package list
- candidate buckets (below)

### Phase 2 — Classify duplicates

Bucket each multi-version package:

| Bucket | Criteria | Action |
|---|---|---|
| **Security bridge** | Advisory on older version; parent cannot reach patched child | Exact-edge override to patched/maintained release |
| **Strong candidate** | Same major (or type-only); newer API is a strict superset / compatible; parent usage verified | Exact-edge override after smoke check |
| **Higher risk** | Major split (`zod` 3/4, `picomatch` 2/4, `google-auth-library` 10/11, etc.) | Investigate separately; do not force unless user opts in |
| **Leave alone** | Different packages that happen to share a name prefix, or intentional dual majors with incompatible APIs still required | Document why |

`pnpm dedupe` only collapses overlaps already allowed by declared semver ranges. It does **not** prove an override is impossible — strong candidates often need exact-edge overrides beyond what dedupe can do.

### Phase 3 — Security first

1. Fix advisories by upgrading direct dependencies when possible.
2. If the vulnerable package is transitive and the parent is already current, add a scoped override:
   ```yaml
   'parent@x.y.z>child': ^patched
   ```
3. Prefer replacing deprecated chains with Node-native equivalents when safe (example: `gaxios>node-fetch` → `npm:node-fetch-native@...` on Node 22+).
4. Re-run `pnpm audit` until clean (or only accepted residual risk remains, documented).

### Phase 4 — Dedupe and consolidate

1. Run `pnpm dedupe` and keep the lockfile change if it reduces versions.
2. For each **strong candidate**:
   - Identify the exact parent→child edges pinning the older version.
   - Diff old vs new package (npm pack / tarball or published changelog) for API breaks.
   - Confirm parents only use preserved APIs (grep parent package source under `.pnpm`).
   - Add exact-edge overrides in `pnpm-workspace.yaml`.
   - Regenerate lockfile:
     ```bash
     pnpm install
     # or offline when the store already has the tarballs:
     pnpm install --frozen-lockfile   # after lockfile regen
     ```
3. Verify each targeted package now has **one** live version.
4. Smoke-test runtime parents by requiring them from their real `.pnpm/.../node_modules/<pkg>` paths (root `require(name)` often fails under pnpm for transitives).

### Phase 5 — Direct dependency currency

For workspace `package.json` dependencies:

- Check registry latest within the supported policy (`engines.node` floor is `>=22.19.0`).
- Align shared runtime deps across manifests (e.g. `@cursor/sdk`).
- Do not downgrade. Raise floors only when a runtime dependency truly requires it; keep Docker/CI pins documented separately from `engines`.

### Phase 6 — Full verification

After any lockfile or override change, run from repo root:

```bash
pnpm install --frozen-lockfile
pnpm typecheck
pnpm lint
pnpm test                 # Chrome bridge remains opt-in via pnpm test:chrome-bridge
pnpm build
pnpm build:docs           # must stay warning-clean
pnpm audit
pnpm dedupe --check
git diff --check
```

All must pass before declaring the review done.

### Phase 7 — Document

Update `tasks/TODO.md` Temporary security overrides:

- Why the edge exists
- Compatibility evidence
- Exit condition ("drop when parent reaches child naturally" / "drop when upstream removes undici", etc.)

Remove entries pnpm reports as unused.

## Override authoring checklist

Good:

```yaml
'@connectrpc/connect-node@1.7.0>undici': ^8.9.0
'serve-handler@6.1.7>bytes': 3.1.2
'google-auth-library@10.9.1>google-logging-utils': 1.2.0
'google-auth-library@11.0.0>google-logging-utils': 1.2.0
```

Bad:

```yaml
undici: ^8.9.0          # bare — forces every consumer
bytes: 3.1.2            # bare — can break unrelated parents
```

When two parents pin the same old child, write **two** edges (see `safe-buffer`, `google-logging-utils`).

## Report format

End with a concise report:

1. **Security** — audit result before/after
2. **Deduped / consolidated** — table of package → before versions → after version
3. **Overrides added/removed** — list with one-line rationale
4. **Remaining multi-version packages** — and why they were left
5. **Verification** — pass/fail for typecheck, lint, test, build, docs, audit, dedupe
6. **Commit** — not made unless requested

## Out of scope unless asked

- Creating generic CI workflows
- Publishing, tagging, or version bumps
- Forcing high-risk major consolidations without an explicit opt-in
- Changing application behavior unrelated to dependency resolution

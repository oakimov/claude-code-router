# Publishing @caeliq packages

This repo publishes:

| Package | Path | Published version |
|---|---|---|
| `@caeliq/ccr-shared` | `packages/shared` | `2.0.0` |
| `@caeliq/llms` | `packages/core` | `1.0.54` |
| `@caeliq/claude-code-router` | `packages/cli` | `2.0.0` |

GitHub repo used for provenance / trusted publishing: `oakimov/claude-code-router`  
npm org: `caeliq`  
npm publisher: `oakimov`

## Status

- [x] Manual first publish completed (2026-07-20)
- [ ] Trusted Publisher (OIDC) configured on all three npm packages
- [ ] `.github/workflows/npm-publish.yml` pushed to GitHub `main`
- [ ] Dry-run workflow_dispatch succeeded
- [ ] Tag-based publish verified on a new version

## 1. One-time npm org setup

1. Confirm you can publish under the org:
   ```bash
   npm whoami
   npm org ls caeliq
   ```
2. Your user (`oakimov`) must have a role that can publish in `caeliq` (typically `developer`+).

## 2. Manual first publish (done)

Trusted publishing can only be attached to packages that already exist on the registry.

```bash
# from repo root
npm login
pnpm install
pnpm build
NPM_DRY_RUN=1 bash scripts/release.sh npm
bash scripts/release.sh npm
```

Verify:

```bash
npm view @caeliq/ccr-shared version
npm view @caeliq/llms version
npm view @caeliq/claude-code-router version
npm install -g @caeliq/claude-code-router
ccr -h
```

## 3. Configure npm Trusted Publishers (OIDC) — do this next

For **each** of the three packages on [npmjs.com](https://www.npmjs.com):

1. Open the package page → **Settings** → **Trusted Publisher**
2. Add a **GitHub Actions** publisher with exactly:
   - **Organization or user:** `oakimov`
   - **Repository:** `claude-code-router`
   - **Workflow filename:** `npm-publish.yml` (filename only, not `.github/workflows/...`)
   - **Environment:** leave empty (unless you later add a GitHub Environment)
3. Save.

Do this for:

- https://www.npmjs.com/package/@caeliq/ccr-shared/access
- https://www.npmjs.com/package/@caeliq/llms/access
- https://www.npmjs.com/package/@caeliq/claude-code-router/access

No `NPM_TOKEN` / `NODE_AUTH_TOKEN` secret is required in GitHub after this.

## 4. Push CI to GitHub

The Azure DevOps remote (`origin`) and GitHub remote (`github`) both need the rebrand + workflow:

```bash
git push origin main
git push github main
```

Confirm the workflow file exists at:

https://github.com/oakimov/claude-code-router/blob/main/.github/workflows/npm-publish.yml

## 5. GitHub Actions publish

Workflow: `.github/workflows/npm-publish.yml`

Triggers:

- Push of a version tag: `v*.*.*` (example: `v2.0.1`)
- Manual **workflow_dispatch** (optional dry-run)

It builds with pnpm, then runs `scripts/release.sh npm` with:

- `id-token: write`
- `npm publish --access public --provenance`
- no `NODE_AUTH_TOKEN`

### Recommended first CI check (dry-run)

1. GitHub → Actions → **Publish npm packages** → **Run workflow**
2. Enable **Dry-run**
3. Confirm the job authenticates via OIDC and reaches `npm publish --dry-run`

### Tag and publish a new version

1. Bump versions in the relevant `packages/*/package.json` files
2. Commit and push to `github` (`main`)
3. Tag and push (prefer matching the CLI version):
   ```bash
   git tag v2.0.1
   git push github v2.0.1
   ```
4. Watch **Actions → Publish npm packages**

Publish order inside the script is always: shared → llms → CLI.

## 6. Later releases

1. Bump versions:
   - shared / CLI when those packages change (`packages/shared`, `packages/cli`)
   - `@caeliq/llms` independently when core changes
2. Ensure `packages/core` still depends on a published `@caeliq/ccr-shared` version range that exists
3. Tag `vX.Y.Z` and push to GitHub

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Automatic provenance generation not supported for provider: null` | Provenance was enabled outside CI. Local publishes must not use `--provenance`. |
| `401` / `needs authentication` in CI | Trusted publisher not configured, or workflow filename mismatch (`npm-publish.yml`) |
| `403` unable to publish under `@caeliq` | npm user not in org / insufficient role |
| `cannot publish over existing version` | bump `version` in that package |
| Provenance errors in CI | npm too old in CI (workflow upgrades npm), or trusted publisher missing |
| Wrong repo in provenance | publisher config must match `oakimov/claude-code-router` |
| Token auth used instead of OIDC | `actions/setup-node` injects `NODE_AUTH_TOKEN`; workflow clears it before publish |
| `npm@latest` engine error on CI | use current Node 22.x (workflow uses `node-version: "22"`) |
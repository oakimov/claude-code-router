# Core Tests

This directory contains parity and unit tests for the core transformation logic.

## Running Tests

From the repo root, the workspace runner executes every test in every package:

```bash
pnpm test              # all packages
pnpm test core         # only packages/core
```

### Execute a Single Test

These tests operate directly on TypeScript source files without requiring a
full build, so we use `tsx` for execution. Run from `packages/core`, not the
repo root — the `@/*` path alias resolves against this package's
`tsconfig.json`:

```bash
cd packages/core
npx tsx src/tests/gemini.parity.ts
```

Each test is a standalone entry point that exits non-zero on failure; there is
no test framework. A new file dropped in this directory is picked up by
`pnpm test` automatically.

### How Parity Testing Works
1. **Baseline Generation**: The script compares current output against "Golden Files" in `__golden__/`. If no golden files exist, it can be configured to generate them.
2. **Deep Equality**: Every field in the resulting JSON and every chunk in the SSE stream is checked for 100% identity.
3. **Zero Regression**: Any difference in output is treated as a failure, ensuring that refactors do not change the API payload.

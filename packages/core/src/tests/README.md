# Core Tests

This directory contains parity and unit tests for the core transformation logic.

## Running Tests

Since these tests operate directly on TypeScript source files without requiring a full build, we use `tsx` for execution.

### Execute Parity Tests
To run the Gemini parity suite:

```bash
npx tsx packages/core/src/tests/gemini.parity.ts
```

### How Parity Testing Works
1. **Baseline Generation**: The script compares current output against "Golden Files" in `__golden__/`. If no golden files exist, it can be configured to generate them.
2. **Deep Equality**: Every field in the resulting JSON and every chunk in the SSE stream is checked for 100% identity.
3. **Zero Regression**: Any difference in output is treated as a failure, ensuring that refactors do not change the API payload.

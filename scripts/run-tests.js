#!/usr/bin/env node

/**
 * Workspace test runner.
 *
 * Every test file under `packages/<pkg>/src/tests/` is a standalone tsx entry
 * point that exits non-zero on failure. This runner discovers them, executes
 * each in its own process, and reports a summary.
 *
 * Tests run from their owning package directory because core resolves the
 * `@/*` path alias from `packages/core/tsconfig.json`; running from the repo
 * root breaks that resolution.
 *
 * Usage:
 *   node scripts/run-tests.js                  # all hermetic tests (CI default)
 *   node scripts/run-tests.js --chrome-bridge  # also run Chrome bridge tests
 *   node scripts/run-tests.js core cli         # only the named packages
 */

const { spawnSync } = require('child_process');
const { readdirSync, existsSync } = require('fs');
const path = require('path');

const repoRoot = path.resolve(__dirname, '..');

// Packages that carry a src/tests directory, in execution order.
const PACKAGES = ['core', 'shared', 'server', 'cli'];

/**
 * Tests this runner cannot satisfy on its own. Each entry names the flag that
 * opts it in and why it is excluded. These never run by default, and never in
 * CI — the required services are operator-provided.
 */
const OPT_IN_ONLY = {
  'cli/ccr-anthropic-flow.test.ts': {
    flag: '--chrome-bridge',
    reason: 'needs a running CCR server plus `ccr chrome-bridge` on a Chrome with Gemini Nano',
  },
};

const PER_TEST_TIMEOUT_MS = 120_000;

const args = process.argv.slice(2);
const enabledFlags = new Set(args.filter((a) => a.startsWith('--')));
const filters = args.filter((a) => !a.startsWith('--'));

function discover(pkg) {
  const dir = path.join(repoRoot, 'packages', pkg, 'src', 'tests');
  if (!existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => f.endsWith('.ts'))
    .sort()
    .map((file) => ({ pkg, file, id: `${pkg}/${file}` }));
}

const discovered = PACKAGES
  .filter((p) => !filters.length || filters.includes(p))
  .flatMap(discover);

const skipped = [];
const tests = discovered.filter((t) => {
  const gate = OPT_IN_ONLY[t.id];
  if (gate && !enabledFlags.has(gate.flag)) {
    skipped.push({ id: t.id, ...gate });
    return false;
  }
  return true;
});

if (!tests.length) {
  console.error('No tests matched.');
  process.exit(1);
}

const failures = [];
const started = process.hrtime.bigint();

for (const test of tests) {
  const result = spawnSync('npx', ['tsx', path.join('src', 'tests', test.file)], {
    cwd: path.join(repoRoot, 'packages', test.pkg),
    encoding: 'utf8',
    timeout: PER_TEST_TIMEOUT_MS,
  });

  const timedOut = result.error && result.error.code === 'ETIMEDOUT';
  if (!timedOut && result.status === 0) {
    console.log(`  ok    ${test.id}`);
    continue;
  }

  const reason = timedOut
    ? `timed out after ${PER_TEST_TIMEOUT_MS / 1000}s`
    : `exit ${result.status !== null ? result.status : 'signal ' + result.signal}`;
  console.log(`  FAIL  ${test.id}  (${reason})`);
  failures.push({
    id: test.id,
    reason,
    output: `${result.stdout || ''}${result.stderr || ''}`,
  });
}

const elapsedSec = Number(process.hrtime.bigint() - started) / 1e9;

for (const s of skipped) {
  console.log(`  skip  ${s.id}  (${s.reason}; run with ${s.flag})`);
}

if (failures.length) {
  console.log(`\n${'='.repeat(72)}`);
  for (const f of failures) {
    console.log(`\n--- ${f.id} (${f.reason}) ---`);
    console.log(f.output.trimEnd() || '(no output)');
  }
  console.log('='.repeat(72));
}

console.log(
  `\n${tests.length - failures.length}/${tests.length} passed` +
  (skipped.length ? `, ${skipped.length} skipped` : '') +
  ` in ${elapsedSec.toFixed(1)}s`
);

process.exit(failures.length ? 1 : 0);

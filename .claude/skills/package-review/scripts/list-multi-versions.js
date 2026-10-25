#!/usr/bin/env bun
/**
 * List package names that resolve to more than one version in bun.lock.
 * Usage (repo root): bun .claude/skills/package-review/scripts/list-multi-versions.js
 */
import { loadBunLock, resolvedPackages } from "./parse-bun-lock.js";

const versions = new Map();
for (const { name, version } of resolvedPackages(loadBunLock())) {
  if (!versions.has(name)) versions.set(name, new Set());
  versions.get(name).add(version);
}

const multi = [...versions.entries()]
  .filter(([, vs]) => vs.size > 1)
  .sort((a, b) => b[1].size - a[1].size || a[0].localeCompare(b[0]));

for (const [name, vs] of multi) {
  console.log(`${name}: ${[...vs].sort().join(", ")}`);
}
console.error(`# ${multi.length} packages with multiple versions`);

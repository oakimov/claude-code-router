#!/usr/bin/env bun
/**
 * Report root package.json overrides whose parent no longer appears in bun.lock
 * (stale exact-edge selectors). Bun does not warn about unused overrides.
 *
 * Usage (repo root): bun .claude/skills/package-review/scripts/check-overrides.js
 */
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { loadBunLock, resolvedPackages } from "./parse-bun-lock.js";

const root = process.cwd();
const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const resolved = resolvedPackages(loadBunLock(root));
const byName = new Map();
for (const row of resolved) {
  if (!byName.has(row.name)) byName.set(row.name, new Set());
  byName.get(row.name).add(row.version);
}

const overrides = pkg.overrides ?? {};
let stale = 0;
let majorScoped = 0;
let ok = 0;

for (const [selector, target] of Object.entries(overrides)) {
  const gt = selector.indexOf(">");
  if (gt < 0) {
    console.log(`BARE\t${selector}\t→\t${target}\t(use parent@version>child)`);
    stale++;
    continue;
  }
  const parentSpec = selector.slice(0, gt);
  const at = parentSpec.lastIndexOf("@");
  if (at <= 0) {
    console.log(`BAD\t${selector}`);
    stale++;
    continue;
  }
  const parentName = parentSpec.slice(0, at);
  const parentVer = parentSpec.slice(at + 1);
  const versions = byName.get(parentName);
  if (!versions || versions.size === 0) {
    console.log(`STALE\t${selector}\t→\t${target}\t(parent absent)`);
    stale++;
    continue;
  }
  const exact = parentVer.includes(".");
  const matched = exact
    ? versions.has(parentVer)
    : [...versions].some((v) => v === parentVer || v.startsWith(parentVer + "."));
  if (!matched) {
    console.log(
      `STALE\t${selector}\t→\t${target}\t(have ${[...versions].sort().join(", ")})`
    );
    stale++;
  } else if (!exact) {
    console.log(`MAJOR\t${selector}\t→\t${target}`);
    majorScoped++;
  } else {
    ok++;
  }
}

console.error(
  `# overrides: ${ok} exact-ok, ${majorScoped} major-scoped, ${stale} stale/bare`
);
if (stale > 0) process.exitCode = 1;

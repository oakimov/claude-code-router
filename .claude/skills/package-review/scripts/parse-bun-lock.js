#!/usr/bin/env bun
/** Shared Bun lockfile (JSONC) parser for package-review scripts. */
import { readFileSync } from "node:fs";

/** Bun's lockfile is JSONC: unquoted keys, trailing commas, comments. */
export function parseBunLockText(src) {
  let i = 0;
  let out = "";
  while (i < src.length) {
    const c = src[i];
    if (c === "\"" || c === "'") {
      const q = c;
      out += c;
      i++;
      while (i < src.length) {
        const ch = src[i];
        out += ch;
        if (ch === "\\") {
          i++;
          if (i < src.length) out += src[i];
          i++;
          continue;
        }
        if (ch === q) {
          i++;
          break;
        }
        i++;
      }
      continue;
    }
    if (c === "/" && src[i + 1] === "/") {
      while (i < src.length && src[i] !== "\n") i++;
      continue;
    }
    if (c === "/" && src[i + 1] === "*") {
      i += 2;
      while (i < src.length && !(src[i] === "*" && src[i + 1] === "/")) i++;
      i += 2;
      continue;
    }
    out += c;
    i++;
  }
  out = out.replace(
    /([{\[,]\s*)([A-Za-z0-9_@/.\-+]+)(\s*:)/g,
    (_m, a, k, colon) => a + JSON.stringify(k) + colon
  );
  out = out.replace(/,(\s*[}\]])/g, "$1");
  return JSON.parse(out);
}

export function loadBunLock(cwd = process.cwd()) {
  const text = readFileSync(`${cwd}/bun.lock`, "utf8");
  return parseBunLockText(text);
}

/**
 * Resolve each packages[] entry to { name, version, lockKey }.
 * Bun stores: "lockKey": ["name@version", …]
 * Peer/nested installs use slashy lockKeys; identity is always the tuple[0].
 */
export function resolvedPackages(lock) {
  const out = [];
  for (const [lockKey, entry] of Object.entries(lock.packages ?? {})) {
    const id = Array.isArray(entry) ? entry[0] : null;
    if (typeof id !== "string" || !id.includes("@")) continue;
    const at = id.lastIndexOf("@");
    if (at <= 0) continue;
    out.push({
      lockKey,
      name: id.slice(0, at),
      version: id.slice(at + 1),
    });
  }
  return out;
}

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = join(import.meta.dirname, "..", "..", "..", "ui", "src");
const files = [
  "lib/api.ts",
  "components/Login.tsx",
  "components/ConfigProvider.tsx",
  "App.tsx",
];

for (const relativePath of files) {
  const source = readFileSync(join(root, relativePath), "utf8");
  assert.doesNotMatch(
    source,
    /(?:localStorage|sessionStorage).*apiKey|apiKey.*(?:localStorage|sessionStorage)/i,
    `${relativePath} must not persist the router API key in web storage`
  );
}

const apiSource = readFileSync(join(root, "lib/api.ts"), "utf8");
assert.match(apiSource, /post<void>\('\/auth\/login', \{ apiKey \}\)/);
assert.match(apiSource, /throw new Error\('Unauthorized'\)/);
assert.doesNotMatch(apiSource, /new Promise\(\(\) => \{\}\)/);
assert.doesNotMatch(apiSource, /tempApiKey|X-Temp-API-Key/);

console.log("UI API-key storage tests passed.");

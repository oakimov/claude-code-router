/** Build aliases preserve dotted basenames and strip only source extensions. */
import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { build } from "esbuild";

async function main() {
  const { pathAliasPlugin } = await import(
    new URL("../../scripts/esbuild-plugin-path-alias.ts", import.meta.url).href
  );
  const root = await mkdtemp(join(tmpdir(), "ccr-alias-build-"));
  try {
    const src = join(root, "src");
    await mkdir(join(src, "folder.name"), { recursive: true });
    await writeFile(join(src, "openai.responses.util.ts"), 'export const dotted = "dotted";');
    await writeFile(join(src, "other.model.ts"), 'export const explicitJs = "explicit-js";');
    await writeFile(join(src, "plain.ts"), 'export const explicitTs = "explicit-ts";');
    await writeFile(join(src, "folder.name", "index.ts"), 'export const directory = "directory";');
    const result = await build({
      stdin: {
        contents: [
          'export { dotted } from "@/openai.responses.util";',
          'export { explicitJs } from "@/other.model.js";',
          'export { explicitTs } from "@/plain.ts";',
          'export { directory } from "@/folder.name";',
        ].join("\n"),
        resolveDir: root,
        loader: "ts",
      },
      bundle: true,
      write: false,
      format: "esm",
      platform: "node",
      plugins: [pathAliasPlugin({ alias: { "@/*": "src/*" }, baseUrl: root })],
      logLevel: "silent",
    });
    assert.deepEqual(result.errors, []);
    assert.deepEqual(result.warnings, []);
    const compiled = await import(`data:text/javascript;base64,${Buffer.from(result.outputFiles![0].text).toString("base64")}`);
    assert.equal(compiled.dotted, "dotted");
    assert.equal(compiled.explicitJs, "explicit-js");
    assert.equal(compiled.explicitTs, "explicit-ts");
    assert.equal(compiled.directory, "directory");
    console.log("path-alias.build: PASS");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

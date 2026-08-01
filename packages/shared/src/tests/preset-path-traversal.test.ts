import assert from "node:assert/strict";
import { mkdtempSync, rmSync, existsSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import AdmZip from "adm-zip";
import { extractPreset, getPresetDir, HOME_DIR } from "../index";

function expectGetPresetDirRejects(name: string, label: string): void {
  assert.throws(
    () => getPresetDir(name),
    (err: unknown) =>
      err instanceof Error &&
      (err.message.includes("path traversal") ||
        err.message.includes("cannot be empty") ||
        err.message.includes("absolute path")),
    label
  );
}

function testPresetNameTraversal(): void {
  const safe = getPresetDir("my-preset");
  assert.ok(safe.startsWith(join(HOME_DIR, "presets")));
  assert.equal(safe, join(HOME_DIR, "presets", "my-preset"));

  expectGetPresetDirRejects("", "empty name");
  expectGetPresetDirRejects("../escape", "parent traversal");
  expectGetPresetDirRejects("foo/bar", "slash in name");
  expectGetPresetDirRejects("foo" + String.fromCharCode(92) + "bar", "backslash in name");
  expectGetPresetDirRejects("/tmp/evil", "absolute path");
  expectGetPresetDirRejects("..", "dot-dot alone");
}

async function testZipSlipRejected(): Promise<void> {
  const tempRoot = mkdtempSync(join(tmpdir(), "ccr-preset-zipslip-"));
  const zipPath = join(tempRoot, "evil.zip");
  const targetDir = join(tempRoot, "target");
  const sentinelOutside = join(tempRoot, "outside-sentinel.txt");

  try {
    const zip = new AdmZip();
    // AdmZip.addFile() strips "../" from the name. Mutate entryName after add
    // so the on-disk ZIP still contains a classic zip-slip path.
    zip.addFile("outside-sentinel.txt", Buffer.from("pwned"));
    const slip = zip.getEntries().find((e) => e.entryName === "outside-sentinel.txt");
    assert.ok(slip, "expected slip entry");
    slip.entryName = "../../outside-sentinel.txt";
    (slip as any).header.fileNameLength = Buffer.byteLength("../../outside-sentinel.txt");
    zip.addFile(
      "manifest.json",
      Buffer.from(JSON.stringify({ name: "evil", version: "1.0.0" }))
    );
    zip.writeZip(zipPath);

    const written = new AdmZip(zipPath).getEntries().map((e) => e.entryName);
    assert.ok(
      written.includes("../../outside-sentinel.txt"),
      `zip must retain slip path, got ${JSON.stringify(written)}`
    );

    await assert.rejects(
      () => extractPreset(zipPath, targetDir),
      (err: unknown) =>
        err instanceof Error && err.message.includes("Path traversal detected"),
      "zip-slip entry must be rejected"
    );

    assert.equal(
      existsSync(sentinelOutside),
      false,
      "zip-slip must not write outside the target directory"
    );
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }
}

async function testSafeExtractStillWorks(): Promise<void> {
  const tempRoot = mkdtempSync(join(tmpdir(), "ccr-preset-safe-"));
  const zipPath = join(tempRoot, "safe.zip");
  const targetDir = join(tempRoot, "target");

  try {
    const zip = new AdmZip();
    zip.addFile(
      "manifest.json",
      Buffer.from(JSON.stringify({ name: "safe", version: "1.0.0" }))
    );
    zip.addFile("readme.md", Buffer.from("# safe preset\n"));
    zip.writeZip(zipPath);

    await extractPreset(zipPath, targetDir);

    const manifest = JSON.parse(
      readFileSync(join(targetDir, "manifest.json"), "utf8")
    );
    assert.equal(manifest.name, "safe");
    assert.equal(existsSync(join(targetDir, "readme.md")), true);
  } finally {
    rmSync(tempRoot, { recursive: true, force: true });
  }
}

async function main(): Promise<void> {
  testPresetNameTraversal();
  await testZipSlipRejected();
  await testSafeExtractStillWorks();
  console.log("preset-path-traversal tests passed.");
}

main().catch((error) => {
  process.stderr.write(
    `${error instanceof Error ? error.stack || error.message : String(error)}\n`
  );
  process.exit(1);
});

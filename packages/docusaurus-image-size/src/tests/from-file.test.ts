import assert from "node:assert/strict";
import { createRequire } from "node:module";
import path from "node:path";

const require = createRequire(import.meta.url);
const { imageSizeFromFile, setConcurrency } = require("../../fromFile.cjs") as {
  imageSizeFromFile: (filePath: string) => Promise<{
    width: number;
    height: number;
    type?: string;
  }>;
  setConcurrency: (value: number) => void;
};

const repoRoot = path.resolve(process.cwd(), "../..");
const pngPath = path.join(
  repoRoot,
  "docs/static/blog-images/claude-code-router-img.png"
);
const jpegPath = path.join(
  repoRoot,
  "docs/static/blog-images/alipay.jpg"
);

async function main() {
  setConcurrency(1);
  const [png, jpeg] = await Promise.all([
    imageSizeFromFile(pngPath),
    imageSizeFromFile(jpegPath),
  ]);

  assert.ok(png.width > 0 && png.height > 0);
  assert.equal(png.type, "png");
  assert.ok(jpeg.width > 0 && jpeg.height > 0);
  assert.equal(jpeg.type, "jpg");
  assert.throws(() => setConcurrency(0), /positive integer/);

  console.log("Docusaurus image-size adapter tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

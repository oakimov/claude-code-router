import * as esbuild from "esbuild";
import * as path from "path";
import * as fs from "fs";
import { fileURLToPath } from "url";
import { execSync } from "child_process";
import { pathAliasPlugin } from "./esbuild-plugin-path-alias";

const watch = process.argv.includes("--watch");

// Get the absolute path to the src directory (ES module compatible)
// @ts-ignore
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const baseUrl = path.resolve(__dirname, "..");

const baseConfig: esbuild.BuildOptions = {
  entryPoints: ["src/server.ts"],
  bundle: true,
  minify: true,
  sourcemap: true,
  platform: "node",
  // Match the package engines floor (>=22.19.0) so esbuild does not downlevel
  // syntax for runtimes this package does not support.
  target: "node22",
  plugins: [
    // Add path alias plugin to resolve @/ imports
    pathAliasPlugin({
      alias: {
        "@/*": "src/*",
      },
      baseUrl,
    }),
  ],
  external: [
    "fastify",
    "dotenv",
    "@fastify/cors",
    "undici",
    "tiktoken",
    "@caeliq/ccr-shared",
    "lru-cache",
    // Native/platform runtime — must resolve from node_modules at runtime.
    "@cursor/sdk",
    "@cursor/sdk/*",
  ],
};

// Emit real .d.ts files via tsc, fix up @/ aliases, and expose them through
// a dist/index.d.ts barrel matching the "types" field in package.json.
function generateTypeDeclarations() {
  console.log("Generating type declarations...");
  const dtsRoot = path.join(baseUrl, "dist");
  execSync(
    "tsc --project tsconfig.json --declaration --emitDeclarationOnly --outDir dist",
    { cwd: baseUrl, stdio: "inherit" }
  );
  replacePathAliases(dtsRoot);
  const barrel = [
    'export * from "./server";',
    'export { default } from "./server";',
    "",
  ].join("\n");
  fs.writeFileSync(path.join(dtsRoot, "index.d.ts"), barrel);
}

// Replace @/ paths with relative paths in .d.ts files. dtsRoot is the root
// the declarations were emitted into (mirrors src/'s layout), so aliases
// resolve relative to it rather than to the original src/ tree.
function replacePathAliases(dtsRoot: string, dir = dtsRoot) {
  const files = fs.readdirSync(dir);

  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);

    if (stat.isDirectory()) {
      replacePathAliases(dtsRoot, fullPath);
    } else if (file.endsWith(".d.ts")) {
      const content = fs.readFileSync(fullPath, "utf-8");

      // Replace @/ imports with relative paths
      const replaced = content.replace(/from\s+["']@(\/[^"']+)["']/g, (_, importPath) => {
        const absolutePath = path.resolve(dtsRoot, importPath.slice(1));
        const currentDir = path.dirname(fullPath);
        let relativePath = path.relative(currentDir, absolutePath).split(path.sep).join("/");
        if (!relativePath.startsWith(".")) relativePath = `./${relativePath}`;
        return `from "${relativePath}"`;
      });

      if (replaced !== content) fs.writeFileSync(fullPath, replaced);
    }
  }
}

const cjsConfig: esbuild.BuildOptions = {
  ...baseConfig,
  outdir: "dist/cjs",
  format: "cjs",
  outExtension: { ".js": ".cjs" },
};

const esmConfig: esbuild.BuildOptions = {
  ...baseConfig,
  outdir: "dist/esm",
  format: "esm",
  outExtension: { ".js": ".mjs" },
};

async function build() {
  console.log("Building CJS and ESM versions...");

  // First, generate type declarations
  generateTypeDeclarations();

  const cjsCtx = await esbuild.context(cjsConfig);
  const esmCtx = await esbuild.context(esmConfig);

  if (watch) {
    console.log("Watching for changes...");
    await Promise.all([
      cjsCtx.watch(),
      esmCtx.watch(),
    ]);
  } else {
    await Promise.all([
      cjsCtx.rebuild(),
      esmCtx.rebuild(),
    ]);

    await Promise.all([
      cjsCtx.dispose(),
      esmCtx.dispose(),
    ]);

    console.log("✅ Build completed successfully!");
    console.log("  - CJS: dist/cjs/server.cjs");
    console.log("  - ESM: dist/esm/server.mjs");
    console.log("  - Types: dist/*.d.ts");
  }
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
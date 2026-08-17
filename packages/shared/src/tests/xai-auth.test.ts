import assert from "node:assert/strict";
import { resolveXaiApiKey } from "../xai-auth";

function main(): void {
  // Literal key.
  assert.equal(resolveXaiApiKey("xai-abc123"), "xai-abc123");
  assert.equal(resolveXaiApiKey("  xai-abc123  "), "xai-abc123");

  // Non-xai literal values never resolve, even with bare-env-name allowed.
  assert.equal(resolveXaiApiKey("no-key"), undefined);
  assert.equal(resolveXaiApiKey("no-key", { allowBareEnvName: true }), undefined);
  assert.equal(resolveXaiApiKey(""), undefined);
  assert.equal(resolveXaiApiKey(undefined), undefined);
  assert.equal(resolveXaiApiKey(123 as unknown as string), undefined);

  // $VAR / ${VAR} references are always resolved regardless of allowBareEnvName.
  const env = { XAI_KEY: "xai-fromenv", OTHER: "not-xai-prefixed" };
  assert.equal(resolveXaiApiKey("$XAI_KEY", { env }), "xai-fromenv");
  assert.equal(resolveXaiApiKey("${XAI_KEY}", { env }), "xai-fromenv");
  assert.equal(resolveXaiApiKey("$OTHER", { env }), undefined);
  assert.equal(resolveXaiApiKey("$MISSING", { env }), undefined);

  // Bare env name only resolves when explicitly opted in.
  const bareEnv = { XAI_API_KEY: "xai-bare" };
  assert.equal(resolveXaiApiKey("XAI_API_KEY", { env: bareEnv }), undefined);
  assert.equal(
    resolveXaiApiKey("XAI_API_KEY", { env: bareEnv, allowBareEnvName: true }),
    "xai-bare"
  );

  // Bare name that isn't a valid identifier never resolves even when opted in.
  assert.equal(
    resolveXaiApiKey("not a var", { env: bareEnv, allowBareEnvName: true }),
    undefined
  );
}

main();

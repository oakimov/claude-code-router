import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resetSessionRegistryForTests } from "../../session-registry";

process.env.CCR_SESSION_REGISTRY_DIR = mkdtempSync(
  join(tmpdir(), "ccr-cursor-")
);
resetSessionRegistryForTests();

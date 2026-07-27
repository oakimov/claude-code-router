import { hostPathRule, type HostEnvironment } from "./host-env";
import { CURSOR_SDK_WORKSPACES_ROOT } from "./shared";

/** Depth/breadth caps so a hostile or huge tool payload cannot stall the walk. */
const MAX_DEPTH = 6;
const MAX_NODES = 512;
const MAX_REPORTED = 4;

export type ScratchPathHit = {
  /** Dotted path to the offending argument, e.g. `edits.0.file_path`. */
  argPath: string;
  /** The offending value, truncated. */
  value: string;
};

function truncate(value: string): string {
  return value.length > 200 ? `${value.slice(0, 200)}…` : value;
}

/**
 * True when the model is aiming a host tool at its own scratch container.
 *
 * Matches anywhere in the string, not just the prefix: shell commands arrive as
 * one argument (`cd <workspace> && ls`), so a prefix test would miss them.
 */
function isScratchReference(value: string, workspaceDir: string): boolean {
  if (value.length < 8) return false;
  return value.includes(workspaceDir) || value.includes(CURSOR_SDK_WORKSPACES_ROOT);
}

/**
 * Detect host tool arguments pointing at the sandbox workspace.
 *
 * This is the observable symptom of the model believing it is confined to its
 * own cwd. Claude Code would execute the call on the host, where that path does
 * not exist (or, worse, exists as an unrelated directory).
 */
export function findScratchPaths(
  args: unknown,
  workspaceDir: string
): ScratchPathHit[] {
  const hits: ScratchPathHit[] = [];
  let nodes = 0;

  const walk = (value: unknown, path: string, depth: number): void => {
    if (hits.length >= MAX_REPORTED || nodes >= MAX_NODES || depth > MAX_DEPTH) {
      return;
    }
    nodes++;

    if (typeof value === "string") {
      if (isScratchReference(value, workspaceDir)) {
        hits.push({ argPath: path || "(argument)", value: truncate(value) });
      }
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item, index) => walk(item, `${path}.${index}`, depth + 1));
      return;
    }
    if (value && typeof value === "object") {
      for (const [key, item] of Object.entries(value)) {
        walk(item, path ? `${path}.${key}` : key, depth + 1);
      }
    }
  };

  walk(args, "", 0);
  return hits;
}

/**
 * Guard against a pathological case: if the user's project genuinely lives
 * under the scratch root, every path would look like a violation.
 */
export function scratchDetectionApplies(hostEnv: HostEnvironment): boolean {
  if (!hostEnv.projectRoot) return true;
  return !hostEnv.projectRoot.startsWith(CURSOR_SDK_WORKSPACES_ROOT);
}

/**
 * Tool result returned in place of executing the call. Deterministic correction
 * where the prompt guidance is only preventive — the model gets the topology
 * again at the exact moment it acted on the wrong belief.
 */
export function buildScratchPathCorrection(
  hits: ScratchPathHit[],
  toolName: string,
  workspaceDir: string,
  hostEnv: HostEnvironment
): string {
  const offenders = hits
    .map((hit) => `  - ${hit.argPath}: ${hit.value}`)
    .join("\n");

  return [
    `Error: this ${toolName} call was not executed because it points at your local scratch container, not at the user's machine.`,
    "Offending arguments:",
    offenders,
    `${workspaceDir} is disposable scratch space on the machine running the model bridge. It is empty and contains none of the user's files.`,
    "Host tools execute on the user's machine, where that path does not exist.",
    hostPathRule(hostEnv),
    "Retry the call with a host path.",
  ].join("\n");
}

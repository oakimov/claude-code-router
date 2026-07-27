import type { UnifiedChatRequest } from "@/types/llm";
import { contentToText, hashSessionFingerprint } from "./shared";

/**
 * Facts about the machine where host tools actually execute.
 *
 * CCR (and therefore the Cursor SDK agent) may run in a container with no
 * access to the user's filesystem, while Claude Code executes every tool call
 * on the user's real machine. Cursor builds its harness prompt server-side from
 * the SDK's workspace root, so the model is told — at system level — that it
 * lives in the scratch workspace. These facts are the counter-evidence: they are
 * lifted verbatim from the host's own environment block.
 */
export type HostEnvironment = {
  /** Absolute path of the host project root, when the host advertised one. */
  projectRoot?: string;
  /** Extra roots the host declared as in-scope. */
  additionalRoots: string[];
  platform?: string;
  osVersion?: string;
  /** Every `Key: value` pair discovered, in discovery order, verbatim. */
  facts: Array<{ key: string; value: string }>;
  /** True when anything at all was discovered. */
  known: boolean;
  /** Stable id of the resolved environment; changes only when the facts change. */
  fingerprint: string;
};

/** Nothing discovered — guidance falls back to conservative path wording. */
export const EMPTY_HOST_ENVIRONMENT: HostEnvironment = {
  additionalRoots: [],
  facts: [],
  known: false,
  fingerprint: "none",
};

const ENV_BLOCK_PATTERN =
  /<(env|environment|system-info|system_info)>([\s\S]*?)<\/\1>/gi;

/** `Key: value` on a single line. Keys stay short so prose is not slurped. */
const KEY_VALUE_PATTERN = /^\s*[-*]?\s*([A-Za-z][A-Za-z0-9 _/'-]{2,48}?)\s*:\s*(\S.*?)\s*$/;

/**
 * Keys worth scanning for outside a delimited env block. Kept broad enough to
 * survive host prompt rewording, narrow enough not to match arbitrary text.
 */
const LOOSE_KEY_PATTERN =
  /working director|workspace root|project root|platform|os version|operating system|git repo/i;

/** Canonical field aliases, most specific first. */
const PROJECT_ROOT_KEYS = [
  /^primary working director/i,
  /^current working director/i,
  /^working director/i,
  /^project root/i,
  /^workspace root/i,
  /^cwd$/i,
];
const ADDITIONAL_ROOT_KEYS = [
  /^additional working director/i,
  /^additional director/i,
  /^extra working director/i,
];
const PLATFORM_KEYS = [/^platform$/i, /^operating system$/i, /^os$/i];
const OS_VERSION_KEYS = [/^os version$/i, /^system version$/i];

/** Keys already rendered as dedicated lines by `describeHostEnvironment`. */
const CANONICAL_KEYS = [
  ...PROJECT_ROOT_KEYS,
  ...ADDITIONAL_ROOT_KEYS,
  ...PLATFORM_KEYS,
  ...OS_VERSION_KEYS,
];

const MAX_SCANNED_CHARS = 200_000;
const MAX_FACTS = 16;
const MAX_VALUE_CHARS = 400;

/** Text the host may hide its environment block in: system turns + first user turn. */
function candidateText(request: UnifiedChatRequest): string {
  const parts: string[] = [];
  let budget = MAX_SCANNED_CHARS;
  let sawUser = false;

  for (const msg of request.messages || []) {
    if (msg.role !== "system" && msg.role !== "user") continue;
    if (msg.role === "user" && sawUser) break;
    if (msg.role === "user") sawUser = true;

    const text = contentToText(msg.content);
    if (!text) continue;
    parts.push(text.slice(0, budget));
    budget -= Math.min(text.length, budget);
    if (budget <= 0) break;
  }

  return parts.join("\n");
}

function collectFacts(text: string): Array<{ key: string; value: string }> {
  const facts: Array<{ key: string; value: string }> = [];
  const seen = new Set<string>();

  const addLine = (line: string) => {
    if (facts.length >= MAX_FACTS) return;
    const match = line.match(KEY_VALUE_PATTERN);
    if (!match) return;
    const key = match[1].trim();
    const value = match[2].trim().slice(0, MAX_VALUE_CHARS);
    if (!key || !value) return;
    const dedupeKey = key.toLowerCase();
    if (seen.has(dedupeKey)) return;
    seen.add(dedupeKey);
    facts.push({ key, value });
  };

  // Preferred source: an explicit environment block.
  let blockFound = false;
  ENV_BLOCK_PATTERN.lastIndex = 0;
  let block: RegExpExecArray | null;
  while ((block = ENV_BLOCK_PATTERN.exec(text))) {
    blockFound = true;
    for (const line of block[2].split(/\r?\n/)) addLine(line);
  }

  // Fallback: the host reworded or dropped the block wrapper.
  if (!blockFound) {
    for (const line of text.split(/\r?\n/)) {
      if (!LOOSE_KEY_PATTERN.test(line)) continue;
      addLine(line);
    }
  }

  return facts;
}

function pick(
  facts: Array<{ key: string; value: string }>,
  patterns: RegExp[]
): string | undefined {
  for (const pattern of patterns) {
    const hit = facts.find((fact) => pattern.test(fact.key));
    if (hit) return hit.value;
  }
  return undefined;
}

function looksAbsolute(value: string): boolean {
  return /^(?:\/|[A-Za-z]:[\\/]|\\\\)/.test(value);
}

function splitRoots(value?: string): string[] {
  if (!value) return [];
  return value
    .split(/[,\n]/)
    .map((part) => part.trim())
    .filter((part) => looksAbsolute(part));
}

export function extractHostEnvironment(
  request: UnifiedChatRequest
): HostEnvironment {
  const facts = collectFacts(candidateText(request));

  const rawRoot = pick(facts, PROJECT_ROOT_KEYS);
  const projectRoot = rawRoot && looksAbsolute(rawRoot) ? rawRoot : undefined;
  const additionalRoots = splitRoots(pick(facts, ADDITIONAL_ROOT_KEYS)).filter(
    (root) => root !== projectRoot
  );
  const platform = pick(facts, PLATFORM_KEYS);
  const osVersion = pick(facts, OS_VERSION_KEYS);

  const known = Boolean(projectRoot || platform || osVersion || facts.length);
  if (!known) return EMPTY_HOST_ENVIRONMENT;

  return {
    projectRoot,
    additionalRoots,
    platform,
    osVersion,
    facts,
    known,
    fingerprint: hashSessionFingerprint([
      projectRoot || "",
      additionalRoots.join("|"),
      platform || "",
      osVersion || "",
      facts.map((fact) => `${fact.key}=${fact.value}`).join("|"),
    ]),
  };
}

/** Bullet lines describing the host machine, for prompt/rules injection. */
export function describeHostEnvironment(env: HostEnvironment): string[] {
  const lines: string[] = [];
  if (env.projectRoot) lines.push(`- Project root: ${env.projectRoot}`);
  for (const root of env.additionalRoots) {
    lines.push(`- Additional in-scope root: ${root}`);
  }
  if (env.platform) lines.push(`- Platform: ${env.platform}`);
  if (env.osVersion) lines.push(`- OS version: ${env.osVersion}`);

  // Pass through anything else the host reported, so new fields need no code change.
  for (const fact of env.facts) {
    if (CANONICAL_KEYS.some((pattern) => pattern.test(fact.key))) continue;
    lines.push(`- ${fact.key}: ${fact.value}`);
  }

  return lines;
}

/**
 * Path rule for the model. Positive and concrete when the host root is known,
 * conservative when it is not — never guess a root.
 */
export function hostPathRule(env: HostEnvironment): string {
  if (env.projectRoot) {
    return `Every path you pass to a host tool must be an absolute path on the host, normally under ${env.projectRoot}.`;
  }
  return "Every path you pass to a host tool must be an absolute host path taken from this conversation — never invent one and never derive one from your local filesystem.";
}

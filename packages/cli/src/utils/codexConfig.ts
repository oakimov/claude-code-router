import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  writeFileSync,
  copyFileSync,
  statSync,
  mkdtempSync,
  rmSync,
} from "node:fs";
import { homedir, tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { CONFIG_FILE, HOME_DIR } from "@caeliq/ccr-shared";
import { readConfigFileRaw } from "./index";
import {
  fetchModelsDevCatalog,
  lookupModel,
  type ModelDevInfo,
  type ModelsDevIndex,
} from "./modelsdev";

/**
 * Generate the Codex-side configuration that makes CCR-routed models appear in
 * Codex natively.
 *
 * Codex does not populate its picker from `GET /v1/models`; it reads a local
 * catalog file named by `model_catalog_json` at startup. So this writes two
 * things: the catalog itself, and a managed block in Codex's `config.toml`
 * that points Codex at CCR and at that catalog.
 */

const ROOT_BEGIN = "# BEGIN ccr-managed";
const ROOT_END = "# END ccr-managed";
const PROVIDER_BEGIN = "# BEGIN ccr-provider-managed";
const PROVIDER_END = "# END ccr-provider-managed";

const CODEX_PROVIDER_ID = "ccr";
const CATALOG_PATH = join(HOME_DIR, "codex", "models.json");

/** Used when models.dev has no context limit for a model. */
const DEFAULT_CONTEXT_WINDOW = 128_000;
/** Codex compacts before the hard limit; mirror its usual headroom. */
const AUTO_COMPACT_RATIO = 0.8;
const DEFAULT_REASONING_LEVELS = ["low", "medium", "high"];

const RESET = "\x1B[0m";
const DIM = "\x1B[2m";
const GREEN = "\x1B[32m";
const YELLOW = "\x1B[33m";
const BOLDCYAN = "\x1B[1m\x1B[36m";

export interface CodexConfigOptions {
  providers?: string[];
  models: string[];
  baseUrl?: string;
  codexHome: string;
  dryRun: boolean;
  force: boolean;
  codexProbe: boolean;
}

export function parseCodexConfigArgs(argv: string[]): CodexConfigOptions {
  const options: CodexConfigOptions = {
    models: ["*"],
    codexHome: process.env.CODEX_HOME || join(homedir(), ".codex"),
    dryRun: false,
    force: false,
    codexProbe: true,
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const next = argv[i + 1];
    const requireValue = (): string => {
      if (!next || next.startsWith("--")) {
        throw new Error(`${arg} requires a value`);
      }
      i++;
      return next;
    };
    switch (arg) {
      case "--providers":
        options.providers = splitList(requireValue());
        break;
      case "--models":
        options.models = splitList(requireValue());
        break;
      case "--base-url":
        options.baseUrl = requireValue();
        break;
      case "--codex-home":
        options.codexHome = requireValue();
        break;
      case "--dry-run":
        options.dryRun = true;
        break;
      case "--force":
        options.force = true;
        break;
      case "--no-codex-probe":
        options.codexProbe = false;
        break;
      default:
        throw new Error(`Unknown option for codex-config: ${arg}`);
    }
  }

  if (!options.models.length) options.models = ["*"];
  return options;
}

function splitList(value: string): string[] {
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

/** Wildcard matcher: `*` spans any run, `?` one character. */
export function globToRegExp(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, "\\$&");
  const expanded = escaped.replace(/\*/g, ".*").replace(/\?/g, ".");
  return new RegExp(`^${expanded}$`, "i");
}

export function matchesModel(
  patterns: string[],
  modelName: string,
  qualified: string
): boolean {
  return patterns.some((pattern) => {
    const re = globToRegExp(pattern);
    return re.test(modelName) || re.test(qualified);
  });
}

export interface CatalogModel {
  providerName: string;
  modelName: string;
  slug: string;
  info: ModelDevInfo | null;
}

export function selectModels(
  config: any,
  options: CodexConfigOptions
): CatalogModel[] {
  const providers = Array.isArray(config?.Providers) ? config.Providers : [];
  const wanted = options.providers?.map((name) => name.toLowerCase());
  const selected: CatalogModel[] = [];
  const seen = new Set<string>();

  for (const provider of providers) {
    const providerName =
      typeof provider?.name === "string" ? provider.name.trim() : "";
    if (!providerName) continue;
    if (wanted && !wanted.includes(providerName.toLowerCase())) continue;
    if (!Array.isArray(provider.models)) continue;

    for (const model of provider.models) {
      if (typeof model !== "string") continue;
      const modelName = model.trim();
      if (!modelName) continue;

      const slug = `${providerName},${modelName}`;
      if (seen.has(slug)) continue;
      if (!matchesModel(options.models, modelName, slug)) continue;

      seen.add(slug);
      selected.push({ providerName, modelName, slug, info: null });
    }
  }

  return selected;
}

/** Descriptions mirror Codex's own catalog wording for known effort tokens. */
const REASONING_DESCRIPTIONS: Record<string, string> = {
  minimal: "Fastest responses with minimal reasoning",
  low: "Fast responses with lighter reasoning",
  medium: "Balances speed and reasoning depth for everyday tasks",
  high: "Greater reasoning depth for complex problems",
  xhigh: "Extra high reasoning depth for complex problems",
  max: "Maximum reasoning depth for the hardest problems",
  ultra: "Maximum reasoning with automatic task delegation",
};

function pickDefaultEffort(levels: string[]): string {
  // Codex requires default_reasoning_level to be one of the supported levels.
  if (levels.includes("medium")) return "medium";
  if (levels.includes("low")) return "low";
  if (levels.length) return levels[0];
  return "medium";
}

/** Codex's schema wants reasoning levels as {effort, description} objects. */
function toReasoningLevels(levels: string[]): Array<{ effort: string; description: string }> {
  if (!levels.length) levels = DEFAULT_REASONING_LEVELS;
  return levels.map((effort) => ({
    effort,
    description: REASONING_DESCRIPTIONS[effort] || `Reasoning effort: ${effort}`,
  }));
}

function inputModalities(info: ModelDevInfo | null): string[] {
  if (!info) return ["text"];
  const allowed = info.modalitiesIn.filter(
    (modality) => modality === "text" || modality === "image"
  );
  if (allowed.length) {
    return allowed.includes("text") ? allowed : ["text", ...allowed];
  }
  // `attachment` is the upstream umbrella flag when modalities are absent.
  return info.attachment ? ["text", "image"] : ["text"];
}

/** Swap Codex's identity text ("based on GPT-5") for the routed model's name. */
function rewriteIdentity(text: string, name: string): string {
  if (typeof text !== "string" || !text) return text;
  const gpt5Model = "GPT-5(?:\\.\\d+)*(?:-[A-Za-z0-9.]+)?";
  return text
    .replace(
      new RegExp(`\\b(?:a coding agent|an agent) based on ${gpt5Model}\\b`, "g"),
      `an agent based on ${name}`
    )
    .replace(new RegExp(`\\bbased on ${gpt5Model}\\b`, "g"), `based on ${name}`);
}

export function buildCatalogEntry(
  model: CatalogModel,
  index: number,
  template: Record<string, any> | null
): Record<string, any> {
  const info = model.info;
  const context = info?.context || DEFAULT_CONTEXT_WINDOW;

  const effortLevels = info?.effortLevels.length
    ? info.effortLevels
    : DEFAULT_REASONING_LEVELS;
  const displayName = info?.name || model.modelName;

  const entry: Record<string, any> = {
    ...(template || {}),
    slug: model.slug,
    display_name: displayName,
    description: info?.name
      ? `${info.name} routed through Claude Code Router (${model.providerName})`
      : `${model.modelName} routed through Claude Code Router (${model.providerName})`,
    priority: 100 + index,
    visibility: "list",
    supported_in_api: true,
    context_window: context,
    max_context_window: context,
    effective_context_window_percent: 95,
    auto_compact_token_limit: Math.floor(context * AUTO_COMPACT_RATIO),
    default_reasoning_level: pickDefaultEffort(effortLevels),
    supported_reasoning_levels: toReasoningLevels(effortLevels),
    // Required by Codex's catalog schema; a synthesized entry without it is
    // rejected at startup ("missing field `shell_type`").
    shell_type: "shell_command",
    input_modalities: inputModalities(info),
    comp_hash: createHash("sha256").update(model.slug).digest("hex").slice(0, 16),
    supports_reasoning_summaries: false,
    default_reasoning_summary: "none",
    support_verbosity: false,
    default_verbosity: null,
    supports_search_tool: false,
    supports_image_detail_original: false,
    use_responses_lite: false,
    additional_speed_tiers: [],
    service_tiers: [],
    availability_nux: null,
    upgrade: null,
    // Codex only allows spawn_agent overrides between models advertising the
    // same backend version; v1 is the conservative choice.
    multi_agent_version: "v1",
  };

  // When cloning a real GPT-5 template, stop telling Codex the external model
  // is "based on GPT-5" — mirror codex-router's identity rewrite.
  if (template && typeof entry.base_instructions === "string") {
    entry.base_instructions = rewriteIdentity(entry.base_instructions, displayName);
  }
  if (
    template &&
    entry.model_messages &&
    typeof entry.model_messages === "object" &&
    typeof entry.model_messages.instructions_template === "string"
  ) {
    entry.model_messages = {
      ...entry.model_messages,
      instructions_template: rewriteIdentity(
        entry.model_messages.instructions_template,
        displayName
      ),
    };
  }

  return entry;
}

/**
 * Locate the Codex binary. It is usually NOT on PATH — the desktop app keeps
 * it at a fixed resource path, and Codex itself advertises it as CODEX_CLI_PATH.
 */
export function findCodexBinary(): string | null {
  const candidates: string[] = [];
  if (process.env.CODEX_CLI_PATH) candidates.push(process.env.CODEX_CLI_PATH);
  if (process.env.CODEX_BIN) candidates.push(process.env.CODEX_BIN);

  try {
    const which = spawnSync("which", ["codex"], { encoding: "utf8" });
    if (which.status === 0 && which.stdout.trim()) {
      candidates.push(which.stdout.trim());
    }
  } catch {
    // fall through
  }

  candidates.push("/Applications/ChatGPT.app/Contents/Resources/codex");

  for (const candidate of candidates) {
    if (candidate && existsSync(candidate)) return candidate;
  }
  return null;
}

/**
 * Clone a real entry from Codex's own catalog when the binary is available.
 * The catalog schema is undocumented and carries fields (base_instructions,
 * model_messages) a synthesized entry would omit, so a real template is the
 * higher-fidelity option when we can get one.
 */
export function captureCodexTemplate(): Record<string, any> | null {
  const binary = findCodexBinary();
  if (!binary) return null;

  let isolatedHome: string | undefined;
  try {
    // Run against an isolated CODEX_HOME so a `model_catalog_json` already
    // pointed at our (possibly stale) catalog cannot make `debug models` fail.
    isolatedHome = mkdtempSync(join(tmpdir(), "ccr-codex-capture-"));
    const result = spawnSync(binary, ["debug", "models"], {
      encoding: "utf8",
      timeout: 30_000,
      maxBuffer: 32 * 1024 * 1024,
      env: { ...process.env, CODEX_HOME: isolatedHome },
    });
    if (result.status !== 0 || !result.stdout) return null;

    const parsed = JSON.parse(result.stdout);
    const models = Array.isArray(parsed?.models) ? parsed.models : [];
    if (!models.length) return null;

    const template =
      models.find((model: any) => model?.visibility === "list") || models[0];
    if (!template || typeof template !== "object") return null;

    // Drop identity fields; buildCatalogEntry sets its own.
    const clone = { ...template };
    delete clone.slug;
    delete clone.display_name;
    delete clone.description;
    return clone;
  } catch {
    return null;
  } finally {
    if (isolatedHome) {
      rmSync(isolatedHome, { recursive: true, force: true });
    }
  }
}

function splitRootAndRest(lines: string[]): { rootEnd: number } {
  const firstTable = lines.findIndex((line) => /^\s*\[/.test(line));
  return { rootEnd: firstTable === -1 ? lines.length : firstTable };
}

function stripBlock(text: string, begin: string, end: string): string {
  const lines = text.split("\n");
  const beginIndexes = lines.flatMap((line, index) =>
    line.trim() === begin ? [index] : []
  );
  const endIndexes = lines.flatMap((line, index) =>
    line.trim() === end ? [index] : []
  );

  if (beginIndexes.length === 0 && endIndexes.length === 0) {
    return text;
  }
  if (
    beginIndexes.length !== 1 ||
    endIndexes.length !== 1 ||
    beginIndexes[0] >= endIndexes[0]
  ) {
    throw new Error(
      `Malformed managed block in config.toml: expected exactly one '${begin}' followed by '${end}'.`
    );
  }

  return [
    ...lines.slice(0, beginIndexes[0]),
    ...lines.slice(endIndexes[0] + 1),
  ].join("\n");
}

const CCR_PROVIDER_TABLE_RE =
  /^\s*\[model_providers\.(?:"ccr"|ccr)\]\s*(?:#.*)?$/;

function findTableRange(
  lines: string[],
  tableHeader: RegExp
): { start: number; end: number } | null {
  const starts = lines.flatMap((line, index) =>
    tableHeader.test(line) ? [index] : []
  );
  if (starts.length > 1) {
    throw new Error("config.toml contains duplicate [model_providers.ccr] tables.");
  }
  if (starts.length === 0) return null;
  const start = starts[0];
  const nextTable = lines.findIndex(
    (line, index) => index > start && /^\s*\[\[?/.test(line)
  );
  return { start, end: nextTable === -1 ? lines.length : nextTable };
}

/** Root-level (pre-first-table) assignment to `key`, if any. */
function findRootKey(text: string, key: string): string | null {
  const lines = text.split("\n");
  const { rootEnd } = splitRootAndRest(lines);
  const re = new RegExp(`^\\s*${key}\\s*=\\s*(.+)$`);

  for (let i = 0; i < rootEnd; i++) {
    const match = lines[i].match(re);
    if (match) return match[1].trim();
  }
  return null;
}

export interface PatchResult {
  text: string;
  changed: boolean;
}

export function patchCodexConfig(
  original: string,
  params: {
    baseUrl: string;
    catalogPath: string;
    force: boolean;
    envKey?: string;
  }
): PatchResult {
  // Remove our previous blocks first so re-running replaces rather than stacks.
  let text = stripBlock(original, ROOT_BEGIN, ROOT_END);
  text = stripBlock(text, PROVIDER_BEGIN, PROVIDER_END);

  const providerLines = text.split("\n");
  const existingProviderTable = findTableRange(
    providerLines,
    CCR_PROVIDER_TABLE_RE
  );
  if (existingProviderTable && !params.force) {
    throw new Error(
      "Refusing to replace user-owned [model_providers.ccr] in config.toml. " +
        "Remove it or re-run with --force."
    );
  }
  if (existingProviderTable) {
    text = [
      ...providerLines.slice(0, existingProviderTable.start),
      ...providerLines.slice(existingProviderTable.end),
    ].join("\n");
  }

  if (!params.force) {
    for (const key of ["model_provider", "model_catalog_json"]) {
      const existing = findRootKey(text, key);
      if (existing) {
        throw new Error(
          `Refusing to replace user-owned ${key} in config.toml (currently ${existing}). ` +
            `Remove it or re-run with --force.`
        );
      }
    }
  } else {
    // `--force` replaces the root-level keys only; a table scoped to some
    // other model_provider must never be touched.
    const lines = text.split("\n");
    const { rootEnd } = splitRootAndRest(lines);
    text = lines
      .filter(
        (line, i) =>
          !(i < rootEnd && /^\s*(model_provider|model_catalog_json)\s*=/.test(line))
      )
      .join("\n");
  }

  const rootBlock = [
    ROOT_BEGIN,
    `model_provider = ${JSON.stringify(CODEX_PROVIDER_ID)}`,
    `model_catalog_json = ${JSON.stringify(params.catalogPath)}`,
    ROOT_END,
  ];

  const providerBlock = [
    PROVIDER_BEGIN,
    `[model_providers.${CODEX_PROVIDER_ID}]`,
    `name = "Claude Code Router"`,
    `base_url = ${JSON.stringify(params.baseUrl)}`,
    ...(params.envKey
      ? [`env_key = ${JSON.stringify(params.envKey)}`]
      : []),
    `wire_api = "responses"`,
    `requires_openai_auth = false`,
    PROVIDER_END,
  ];

  // Root keys must precede the first [table] header or TOML assigns them to it.
  const lines = text.split("\n");
  const { rootEnd } = splitRootAndRest(lines);
  const head = lines.slice(0, rootEnd);
  const tail = lines.slice(rootEnd);

  const merged = [
    ...head,
    ...rootBlock,
    "",
    ...tail,
    "",
    ...providerBlock,
    "",
  ];

  const result = merged
    .join("\n")
    // Normalize only the separators adjacent to our own blocks. Collapsing
    // blank lines globally would rewrite unrelated user-owned TOML.
    .replace(/^\n+(# BEGIN ccr-managed)/, "$1")
    .replace(/\n{2,}(# BEGIN ccr-managed)/, "\n\n$1")
    .replace(/(# END ccr-managed)\n{2,}/, "$1\n\n")
    .replace(/\n{2,}(# BEGIN ccr-provider-managed)/, "\n\n$1")
    .replace(/(# END ccr-provider-managed)\n*$/, "$1\n");

  return { text: result, changed: result !== original };
}

function writeFileAtomic(
  target: string,
  content: string,
  mode = 0o600
): void {
  mkdirSync(dirname(target), { recursive: true, mode: 0o700 });
  const temp = `${target}.tmp.${process.pid}`;
  try {
    writeFileSync(temp, content, { encoding: "utf8", mode });
    renameSync(temp, target);
  } finally {
    rmSync(temp, { force: true });
  }
}

function resolvePort(value: unknown): number {
  let candidate: unknown = value ?? 3456;
  if (typeof candidate === "string") {
    const envMatch = candidate.match(/^\$(?:\{([^}]+)\}|([A-Z_][A-Z0-9_]*))$/);
    if (envMatch) {
      candidate = process.env[envMatch[1] || envMatch[2]];
    }
  }
  const port =
    typeof candidate === "number"
      ? candidate
      : typeof candidate === "string" && /^\d+$/.test(candidate)
        ? Number(candidate)
        : NaN;
  if (!Number.isInteger(port) || port < 1 || port > 65_535) {
    throw new Error(
      "Configured PORT is not a valid TCP port. Set its environment variable or pass --base-url."
    );
  }
  return port;
}

export async function codexConfigCommand(argv: string[]): Promise<void> {
  let options: CodexConfigOptions;
  try {
    options = parseCodexConfigArgs(argv);
  } catch (error) {
    console.error((error as Error).message);
    process.exit(1);
  }

  if (options.dryRun && !existsSync(CONFIG_FILE)) {
    console.error(
      `Cannot dry-run without an existing CCR config at ${CONFIG_FILE}.`
    );
    process.exit(1);
  }

  const config = await readConfigFileRaw();
  const selected = selectModels(config, options);

  if (!selected.length) {
    console.error(
      "No models matched. Check --providers / --models against your configured Providers."
    );
    process.exit(1);
  }

  const baseUrl =
    options.baseUrl || `http://127.0.0.1:${resolvePort(config?.PORT)}/v1`;
  try {
    new URL(baseUrl);
  } catch {
    console.error(`Invalid --base-url: ${baseUrl}`);
    process.exit(1);
  }

  const index: ModelsDevIndex | null = await fetchModelsDevCatalog();
  for (const model of selected) {
    model.info = lookupModel(index, model.providerName, model.modelName);
  }

  const template = options.codexProbe ? captureCodexTemplate() : null;
  if (options.codexProbe && !template) {
    console.warn(
      `${YELLOW}Warning:${RESET} could not read Codex's native catalog via \`codex debug models\`. ` +
        `Writing synthesized entries; the picker may reject them if the schema differs.`
    );
  }

  const catalog = {
    models: selected.map((model, i) => buildCatalogEntry(model, i, template)),
  };

  const configTomlPath = join(options.codexHome, "config.toml");
  const originalToml = existsSync(configTomlPath)
    ? readFileSync(configTomlPath, "utf8")
    : "";

  let patched: PatchResult;
  try {
    patched = patchCodexConfig(originalToml, {
      baseUrl,
      catalogPath: CATALOG_PATH,
      force: options.force,
      envKey: config?.APIKEY ? "CCR_API_KEY" : undefined,
    });
  } catch (error) {
    console.error((error as Error).message);
    process.exit(1);
  }

  console.log(`\n${BOLDCYAN}Models:${RESET} ${selected.length}`);
  for (const model of selected) {
    const info = model.info;
    const detail = info
      ? `${info.name || model.modelName} · ${info.context || DEFAULT_CONTEXT_WINDOW} ctx${
          info.effortLevels.length ? ` · ${info.effortLevels.join("/")}` : ""
        }`
      : `${model.modelName} ${DIM}(no models.dev metadata)${RESET}`;
    console.log(`  ${model.slug}  ${DIM}→${RESET} ${detail}`);
  }

  if (options.dryRun) {
    console.log(`\n${DIM}--dry-run: no files written.${RESET}`);
    console.log(`\n${BOLDCYAN}Catalog${RESET} ${CATALOG_PATH}`);
    console.log(`${BOLDCYAN}Codex config${RESET} ${configTomlPath}`);
    console.log(`\n${patched.text}`);
    return;
  }

  writeFileAtomic(CATALOG_PATH, `${JSON.stringify(catalog, null, 2)}\n`);
  console.log(`\n${GREEN}✓${RESET} Wrote catalog to ${CATALOG_PATH}`);

  if (patched.changed) {
    if (originalToml) {
      const backup = `${configTomlPath}.bak`;
      copyFileSync(configTomlPath, backup);
      console.log(`${DIM}Backed up config.toml to ${backup}${RESET}`);
    }
    mkdirSync(options.codexHome, { recursive: true });
    // config.toml can carry sensitive values (MCP bearer tokens); keep the
    // existing file's mode, or default to 0600 when creating a new one.
    const existingMode = originalToml
      ? statSync(configTomlPath).mode & 0o777
      : 0o600;
    writeFileAtomic(configTomlPath, patched.text, existingMode);
    console.log(`${GREEN}✓${RESET} Updated ${configTomlPath}`);
  } else {
    console.log(`${DIM}config.toml already up to date.${RESET}`);
  }

  console.log(
    `\n${YELLOW}Codex reads its model catalog only at startup —` +
      ` fully quit and reopen Codex to see these models.${RESET}`
  );
}

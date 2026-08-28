import { get_encoding } from "tiktoken";
import { sessionUsageCache, Usage } from "./cache";
import { readFile } from "fs/promises";
import { opendir, stat } from "fs/promises";
import { join, resolve, relative, basename } from "path";
import { createHash } from "crypto";
import { CLAUDE_PROJECTS_DIR, HOME_DIR } from "@caeliq/ccr-shared";
import { LRUCache } from "lru-cache";
import { ConfigService } from "../services/config";
import { TokenizerService } from "../services/tokenizer";
import { isReasoningDisabled } from "./reasoning-effort";
import {
  estimateTokenizePayloadChars,
  estimateTokensFromChars,
  countTokensInWorker,
  TOKEN_COUNT_WORKER_THRESHOLD_CHARS,
} from "./token-count-worker";
import { ensureRequestLatency, markLatency } from "./request-latency";

// Types from @anthropic-ai/sdk
interface Tool {
  name: string;
  description?: string;
  input_schema: object;
}

interface ContentBlockParam {
  type: string;
  [key: string]: any;
}

interface MessageParam {
  role: string;
  content: string | ContentBlockParam[];
}

interface MessageCreateParamsBase {
  messages?: MessageParam[];
  system?: string | any[];
  tools?: Tool[];
  [key: string]: any;
}

const enc = get_encoding("cl100k_base");

// disallowed_special: [] treats literal special-token-like substrings
// (e.g. "<|fim_prefix|>") in arbitrary message/tool content as plain
// text instead of throwing, since this is just counting tokens, not
// feeding the result back into the model.
const encodeSafe = (text: string) => enc.encode(text, undefined, []);

const CCR_SUBAGENT_MODEL_OPEN_TAG = "<CCR-SUBAGENT-MODEL>";
const CCR_SUBAGENT_MODEL_CLOSE_TAG = "</CCR-SUBAGENT-MODEL>";
export const CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX =
  "x-anthropic-billing-header";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

/** Normalize `provider/model` or `provider,model` into our `provider,model` form. */
export function normalizeModelSelector(
  value: string | undefined | null
): string | undefined {
  const trimmed = value?.trim();
  if (!trimmed) return undefined;
  if (trimmed.includes(",")) return trimmed;
  const slash = trimmed.indexOf("/");
  if (slash > 0) {
    return `${trimmed.slice(0, slash)},${trimmed.slice(slash + 1)}`;
  }
  return trimmed;
}

export function claudeCodeBillingMetadataIsSubagent(text: string): boolean {
  const prefix = `${CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX}:`;
  if (!text.startsWith(prefix)) return false;
  const payload = text.slice(prefix.length).trim();
  if (!payload) return false;

  if (payload.startsWith("{")) {
    try {
      const metadata = JSON.parse(payload) as unknown;
      return isRecord(metadata) && metadata.cc_is_subagent === true;
    } catch {
      return false;
    }
  }

  const values = payload
    .split(";")
    .map((part) => part.trim())
    .filter(Boolean)
    .flatMap((part) => {
      const separator = part.indexOf("=");
      if (
        separator < 0 ||
        part.slice(0, separator).trim() !== "cc_is_subagent"
      ) {
        return [];
      }
      return [part.slice(separator + 1).trim()];
    });
  return values.length === 1 && values[0] === "true";
}

/**
 * Strip Claude Code's billing helper system block.
 * Returns true when that block marks the request as a subagent.
 */
export function removeClaudeCodeBillingSystemHeader(body: any): boolean {
  const system = body?.system;
  if (!Array.isArray(system) || system.length === 0) return false;

  const firstBlock = system[0];
  const firstText =
    typeof firstBlock === "string"
      ? firstBlock
      : isRecord(firstBlock) &&
          firstBlock.type === "text" &&
          typeof firstBlock.text === "string"
        ? firstBlock.text
        : undefined;

  if (!firstText?.startsWith(CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX)) {
    return false;
  }

  const isSubagent = claudeCodeBillingMetadataIsSubagent(firstText);
  system.shift();
  if (system.length === 0) {
    delete body.system;
  }
  return isSubagent;
}

function extractAndRemoveSubagentModelTagFromText(
  text: string,
  replace: (next: string) => void
): string | undefined {
  const openIndex = text.indexOf(CCR_SUBAGENT_MODEL_OPEN_TAG);
  if (openIndex < 0) return undefined;
  const modelStart = openIndex + CCR_SUBAGENT_MODEL_OPEN_TAG.length;
  const closeIndex = text.indexOf(CCR_SUBAGENT_MODEL_CLOSE_TAG, modelStart);
  if (closeIndex < 0) return undefined;

  const model = normalizeModelSelector(text.slice(modelStart, closeIndex));
  if (!model) return undefined;

  replace(
    `${text.slice(0, openIndex)}${text.slice(
      closeIndex + CCR_SUBAGENT_MODEL_CLOSE_TAG.length
    )}`
  );
  return model;
}

function extractAndRemoveSubagentModelTagFromContentBlock(
  block: unknown,
  replace: (next: string) => void
): string | undefined {
  if (typeof block === "string") {
    return extractAndRemoveSubagentModelTagFromText(block, replace);
  }
  if (!isRecord(block) || typeof block.text !== "string") {
    return undefined;
  }
  return extractAndRemoveSubagentModelTagFromText(block.text, replace);
}

function extractAndRemoveSystemSubagentModelTag(
  body: any
): string | undefined {
  const system = body?.system;
  if (typeof system === "string") {
    return extractAndRemoveSubagentModelTagFromText(system, (text) => {
      body.system = text;
    });
  }
  if (!Array.isArray(system)) return undefined;

  for (let index = 0; index < system.length; index += 1) {
    const block = system[index];
    const model = extractAndRemoveSubagentModelTagFromContentBlock(
      block,
      (text) => {
        if (typeof block === "string") {
          system[index] = text;
        } else if (isRecord(block)) {
          block.text = text;
        }
      }
    );
    if (model) return model;
  }
  return undefined;
}

function extractAndRemoveSubagentModelTagFromMessage(
  message: Record<string, unknown>
): string | undefined {
  if (typeof message.content === "string") {
    return extractAndRemoveSubagentModelTagFromText(
      message.content,
      (text) => {
        message.content = text;
      }
    );
  }
  if (!Array.isArray(message.content)) return undefined;

  const content = message.content;
  for (let index = 0; index < content.length; index += 1) {
    const block = content[index];
    const model = extractAndRemoveSubagentModelTagFromContentBlock(
      block,
      (text) => {
        if (typeof block === "string") {
          content[index] = text;
        } else if (isRecord(block)) {
          block.text = text;
        }
      }
    );
    if (model) return model;
  }
  return undefined;
}

function extractAndRemoveMessageSubagentModelTag(
  body: any
): string | undefined {
  if (!Array.isArray(body?.messages)) return undefined;
  const limit = Math.min(body.messages.length, 2);
  for (let index = 0; index < limit; index += 1) {
    const message = body.messages[index];
    if (!isRecord(message) || message.role !== "user") continue;
    const model = extractAndRemoveSubagentModelTagFromMessage(message);
    if (model) return model;
  }
  return undefined;
}

/** Prefer system tag, then early user-message tag; strip whichever matched. */
export function extractAndRemoveClaudeCodeSubagentModelTag(
  body: any
): string | undefined {
  return (
    extractAndRemoveSystemSubagentModelTag(body) ||
    extractAndRemoveMessageSubagentModelTag(body)
  );
}

/** Read Claude Code's billing/subagent marker without mutating the source body. */
export function inspectClaudeCodeBillingSystemHeader(body: any): {
  present: boolean;
  isSubagent: boolean;
} {
  const system = body?.system;
  if (!Array.isArray(system) || system.length === 0) {
    return { present: false, isSubagent: false };
  }
  const firstBlock = system[0];
  const firstText =
    typeof firstBlock === "string"
      ? firstBlock
      : isRecord(firstBlock) &&
          firstBlock.type === "text" &&
          typeof firstBlock.text === "string"
        ? firstBlock.text
        : undefined;
  if (!firstText?.startsWith(CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX)) {
    return { present: false, isSubagent: false };
  }
  return {
    present: true,
    isSubagent: claudeCodeBillingMetadataIsSubagent(firstText),
  };
}


export const calculateTokenCount = (
  messages: MessageParam[],
  system: any,
  tools: Tool[]
) => {
  let tokenCount = 0;
  if (Array.isArray(messages)) {
    messages.forEach((message) => {
      if (typeof message.content === "string") {
        tokenCount += encodeSafe(message.content).length;
      } else if (Array.isArray(message.content)) {
        message.content.forEach((contentPart: any) => {
          if (contentPart.type === "text") {
            tokenCount += encodeSafe(contentPart.text).length;
          } else if (contentPart.type === "tool_use") {
            tokenCount += encodeSafe(JSON.stringify(contentPart.input)).length;
          } else if (contentPart.type === "tool_result") {
            tokenCount += encodeSafe(
              typeof contentPart.content === "string"
                ? contentPart.content
                : JSON.stringify(contentPart.content)
            ).length;
          }
        });
      }
    });
  }
  if (typeof system === "string") {
    tokenCount += encodeSafe(system).length;
  } else if (Array.isArray(system)) {
    system.forEach((item: any) => {
      if (item.type !== "text") return;
      if (typeof item.text === "string") {
        tokenCount += encodeSafe(item.text).length;
      } else if (Array.isArray(item.text)) {
        item.text.forEach((textPart: any) => {
          tokenCount += encodeSafe(textPart || "").length;
        });
      }
    });
  }
  if (tools) {
    tools.forEach((tool: Tool) => {
      if (tool.description) {
        tokenCount += encodeSafe(tool.name + tool.description).length;
      }
      if (tool.input_schema) {
        tokenCount += encodeSafe(JSON.stringify(tool.input_schema)).length;
      }
    });
  }
  return tokenCount;
};

const getProjectSpecificRouter = async (
  req: any,
  _configService: ConfigService
) => {
  // Check if there is project-specific configuration
  if (req.sessionId) {
    const project = await searchProjectBySession(req.sessionId);
    if (project) {
      const homeRoot = resolve(HOME_DIR);
      const projectDir = resolve(homeRoot, basename(project));
      const projectRel = relative(homeRoot, projectDir);
      if (!projectRel || projectRel.startsWith("..")) {
        return undefined;
      }
      const sessionName = `${basename(String(req.sessionId))}.json`;
      const projectConfigPath = join(projectDir, "config.json");
      const sessionConfigPath = join(projectDir, sessionName);

      // First try to read sessionConfig file
      try {
        const sessionConfig = JSON.parse(await readFile(sessionConfigPath, "utf8"));
        if (sessionConfig && sessionConfig.Router) {
          return sessionConfig.Router;
        }
      } catch {}
      try {
        const projectConfig = JSON.parse(await readFile(projectConfigPath, "utf8"));
        if (projectConfig && projectConfig.Router) {
          return projectConfig.Router;
        }
      } catch {}
    }
  }
  return undefined; // Return undefined to use original configuration
};

/**
 * Routing body: prefer Unified projection when present so scenario detection
 * does not interpret Responses wire shapes as Anthropic bodies.
 */
function routingBody(req: any): any {
  return req.unifiedBody && typeof req.unifiedBody === "object"
    ? req.unifiedBody
    : req.body;
}

function isAnthropicClient(req: any): boolean {
  return (
    !req.clientProtocol || req.clientProtocol === "anthropic_messages"
  );
}

function hasWebSearchTool(tools: any[] | undefined): boolean {
  if (!Array.isArray(tools)) return false;
  return tools.some((tool: any) => {
    if (!tool || typeof tool !== "object") return false;
    if (
      typeof tool.type === "string" &&
      (tool.type === "web_search" || tool.type.startsWith("web_search"))
    ) {
      return true;
    }
    const name =
      tool.function?.name || tool.name || tool.function?.function?.name;
    return (
      typeof name === "string" &&
      (name === "web_search" || name.toLowerCase() === "websearch")
    );
  });
}

function hasThinkSignal(body: any): boolean {
  if (isReasoningDisabled(body?.reasoning, body?.thinking)) return false;
  if (body?.thinking) return true;
  const reasoning = body?.reasoning;
  if (reasoning && typeof reasoning === "object") {
    if (reasoning.enabled === true) return true;
    if (
      typeof reasoning.effort === "string" &&
      reasoning.effort.trim().length > 0
    ) {
      return true;
    }
  }
  return false;
}

const getUseModel = async (
  req: any,
  tokenCount: number,
  configService: ConfigService,
  lastUsage?: Usage | undefined
): Promise<{ model: string; scenarioType: RouterScenarioType }> => {
  const projectSpecificRouter = await getProjectSpecificRouter(req, configService);
  const providers = configService.get<any[]>("providers") || [];
  const Router = projectSpecificRouter || configService.get("Router");
  const body = routingBody(req);
  const modelValue =
    typeof body?.model === "string" ? body.model : String(body?.model ?? "");

  if (modelValue.includes(",")) {
    const [provider, model] = modelValue.split(",");
    const finalProvider = providers.find(
      (p: any) => p.name.toLowerCase() === provider.toLowerCase()
    );
    const finalModel = finalProvider?.models?.find(
      (m: any) => m.toLowerCase() === model.toLowerCase()
    );
    if (finalProvider && finalModel) {
      return { model: `${finalProvider.name},${finalModel}`, scenarioType: 'default' };
    }
    return { model: modelValue, scenarioType: 'default' };
  }

  // Scenario precedence: explicit provider,model (above) → subagent →
  // background → webSearch → think → longContext → default. The long-context
  // check intentionally runs after the higher-priority scenarios so a large
  // subagent/background/web-search/thinking request keeps its scenario route.
  const longContextThreshold = Router?.longContextThreshold || 60000;
  const lastUsageThreshold =
    lastUsage &&
    lastUsage.input_tokens > longContextThreshold &&
    tokenCount > 20000;
  const tokenCountThreshold = tokenCount > longContextThreshold;

  // subagent / background: Anthropic Messages clients only (Claude Code signals).
  if (isAnthropicClient(req)) {
    const taggedSubagentModel = req.protocolContext?.taggedSubagentModel;
    const isClaudeCodeSubagent =
      req.protocolContext?.claudeCodeSubagent === true;
    if (taggedSubagentModel) {
      req.log.info(`Using CCR subagent tag model: ${taggedSubagentModel}`);
      return { model: taggedSubagentModel, scenarioType: 'subagent' };
    }
    if (isClaudeCodeSubagent) {
      const envModel = normalizeModelSelector(
        process.env.CLAUDE_CODE_SUBAGENT_MODEL
      );
      if (envModel) {
        req.log.info(
          `Using CLAUDE_CODE_SUBAGENT_MODEL for Claude Code subagent: ${envModel}`
        );
        return { model: envModel, scenarioType: 'subagent' };
      }
    }

    const globalRouter = configService.get("Router");
    if (
      modelValue?.includes("claude") &&
      modelValue?.includes("haiku") &&
      globalRouter?.background
    ) {
      req.log.info(`Using background model for ${modelValue}`);
      return { model: globalRouter.background, scenarioType: 'background' };
    }
  }

  // The priority of websearch must be higher than thinking.
  if (hasWebSearchTool(body?.tools) && Router?.webSearch) {
    return { model: Router.webSearch, scenarioType: 'webSearch' };
  }
  if (hasThinkSignal(body) && Router?.think) {
    req.log.info(`Using think model for reasoning/thinking request`);
    return { model: Router.think, scenarioType: 'think' };
  }

  // if tokenCount is greater than the configured threshold, use the long context model
  if ((lastUsageThreshold || tokenCountThreshold) && Router?.longContext) {
    req.log.info(
      `Using long context model due to token count: ${tokenCount}, threshold: ${longContextThreshold}`
    );
    return { model: Router.longContext, scenarioType: 'longContext' };
  }

  // No scenario matched and no default route is configured: keep the
  // caller's original model instead of wiping it out.
  return { model: Router?.default || modelValue, scenarioType: 'default' };
};

export interface RouterContext {
  configService: ConfigService;
  tokenizerService?: TokenizerService;
  event?: any;
}

export type RouterScenarioType =
  | 'default'
  | 'background'
  | 'think'
  | 'longContext'
  | 'webSearch'
  | 'subagent';

export interface RouterFallbackConfig {
  default?: string[];
  background?: string[];
  think?: string[];
  longContext?: string[];
  webSearch?: string[];
  subagent?: string[];
}

const parseSessionId = (userId: unknown): string | undefined => {
  if (typeof userId !== "string" || !userId) {
    return undefined;
  }

  const parts = userId.split("_session_");
  if (parts.length > 1 && parts[1]) {
    return parts[1];
  }

  try {
    const parsed = JSON.parse(userId);
    if (parsed && typeof parsed.session_id === "string" && parsed.session_id) {
      return parsed.session_id;
    }
  } catch {
    // Ignore non-JSON user_id formats.
  }

  return undefined;
};

const sessionTokenPrefixCache = new LRUCache<
  string,
  { fingerprint: string; tokenCount: number }
>({ max: 500 });

function fingerprintMessagePrefix(messages: MessageParam[]): string {
  if (!messages.length) return "0";
  const prefix = messages.slice(0, -1);
  const head = JSON.stringify(prefix.slice(0, 2));
  const tail = JSON.stringify(prefix.slice(-1));
  return createHash("sha256")
    .update(`${prefix.length}|${head}|${tail}`)
    .digest("hex")
    .slice(0, 24);
}

function routerNeedsExactTokenCount(
  routerConfig: any,
  customRouterPath: unknown
): boolean {
  if (customRouterPath) return true;
  if (routerConfig?.longContext) return true;
  return false;
}

async function resolveExactTokenCount(options: {
  messages: MessageParam[];
  system: any;
  tools: Tool[];
  tokenizerService?: TokenizerService;
  tokenizerConfig?: any;
  sessionId?: string;
  lastUsage?: Usage;
  longContextThreshold: number;
  latency?: ReturnType<typeof ensureRequestLatency>;
}): Promise<number> {
  const {
    messages,
    system,
    tools,
    tokenizerService,
    tokenizerConfig,
    sessionId,
    lastUsage,
    longContextThreshold,
    latency,
  } = options;

  markLatency(latency, "tokenizeStart");

  const charCount = estimateTokenizePayloadChars(messages, system, tools);
  // One char can be one BPE token in pathological cases. If charCount is
  // strictly below the threshold, exact tokens cannot reach it.
  if (charCount < longContextThreshold) {
    const estimated = estimateTokensFromChars(charCount);
    markLatency(latency, "tokenizeEnd");
    return estimated;
  }

  // Incremental: stable transcript prefix + provider-reported prior usage.
  if (sessionId && lastUsage && messages.length > 1) {
    const fingerprint = fingerprintMessagePrefix(messages);
    const cached = sessionTokenPrefixCache.get(sessionId);
    if (cached && cached.fingerprint === fingerprint) {
      const appended = messages.slice(-1);
      const delta = calculateTokenCount(appended, [], []);
      const combined = lastUsage.input_tokens + delta;
      markLatency(latency, "tokenizeEnd");
      return combined;
    }
  }

  let tokenCount: number;
  const useWorker = charCount >= TOKEN_COUNT_WORKER_THRESHOLD_CHARS;
  if (useWorker) {
    try {
      tokenCount = await countTokensInWorker({ messages, system, tools });
    } catch {
      tokenCount = calculateTokenCount(messages, system, tools);
    }
  } else if (tokenizerService) {
    const result = await tokenizerService.countTokens(
      { messages, system, tools },
      tokenizerConfig
    );
    tokenCount = result.tokenCount;
  } else {
    tokenCount = calculateTokenCount(messages, system, tools);
  }

  if (sessionId && messages.length > 1) {
    sessionTokenPrefixCache.set(sessionId, {
      fingerprint: fingerprintMessagePrefix(messages),
      tokenCount,
    });
  }

  markLatency(latency, "tokenizeEnd");
  return tokenCount;
}

export const router = async (req: any, _res: any, context: RouterContext) => {
  const { configService, event } = context;
  const body = routingBody(req);
  const latency = ensureRequestLatency(req);

  // Session identity: Anthropic metadata.user_id, or Unified metadata if present.
  const sessionSource =
    req.body?.metadata?.user_id ?? body?.metadata?.user_id;
  req.sessionId = parseSessionId(sessionSource);
  const lastMessageUsage = sessionUsageCache.get(req.sessionId);

  // Token counting always uses the Unified projection when available.
  const messages = (body.messages || []) as MessageParam[];
  const system = body.system ?? [];
  const tools = (body.tools || []) as Tool[];

  try {
    const modelForTokenizer =
      typeof body.model === "string" ? body.model : String(body.model ?? "");
    const customRouterPath = configService.get("CUSTOM_ROUTER_PATH");
    const providers = configService.get<any[]>("providers") || [];
    const globalRouter = configService.get("Router");
    markLatency(latency, "projectLookup");
    const projectRouter = await getProjectSpecificRouter(req, configService);
    const activeRouter = projectRouter || globalRouter;
    const longContextThreshold = activeRouter?.longContextThreshold || 60000;

    // Explicit provider,model routes need no token count unless a custom router
    // might still consume it. getUseModel returns those routes before longContext.
    if (!customRouterPath && modelForTokenizer.includes(",")) {
      const [provider, model] = modelForTokenizer.split(",");
      const finalProvider = providers.find(
        (p: any) => p.name.toLowerCase() === provider.toLowerCase()
      );
      const finalModel = finalProvider?.models?.find(
        (m: any) => m.toLowerCase() === model.toLowerCase()
      );
      req.tokenCount = 0;
      body.model =
        finalProvider && finalModel
          ? `${finalProvider.name},${finalModel}`
          : modelForTokenizer;
      if (req.unifiedBody && typeof req.unifiedBody === "object") {
        req.unifiedBody.model = body.model;
      }
      req.scenarioType = "default";
      markLatency(latency, "routeSelected");
      return;
    }

    let tokenCount = 0;
    if (routerNeedsExactTokenCount(activeRouter, customRouterPath)) {
      const [providerName, modelName] = modelForTokenizer.split(",");
      const tokenizerConfig =
        context.tokenizerService?.getTokenizerConfigForModel(
          providerName,
          modelName
        );
      tokenCount = await resolveExactTokenCount({
        messages,
        system,
        tools,
        tokenizerService: context.tokenizerService,
        tokenizerConfig,
        sessionId: req.sessionId,
        lastUsage: lastMessageUsage,
        longContextThreshold,
        latency,
      });
    }

    req.tokenCount = tokenCount;

    let model;
    if (customRouterPath) {
      try {
        const customRouter = require(customRouterPath);
        // Anthropic custom routers continue to receive original Anthropic req.body.
        // Multi-protocol routers can read req.clientProtocol / req.unifiedBody.
        model = await customRouter(req, configService.getAll(), {
          event,
        });
      } catch (e: any) {
        req.log.error(`failed to load custom router: ${e.message}`);
      }
    }
    if (!model) {
      const result = await getUseModel(
        req,
        tokenCount,
        configService,
        lastMessageUsage
      );
      model = result.model;
      req.scenarioType = result.scenarioType;
    } else {
      req.scenarioType = "default";
    }
    markLatency(latency, "routeSelected");

    // Apply selected model to canonical Unified. Do not mutate immutable client wire provenance.
    body.model = model;
    if (req.unifiedBody && typeof req.unifiedBody === "object") {
      req.unifiedBody.model = model;
    }
  } catch (error: any) {
    req.log.error(`Error in router middleware: ${error.message}`);
    const Router = configService.get("Router");
    const fallbackModel = Router?.default || body.model;
    body.model = fallbackModel;
    if (req.unifiedBody && typeof req.unifiedBody === "object") {
      req.unifiedBody.model = fallbackModel;
    }
    req.scenarioType = "default";
  }
  return;
};

// Memory cache for sessionId to project name mapping
// null value indicates previously searched but not found
// Uses LRU cache with max 1000 entries
const sessionProjectCache = new LRUCache<string, string>({
  max: 1000,
});

export const searchProjectBySession = async (
  sessionId: string,
  logger?: any
): Promise<string | null> => {
  // Check cache first
  if (sessionProjectCache.has(sessionId)) {
    const result = sessionProjectCache.get(sessionId);
    if (!result || result === '') {
      return null;
    }
    return result;
  }

  try {
    const projectsDirStat = await stat(CLAUDE_PROJECTS_DIR);
    if (!projectsDirStat.isDirectory()) {
      sessionProjectCache.set(sessionId, '');
      return null;
    }
  } catch (error: any) {
    if (error?.code === "ENOENT") {
      sessionProjectCache.set(sessionId, '');
      return null;
    }
    (logger?.error ?? console.error)("Error checking Claude projects directory:", error);
    sessionProjectCache.set(sessionId, '');
    return null;
  }

  try {
    const dir = await opendir(CLAUDE_PROJECTS_DIR);
    const folderNames: string[] = [];

    // Collect all folder names
    for await (const dirent of dir) {
      if (dirent.isDirectory()) {
        folderNames.push(dirent.name);
      }
    }

    // Concurrently check each project folder for sessionId.jsonl file
    const projectsRoot = resolve(CLAUDE_PROJECTS_DIR);
    const safeSessionFile = `${basename(String(sessionId))}.jsonl`;
    const checkPromises = folderNames.map(async (folderName) => {
      const projectDir = resolve(projectsRoot, basename(folderName));
      const projectRel = relative(projectsRoot, projectDir);
      if (!projectRel || projectRel.startsWith("..")) {
        return null;
      }
      const sessionFilePath = join(projectDir, safeSessionFile);
      try {
        const fileStat = await stat(sessionFilePath);
        return fileStat.isFile() ? folderName : null;
      } catch {
        // File does not exist, continue checking next
        return null;
      }
    });

    const results = await Promise.all(checkPromises);

    // Return the first existing project directory name
    for (const result of results) {
      if (result) {
        // Cache the found result
        sessionProjectCache.set(sessionId, result);
        return result;
      }
    }

    // Cache not found result (null value means previously searched but not found)
    sessionProjectCache.set(sessionId, '');
    return null; // No matching project found
  } catch (error) {
    (logger?.error ?? console.error)("Error searching for project by session:", error);
    // Cache null result on error to avoid repeated errors
    sessionProjectCache.set(sessionId, '');
    return null;
  }
};

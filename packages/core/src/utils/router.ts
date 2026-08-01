import { get_encoding } from "tiktoken";
import { sessionUsageCache, Usage } from "./cache";
import { readFile } from "fs/promises";
import { opendir, stat } from "fs/promises";
import { join, resolve, relative, basename } from "path";
import { CLAUDE_PROJECTS_DIR, HOME_DIR } from "@caeliq/ccr-shared";
import { LRUCache } from "lru-cache";
import { ConfigService } from "../services/config";
import { TokenizerService } from "../services/tokenizer";

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
const CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX = "x-anthropic-billing-header";

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

function claudeCodeBillingMetadataIsSubagent(text: string): boolean {
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

const getUseModel = async (
  req: any,
  tokenCount: number,
  configService: ConfigService,
  lastUsage?: Usage | undefined
): Promise<{ model: string; scenarioType: RouterScenarioType }> => {
  const projectSpecificRouter = await getProjectSpecificRouter(req, configService);
  const providers = configService.get<any[]>("providers") || [];
  const Router = projectSpecificRouter || configService.get("Router");

  if (req.body.model.includes(",")) {
    const [provider, model] = req.body.model.split(",");
    const finalProvider = providers.find(
      (p: any) => p.name.toLowerCase() === provider
    );
    const finalModel = finalProvider?.models?.find(
      (m: any) => m.toLowerCase() === model
    );
    if (finalProvider && finalModel) {
      return { model: `${finalProvider.name},${finalModel}`, scenarioType: 'default' };
    }
    return { model: req.body.model, scenarioType: 'default' };
  }

  // if tokenCount is greater than the configured threshold, use the long context model
  const longContextThreshold = Router?.longContextThreshold || 60000;
  const lastUsageThreshold =
    lastUsage &&
    lastUsage.input_tokens > longContextThreshold &&
    tokenCount > 20000;
  const tokenCountThreshold = tokenCount > longContextThreshold;
  if ((lastUsageThreshold || tokenCountThreshold) && Router?.longContext) {
    req.log.info(
      `Using long context model due to token count: ${tokenCount}, threshold: ${longContextThreshold}`
    );
    return { model: Router.longContext, scenarioType: 'longContext' };
  }

  // Claude Code subagent signals: strip billing helper system text, then prefer
  // explicit <CCR-SUBAGENT-MODEL> tag, else CLAUDE_CODE_SUBAGENT_MODEL env.
  const isClaudeCodeSubagent = removeClaudeCodeBillingSystemHeader(req.body);
  const taggedSubagentModel = extractAndRemoveClaudeCodeSubagentModelTag(req.body);
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

  // Use the background model for any Claude Haiku variant
  const globalRouter = configService.get("Router");
  if (
    req.body.model?.includes("claude") &&
    req.body.model?.includes("haiku") &&
    globalRouter?.background
  ) {
    req.log.info(`Using background model for ${req.body.model}`);
    return { model: globalRouter.background, scenarioType: 'background' };
  }
  // The priority of websearch must be higher than thinking.
  if (
    Array.isArray(req.body.tools) &&
    req.body.tools.some((tool: any) => tool.type?.startsWith("web_search")) &&
    Router?.webSearch
  ) {
    return { model: Router.webSearch, scenarioType: 'webSearch' };
  }
  // if exits thinking, use the think model
  if (req.body.thinking && Router?.think) {
    req.log.info(`Using think model for ${req.body.thinking}`);
    return { model: Router.think, scenarioType: 'think' };
  }
  // No scenario matched and no default route is configured: keep the
  // caller's original model instead of wiping it out.
  return { model: Router?.default || req.body.model, scenarioType: 'default' };
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

export const router = async (req: any, _res: any, context: RouterContext) => {
  const { configService, event } = context;
  req.sessionId = parseSessionId(req.body.metadata?.user_id);
  const lastMessageUsage = sessionUsageCache.get(req.sessionId);
  const { messages, system = [], tools }: MessageCreateParamsBase = req.body;
  const rewritePrompt = configService.get("REWRITE_SYSTEM_PROMPT");
  if (
    rewritePrompt &&
    system.length > 1 &&
    system[1]?.text?.includes("<env>")
  ) {
    const prompt = await readFile(rewritePrompt, "utf-8");
    system[1].text = `${prompt}<env>${system[1].text.split("<env>").pop()}`;
  }

  try {
    // Try to get tokenizer config for the current model
    const [providerName, modelName] = req.body.model.split(",");
    const tokenizerConfig = context.tokenizerService?.getTokenizerConfigForModel(
      providerName,
      modelName
    );

    // Use TokenizerService if available, otherwise fall back to legacy method
    let tokenCount: number;

    if (context.tokenizerService) {
      const result = await context.tokenizerService.countTokens(
        {
          messages: messages as MessageParam[],
          system,
          tools: tools as Tool[],
        },
        tokenizerConfig
      );
      tokenCount = result.tokenCount;
    } else {
      // Legacy fallback
      tokenCount = calculateTokenCount(
        messages as MessageParam[],
        system,
        tools as Tool[]
      );
    }

    let model;
    const customRouterPath = configService.get("CUSTOM_ROUTER_PATH");
    if (customRouterPath) {
      try {
        const customRouter = require(customRouterPath);
        req.tokenCount = tokenCount; // Pass token count to custom router
        model = await customRouter(req, configService.getAll(), {
          event,
        });
      } catch (e: any) {
        req.log.error(`failed to load custom router: ${e.message}`);
      }
    }
    if (!model) {
      const result = await getUseModel(req, tokenCount, configService, lastMessageUsage);
      model = result.model;
      req.scenarioType = result.scenarioType;
    } else {
      // Custom router doesn't provide scenario type, default to 'default'
      req.scenarioType = 'default';
    }
    req.body.model = model;
  } catch (error: any) {
    req.log.error(`Error in router middleware: ${error.message}`);
    const Router = configService.get("Router");
    // Fall back to the caller's original model rather than wiping it out
    // when no default route is configured.
    req.body.model = Router?.default || req.body.model;
    req.scenarioType = 'default';
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

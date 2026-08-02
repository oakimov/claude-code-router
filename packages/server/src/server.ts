import Server, {
  calculateTokenCount,
  TokenizerService,
  isClientAbortError,
  sanitizeHeadersForLog,
} from "@caeliq/llms";
import { readConfigFile, writeConfigFile, backupConfigFile } from "./utils";
import { join, resolve, relative, sep, basename } from "path";
import fastifyStatic from "@fastify/static";
import fastifyCookie from "@fastify/cookie";
import type {} from "@fastify/rate-limit";
import { readdirSync, statSync, readFileSync, writeFileSync, existsSync, mkdirSync, unlinkSync, rmSync } from "fs";
import { homedir } from "os";
import {
  getPresetDir,
  readManifestFromDir,
  manifestToPresetFile,
  saveManifest,
  isPresetInstalled,
  extractPreset,
  HOME_DIR,
  extractMetadata,
  loadConfigFromManifest,
  downloadPresetToTemp,
  getTempDir,
  findMarketPresetByName,
  getMarketPresets,
  RATE_LIMIT_CONFIG,
  type PresetFile,
  type ManifestFile,
  type PresetMetadata,
} from "@caeliq/ccr-shared";
import fastifyMultipart from "@fastify/multipart";
import AdmZip from "adm-zip";
import { registerCodexAuthRoutes } from "./routes/codex-auth";
import { registerQwenAuthRoutes } from "./routes/qwen-auth";
import { registerClaudeAuthRoutes } from "./routes/claude-auth";
import { registerAntigravityAuthRoutes } from "./routes/antigravity-auth";
import {
  apiKeysMatch,
  clearUiSessionCookie,
  createUiSession,
  revokeUiSession,
  setUiSessionCookie,
} from "./auth/ui-session";

const LOG_DIR = join(homedir(), ".claude-code-router", "logs");

/** Resolve a log file path, rejecting anything outside LOG_DIR. */
export function resolveLogFilePath(filePath?: string): string {
  const logDir = resolve(LOG_DIR);
  // Prefer basename so callers cannot pass absolute paths outside the log dir.
  const name = filePath ? basename(filePath) : "app.log";
  if (!name || name === "." || name === ".." || name.includes("..") || !name.endsWith(".log")) {
    throw new Error("Log file path must be under the logs directory");
  }
  const candidate = resolve(logDir, name);
  const rel = relative(logDir, candidate);
  if (!rel || rel.startsWith("..") || rel.includes(`..${sep}`)) {
    throw new Error("Log file path must be under the logs directory");
  }
  return candidate;
}

export const createServer = async (config: any): Promise<any> => {
  const server = new Server(config);
  const app = server.app;
  const rateLimitOptions = {
    config: { rateLimit: { ...RATE_LIMIT_CONFIG } },
  };

  // Intercept all fetch calls to log provider interactions
  const originalFetch = global.fetch;
  const headersFromFetchArgs = (input: RequestInfo | URL, init?: RequestInit) => {
    // Prefer init.headers; fall back to Request headers when input is a Request.
    if (init?.headers) return init.headers;
    if (input instanceof Request) return input.headers;
    return undefined;
  };
  const methodFromFetchArgs = (input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method) return init.method;
    if (input instanceof Request) return input.method;
    return "GET";
  };

  global.fetch = async (...args) => {
    const input = args[0] as RequestInfo | URL;
    const init = args[1] as RequestInit | undefined;
    const url = input instanceof Request ? input.url : String(input);

    // Filter out localhost/internal requests to reduce noise
    if (url.includes("localhost") || url.includes("127.0.0.1")) {
      return originalFetch(...args);
    }

    const requestHeaders = sanitizeHeadersForLog(
      headersFromFetchArgs(input, init) as any
    );

    app.log.debug(
      {
        url,
        method: methodFromFetchArgs(input, init),
        headers: requestHeaders,
      },
      "Upstream Provider Request"
    );

    try {
      const response = await originalFetch(...args);
      const responseHeaders = sanitizeHeadersForLog(response.headers);

      app.log.debug(
        {
          url,
          status: response.status,
          headers: responseHeaders,
        },
        "Upstream Provider Response"
      );

      if (response.status >= 400) {
        const errorBody = await response.clone().text();
        app.log.error(
          {
            url,
            status: response.status,
            body: errorBody,
          },
          "Upstream Provider Error Body"
        );
      }

      return response;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      // Client disconnect aborts are expected (AbortSignal from SSE/socket close).
      // Log at debug so OpenCode/Cursor noise does not look like provider 500s.
      // Use shared classifier so timeouts stay as errors (not quiet disconnects).
      if (isClientAbortError(error)) {
        app.log.debug({ url, error: message }, "Upstream provider fetch aborted");
      } else {
        app.log.error(
          {
            url,
            error: message,
          },
          "Upstream Provider Fetch Exception"
        );
      }
      throw error;
    }
  };

  await app.register(fastifyCookie);

  app.register(fastifyMultipart, {
    limits: {
      fileSize: 50 * 1024 * 1024, // 50MB
    },
  });

  app.post(
    "/api/auth/login",
    rateLimitOptions,
    async (req: any, reply: any) => {
      const configuredApiKey = server.configService.get("APIKEY");
      if (!configuredApiKey || !apiKeysMatch(req.body?.apiKey, configuredApiKey)) {
        return reply.status(401).send({ error: "Invalid API key" });
      }

      const sessionId = createUiSession();
      setUiSessionCookie(reply, sessionId);
      return { success: true };
    }
  );

  app.post(
    "/api/auth/logout",
    rateLimitOptions,
    async (req: any, reply: any) => {
      revokeUiSession(req);
      clearUiSessionCookie(reply);
      return { success: true };
    }
  );

  // Register Codex OAuth callback routes
  await registerCodexAuthRoutes(app);

  // Register Qwen Chat JWT auth routes
  await registerQwenAuthRoutes(app);

  // Register Claude OAuth callback route (http://127.0.0.1:8080/callback)
  await registerClaudeAuthRoutes(app);

  // Antigravity OAuth: host 51121 → this server (compose 51121:3456)
  await registerAntigravityAuthRoutes(app);

  app.post("/v1/messages/count_tokens", rateLimitOptions, async (req: any, reply: any) => {
    const { messages, tools, system, model } = req.body;
    const tokenizerService = (app as any)._server!.tokenizerService as TokenizerService;

    // If model is specified in "providerName,modelName" format, use the configured tokenizer
    if (model && model.includes(",") && tokenizerService) {
      try {
        const [provider, modelName] = model.split(",");
        req.log?.info(`Looking up tokenizer for provider: ${provider}, model: ${modelName}`);

        const tokenizerConfig = tokenizerService.getTokenizerConfigForModel(provider, modelName);

        if (!tokenizerConfig) {
          req.log?.debug(`No tokenizer config found for ${provider},${modelName}; using default tiktoken`);
        } else {
          req.log?.info(`Using tokenizer config: ${JSON.stringify(tokenizerConfig)}`);
        }

        const result = await tokenizerService.countTokens(
          { messages, system, tools },
          tokenizerConfig
        );

        return {
          "input_tokens": result.tokenCount,
          "tokenizer": result.tokenizerUsed,
        };
      } catch (error: any) {
        req.log?.error(`Error using configured tokenizer: ${error.message}`);
        req.log?.error(error.stack);
        // Fall back to default calculation
      }
    } else {
      if (!model) {
        req.log?.info(`No model specified, using default tiktoken`);
      } else if (!model.includes(",")) {
        req.log?.info(`Model "${model}" does not contain comma, using default tiktoken`);
      } else if (!tokenizerService) {
        req.log?.warn(`TokenizerService not available, using default tiktoken`);
      }
    }

    // Default to tiktoken calculation
    const tokenCount = calculateTokenCount(messages, system, tools);
    return { "input_tokens": tokenCount }
  });

  // Add endpoint to read config.json with access control
  app.get("/api/config", rateLimitOptions, async (req: any, reply: any) => {
    return await readConfigFile(false);
  });

  app.get("/api/transformers", rateLimitOptions, async (req: any, reply: any) => {
    const transformerService = (app as any)._server!.transformerService;
    const transformers = transformerService.getAllTransformers();
    // Transformers registered by class (so they can take options) expose
    // `endPoint` only on an instance, so fall back to the routes actually
    // registered for them.
    const registeredEndpoints = new Map<string, string | undefined>(
      transformerService
        .getTransformersWithEndpoint()
        .map((entry: any) => [entry.name, entry.transformer.endPoint])
    );
    const transformerList = Array.from(transformers.entries()).map(
      ([name, transformer]: any) => ({
        name,
        endpoint:
          transformer.endPoint || registeredEndpoints.get(name) || null,
      })
    );
    return { transformers: transformerList };
  });

  // Add endpoint to save config.json with access control
  app.post("/api/config", rateLimitOptions, async (req: any, reply: any) => {
    const newConfig = req.body;

    // Backup existing config file if it exists
    const backupPath = await backupConfigFile();
    if (backupPath) {
      app.log.info(`Backed up existing configuration file to ${backupPath}`);
    }

    await writeConfigFile(newConfig);
    return { success: true, message: "Config saved successfully" };
  });

  // Register static file serving with caching
  app.register(fastifyStatic, {
    root: join(__dirname, "..", "dist"),
    prefix: "/ui/",
    maxAge: "1h",
  });

  // Redirect /ui to /ui/ for proper static file serving
  app.get("/ui", rateLimitOptions, async (_: any, reply: any) => {
    return reply.redirect("/ui/");
  });

  // Get log file list endpoint
  app.get("/api/logs/files", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const logFiles: Array<{ name: string; path: string; size: number; lastModified: string }> = [];

      if (existsSync(LOG_DIR)) {
        const files = readdirSync(LOG_DIR);

        for (const file of files) {
          if (file.endsWith('.log')) {
            const filePath = join(LOG_DIR, file);
            const stats = statSync(filePath);

            logFiles.push({
              name: file,
              path: filePath,
              size: stats.size,
              lastModified: stats.mtime.toISOString()
            });
          }
        }

        // Sort by modification time in descending order
        logFiles.sort((a, b) => new Date(b.lastModified).getTime() - new Date(a.lastModified).getTime());
      }

      return logFiles;
    } catch (error) {
      app.log.error({ err: error }, "Failed to get log files");
      reply.status(500).send({ error: "Failed to get log files" });
    }
  });

  // Get log content endpoint
  app.get("/api/logs", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const logFilePath = resolveLogFilePath((req.query as any).file as string | undefined);

      if (!existsSync(logFilePath)) {
        return [];
      }

      const logContent = readFileSync(logFilePath, 'utf8');
      const logLines = logContent.split('\n').filter(line => line.trim())

      return logLines;
    } catch (error) {
      if (error instanceof Error && error.message.includes("logs directory")) {
        reply.status(400).send({ error: error.message });
        return;
      }
      app.log.error({ err: error }, "Failed to get logs");
      reply.status(500).send({ error: "Failed to get logs" });
    }
  });

  // Clear log content endpoint
  app.delete("/api/logs", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const logFilePath = resolveLogFilePath((req.query as any).file as string | undefined);

      if (existsSync(logFilePath)) {
        writeFileSync(logFilePath, '', 'utf8');
      }

      return { success: true, message: "Logs cleared successfully" };
    } catch (error) {
      if (error instanceof Error && error.message.includes("logs directory")) {
        reply.status(400).send({ error: error.message });
        return;
      }
      app.log.error({ err: error }, "Failed to clear logs");
      reply.status(500).send({ error: "Failed to clear logs" });
    }
  });

  // Get presets list
  app.get("/api/presets", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const presetsDir = join(HOME_DIR, "presets");

      if (!existsSync(presetsDir)) {
        return { presets: [] };
      }

      const entries = readdirSync(presetsDir, { withFileTypes: true });
      const presetDirs = entries.filter(e => e.isDirectory() && !e.name.startsWith('.')).map(e => e.name);

      const presets: Array<PresetMetadata & { installed: boolean; id: string }> = [];

      for (const dirName of presetDirs) {
        const presetDir = join(presetsDir, dirName);
        try {
          const manifestPath = join(presetDir, "manifest.json");
          const content = readFileSync(manifestPath, 'utf-8');
          const manifest = JSON.parse(content);

          // Extract metadata fields
          const { Providers, Router, PORT, HOST, API_TIMEOUT_MS, PROXY_URL, LOG, LOG_LEVEL, StatusLine, NON_INTERACTIVE_MODE, ...metadata } = manifest;

          presets.push({
            id: dirName,  // Use directory name as unique identifier
            name: metadata.name || dirName,
            version: metadata.version || '1.0.0',
            description: metadata.description,
            author: metadata.author,
            homepage: metadata.homepage,
            repository: metadata.repository,
            license: metadata.license,
            keywords: metadata.keywords,
            ccrVersion: metadata.ccrVersion,
            source: metadata.source,
            sourceType: metadata.sourceType,
            checksum: metadata.checksum,
            installed: true,
          });
        } catch (error) {
          app.log.error({ err: error }, `Failed to read preset ${dirName}`);
        }
      }

      return { presets };
    } catch (error) {
      app.log.error({ err: error }, "Failed to get presets");
      reply.status(500).send({ error: "Failed to get presets" });
    }
  });

  // Get preset details
  app.get("/api/presets/:name", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const { name } = req.params;
      const presetDir = getPresetDir(name);

      if (!existsSync(presetDir)) {
        reply.status(404).send({ error: "Preset not found" });
        return;
      }

      const manifest = await readManifestFromDir(presetDir);
      const presetFile = manifestToPresetFile(manifest);

      // Return preset info, config uses the applied userValues configuration
      return {
        ...presetFile,
        config: loadConfigFromManifest(manifest, presetDir),
        userValues: manifest.userValues || {},
      };
    } catch (error: any) {
      app.log.error({ err: error }, "Failed to get preset");
      reply.status(500).send({ error: error.message || "Failed to get preset" });
    }
  });

  // Apply preset (configure sensitive information)
  app.post("/api/presets/:name/apply", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const { name } = req.params;
      const { secrets } = req.body;

      const presetDir = getPresetDir(name);

      if (!existsSync(presetDir)) {
        reply.status(404).send({ error: "Preset not found" });
        return;
      }

      // Read existing manifest
      const manifest = await readManifestFromDir(presetDir);

      // Save user input to userValues (keep original config unchanged)
      const updatedManifest: ManifestFile = { ...manifest };

      // Save or update userValues
      if (secrets && Object.keys(secrets).length > 0) {
        updatedManifest.userValues = {
          ...updatedManifest.userValues,
          ...secrets,
        };
      }

      // Save updated manifest
      await saveManifest(name, updatedManifest);

      return { success: true, message: "Preset applied successfully" };
    } catch (error: any) {
      app.log.error({ err: error }, "Failed to apply preset");
      reply.status(500).send({ error: error.message || "Failed to apply preset" });
    }
  });

  // Delete preset
  app.delete("/api/presets/:name", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const { name } = req.params;
      const presetDir = getPresetDir(name);

      if (!existsSync(presetDir)) {
        reply.status(404).send({ error: "Preset not found" });
        return;
      }

      // Recursively delete entire directory
      rmSync(presetDir, { recursive: true, force: true });

      return { success: true, message: "Preset deleted successfully" };
    } catch (error: any) {
      app.log.error({ err: error }, "Failed to delete preset");
      reply.status(500).send({ error: error.message || "Failed to delete preset" });
    }
  });

  // Get preset market list
  app.get("/api/presets/market", rateLimitOptions, async (req: any, reply: any) => {
    try {
      // Use market presets function
      const marketPresets = await getMarketPresets();
      return { presets: marketPresets };
    } catch (error: any) {
      app.log.error({ err: error }, "Failed to get market presets");
      reply.status(500).send({ error: error.message || "Failed to get market presets" });
    }
  });

  // Install preset from GitHub repository by preset name
  app.post("/api/presets/install/github", rateLimitOptions, async (req: any, reply: any) => {
    try {
      const { presetName } = req.body;

      if (!presetName) {
        reply.status(400).send({ error: "Preset name is required" });
        return;
      }

      // Check if preset is in the marketplace
      const marketPreset = await findMarketPresetByName(presetName);
      if (!marketPreset) {
        reply.status(400).send({
          error: "Preset not found in marketplace",
          message: `Preset '${presetName}' is not available in the official marketplace. Please check the available presets.`
        });
        return;
      }

      // Get repository from market preset
      if (!marketPreset.repo) {
        reply.status(400).send({
          error: "Invalid preset data",
          message: `Preset '${presetName}' does not have repository information`
        });
        return;
      }

      // Parse GitHub repository URL
      const githubRepoMatch = marketPreset.repo.match(/(?:github\.com[:/]|^)([^/]+)\/([^/\s#]+?)(?:\.git)?$/);
      if (!githubRepoMatch) {
        reply.status(400).send({ error: "Invalid GitHub repository URL" });
        return;
      }

      const [, owner, repoName] = githubRepoMatch;

      // Use preset name from market
      const installedPresetName = marketPreset.name || presetName;

      // Check if already installed BEFORE downloading
      if (await isPresetInstalled(installedPresetName)) {
        reply.status(409).send({
          error: "Preset already installed",
          message: `Preset '${installedPresetName}' is already installed. To update or reconfigure, please delete it first using the delete button.`,
          presetName: installedPresetName
        });
        return;
      }

      // Download GitHub repository ZIP file
      const downloadUrl = `https://github.com/${owner}/${repoName}/archive/refs/heads/main.zip`;
      const tempFile = await downloadPresetToTemp(downloadUrl);

      // Load preset to validate structure
      const preset = await loadPresetFromZip(tempFile);

      // Double-check if already installed (in case of race condition)
      if (await isPresetInstalled(installedPresetName)) {
        unlinkSync(tempFile);
        reply.status(409).send({
          error: "Preset already installed",
          message: `Preset '${installedPresetName}' was installed while downloading. Please try again.`,
          presetName: installedPresetName
        });
        return;
      }

      // Extract to target directory
      const targetDir = getPresetDir(installedPresetName);
      await extractPreset(tempFile, targetDir);

      // Read manifest and add repo information
      const manifest = await readManifestFromDir(targetDir);

      // Add repo information to manifest from market data
      manifest.repository = marketPreset.repo;
      if (marketPreset.url) {
        manifest.source = marketPreset.url;
      }

      // Save updated manifest
      await saveManifest(installedPresetName, manifest);

      // Clean up temp file
      unlinkSync(tempFile);

      return {
        success: true,
        presetName: installedPresetName,
        preset: {
          ...preset.metadata,
          installed: true,
        }
      };
    } catch (error: any) {
      app.log.error({ err: error }, "Failed to install preset from GitHub");
      reply.status(500).send({ error: error.message || "Failed to install preset from GitHub" });
    }
  });

  // Helper function: Load preset from ZIP
  async function loadPresetFromZip(zipFile: string): Promise<PresetFile> {
    const zip = new AdmZip(zipFile);

    // First try to find manifest.json in root directory
    let entry = zip.getEntry('manifest.json');

    // If not in root, try to find in subdirectories (handle GitHub repo archive structure)
    if (!entry) {
      const entries = zip.getEntries();
      // Find any manifest.json file
      entry = entries.find(e => e.entryName.includes('manifest.json')) || null;
    }

    if (!entry) {
      throw new Error('Invalid preset file: manifest.json not found');
    }

    const manifest = JSON.parse(entry.getData().toString('utf-8')) as ManifestFile;
    return manifestToPresetFile(manifest);
  }

  return server;
};

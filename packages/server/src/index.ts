import { existsSync } from "fs";
import { writeFile } from "fs/promises";
import { randomBytes } from "crypto";
import { homedir } from "os";
import { join } from "path";
import { initConfig, initDir } from "./utils";
import {
  HealthHeartbeat,
  resolveHeartbeatIntervalMs,
} from "./utils/health-heartbeat";
import {
  ACTIVE_SERVER_LOG_NAME,
  DEFAULT_LOG_MAX_FILES,
  DEFAULT_LOG_MAX_TOTAL_BYTES,
  LogRetentionScheduler,
  SERVER_LOG_HISTORY_NAME,
} from "./utils/log-retention";
import { createServer } from "./server";
import { apiKeyAuth, detectClientProtocol } from "./middleware/auth";
import {
  CONFIG_FILE,
  HEALTH_FILE,
  HOME_DIR,
  RATE_LIMIT_CONFIG,
  listPresets,
} from "@caeliq/ccr-shared";
import { createStream } from 'rotating-file-stream';
import { sessionUsageCache, setHealthReporter, SSEParserTransform, SSESerializerTransform, rewriteStream, closeProxyDispatchers, closeTokenCountWorkers } from "@caeliq/llms";
import JSON5 from "json5";
import { IAgent, ITool } from "./agents/type";
import agentsManager from "./agents";
import { EventEmitter } from "node:events";
import { pluginManager, tokenSpeedPlugin } from "@caeliq/llms";

const event = new EventEmitter()

async function initializeClaudeConfig() {
  const homeDir = homedir();
  const configPath = join(homeDir, ".claude.json");
  if (!existsSync(configPath)) {
    const userID = randomBytes(32).toString("hex");
    const configContent = {
      numStartups: 184,
      autoUpdaterStatus: "enabled",
      userID,
      hasCompletedOnboarding: true,
      lastOnboardingVersion: "1.0.17",
      projects: {},
    };
    await writeFile(configPath, JSON.stringify(configContent, null, 2));
  }
}

interface RunOptions {
  port?: number;
  logger?: any;
}

/**
 * Plugin configuration from config file
 */
interface PluginConfig {
  name: string;
  enabled?: boolean;
  options?: Record<string, any>;
}

/**
 * Register plugins from configuration
 * @param serverInstance Server instance
 * @param config Application configuration
 */
async function registerPluginsFromConfig(serverInstance: any, config: any): Promise<void> {
  // Get plugins configuration from config file
  const pluginsConfig: PluginConfig[] = config.plugins || config.Plugins || [];

  for (const pluginConfig of pluginsConfig) {
      const { name, enabled = false, options = {} } = pluginConfig;

      switch (name) {
        case 'token-speed':
          pluginManager.registerPlugin(tokenSpeedPlugin, {
            enabled,
            outputHandlers: [
              {
                type: 'temp-file',
                enabled: true
              }
            ],
            ...options
          });
          break;

        default:
          serverInstance.app.log.warn(`Unknown plugin: ${name}`);
          break;
      }
    }
  // Enable all registered plugins
  await pluginManager.enablePlugins(serverInstance);
}

/**
 * Background consumer for Anthropic message_delta usage frames.
 * Uses one incremental TextDecoder and buffers incomplete SSE events so a
 * split `event:` / `data:` pair across chunks is still recognized.
 */
async function consumeMessageDeltaUsage(
  stream: ReadableStream<Uint8Array>,
  sessionId: string,
  heartbeat: HealthHeartbeat,
  log: { debug: (...args: any[]) => void; error: (...args: any[]) => void }
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleEventBlock = (block: string) => {
    let eventName = "";
    let dataLine = "";
    for (const line of block.split(/\r?\n/)) {
      if (line.startsWith("event:")) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith("data:")) {
        dataLine = line.slice(5).trimStart();
      }
    }
    if (eventName !== "message_delta" || !dataLine) return;
    try {
      const message = JSON.parse(dataLine);
      if (message?.usage) {
        sessionUsageCache.put(sessionId, message.usage);
        heartbeat.recordUsage(sessionId, message.usage);
      }
    } catch {
      // Ignore malformed usage frames.
    }
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder
        .decode(value, { stream: true })
        .replace(/\r\n/g, "\n")
        .replace(/\r/g, "\n");
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        if (block.trim()) handleEventBlock(block);
      }
    }
    buffer += decoder.decode().replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    if (buffer.trim()) handleEventBlock(buffer);
  } catch (readError: any) {
    if (
      readError?.name === "AbortError" ||
      readError?.code === "ERR_STREAM_PREMATURE_CLOSE"
    ) {
      log.debug("Background usage stream closed prematurely");
    } else {
      log.error("Error in background usage stream reading:", readError);
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // already released
    }
  }
}

async function getServer(options: RunOptions = {}) {
  await initializeClaudeConfig();
  await initDir();
  const config = await initConfig();

  // Check if Providers is configured
  const providers = config.Providers || config.providers || [];
  const hasProviders = providers && providers.length > 0;

  let HOST = config.HOST || "127.0.0.1";

  if (hasProviders) {
    HOST = config.HOST;
    if (!config.APIKEY) {
      HOST = "127.0.0.1";
    }
  } else {
    // When no providers are configured, listen on 0.0.0.0 without authentication
    HOST = "0.0.0.0";
    console.log("ℹ️  No providers configured. Listening on 0.0.0.0 without authentication.");
  }

  const port = config.PORT || 3456;

  // Use port from environment variable if set (for background process)
  const servicePort = process.env.SERVICE_PORT
    ? parseInt(process.env.SERVICE_PORT)
    : port;

  // Configure logger based on config settings or external options
  const pad = (num: number) => (num > 9 ? "" : "0") + num;
  // Stable active name + timestamped rotated names so RFS history survives
  // restarts. Daily LogRetentionScheduler also prunes orphaned ccr-*.log files.
  const generator = (time: number | Date | undefined, index: number | undefined) => {
    if (!time) return `./logs/${ACTIVE_SERVER_LOG_NAME}`;

    const date = typeof time === "number" ? new Date(time) : time;
    const month = date.getFullYear() + "" + pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hour = pad(date.getHours());
    const minute = pad(date.getMinutes());
    const seconds = pad(date.getSeconds());
    return `./logs/ccr-${month}${day}${hour}${minute}${seconds}${index ? `_${index}` : ""}.log`;
  };

  let loggerConfig: any;

  // Use external logger configuration if provided
  if (options.logger !== undefined) {
    loggerConfig = options.logger;
  } else {
    // Enable logger if not provided and config.LOG !== false
    if (config.LOG !== false) {
      // Set config.LOG to true (if not already set)
      if (config.LOG === undefined) {
        config.LOG = true;
      }
      loggerConfig = {
        level: config.LOG_LEVEL || "info",
        stream: createStream(generator, {
          path: HOME_DIR,
          history: `./logs/${SERVER_LOG_HISTORY_NAME}`,
          // `size` rotates the *active* file; `maxSize` caps total rotated bytes.
          size: "50M",
          maxSize: "150M",
          maxFiles: DEFAULT_LOG_MAX_FILES,
          interval: "1d",
          compress: false,
        }),
      };
    } else {
      loggerConfig = false;
    }
  }

  const presets = await listPresets();

  const serverInstance = await createServer({
    jsonPath: CONFIG_FILE,
    initialConfig: {
      // ...config,
      providers: config.Providers || config.providers,
      APIKEY: config.APIKEY,
      HOST: HOST,
      PORT: servicePort,
      LOG_REQUEST_BODY_PARTS: config.LOG_REQUEST_BODY_PARTS,
      LOG_REQUEST_BODY_MAX_BYTES: config.LOG_REQUEST_BODY_MAX_BYTES,
      LOG_SSE_EVENTS: config.LOG_SSE_EVENTS,
      LOG_FILE: join(
        homedir(),
        ".claude-code-router",
        "claude-code-router.log"
      ),
    },
    logger: loggerConfig,
  });

  await Promise.allSettled(
      presets.map(async preset => await serverInstance.registerNamespace(`/preset/${preset.name}`, preset.config))
  )

  // Register and configure plugins from config
  await registerPluginsFromConfig(serverInstance, config);

  const heartbeat = new HealthHeartbeat({
    intervalMs: resolveHeartbeatIntervalMs(config),
    logger: serverInstance.app.log,
    snapshotFile: HEALTH_FILE,
  });

  const logRetention = new LogRetentionScheduler({
    logDir: join(HOME_DIR, "logs"),
    maxFiles: DEFAULT_LOG_MAX_FILES,
    maxTotalBytes: DEFAULT_LOG_MAX_TOTAL_BYTES,
    logger: serverInstance.app.log,
  });

  // Enrich the existing `/health` liveness probe rather than adding a second
  // endpoint; the UI status bar reads its `vitals` field.
  setHealthReporter(() => heartbeat.getState());

  // Pathname + protocol detection MUST run before auth so 401/403 can use
  // the client protocol error envelope.
  serverInstance.addHook("onRequest", async (req: any, reply: any) => {
    const url = new URL(`http://127.0.0.1${req.url}`);
    req.pathname = url.pathname;
    detectClientProtocol(req);
    // Preset namespace for any registered client protocol, not only /v1/messages.
    const presetMatch = req.pathname.match(/^\/preset\/([^/]+)\//);
    if (presetMatch) {
      req.preset = presetMatch[1];
    }
    // Only routed LLM traffic counts; UI/status polling would drown the report.
    if (req.protocolMatch) {
      heartbeat.trackRequest(req, reply);
    }
  });

  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    return new Promise<void>((resolve, reject) => {
      const done = (err?: Error) => {
        if (err) reject(err);
        else resolve();
      };
      apiKeyAuth(config)(req, reply, done).catch(reject);
    });
  });

  serverInstance.addHook("preHandler", async (req: any, reply: any) => {
    if (req.pathname.endsWith("/v1/messages")) {
      const useAgents = []

      for (const agent of agentsManager.getAllAgents()) {
        if (agent.shouldHandle(req, config)) {
          // Set agent identifier
          useAgents.push(agent.name)

          // change request body
          agent.reqHandler(req, config);

          // append agent tools
          if (agent.tools.size) {
            if (!req.body?.tools?.length) {
              req.body.tools = []
            }
            req.body.tools.unshift(...Array.from(agent.tools.values()).map(item => {
              return {
                name: item.name,
                description: item.description,
                input_schema: item.input_schema
              }
            }))
          }
        }
      }

      if (useAgents.length) {
        req.agents = useAgents;
      }
    }
  });
  serverInstance.addHook("onError", async (request: any, reply: any, error: any) => {
    event.emit('onError', request, reply, error);
  })
  serverInstance.addHook("onSend", (req: any, reply: any, payload: any, done: any) => {
    if (req.sessionId && req.pathname.endsWith("/v1/messages")) {
      if (payload instanceof ReadableStream) {
        if (req.agents) {
          const abortController = new AbortController();
          const eventStream = payload.pipeThrough(new SSEParserTransform())
          let currentAgent: undefined | IAgent;
          let currentToolIndex = -1
          let currentToolName = ''
          let currentToolArgs = ''
          let currentToolId = ''
          const toolMessages: any[] = []
          const assistantMessages: any[] = []
          // Store Anthropic format message body, distinguishing text and tool types
          return done(null, rewriteStream(eventStream, async (data, controller) => {
            try {
              // Detect tool call start
              if (data.event === 'content_block_start' && data?.data?.content_block?.name) {
                const agent = req.agents.find((name: string) => agentsManager.getAgent(name)?.tools.get(data.data.content_block.name))
                if (agent) {
                  currentAgent = agentsManager.getAgent(agent)
                  currentToolIndex = data.data.index
                  currentToolName = data.data.content_block.name
                  currentToolId = data.data.content_block.id
                  return undefined;
                }
              }

              // Collect tool arguments
              if (currentToolIndex > -1 && data.data.index === currentToolIndex && data.data?.delta?.type === 'input_json_delta') {
                currentToolArgs += data.data?.delta?.partial_json;
                return undefined;
              }

              // Tool call completed, handle agent invocation
              if (currentToolIndex > -1 && data.data.index === currentToolIndex && data.data.type === 'content_block_stop') {
                try {
                  const args = JSON5.parse(currentToolArgs);
                  assistantMessages.push({
                    type: "tool_use",
                    id: currentToolId,
                    name: currentToolName,
                    input: args
                  })
                  const toolResult = await currentAgent?.tools.get(currentToolName)?.handler(args, {
                    req,
                    config
                  });
                  toolMessages.push({
                    "tool_use_id": currentToolId,
                    "type": "tool_result",
                    "content": toolResult
                  })
                  currentAgent = undefined
                  currentToolIndex = -1
                  currentToolName = ''
                  currentToolArgs = ''
                  currentToolId = ''
                } catch (e) {
                  serverInstance.app.log.error({ err: e }, "Agent tool execution error");
                }
                return undefined;
              }

              if (data.event === 'message_delta' && toolMessages.length) {
                req.body.messages.push({
                  role: 'assistant',
                  content: assistantMessages
                })
                req.body.messages.push({
                  role: 'user',
                  content: toolMessages
                })
                const response = await fetch(`http://127.0.0.1:${config.PORT || 3456}/v1/messages`, {
                  method: "POST",
                  headers: {
                    'x-api-key': config.APIKEY,
                    'content-type': 'application/json',
                  },
                  body: JSON.stringify(req.body),
                })
                if (!response.ok) {
                  return undefined;
                }
                const stream = response.body!.pipeThrough(new SSEParserTransform() as any)
                const reader = stream.getReader()
                while (true) {
                  try {
                    const {value, done} = await reader.read();
                    if (done) {
                      break;
                    }
                    const eventData = value as any;
                    if (['message_start', 'message_stop'].includes(eventData.event)) {
                      continue
                    }

                    // Check if stream is still writable
                    if (!controller.desiredSize) {
                      break;
                    }

                    controller.enqueue(eventData)
                  }catch (readError: any) {
                    if (readError.name === 'AbortError' || readError.code === 'ERR_STREAM_PREMATURE_CLOSE') {
                      abortController.abort(); // Abort all related operations
                      break;
                    }
                    throw readError;
                  }

                }
                return undefined
              }
              return data
            }catch (error: any) {
              // Premature close is expected on client disconnect.
              if (error.code === 'ERR_STREAM_PREMATURE_CLOSE' || error.name === 'AbortError') {
                abortController.abort();
                return undefined;
              }

              serverInstance.app.log.error('Unexpected error in stream processing:', error);

              // Re-throw other errors
              throw error;
            }
          }).pipeThrough(new SSESerializerTransform()))
        }

        // Nonblocking usage tap: forward bytes to the client immediately and
        // parse usage on a side branch whose writable has infinite HWM so
        // tee()-coupled backpressure cannot stall the client stream.
        const usageTunnel = new TransformStream<Uint8Array, Uint8Array>(
          undefined,
          { highWaterMark: Infinity },
          { highWaterMark: 1 }
        );
        const usageWriter = usageTunnel.writable.getWriter();
        void consumeMessageDeltaUsage(
          usageTunnel.readable,
          req.sessionId,
          heartbeat,
          serverInstance.app.log
        );

        let usageAlive = true;
        const abandonUsage = () => {
          if (!usageAlive) return;
          usageAlive = false;
          void usageWriter.close().catch(() => {});
        };

        const clientStream = payload.pipeThrough(
          new TransformStream<Uint8Array, Uint8Array>({
            transform(chunk, controller) {
              controller.enqueue(chunk);
              if (!usageAlive) return;
              void usageWriter.write(chunk.slice()).catch(() => {
                abandonUsage();
              });
            },
            flush() {
              abandonUsage();
            },
            cancel() {
              abandonUsage();
            },
          } as Transformer<Uint8Array, Uint8Array>)
        );
        return done(null, clientStream)
      }
      if (typeof payload === 'object' && payload !== null) {
        sessionUsageCache.put(req.sessionId, payload.usage);
        heartbeat.recordUsage(req.sessionId, payload.usage);
      }
    }
    done(null, payload)
  });
  serverInstance.addHook("onSend", async (req: any, reply: any, payload: any) => {
    event.emit('onSend', req, reply, payload);
    return payload;
  });

  // Report once the port is bound, then on every interval.
  serverInstance.addHook("onListen", async () => {
    heartbeat.start();
    logRetention.start();
  });
  serverInstance.addHook("onClose", async () => {
    heartbeat.stop();
    logRetention.stop();
    closeProxyDispatchers();
    await closeTokenCountWorkers();
  });

  // Add global error handlers to prevent the service from crashing
  process.on("uncaughtException", (err) => {
    serverInstance.app.log.error("Uncaught exception:", err);
  });

  process.on("unhandledRejection", (reason, promise) => {
    // Cursor SDK cancel/close can reject late with AbortError after CCR already
    // moved on. Log at debug so Docker/Node do not spam "You have triggered an
    // unhandledRejection" noise for expected aborts.
    const err = reason as { name?: string; code?: string; message?: string } | null;
    const isAbort =
      err?.name === "AbortError" ||
      err?.code === "ABORT_ERR" ||
      (typeof err?.message === "string" &&
        err.message.includes("This operation was aborted"));
    if (isAbort) {
      serverInstance.app.log.debug(
        { err: reason },
        "Unhandled AbortError rejection (expected on cancel)"
      );
      return;
    }
    serverInstance.app.log.error("Unhandled rejection at:", promise, "reason:", reason);
  });

  return serverInstance;
}

async function run() {
  const server = await getServer();
  server.app.post(
    "/api/restart",
    { config: { rateLimit: { ...RATE_LIMIT_CONFIG } } },
    async () => {
      setTimeout(async () => {
        process.exit(0);
      }, 100);

      return { success: true, message: "Service restart initiated" };
    }
  );
  await server.start();
}

export { getServer };
export type { RunOptions };
export type { IAgent, ITool } from "./agents/type";
export { initDir, initConfig, readConfigFile, writeConfigFile, backupConfigFile } from "./utils";
export { pluginManager, tokenSpeedPlugin } from "@caeliq/llms";

// Start service if this file is run directly
if (require.main === module) {
  run().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
  });
}

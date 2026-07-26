import {
  Cursor,
  type ModelListItem,
  type ModelSelection,
  type SDKMessage,
} from "@cursor/sdk";
import type { UnifiedChatRequest } from "@/types/llm";
import { resolveCursorApiKey } from "@/utils/cursor-auth";
import { accumulateChatCompletion, createSseHelpers } from "./events-to-sse";
import {
  extractTrailingToolResults,
  progressOnlyContinuationPrompt,
  shouldContinueProgressOnlyTurn,
  toSdkPrompt,
} from "./prompt";
import {
  buildSessionKey,
  cancelActiveRun,
  globalSessionManager,
  touchSession,
  withSessionSendLock,
  type CursorSdkSession,
} from "./session";
import {
  isCursorAgentBusyError,
  isCursorSendPoisonError,
  sendCursorPrompt,
} from "./send";
import {
  DEFAULT_CURSOR_MODE,
  coerceThinkingText,
  extractEffort,
  type CursorSdkMode,
} from "./shared";
import { toCustomTools } from "./tools";
import {
  cacheReadFromSdkDelta,
  estimateRequestPromptTokens,
  requestUsageFromEstimate,
  usageFromSdk,
  type OpenAiUsage,
} from "./usage";

export interface CursorSdkRunnerOptions {
  cursorMode?: CursorSdkMode;
  cursorCwd?: string;
  /** Opt-in only; ignored/forced off in Docker. Default false. */
  sandboxEnabled?: boolean;
  abortSignal?: AbortSignal;
  logger?: any;
}

/** Process-local only — model IDs themselves live in Providers[].models via `ccr model get`. */
const MODEL_CATALOG_TTL_MS = 15 * 60 * 1000;

type InMemoryModelCatalog = {
  fetchedAt: number;
  apiKeyFingerprint: string;
  models: ModelListItem[];
};

let modelCatalog: InMemoryModelCatalog | null = null;

function apiKeyFingerprint(apiKey: string): string {
  return `${apiKey.length}:${apiKey.slice(0, 8)}:${apiKey.slice(-4)}`;
}

async function getModelCatalog(apiKey: string): Promise<ModelListItem[]> {
  const fingerprint = apiKeyFingerprint(apiKey);
  if (
    modelCatalog &&
    modelCatalog.apiKeyFingerprint === fingerprint &&
    Date.now() - modelCatalog.fetchedAt < MODEL_CATALOG_TTL_MS
  ) {
    return modelCatalog.models;
  }

  const models = (await Cursor.models.list({ apiKey })) || [];
  modelCatalog = {
    fetchedAt: Date.now(),
    apiKeyFingerprint: fingerprint,
    models,
  };
  return models;
}

/**
 * Map a configured CCR model id (+ optional effort) onto SDK ModelSelection.
 * Catalog fetch is best-effort for variant params only — never writes a side cache file.
 * Canonical model ids are discovered with `ccr model get <cursor-provider>`.
 */
async function resolveModelSelection(
  apiKey: string,
  modelId: string,
  effort?: string
): Promise<ModelSelection> {
  let models: ModelListItem[] = [];
  try {
    models = await getModelCatalog(apiKey);
  } catch {
    return { id: modelId };
  }

  const found = models.find(
    (m) =>
      m.id === modelId ||
      m.displayName === modelId ||
      (Array.isArray(m.aliases) && m.aliases.includes(modelId))
  );
  if (!found) return { id: modelId };

  if (effort && Array.isArray(found.variants) && found.variants.length) {
    const match =
      found.variants.find((v) =>
        v.params?.some(
          (p) =>
            /effort|reasoning/i.test(p.id) &&
            String(p.value).toLowerCase() === String(effort).toLowerCase()
        )
      ) ||
      found.variants.find((v) =>
        (v.displayName || "")
          .toLowerCase()
          .includes(String(effort).toLowerCase())
      );
    if (match) {
      return { id: found.id, params: match.params };
    }
  }

  const defaultVariant =
    found.variants?.find((v) => v.isDefault) || found.variants?.[0];
  if (defaultVariant?.params?.length) {
    return { id: found.id, params: defaultVariant.params };
  }
  return { id: found.id };
}

function firstSystemAndUserText(request: UnifiedChatRequest): string {
  const parts: string[] = [];
  for (const msg of request.messages || []) {
    if (msg.role === "system" || msg.role === "user") {
      if (typeof msg.content === "string") parts.push(msg.content);
      else if (Array.isArray(msg.content)) {
        parts.push(
          msg.content
            .map((p: any) => (typeof p === "string" ? p : p?.text || ""))
            .join("")
        );
      }
      if (msg.role === "user") break;
    }
  }
  return parts.join("\n");
}

async function* streamSessionEvents(
  session: CursorSdkSession,
  mode: CursorSdkMode,
  runToken: symbol,
  abortSignal?: AbortSignal
): AsyncGenerator<
  | { kind: "sdk"; message: SDKMessage; source: "stream" | "delta" }
  | { kind: "host_tool"; tool: { id: string; name: string; args: Record<string, unknown> } }
  | { kind: "end" }
> {
  const iterator = session.streamIterator;
  if (!iterator) {
    yield { kind: "end" };
    return;
  }

  let pendingNext = iterator.next();
  let emitWait = session.waitForEmit().then(() => "emit" as const);
  let sdkMessageWait = session.waitForSdkMessage().then(
    () => "sdk_message" as const
  );

  const shiftSdkDelta = () => {
    const idx = session.pendingSdkMessages.findIndex(
      (entry) => entry.runToken === runToken
    );
    if (idx === -1) return undefined;
    const [entry] = session.pendingSdkMessages.splice(idx, 1);
    return entry;
  };

  while (true) {
    if (abortSignal?.aborted || session.activeRunToken !== runToken) {
      yield { kind: "end" };
      return;
    }

    const queuedSdkDelta = shiftSdkDelta();
    if (queuedSdkDelta) {
      yield {
        kind: "sdk",
        message: queuedSdkDelta.message,
        source: queuedSdkDelta.source,
      };
      continue;
    }

    const raced = await Promise.race([
      sdkMessageWait.then((v) => ({ type: v })),
      pendingNext.then((r) => ({ type: "msg" as const, r })),
      emitWait.then((v) => ({ type: v })),
    ]);

    if (raced.type === "emit") {
      while (
        session.pendingEmit.length &&
        session.pendingEmit[0]?.runToken === runToken
      ) {
        const tool = session.pendingEmit.shift()!;
        yield { kind: "host_tool", tool };
      }
      emitWait = session.waitForEmit().then(() => "emit" as const);
      continue;
    }

    if (raced.type === "sdk_message") {
      while (true) {
        const entry = shiftSdkDelta();
        if (!entry) break;
        yield { kind: "sdk", message: entry.message, source: entry.source };
      }
      sdkMessageWait = session.waitForSdkMessage().then(
        () => "sdk_message" as const
      );
      continue;
    }

    const { r } = raced;
    if (r.done) {
      yield { kind: "end" };
      return;
    }

    const message = r.value as SDKMessage;
    if (abortSignal?.aborted || session.activeRunToken !== runToken) {
      yield { kind: "end" };
      return;
    }
    if (message?.type === "tool_call" && mode === "bridge") {
      session.metrics.builtinToolCallsSeen += 1;
      // Do not forward Cursor built-ins as Claude Code tools.
    }
    yield { kind: "sdk", message, source: "stream" };
    pendingNext = iterator.next();
  }
}

function sdkMessageFromDelta(
  session: CursorSdkSession,
  update: any
): SDKMessage | undefined {
  if (!update || update.type !== "thinking-delta") return undefined;
  const text = coerceThinkingText(update.text);
  if (!text) return undefined;
  return {
    type: "thinking",
    agent_id: session.agentId,
    run_id: session.run?.id || "",
    text,
  } as SDKMessage;
}

export async function runCursor(
  request: UnifiedChatRequest,
  provider: any,
  context: any,
  options: CursorSdkRunnerOptions = {}
): Promise<Response> {
  const logger = options.logger || provider?.logger;
  if (logger) globalSessionManager.setLogger(logger);
  const mode: CursorSdkMode = options.cursorMode || DEFAULT_CURSOR_MODE;
  const wantsStream = request.stream === true;
  const modelId = request.model || "composer-2";

  let apiKey: string;
  try {
    apiKey = resolveCursorApiKey(provider);
  } catch (err: any) {
    throw Object.assign(err instanceof Error ? err : new Error(String(err)), {
      statusCode: err?.statusCode || 401,
      code: "provider_response_error",
      type: "api_error",
    });
  }

  const headerSession =
    context?.req?.headers?.["x-ccr-cursor-session"] ||
    context?.req?.headers?.["X-Ccr-Cursor-Session"];
  const sessionKey = buildSessionKey({
    headerSession: typeof headerSession === "string" ? headerSession : undefined,
    metadataUserId: (request as any)?.metadata?.user_id,
    model: modelId,
    systemAndFirstUser: firstSystemAndUserText(request),
  });

  const effort = extractEffort(request);
  const model = await resolveModelSelection(apiKey, modelId, effort);

  const sessionOptions = {
    key: sessionKey,
    apiKey,
    model,
    mode,
    cursorCwd: options.cursorCwd,
    sandboxEnabled: options.sandboxEnabled,
  };

  let session = await globalSessionManager.getOrCreate(sessionOptions);

  const toolResults = extractTrailingToolResults(request);
  const hadParked = session.parked.length > 0;
  if (toolResults.length && hadParked) {
    globalSessionManager.resolveParkedTools(session, toolResults);
  }

  // Resume the parked customTools.execute stream only when we still have an
  // iterator AND we just (or previously) parked tools. Otherwise this is a
  // new/follow-up turn — any leftover active run must be cancelled first or
  // Cursor throws "Agent … already has active run" (compact/retry/disconnect).
  const shouldSendNewPrompt = !session.streamIterator || !hadParked;
  // Dead-run recovery: iterator is gone but the Cursor agent session still holds
  // prior turns. Prefer a slim follow-up (trailing tool results / last user turn)
  // over re-embedding the full CCR transcript as plain text.
  const followUpOnly = session.hasSentPrompt;
  let runToken = session.activeRunToken;

  const clearQueuedSdkMessages = (
    targetSession: CursorSdkSession,
    targetRunToken: symbol
  ) => {
    targetSession.pendingSdkMessages = targetSession.pendingSdkMessages.filter(
      (entry) => entry.runToken !== targetRunToken
    );
    targetSession.notifySdkMessage();
  };

  const sdkSendOptionsForSession = (
    targetSession: CursorSdkSession,
    targetRunToken: symbol
  ) => {
    const customTools =
      mode === "bridge" ? toCustomTools(request, targetSession) : undefined;
    return {
      model,
      mode: mode === "plan" ? ("plan" as const) : ("agent" as const),
      local: customTools ? { customTools } : undefined,
      onDelta: ({ update }: { update: any }) => {
        const message = sdkMessageFromDelta(targetSession, update);
        if (message) {
          targetSession.enqueueSdkMessage(message, targetRunToken);
        }
      },
    };
  };

  const toProviderError = (err: any) =>
    Object.assign(new Error(`Cursor SDK error: ${err?.message || err}`), {
      statusCode: 502,
      code: "provider_response_error",
      type: "api_error",
    });

  const shouldReplayWithFreshSession = (err: any, targetSession: CursorSdkSession) =>
    targetSession.hasSentPrompt &&
    !options.abortSignal?.aborted &&
    (err?.retryFreshCursorSession === true ||
      isCursorAgentBusyError(err) ||
      isCursorSendPoisonError(err));

  const startNewPrompt = async (
    targetSession: CursorSdkSession,
    followUpOnlyForPrompt: boolean
  ): Promise<symbol> => {
    let nextRunToken: symbol | undefined;
    await withSessionSendLock(targetSession, async () => {
      // Re-check under the lock: a concurrent request may have started a run.
      if (targetSession.run || targetSession.streamIterator) {
        const cancelResult = await cancelActiveRun(targetSession, {
          rejectParked: true,
          reason: "cursor-sdk superseded by new prompt",
          timeoutMs: 2_000,
          poisonOnFailure: true,
        });
        if (cancelResult.failed) {
          globalSessionManager.invalidate(
            targetSession,
            "cursor-sdk active run did not finish cancellation before new prompt"
          );
          throw Object.assign(
            new Error("cursor-sdk active run did not finish cancellation"),
            { retryFreshCursorSession: true }
          );
        }
      }

      const prompt = toSdkPrompt(request, {
        mode,
        workspaceDir: targetSession.workspaceDir,
        followUpOnly: followUpOnlyForPrompt,
      });

      nextRunToken = Symbol("cursor-sdk-run");
      targetSession.activeRunToken = nextRunToken;

      try {
        const run = await sendCursorPrompt(
          targetSession,
          prompt,
          sdkSendOptionsForSession(targetSession, nextRunToken),
          {
            abortSignal: options.abortSignal,
            logger,
          }
        );
        targetSession.run = run;
        targetSession.activeRunToken = nextRunToken;
        targetSession.streamIterator = run.stream()[Symbol.asyncIterator]();
        targetSession.hasSentPrompt = true;
      } catch (err: any) {
        if (nextRunToken && targetSession.activeRunToken === nextRunToken) {
          targetSession.activeRunToken = undefined;
          clearQueuedSdkMessages(targetSession, nextRunToken);
        }
        if (options.abortSignal?.aborted || isCursorSendPoisonError(err)) {
          globalSessionManager.invalidate(
            targetSession,
            options.abortSignal?.aborted
              ? "cursor-sdk send aborted"
              : "cursor-sdk send left session state unsafe"
          );
        }
        logger?.error?.({ err }, "cursor-sdk agent.send failed");
        throw err;
      }
    });
    if (!nextRunToken) {
      throw new Error("cursor-sdk agent.send did not produce a run token");
    }
    return nextRunToken;
  };

  if (shouldSendNewPrompt) {
    if (followUpOnly && toolResults.length && !session.streamIterator) {
      logger?.warn?.(
        {
          toolResultCount: toolResults.length,
          hadParked,
        },
        "cursor-sdk dead-run recovery: sending follow-up with tool results"
      );
    }
    try {
      runToken = await startNewPrompt(session, followUpOnly);
    } catch (err: any) {
      if (!shouldReplayWithFreshSession(err, session)) {
        throw toProviderError(err);
      }
      logger?.warn?.(
        {
          err,
          sessionKey: session.key,
          agentId: session.agentId,
        },
        "cursor-sdk retrying failed resumed send with fresh session"
      );
      globalSessionManager.invalidate(
        session,
        "cursor-sdk resumed send failed before streaming"
      );
      session = await globalSessionManager.getOrCreate(sessionOptions);
      try {
        runToken = await startNewPrompt(session, false);
      } catch (retryErr: any) {
        throw toProviderError(retryErr);
      }
    }
  }

  if (!runToken) {
    runToken = Symbol("cursor-sdk-run");
    session.activeRunToken = runToken;
  }

  const helpers = createSseHelpers(modelId, new TextEncoder());
  const collected: Array<Record<string, unknown>> = [];
  // Host-facing usage must be per CCR request. Cursor SDK usage events / run.usage
  // are cumulative across the held-open agent session and arrive only at true turn
  // end — after we already park for Claude Code tools — so they cannot drive
  // Claude Code context accounting (same fix as cursor-opencode-provider).
  const promptTokens = estimateRequestPromptTokens(request);
  let outputChars = 0;
  let sdkUsageRaw: OpenAiUsage | undefined;

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const enqueue = (chunk: Record<string, unknown>) => {
        collected.push(chunk);
        if (wantsStream) {
          controller.enqueue(helpers.encode(chunk));
        }
      };

      const noteOutput = (text: string | undefined | null) => {
        if (typeof text === "string" && text.length) outputChars += text.length;
      };

      const finishUsage = (): OpenAiUsage => {
        const cacheReadTokens = cacheReadFromSdkDelta(
          sdkUsageRaw,
          session.lastSdkUsageRaw,
          promptTokens
        );
        const usage = requestUsageFromEstimate(
          promptTokens,
          outputChars,
          cacheReadTokens
        );
        if (sdkUsageRaw) session.lastSdkUsageRaw = sdkUsageRaw;
        return usage;
      };

      const emitFinish = (reason: "stop" | "tool_calls") => {
        const usage = finishUsage();
        logger?.debug?.(
          {
            reason,
            promptTokens: usage.prompt_tokens,
            completionTokens: usage.completion_tokens,
            outputChars,
            sdkUsageRaw,
            runUsage: session.run?.usage,
          },
          "cursor-sdk finish usage (request estimate; raw SDK is diagnostic only)"
        );
        enqueue(helpers.finish(reason, usage));
      };

      try {
        let emittedHostTools = 0;
        let sawAssistantText = false;
        // Anthropic/Claude Code require thinking blocks before text/tool_use.
        // Cursor can emit more thinking after assistant text; dropping those
        // late deltas avoids "Content block is not a thinking block".
        let allowThinking = true;
        let emittedThinking = false;
        let thinkingSigned = false;
        let thinkingSource: "stream" | "delta" | undefined;
        let currentRunAssistantText = "";
        let progressContinuationAttempts = 0;

        const flushThinkingSignature = () => {
          if (!emittedThinking || thinkingSigned) return;
          enqueue(helpers.thinkingSignature());
          thinkingSigned = true;
          allowThinking = false;
        };

        while (true) {
          let continuedProgressTurn = false;
          for await (const event of streamSessionEvents(
            session,
            mode,
            runToken!,
            options.abortSignal
          )) {
            // Keep the session out of idle/LRU eviction while SSE is live.
            touchSession(session);
            if (event.kind === "host_tool") {
              if (mode !== "bridge") continue;
              flushThinkingSignature();
              allowThinking = false;
              noteOutput(event.tool.name);
              noteOutput(JSON.stringify(event.tool.args ?? {}));
              enqueue(helpers.toolCall(event.tool, emittedHostTools));
              emittedHostTools += 1;

              // Emit all currently pending host tools, then finish this CCR turn.
              while (session.pendingEmit.length) {
                const tool = session.pendingEmit.shift()!;
                noteOutput(tool.name);
                noteOutput(JSON.stringify(tool.args ?? {}));
                enqueue(helpers.toolCall(tool, emittedHostTools));
                emittedHostTools += 1;
              }
              emitFinish("tool_calls");
              if (wantsStream) controller.enqueue(helpers.encodeDone());
              controller.close();
              return;
            }

            if (event.kind === "end") {
              if (session.activeRunToken === runToken) {
                session.streamIterator = undefined;
                session.run = undefined;
                session.activeRunToken = undefined;
              }

              if (
                shouldContinueProgressOnlyTurn({
                  mode,
                  assistantText: currentRunAssistantText,
                  emittedHostTools,
                  continuationAttempts: progressContinuationAttempts,
                })
              ) {
                progressContinuationAttempts += 1;
                logger?.warn?.(
                  {
                    assistantText: currentRunAssistantText,
                    attempt: progressContinuationAttempts,
                  },
                  "cursor-sdk continuing progress-only terminal turn"
                );
                const continuationRunToken = Symbol("cursor-sdk-run");
                session.activeRunToken = continuationRunToken;
                let run;
                try {
                  run = await withSessionSendLock(session, () =>
                    sendCursorPrompt(
                      session,
                      progressOnlyContinuationPrompt(),
                      sdkSendOptionsForSession(session, continuationRunToken),
                      {
                        abortSignal: options.abortSignal,
                        logger,
                      }
                    )
                  );
                } catch (err) {
                  if (session.activeRunToken === continuationRunToken) {
                    session.activeRunToken = undefined;
                    clearQueuedSdkMessages(session, continuationRunToken);
                  }
                  throw err;
                }
                runToken = continuationRunToken;
                session.run = run;
                session.activeRunToken = runToken;
                session.streamIterator = run.stream()[Symbol.asyncIterator]();
                session.hasSentPrompt = true;
                currentRunAssistantText = "";
                continuedProgressTurn = true;
                break;
              }

              flushThinkingSignature();
              emitFinish("stop");
              if (wantsStream) controller.enqueue(helpers.encodeDone());
              controller.close();
              return;
            }

            const message = event.message;
            if (!message) continue;

            if (message.type === "assistant") {
              for (const block of message.message?.content || []) {
                if (block.type === "text" && block.text) {
                  flushThinkingSignature();
                  allowThinking = false;
                  sawAssistantText = true;
                  currentRunAssistantText += block.text;
                  noteOutput(block.text);
                  enqueue(helpers.content(block.text));
                }
                // tool_use blocks for custom tools are handled via parkHostTool emit path
              }
            } else if (message.type === "thinking") {
              if (!allowThinking) continue;
              const source = event.source || "stream";
              if (thinkingSource && thinkingSource !== source) continue;
              thinkingSource = source;
              const text = coerceThinkingText((message as any).text);
              if (text) {
                emittedThinking = true;
                noteOutput(text);
                enqueue(helpers.thinking(text));
              }
            } else if (message.type === "usage") {
              // Raw SDK usage is cumulative; finishUsage maps only its cache ratio.
              sdkUsageRaw = usageFromSdk(message);
            } else if (message.type === "status") {
              if (message.status === "ERROR" || message.status === "CANCELLED") {
                throw Object.assign(
                  new Error(
                    message.message ||
                      `Cursor run ${message.status.toLowerCase()}`
                  ),
                  {
                    statusCode: 502,
                    code: "provider_response_error",
                    type: "api_error",
                  }
                );
              }
              // Drain the SDK iterator after FINISHED. The SDK emits this status
              // before it persists terminal metadata and closes the event buffer;
              // the `end` branch above is the safe point for completion/recovery.
            } else if (message.type === "tool_call" && mode !== "bridge") {
              // plan/agent: ignore or narrate lightly in agent mode
              if (mode === "agent" && message.name) {
                flushThinkingSignature();
                allowThinking = false;
                const narrate = `\n[cursor tool ${message.status || "running"}: ${message.name}]\n`;
                noteOutput(narrate);
                enqueue(helpers.content(narrate));
              }
            }
          }
          if (!continuedProgressTurn) break;
        }

        if (!sawAssistantText && !emittedHostTools) {
          flushThinkingSignature();
          emitFinish("stop");
        }
        if (wantsStream) controller.enqueue(helpers.encodeDone());
        controller.close();
      } catch (err: any) {
        logger?.error?.(
          {
            err,
            metrics: session.metrics,
          },
          "cursor-sdk stream failed"
        );
        // Ensure a failed/aborted stream cannot block the next agent.send.
        void cancelActiveRun(session, {
          rejectParked: true,
          reason: "cursor-sdk stream failed",
          timeoutMs: 2_000,
          onlyRunToken: runToken,
          poisonOnFailure: true,
        }).then((result) => {
          if (result.skipped) return;
          globalSessionManager.invalidate(
            session,
            "cursor-sdk stream failed before terminal state"
          );
        });
        try {
          controller.error(err);
        } catch {
          // already closed
        }
      }
    },
    cancel() {
      void cancelActiveRun(session, {
        rejectParked: true,
        reason: "cursor-sdk client cancelled stream",
        timeoutMs: 2_000,
        onlyRunToken: runToken,
        poisonOnFailure: true,
      }).then((result) => {
        if (result.skipped) return;
        globalSessionManager.invalidate(
          session,
          "cursor-sdk client cancelled stream"
        );
      });
    },
  });

  if (!wantsStream) {
    // Materialize the stream into a single chat.completion JSON.
    const reader = stream.getReader();
    while (true) {
      const { done } = await reader.read();
      if (done) break;
    }
    const completion = accumulateChatCompletion(modelId, collected);
    return new Response(JSON.stringify(completion), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  });
}

import {
  Cursor,
  type ModelListItem,
  type ModelSelection,
  type SDKMessage,
} from "@cursor/sdk";
import type { UnifiedChatRequest, UnifiedMessage } from "@/types/llm";
import type { UnifiedTurnIntent } from "@/types/turn-intent";
import { resolveCursorApiKey } from "@/utils/cursor-auth";
import { accumulateChatCompletion, createSseHelpers } from "./events-to-sse";
import {
  analyzeTrailingCursorToolTurn,
  progressOnlyContinuationPrompt,
  shouldContinueProgressOnlyTurn,
  toSdkPrompt,
  type TrailingCursorToolTurn,
} from "./prompt";
import { extractHostEnvironment } from "./host-env";
import {
  buildSessionKey,
  cancelActiveRun,
  globalSessionManager,
  markSessionPoisoned,
  refreshWorkspaceGuidance,
  shouldEnableCursorSandbox,
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
  contentToText,
  coerceThinkingText,
  extractEffort,
  hashSessionFingerprint,
  type CursorSdkMode,
} from "./shared";
import { createTurnToolMetrics, toCustomTools } from "./tools";
import {
  cacheReadFromSdkDelta,
  estimateRequestPromptTokens,
  requestUsageFromEstimate,
  usageFromSdk,
  type OpenAiUsage,
} from "./usage";
import {
  createCursorCompatibilityStamp,
  createCursorTranscriptCommit,
  fingerprintCursorTurn,
  getStrictCursorTranscriptSuffix,
} from "./turn-identity";
import { planCursorLifecycle } from "./lifecycle-planner";
import {
  globalCursorTurnRegistry,
  type CursorTurnLease,
} from "./turn-output";

export interface CursorSdkRunnerOptions {
  cursorMode?: CursorSdkMode;
  cursorCwd?: string;
  /** Opt-in only; ignored/forced off in Docker. Default false. */
  sandboxEnabled?: boolean;
  abortSignal?: AbortSignal;
  logger?: any;
  /** Request-local protocol semantics; never serialized into the provider body. */
  turnIntent?: UnifiedTurnIntent;
  /** Stable source conversation identity recovered by the protocol adapter. */
  sourceSessionIdentity?: string;
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
  return hashSessionFingerprint([apiKey]);
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

function isSupportedIdleTranscriptSuffix(
  suffix: readonly UnifiedMessage[]
): boolean {
  return suffix.length === 1 && suffix[0]?.role === "user";
}

function isSupportedParkedTranscriptSuffix(
  suffix: readonly UnifiedMessage[],
  trailingTurn: TrailingCursorToolTurn,
  turnIntent?: UnifiedTurnIntent
): boolean {
  if (trailingTurn.hasTrailingUserInput) return false;
  const hasSyntheticTrailingUser =
    turnIntent?.interruption === "synthetic_client_interrupt" &&
    turnIntent.steering === "none";
  const expectedLength =
    trailingTurn.toolResults.length + (hasSyntheticTrailingUser ? 1 : 0);
  if (suffix.length !== expectedLength) return false;

  for (let index = 0; index < trailingTurn.toolResults.length; index += 1) {
    const message = suffix[index];
    const expected = trailingTurn.toolResults[index];
    if (
      message?.role !== "tool" ||
      message.tool_call_id !== expected.toolCallId ||
      contentToText(message.content) !== expected.content
    ) {
      return false;
    }
  }

  return (
    !hasSyntheticTrailingUser ||
    suffix[suffix.length - 1]?.role === "user"
  );
}

function throwIfCursorProducerAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw Object.assign(
    new Error("cursor-sdk response producer aborted before lifecycle mutation"),
    { name: "AbortError" }
  );
}

export async function* streamSessionEvents(
  session: CursorSdkSession,
  mode: CursorSdkMode,
  runToken: symbol,
  abortSignal?: AbortSignal
): AsyncGenerator<
  | { kind: "sdk"; message: SDKMessage; source: "stream" | "delta" }
  | { kind: "host_tool"; tool: { id: string; name: string; args: Record<string, unknown> } }
  | { kind: "end"; aborted?: boolean }
> {
  const iterator = session.streamIterator;
  if (!iterator) {
    yield { kind: "end" };
    return;
  }

  // A pending next() is rejected by the SDK's internal AbortController when
  // run.cancel() runs. This generator can return before racing that promise
  // (abort / run-token change below), so keep a handler attached from creation
  // instead of relying on a later race or on cancelActiveRun's catch.
  const trackNext = (next: Promise<IteratorResult<any>>) => {
    void next.catch(() => undefined);
    session.streamNext = next;
    session.streamNextRunToken = runToken;
    return next;
  };

  let pendingNext = trackNext(
    session.streamNextRunToken === runToken && session.streamNext
      ? session.streamNext
      : iterator.next()
  );
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
      yield { kind: "end", aborted: abortSignal?.aborted === true };
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
    if (session.streamNext === pendingNext) {
      session.streamNext = undefined;
      session.streamNextRunToken = undefined;
    }
    if (r.done) {
      yield { kind: "end" };
      return;
    }

    const message = r.value as SDKMessage;
    if (abortSignal?.aborted || session.activeRunToken !== runToken) {
      yield { kind: "end", aborted: abortSignal?.aborted === true };
      return;
    }
    if (message?.type === "tool_call" && mode === "bridge") {
      session.metrics.builtinToolCallsSeen += 1;
      // Do not forward Cursor built-ins as Claude Code tools.
    }
    yield { kind: "sdk", message, source: "stream" };
    pendingNext = trackNext(iterator.next());
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

/**
 * Remove an interrupted session from reuse before waiting on SDK cleanup.
 *
 * `cancelActiveRun` clears local handles before its awaited iterator/run
 * cancellation completes. Detaching and poisoning first prevents a reconnect
 * from observing that temporary handle-free state and sending on the still-busy
 * SDK agent, while preserving the old handles for the retirement task itself.
 */
async function retireCursorSession(
  session: CursorSdkSession,
  options: {
    reason: string;
    onlyRunToken?: symbol;
  }
): Promise<boolean> {
  if (
    options.onlyRunToken &&
    session.activeRunToken !== options.onlyRunToken
  ) {
    return false;
  }

  return globalSessionManager.retireSession(
    session,
    options.reason,
    () =>
      withSessionSendLock(session, async () => {
        await cancelActiveRun(session, {
          rejectParked: true,
          reason: options.reason,
          timeoutMs: 2_000,
          poisonOnFailure: true,
        });
        globalSessionManager.invalidate(session, options.reason);
        return true;
      })
  );
}

type ResolvedCursorRequest = {
  apiKey: string;
  hostEnv: ReturnType<typeof extractHostEnvironment>;
  mode: CursorSdkMode;
  modelId: string;
  sessionKey: string;
  turnIntent?: UnifiedTurnIntent;
};

function resolveCursorRequest(
  request: UnifiedChatRequest,
  provider: any,
  context: any,
  options: CursorSdkRunnerOptions
): ResolvedCursorRequest {
  const mode: CursorSdkMode = options.cursorMode || DEFAULT_CURSOR_MODE;
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
  const sourceSessionIdentity =
    options.sourceSessionIdentity ||
    context?.unifiedRequest?.sourceSessionIdentity ||
    (request as any)?.metadata?.user_id;
  const sessionKey = buildSessionKey({
    headerSession:
      typeof headerSession === "string" ? headerSession : undefined,
    metadataUserId:
      typeof sourceSessionIdentity === "string"
        ? sourceSessionIdentity
        : undefined,
    model: modelId,
    systemAndFirstUser: firstSystemAndUserText(request),
  });

  return {
    apiKey,
    hostEnv: extractHostEnvironment(request),
    mode,
    modelId,
    sessionKey,
    turnIntent:
      options.turnIntent || context?.unifiedRequest?.turnIntent,
  };
}

/**
 * Coordinate identical host retries before entering the session lifecycle.
 *
 * The shared turn owns the one response producer. Every caller receives an
 * independent bounded replay stream, so no two HTTP responses can consume the
 * same Cursor iterator.
 */
export async function runCursor(
  request: UnifiedChatRequest,
  provider: any,
  context: any,
  options: CursorSdkRunnerOptions = {}
): Promise<Response> {
  const resolved = resolveCursorRequest(request, provider, context, options);
  const logicalWorkspace =
    options.cursorCwd || `managed-session:${resolved.sessionKey}`;
  const admissionCompatibility = createCursorCompatibilityStamp({
    credentialFingerprint: apiKeyFingerprint(resolved.apiKey),
    guidanceFingerprint: resolved.hostEnv.fingerprint,
    mode: resolved.mode,
    model: {
      id: resolved.modelId,
      effort: extractEffort(request) || "",
    },
    sandboxEnabled: shouldEnableCursorSandbox(options.sandboxEnabled),
    tools: request.tools,
    workspaceDir: logicalWorkspace,
  });
  const fingerprint = fingerprintCursorTurn(request, {
    compatibilityStamp: admissionCompatibility,
    turnIntent: resolved.turnIntent,
  });
  const lease: CursorTurnLease = await globalCursorTurnRegistry.admit({
    sessionKey: resolved.sessionKey,
    fingerprint,
    responseKind: request.stream === true ? "stream" : "json",
    signal: options.abortSignal,
  });

  if (lease.kind !== "leader") {
    return lease.response();
  }

  try {
    const response = await runCursorOnce(
      request,
      provider,
      context,
      {
        ...options,
        // Subscriber aborts are reference-counted by the shared turn. The
        // producer is aborted only when every subscriber leaves or a different
        // logical turn supersedes it.
        abortSignal: lease.producerSignal,
        turnIntent: resolved.turnIntent,
      },
      resolved
    );
    await lease.attach(response);
  } catch (error) {
    lease.fail(error);
    throw error;
  }
  // A caller can disconnect after the shared producer is attached while an
  // identical subscriber remains. Its local response failure must not fail the
  // shared turn for the surviving subscriber.
  return lease.response();
}

async function runCursorOnce(
  request: UnifiedChatRequest,
  provider: any,
  context: any,
  options: CursorSdkRunnerOptions,
  resolved: ResolvedCursorRequest
): Promise<Response> {
  throwIfCursorProducerAborted(options.abortSignal);
  const logger = options.logger || provider?.logger;
  if (logger) globalSessionManager.setLogger(logger);
  const mode = resolved.mode;
  const wantsStream = request.stream === true;
  const modelId = resolved.modelId;
  const apiKey = resolved.apiKey;
  const sessionKey = resolved.sessionKey;

  const effort = extractEffort(request);
  const model = await resolveModelSelection(apiKey, modelId, effort);
  throwIfCursorProducerAborted(options.abortSignal);

  // Host facts are re-read every turn: the project root can change between
  // requests on the same session, and only the host knows where tools land.
  const hostEnv = resolved.hostEnv;

  const sessionOptions = {
    key: sessionKey,
    apiKey,
    model,
    mode,
    cursorCwd: options.cursorCwd,
    sandboxEnabled: options.sandboxEnabled,
    hostEnv,
  };

  throwIfCursorProducerAborted(options.abortSignal);
  let session = await globalSessionManager.getOrCreate(sessionOptions);
  throwIfCursorProducerAborted(options.abortSignal);
  const hostEnvForSession = (targetSession: CursorSdkSession) =>
    !hostEnv.known && targetSession.hostEnv?.known
      ? targetSession.hostEnv
      : hostEnv;
  const compatibilityForSession = (targetSession: CursorSdkSession) =>
    createCursorCompatibilityStamp({
      credentialFingerprint: apiKeyFingerprint(apiKey),
      guidanceFingerprint: hostEnvForSession(targetSession).fingerprint,
      mode,
      model,
      sandboxEnabled: shouldEnableCursorSandbox(options.sandboxEnabled),
      tools: request.tools,
      workspaceDir: targetSession.workspaceDir,
    });

  let promptHostEnv = hostEnvForSession(session);
  let compatibilityStamp = compatibilityForSession(session);
  const trailingTurn = analyzeTrailingCursorToolTurn(
    request,
    options.turnIntent
  );
  const toolResults = trailingTurn.toolResults;

  const runSnapshot =
    session.parked.length > 0
      ? {
          kind: "parked" as const,
          live: Boolean(
            session.run &&
              session.streamIterator &&
              session.activeRunToken &&
              !session.poisoned
          ),
          tools: session.parked,
        }
      : session.run ||
          session.streamIterator ||
          session.streamNext ||
          session.activeRunToken
        ? { kind: "active-different-turn" as const }
        : { kind: "idle" as const };
  const strictSuffix =
    session.transcriptCommit &&
    session.compatibilityStamp === compatibilityStamp
      ? getStrictCursorTranscriptSuffix(session.transcriptCommit, request)
      : undefined;
  const supportedSuffix =
    strictSuffix === undefined
      ? false
      : runSnapshot.kind === "parked"
        ? isSupportedParkedTranscriptSuffix(
            strictSuffix,
            trailingTurn,
            options.turnIntent
          )
        : runSnapshot.kind === "idle"
          ? isSupportedIdleTranscriptSuffix(strictSuffix)
          : true;
  const alignment =
    !session.hasSentPrompt || !session.transcriptCommit
      ? "unknown"
      : supportedSuffix
        ? "strict"
        : "divergent";
  let lifecyclePlan = planCursorLifecycle({
    session: {
      alignment,
      hasSentPrompt: session.hasSentPrompt,
      poisoned: session.poisoned === true,
      run: runSnapshot,
    },
    turn: {
      hasMeaningfulSteering: trailingTurn.hasTrailingUserInput,
      toolResults,
    },
  });

  if (lifecyclePlan.action === "retire-and-replay-full") {
    logger?.info?.(
      {
        sessionKey: session.key,
        agentId: session.agentId,
        alignment,
        lifecycleReason: lifecyclePlan.reason,
        toolResultCount: toolResults.length,
      },
      "cursor-sdk retiring session before full transcript replay"
    );
    await retireCursorSession(session, {
      reason: `cursor-sdk lifecycle: ${lifecyclePlan.reason}`,
      onlyRunToken: session.activeRunToken,
    });
    throwIfCursorProducerAborted(options.abortSignal);
    session = await globalSessionManager.getOrCreate(sessionOptions);
    throwIfCursorProducerAborted(options.abortSignal);
    promptHostEnv = hostEnvForSession(session);
    compatibilityStamp = compatibilityForSession(session);
    lifecyclePlan = {
      action: "send-full",
      reason: "unused-session",
    };
  }

  throwIfCursorProducerAborted(options.abortSignal);
  refreshWorkspaceGuidance(session, promptHostEnv);

  if (lifecyclePlan.action === "resume-parked") {
    throwIfCursorProducerAborted(options.abortSignal);
    const resolvedCount = globalSessionManager.resolveParkedTools(
      session,
      toolResults
    );
    if (resolvedCount !== toolResults.length || session.parked.length !== 0) {
      throw new Error(
        "cursor-sdk lifecycle changed while resolving exact parked tools"
      );
    }
  }

  const shouldSendNewPrompt = lifecyclePlan.action !== "resume-parked";
  const followUpOnly = lifecyclePlan.action === "send-incremental";
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
      mode === "bridge"
        ? toCustomTools(request, targetSession, logger, turnToolMetrics)
        : undefined;
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

  // Owned by this request, not the session: `execute` can fire before the
  // response stream exists, so it must be counted independently of both the
  // cumulative session metrics and any mid-request session replacement.
  const turnToolMetrics = createTurnToolMetrics();

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
      if (targetSession.poisoned) {
        throw Object.assign(
          new Error("cursor-sdk session is no longer safe for sends"),
          { retryFreshCursorSession: true }
        );
      }

      // Admission should have made this an idle, aligned session. Never use
      // cancellation as permission to append to an opaque Cursor checkpoint.
      if (
        targetSession.run ||
        targetSession.streamIterator ||
        targetSession.streamNext ||
        targetSession.activeRunToken ||
        targetSession.parked.length > 0
      ) {
        markSessionPoisoned(
          targetSession,
          "cursor-sdk session became active after lifecycle admission"
        );
        throw Object.assign(
          new Error("cursor-sdk session became active before prompt send"),
          { retryFreshCursorSession: true }
        );
      }

      // Stream cancellation poisons synchronously before it waits for this same
      // lock. Do not send if cancellation started while the prior run was being
      // retired.
      if (targetSession.poisoned) {
        throw Object.assign(
          new Error("cursor-sdk session was retired during cancellation"),
          { retryFreshCursorSession: true }
        );
      }

      const prompt = toSdkPrompt(request, {
        mode,
        workspaceDir: targetSession.workspaceDir,
        followUpOnly: followUpOnlyForPrompt,
        hostEnv: promptHostEnv,
        turnIntent: options.turnIntent,
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
        targetSession.streamNext = undefined;
        targetSession.streamNextRunToken = undefined;
        targetSession.hasSentPrompt = true;
        throwIfCursorProducerAborted(options.abortSignal);
      } catch (err: any) {
        if (nextRunToken && targetSession.activeRunToken === nextRunToken) {
          targetSession.activeRunToken = undefined;
          clearQueuedSdkMessages(targetSession, nextRunToken);
        }
        if (
          options.abortSignal?.aborted ||
          isCursorSendPoisonError(err) ||
          isCursorAgentBusyError(err)
        ) {
          markSessionPoisoned(
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
    try {
      runToken = await startNewPrompt(session, followUpOnly);
    } catch (err: any) {
      const shouldReplay = shouldReplayWithFreshSession(err, session);
      await retireCursorSession(session, {
        reason: "cursor-sdk prompt send failed before streaming",
      });
      if (!shouldReplay) {
        throw toProviderError(err);
      }
      throwIfCursorProducerAborted(options.abortSignal);
      logger?.warn?.(
        {
          err,
          sessionKey: session.key,
          agentId: session.agentId,
        },
        "cursor-sdk retrying failed resumed send with fresh session"
      );
      session = await globalSessionManager.getOrCreate(sessionOptions);
      throwIfCursorProducerAborted(options.abortSignal);
      promptHostEnv = hostEnvForSession(session);
      compatibilityStamp = compatibilityForSession(session);
      refreshWorkspaceGuidance(session, promptHostEnv);
      try {
        runToken = await startNewPrompt(session, false);
      } catch (retryErr: any) {
        await retireCursorSession(session, {
          reason: "cursor-sdk fresh replay send failed before streaming",
        });
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
      /**
       * Must run on every exit path. A bridge turn normally ends by emitting
       * host tool calls and returning early, so a report placed on the
       * run-completed path alone never fires for the common case.
       */
      const reportScratchViolations = () => {
        if (turnToolMetrics.scratchViolations <= 0) return;
        // Surfaces the "model thinks it is confined to its sandbox" failure
        // per model, without needing debug-level logs.
        logger?.warn?.(
          {
            sessionKey: session.key,
            model: modelId,
            scratchViolations: turnToolMetrics.scratchViolations,
            scratchCorrections: turnToolMetrics.scratchCorrections,
            metrics: session.metrics,
            hostProjectRoot: promptHostEnv.projectRoot,
          },
          "cursor-sdk turn produced scratch-workspace tool paths"
        );
      };

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

      const commitHostTranscript = () => {
        if (session.activeRunToken !== runToken) return;
        const completion = accumulateChatCompletion(modelId, collected) as any;
        const assistantMessage = completion?.choices?.[0]?.message;
        if (!assistantMessage || assistantMessage.role !== "assistant") return;
        session.transcriptCommit = createCursorTranscriptCommit(
          request,
          assistantMessage as UnifiedMessage
        );
        session.compatibilityStamp = compatibilityStamp;
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
        commitHostTranscript();
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
              if (event.aborted) {
                throw Object.assign(
                  new Error("cursor-sdk response producer aborted"),
                  { name: "AbortError" }
                );
              }
              if (
                shouldContinueProgressOnlyTurn({
                  mode,
                  assistantText: currentRunAssistantText,
                  emittedHostTools,
                  continuationAttempts: progressContinuationAttempts,
                })
              ) {
                if (session.activeRunToken === runToken) {
                  session.streamIterator = undefined;
                  session.streamNext = undefined;
                  session.streamNextRunToken = undefined;
                  session.run = undefined;
                  session.activeRunToken = undefined;
                }
                progressContinuationAttempts += 1;
                logger?.warn?.(
                  {
                    assistantText: currentRunAssistantText,
                    attempt: progressContinuationAttempts,
                  },
                  "cursor-sdk continuing progress-only terminal turn"
                );
                const continuationRunToken = Symbol("cursor-sdk-run");
                // The response cancel callback closes over runToken. Transfer
                // ownership before awaiting send so cancellation cannot mistake
                // this continuation for the already-finished prior run.
                runToken = continuationRunToken;
                session.activeRunToken = continuationRunToken;
                try {
                  await withSessionSendLock(session, async () => {
                    if (session.poisoned || options.abortSignal?.aborted) {
                      throw new Error(
                        "cursor-sdk progress continuation aborted before send"
                      );
                    }

                    const run = await sendCursorPrompt(
                      session,
                      progressOnlyContinuationPrompt(promptHostEnv),
                      sdkSendOptionsForSession(session, continuationRunToken),
                      {
                        abortSignal: options.abortSignal,
                        logger,
                      }
                    );
                    // Publish the handles under the lock so a concurrent
                    // retirement can always cancel a run whose send completed.
                    session.run = run;
                    session.activeRunToken = continuationRunToken;
                    session.streamIterator =
                      run.stream()[Symbol.asyncIterator]();
                    session.streamNext = undefined;
                    session.streamNextRunToken = undefined;
                    session.hasSentPrompt = true;

                    if (session.poisoned || options.abortSignal?.aborted) {
                      throw new Error(
                        "cursor-sdk progress continuation retired during send"
                      );
                    }
                  });
                } catch (err) {
                  // The outer stream failure path owns retirement. Preserve the
                  // token and any returned run handles for that cleanup.
                  throw err;
                }
                currentRunAssistantText = "";
                continuedProgressTurn = true;
                break;
              }

              flushThinkingSignature();
              emitFinish("stop");
              if (session.activeRunToken === runToken) {
                session.streamIterator = undefined;
                session.streamNext = undefined;
                session.streamNextRunToken = undefined;
                session.run = undefined;
                session.activeRunToken = undefined;
              }
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
        // Retire before surfacing the error so a reconnect cannot overtake SDK
        // cleanup and send on the same still-active agent.
        await retireCursorSession(session, {
          reason: "cursor-sdk stream failed before terminal state",
          onlyRunToken: runToken,
        });
        try {
          controller.error(err);
        } catch {
          // already closed
        }
      } finally {
        reportScratchViolations();
      }
    },
    async cancel() {
      // Returning this promise makes Web/Node stream cancellation a teardown
      // barrier instead of detached cleanup racing the user's next request.
      await retireCursorSession(session, {
        reason: "cursor-sdk client cancelled stream",
        onlyRunToken: runToken,
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

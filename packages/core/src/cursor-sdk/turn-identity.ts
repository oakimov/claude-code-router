import { createHash } from "crypto";
import type {
  UnifiedChatRequest,
  UnifiedMessage,
  UnifiedTool,
} from "@/types/llm";
import type { UnifiedTurnIntent } from "@/types/turn-intent";
import type { CursorSdkMode } from "./shared";

type CanonicalValue =
  | null
  | boolean
  | number
  | string
  | CanonicalValue[]
  | { [key: string]: CanonicalValue };

/**
 * Request fields that identify transport delivery rather than the logical turn.
 *
 * Tool-call IDs are deliberately not listed here: they link assistant calls to
 * tool results and are therefore part of transcript semantics.
 */
const TURN_TRANSPORT_FIELDS = new Set([
  "id",
  "idempotency_key",
  "metadata",
  "anthropic_metadata",
  "request_id",
  "req_id",
  "stream",
  "stream_options",
  "trace_id",
  "user",
]);

function canonicalizeUnknown(value: unknown): CanonicalValue | undefined {
  if (value === null) return null;
  if (
    typeof value === "string" ||
    typeof value === "boolean" ||
    typeof value === "number"
  ) {
    return value;
  }
  if (typeof value === "bigint") return String(value);
  if (Array.isArray(value)) {
    return value
      .map((entry) => canonicalizeUnknown(entry))
      .filter((entry): entry is CanonicalValue => entry !== undefined);
  }
  if (typeof value !== "object") return undefined;

  const result: Record<string, CanonicalValue> = {};
  for (const key of Object.keys(value as Record<string, unknown>).sort()) {
    if (key === "cache_control") continue;
    const canonical = canonicalizeUnknown(
      (value as Record<string, unknown>)[key]
    );
    if (canonical !== undefined) result[key] = canonical;
  }
  return result;
}

function canonicalContent(content: unknown): CanonicalValue[] {
  if (content == null) return [];
  if (typeof content === "string") {
    return [{ text: content, type: "text" }];
  }
  if (!Array.isArray(content)) {
    const canonical = canonicalizeUnknown(content);
    return canonical === undefined ? [] : [canonical];
  }

  return content
    .map((part) => {
      if (typeof part === "string") {
        return { text: part, type: "text" } satisfies CanonicalValue;
      }
      if (!part || typeof part !== "object") {
        return canonicalizeUnknown(part);
      }

      const record = part as Record<string, unknown>;
      if (record.type === "text") {
        return {
          text: String(record.text || ""),
          type: "text",
        } satisfies CanonicalValue;
      }
      return canonicalizeUnknown(record);
    })
    .filter((part): part is CanonicalValue => part !== undefined);
}

function canonicalToolArguments(value: unknown): CanonicalValue {
  if (typeof value !== "string") {
    return canonicalizeUnknown(value) ?? null;
  }
  try {
    return canonicalizeUnknown(JSON.parse(value)) ?? null;
  } catch {
    return value;
  }
}

function canonicalMessage(message: UnifiedMessage): CanonicalValue {
  const source = message as UnifiedMessage & {
    is_error?: boolean;
  };
  const result: Record<string, CanonicalValue> = {
    content: canonicalContent(source.content),
    role: source.role,
  };

  if (source.reasoning_content) {
    result.reasoning_content = source.reasoning_content;
  }
  if (source.thinking?.content) {
    // The signature is an opaque replay credential, not conversation content.
    result.thinking = { content: source.thinking.content };
  }
  if (source.tool_call_id) result.tool_call_id = source.tool_call_id;
  if (source.is_error !== undefined) result.is_error = source.is_error;

  if (Array.isArray(source.tool_calls) && source.tool_calls.length) {
    result.tool_calls = source.tool_calls.map((toolCall) => ({
      function: {
        arguments: canonicalToolArguments(toolCall.function.arguments),
        name: toolCall.function.name,
      },
      id: toolCall.id,
      type: toolCall.type,
    }));
  }
  return result;
}

function canonicalTools(tools: UnifiedTool[] | undefined): CanonicalValue[] {
  if (!Array.isArray(tools)) return [];
  return tools
    .map((tool) => canonicalizeUnknown(tool))
    .filter((tool): tool is CanonicalValue => tool !== undefined)
    .sort((left, right) =>
      stableStringify(left).localeCompare(stableStringify(right))
    );
}

function stableStringify(value: CanonicalValue): string {
  return JSON.stringify(value);
}

function fingerprint(value: CanonicalValue): string {
  return createHash("sha256").update(stableStringify(value)).digest("hex");
}

function canonicalTranscript(request: UnifiedChatRequest): {
  system: CanonicalValue[];
  messages: CanonicalValue[];
} {
  return {
    system: canonicalContent(request.system),
    messages: (request.messages || []).map(canonicalMessage),
  };
}

function canonicalTurn(request: UnifiedChatRequest): CanonicalValue {
  const source = request as UnifiedChatRequest & Record<string, unknown>;
  const result: Record<string, CanonicalValue> = {};

  for (const key of Object.keys(source).sort()) {
    if (TURN_TRANSPORT_FIELDS.has(key) || key === "cache_control") continue;

    let value: CanonicalValue | undefined;
    if (key === "messages") {
      value = (request.messages || []).map(canonicalMessage);
    } else if (key === "system") {
      value = canonicalContent(request.system);
    } else if (key === "tools") {
      value = canonicalTools(request.tools);
    } else {
      value = canonicalizeUnknown(source[key]);
    }
    if (value !== undefined) result[key] = value;
  }

  return result;
}

/**
 * Fingerprint a logical model turn independently of streaming, request tracing,
 * cache hints, object insertion order, and JSON argument formatting.
 */
export function fingerprintCursorTurn(
  request: UnifiedChatRequest,
  runtime: {
    compatibilityStamp?: string;
    turnIntent?: UnifiedTurnIntent;
  } = {}
): string {
  return fingerprint({
    compatibilityStamp: runtime.compatibilityStamp || "",
    request: canonicalTurn(request),
    turnIntent: runtime.turnIntent
      ? {
          interruption: runtime.turnIntent.interruption,
          source: runtime.turnIntent.source,
          steering: runtime.turnIntent.steering,
          trailingToolResults: runtime.turnIntent.trailingToolResults.map(
            (result) => ({
              content: result.content,
              isError: result.isError,
              toolCallId: result.toolCallId,
            })
          ),
        }
      : null,
  });
}

export type CursorCompatibilityInput = {
  model: unknown;
  mode: CursorSdkMode;
  workspaceDir: string;
  guidanceFingerprint?: string;
  sandboxEnabled?: boolean;
  /**
   * A one-way credential/account fingerprint supplied by the caller. Never pass
   * a raw API key.
   */
  credentialFingerprint?: string;
  tools?: UnifiedTool[];
};

/**
 * Fingerprint the properties that must remain compatible when an SDK agent is
 * reused. The turn fingerprint separately covers prompt/tool semantics.
 */
export function createCursorCompatibilityStamp(
  input: CursorCompatibilityInput
): string {
  return fingerprint({
    credentialFingerprint: input.credentialFingerprint || "",
    guidanceFingerprint: input.guidanceFingerprint || "",
    mode: input.mode,
    model: canonicalizeUnknown(input.model) ?? null,
    sandboxEnabled: input.sandboxEnabled === true,
    tools: canonicalTools(input.tools),
    workspaceDir: input.workspaceDir,
  });
}

export type CursorTranscriptCommit = Readonly<{
  transcriptHash: string;
  messageCount: number;
}>;

function transcriptHash(
  system: CanonicalValue[],
  messages: CanonicalValue[]
): string {
  return fingerprint({ messages, system });
}

/**
 * Commit the host-visible transcript after a successful assistant turn.
 *
 * The caller supplies the assistant message assembled from what was actually
 * emitted to the host, rather than relying on Cursor's opaque checkpoint.
 */
export function createCursorTranscriptCommit(
  request: UnifiedChatRequest,
  assistantMessage?: UnifiedMessage
): CursorTranscriptCommit {
  if (assistantMessage && assistantMessage.role !== "assistant") {
    throw new TypeError("Cursor transcript commits require an assistant message");
  }

  const transcript = canonicalTranscript(request);
  if (assistantMessage) {
    transcript.messages.push(canonicalMessage(assistantMessage));
  }
  return Object.freeze({
    transcriptHash: transcriptHash(transcript.system, transcript.messages),
    messageCount: transcript.messages.length,
  });
}

/**
 * True only when the incoming transcript contains the exact committed prefix
 * and at least one additional semantic message.
 */
export function getStrictCursorTranscriptSuffix(
  commit: CursorTranscriptCommit,
  request: UnifiedChatRequest
): readonly UnifiedMessage[] | undefined {
  if (
    !Number.isInteger(commit.messageCount) ||
    commit.messageCount < 0 ||
    typeof commit.transcriptHash !== "string"
  ) {
    return undefined;
  }

  const transcript = canonicalTranscript(request);
  if (transcript.messages.length <= commit.messageCount) return undefined;
  if (
    transcriptHash(
      transcript.system,
      transcript.messages.slice(0, commit.messageCount)
    ) !== commit.transcriptHash
  ) {
    return undefined;
  }
  return (request.messages || []).slice(commit.messageCount);
}

export function isStrictCursorTranscriptExtension(
  commit: CursorTranscriptCommit,
  request: UnifiedChatRequest
): boolean {
  return getStrictCursorTranscriptSuffix(commit, request) !== undefined;
}

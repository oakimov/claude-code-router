import type { UnifiedChatRequest, UnifiedMessage } from "@/types/llm";
import type { SDKImage, SDKUserMessage } from "@cursor/sdk";
import { contentToText, CUSTOM_USER_TOOLS_SERVER } from "./shared";
import {
  describeHostEnvironment,
  extractHostEnvironment,
  hostPathRule,
  type HostEnvironment,
} from "./host-env";

function extractImages(content: UnifiedMessage["content"]): SDKImage[] {
  if (!Array.isArray(content)) return [];
  const images: SDKImage[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    if ((part as any).type === "image_url" && (part as any).image_url?.url) {
      const url = String((part as any).image_url.url);
      const dataMatch = url.match(/^data:([^;]+);base64,(.+)$/);
      if (dataMatch) {
        images.push({ data: dataMatch[2], mimeType: dataMatch[1] });
      } else {
        images.push({ url });
      }
    }
  }
  return images;
}

function toolCatalog(request: UnifiedChatRequest): string {
  if (!Array.isArray(request.tools) || request.tools.length === 0) {
    return "(no host tools advertised on this request)";
  }
  return request.tools
    .map((tool) => {
      const name = tool.function?.name || "unknown";
      const desc = tool.function?.description || "";
      return `- ${name}: ${desc}`;
    })
    .join("\n");
}

/**
 * Lines that must outrank Cursor's server-built harness prompt.
 *
 * Cursor tells the model — at system level — that its workspace root is the
 * local scratch directory. That claim is self-consistent (the sandbox really is
 * Linux, really is at that path, really is empty), so a model resolving the
 * conflict against the host's own environment block tends to trust the sandbox
 * and refuse to leave it. The fix is not "ignore your cwd" but the actual
 * topology: tools execute on a different machine.
 */
function remoteExecutionRules(
  workspaceDir: string,
  hostEnv: HostEnvironment
): string[] {
  const lines = [
    "You are a remote reasoning agent driving Claude Code (the host).",
    "Tools do not run where you are running: every tool call is executed by the host on the user's machine, and only the result is sent back to you.",
    `Your local environment is disposable scratch space: ${workspaceDir}`,
    "It is empty and contains none of the user's files. Never read, list, search, or verify paths against it, and never prefix a host tool path with it.",
    "Your own cwd, OS, and workspace root describe that scratch container only — they say nothing about the user's project, so never reason from them or report them to the user.",
  ];

  const hostLines = describeHostEnvironment(hostEnv);
  if (hostLines.length) {
    lines.push(
      "The user's project lives on the host machine, which is where every tool call lands:"
    );
    lines.push(...hostLines);
  }
  lines.push(hostPathRule(hostEnv));

  lines.push(
    `Host tools are exposed via MCP server "${CUSTOM_USER_TOOLS_SERVER}" using bare tool names.`,
    "Cursor built-in tools are denied here. That denial only routes tool use through the host — it is not a restriction on the user's project, which stays fully reachable through the host tools."
  );
  return lines;
}

export function buildBridgeSystemGuidance(
  request: UnifiedChatRequest,
  workspaceDir: string,
  hostEnv: HostEnvironment = extractHostEnvironment(request)
): string {
  return [
    ...remoteExecutionRules(workspaceDir, hostEnv),
    "A progress update such as 'Checking...', 'Inspecting...', or 'Let me look...' is not a final answer.",
    "After progress narration, call a host tool in the same turn; if no tool is needed, provide the complete user-facing answer before finishing.",
    "Never end a turn with progress narration alone.",
    "Available host tools:",
    toolCatalog(request),
  ].join("\n");
}

/**
 * Short restatement appended after the transcript. The guidance above is far
 * from the generation point once history is flattened in; this is the last
 * thing the model reads before answering.
 */
export function buildBridgeTailReminder(
  workspaceDir: string,
  hostEnv: HostEnvironment
): string {
  const lines = [
    "[bridge reminder]",
    `Tools run on the host machine, not here. ${workspaceDir} is empty scratch space — never treat it as the project, and never build a tool path from it.`,
    hostPathRule(hostEnv),
  ];
  return lines.join("\n");
}

/** Rules document for the sandbox workspace, injected by Cursor as project rules. */
export function buildWorkspaceRulesDocument(
  workspaceDir: string,
  hostEnv: HostEnvironment
): string {
  return [
    "# Agent rules",
    "",
    ...remoteExecutionRules(workspaceDir, hostEnv).map((line) =>
      line.startsWith("- ") ? line : `- ${line}`
    ),
    "",
  ].join("\n");
}

/** One-line rationale attached to Cursor built-in denials. */
export function buildDenyGuidance(hostEnv: HostEnvironment): string {
  return [
    "Your local filesystem is empty scratch space on a different machine from the user's project.",
    "The project is reachable only through the host tools, and it is not restricted.",
    hostPathRule(hostEnv),
  ].join(" ");
}

const PROGRESS_ONLY_PATTERNS = [
  /^(?:checking|inspecting|reviewing|examining|investigating|verifying|reading|searching)\s+(?:the|this|that|these|those|your|our|which|whether|for|how|why|what|where)\b/i,
  /^looking\s+(?:at|into|for|through|up)\b/i,
  /^(?:i(?:'ll| will)|let me)\s+(?:check|inspect|review|examine|investigate|verify|read|search|look)\b/i,
];

/**
 * Detect a short terminal message that only announces work which never occurred.
 * Keep this deliberately narrow: it is a recovery signal, not a general quality score.
 */
export function isProgressOnlyAssistantText(text: string): boolean {
  const trimmed = String(text || "").trim();
  if (!trimmed || trimmed.length > 240) return false;
  if (/\n\s*\n/.test(trimmed)) return false;

  const normalized = trimmed.replace(/\s+/g, " ");
  const sentenceEnds = normalized.match(/[.!?](?:\s|$)/g)?.length || 0;
  if (sentenceEnds > 1) return false;
  return PROGRESS_ONLY_PATTERNS.some((pattern) => pattern.test(normalized));
}

export function shouldContinueProgressOnlyTurn(input: {
  mode: "bridge" | "plan" | "agent";
  assistantText: string;
  emittedHostTools: number;
  continuationAttempts: number;
}): boolean {
  return (
    input.mode === "bridge" &&
    input.emittedHostTools === 0 &&
    input.continuationAttempts < 1 &&
    isProgressOnlyAssistantText(input.assistantText)
  );
}

export function progressOnlyContinuationPrompt(
  hostEnv?: HostEnvironment
): SDKUserMessage {
  const lines = [
    "Continue the same turn now.",
    "Your previous assistant message was only progress narration and ended without a host tool call or a complete answer.",
    "Do not repeat the progress update.",
    "If evidence is still needed, call an available host tool immediately; otherwise provide the complete user-facing answer now.",
    "Do not finish until the requested work or answer is complete.",
  ];
  if (hostEnv?.known) {
    lines.push(hostPathRule(hostEnv));
  }
  return { text: lines.join(" ") };
}

/**
 * Flatten Unified chat history into a single SDK prompt for a fresh/continued send.
 * Live parked tool results are resolved via customTools.execute — not only this text.
 */
export function toSdkPrompt(
  request: UnifiedChatRequest,
  options: {
    mode: "bridge" | "plan" | "agent";
    workspaceDir: string;
    /** When true, only the latest user turn (+ guidance) is sent — session already has history. */
    followUpOnly?: boolean;
    hostEnv?: HostEnvironment;
  }
): SDKUserMessage {
  const messages = request.messages || [];
  const parts: string[] = [];
  const hostEnv = options.hostEnv || extractHostEnvironment(request);

  if (options.mode === "bridge") {
    parts.push(
      buildBridgeSystemGuidance(request, options.workspaceDir, hostEnv)
    );
    parts.push("");
  } else if (options.mode === "plan") {
    parts.push(
      "You are a planning/chat assistant. Respond with text and reasoning only. Do not execute tools."
    );
    parts.push("");
  }

  const systemMsgs = messages.filter((m) => m.role === "system");
  for (const msg of systemMsgs) {
    const text = contentToText(msg.content);
    if (text) parts.push(`[system]\n${text}`);
  }

  const nonSystem = messages.filter((m) => m.role !== "system");
  const history = options.followUpOnly
    ? nonSystem.slice(-Math.max(1, countTrailingToolRoundtrip(nonSystem) + 1))
    : nonSystem;

  let images: SDKImage[] = [];

  for (const msg of history) {
    if (msg.role === "assistant") {
      const text = contentToText(msg.content);
      if (text) parts.push(`[assistant]\n${text}`);
      if (Array.isArray(msg.tool_calls) && msg.tool_calls.length) {
        for (const tc of msg.tool_calls) {
          parts.push(
            `[assistant tool_call id=${tc.id} name=${tc.function.name}]\n${tc.function.arguments || "{}"}`
          );
        }
      }
      continue;
    }

    if (msg.role === "tool") {
      parts.push(
        `[tool_result id=${msg.tool_call_id || "unknown"}]\n${contentToText(msg.content)}`
      );
      continue;
    }

    if (msg.role === "user") {
      const text = contentToText(msg.content);
      images = [...images, ...extractImages(msg.content)];
      if (text) parts.push(`[user]\n${text}`);
    }
  }

  // Recency guard: the guidance above is far from the generation point once the
  // transcript is flattened in, so restate the topology rule last.
  if (options.mode === "bridge") {
    parts.push(buildBridgeTailReminder(options.workspaceDir, hostEnv));
  }

  // Prefer the last user text as the primary prompt body if follow-up only and we have it.
  const text = parts.join("\n\n").trim() || "Continue.";
  return images.length ? { text, images } : { text };
}

function countTrailingToolRoundtrip(messages: UnifiedMessage[]): number {
  let n = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "tool") {
      n++;
      continue;
    }
    break;
  }
  return n;
}

export function extractTrailingToolResults(
  request: UnifiedChatRequest
): Array<{ toolCallId: string; content: string }> {
  const results: Array<{ toolCallId: string; content: string }> = [];
  const messages = request.messages || [];
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== "tool") break;
    results.unshift({
      toolCallId: String(msg.tool_call_id || ""),
      content: contentToText(msg.content),
    });
  }
  return results;
}

import type { UnifiedChatRequest, UnifiedMessage } from "@/types/llm";
import type { SDKImage, SDKUserMessage } from "@cursor/sdk";
import { CUSTOM_USER_TOOLS_SERVER } from "./shared";

function contentToText(content: UnifiedMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return String(content);
  return content
    .map((part) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object" && "text" in part) {
        return String((part as any).text || "");
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

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

export function buildBridgeSystemGuidance(
  request: UnifiedChatRequest,
  workspaceDir: string
): string {
  return [
    "You are running inside the Claude Code Router Cursor SDK bridge.",
    "Claude Code (the host) owns filesystem, shell, and project tools.",
    "Cursor built-in tools are denied in this workspace — do not call them.",
    `Host tools are exposed via MCP server "${CUSTOM_USER_TOOLS_SERVER}" using bare tool names.`,
    `Isolated agent cwd (do not treat as the user project): ${workspaceDir}`,
    "Pass absolute paths into host tools when referring to the user's real project files.",
    "Available host tools:",
    toolCatalog(request),
  ].join("\n");
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
  }
): SDKUserMessage {
  const messages = request.messages || [];
  const parts: string[] = [];

  if (options.mode === "bridge") {
    parts.push(buildBridgeSystemGuidance(request, options.workspaceDir));
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

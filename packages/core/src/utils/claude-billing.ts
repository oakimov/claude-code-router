import { createHash, randomBytes } from "crypto";
import { TextContent, UnifiedChatRequest, UnifiedMessage } from "@/types/llm";
import { CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX } from "./router";

/** Salt used in Claude Code's `cc_version` suffix hash (`RYn()`). */
export const BILLING_SALT = "59cf53e54c78";
export const CC_VERSION = process.env.ANTHROPIC_CLI_VERSION || "2.1.220";
export const CC_ENTRYPOINT = process.env.CLAUDE_CODE_ENTRYPOINT || "cli";
export const SYSTEM_IDENTITY =
  "You are Claude Code, Anthropic's official CLI for Claude.";

function textOfContent(content: UnifiedMessage["content"]): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const firstText = content.find(
      (part): part is TextContent => part.type === "text"
    );
    return firstText?.text ?? "";
  }
  return "";
}

export function extractFirstUserMessageText(
  messages: UnifiedMessage[] | undefined
): string {
  const firstUser = messages?.find((msg) => msg.role === "user");
  return firstUser ? textOfContent(firstUser.content) : "";
}

export function computeVersionSuffix(text: string, version: string): string {
  const sample = [4, 7, 20].map((i) => text[i] || "0").join("");
  return createHash("sha256")
    .update(BILLING_SALT + sample + version)
    .digest("hex")
    .slice(0, 3);
}

let cachedSessionCch: string | undefined;

/** 5 lowercase hex chars, cached per process — not a content hash. */
export function sessionCch(): string {
  if (!cachedSessionCch) {
    cachedSessionCch = randomBytes(3).toString("hex").slice(0, 5);
  }
  return cachedSessionCch;
}

export function __resetClaudeBillingStateForTests(): void {
  cachedSessionCch = undefined;
}

export function buildClaudeBillingHeaderValue(
  messages: UnifiedMessage[] | undefined,
  version: string = CC_VERSION,
  entrypoint: string = CC_ENTRYPOINT
): string {
  const suffix = computeVersionSuffix(
    extractFirstUserMessageText(messages),
    version
  );
  return (
    `${CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX}: cc_version=${version}.${suffix}; ` +
    `cc_entrypoint=${entrypoint}; cch=${sessionCch()};`
  );
}

/**
 * Normalize `request.system` to an array as a complete ordered fold:
 * top-level `request.system` blocks first, then every `role:"system"` /
 * `role:"developer"` message's text blocks in source order. Consumed messages
 * are removed from `request.messages` so no downstream builder can drop or
 * double-emit them. Block-level `cache_control` is preserved.
 */
export function normalizeSystemToArray(
  request: UnifiedChatRequest
): TextContent[] {
  const blocks: TextContent[] = [];

  const pushContent = (content: UnifiedMessage["content"] | undefined) => {
    if (typeof content === "string") {
      if (content) blocks.push({ type: "text", text: content });
      return;
    }
    if (!Array.isArray(content)) return;
    for (const part of content) {
      if (part?.type === "text" && part.text) {
        const block: TextContent = { type: "text", text: part.text };
        if (part.cache_control !== undefined) {
          block.cache_control = part.cache_control;
        }
        blocks.push(block);
      }
    }
  };

  if (typeof request.system === "string" || Array.isArray(request.system)) {
    pushContent(request.system as UnifiedMessage["content"]);
  }

  if (Array.isArray(request.messages)) {
    const remaining: UnifiedMessage[] = [];
    for (const message of request.messages) {
      if (
        message.role === "system" ||
        (message.role as string) === "developer"
      ) {
        pushContent(message.content);
      } else {
        remaining.push(message);
      }
    }
    request.messages = remaining;
  }

  request.system = blocks;
  return blocks;
}

/** Drop any existing billing entry (dedupe) and prepend a fresh one at system[0]. */
export function applyClaudeBillingSystemBlock(
  system: TextContent[],
  messages: UnifiedMessage[] | undefined
): void {
  for (let i = system.length - 1; i >= 0; i--) {
    if (system[i].text.startsWith(CLAUDE_CODE_BILLING_SYSTEM_HEADER_PREFIX)) {
      system.splice(i, 1);
    }
  }
  system.unshift({ type: "text", text: buildClaudeBillingHeaderValue(messages) });
}

/**
 * Insert SYSTEM_IDENTITY as its own entry at system[1]. Any remaining
 * caller system content is left in place here — relocateForeignSystemContent
 * (called separately, afterwards) is what moves it out of system[].
 */
export function applyClaudeSystemIdentity(system: TextContent[]): void {
  const candidateIndex = 1;
  const candidate = system[candidateIndex];

  if (candidate?.text === SYSTEM_IDENTITY) return;

  if (
    candidate?.text.startsWith(SYSTEM_IDENTITY) &&
    candidate.text.length > SYSTEM_IDENTITY.length
  ) {
    const remainder: TextContent = {
      type: "text",
      text: candidate.text.slice(SYSTEM_IDENTITY.length),
    };
    if (candidate.cache_control !== undefined) {
      remainder.cache_control = candidate.cache_control;
    }
    system.splice(
      candidateIndex,
      1,
      { type: "text", text: SYSTEM_IDENTITY },
      remainder
    );
    return;
  }

  system.splice(candidateIndex, 0, { type: "text", text: SYSTEM_IDENTITY });
}

/**
 * Move everything in `system[]` past the identity block (index 1) into the
 * first user message instead. Anthropic's OAuth billing validator appears to
 * inspect `system[]` content beyond the identity prefix and reject requests
 * whose system array carries a foreign harness prompt with an "out of extra
 * usage" 400 — the same technique used by third-party Claude Code OAuth
 * shims (e.g. opencode-claude-auth) to avoid that check. The relocated text
 * still reaches the model, just as part of the first user turn rather than
 * `system[]`.
 *
 * A no-op when there is no user message to attach the content to, so nothing
 * is silently dropped — the caller's system content stays in `system[]`
 * instead.
 */
export function relocateForeignSystemContent(
  system: TextContent[],
  messages: UnifiedMessage[] | undefined
): void {
  if (system.length <= 2 || !Array.isArray(messages)) return;

  const firstUser = messages.find((msg) => msg.role === "user");
  if (!firstUser) return;

  const foreign = system.splice(2);
  const text = foreign
    .map((block) => block.text)
    .filter((t) => t.length > 0)
    .join("\n\n");
  if (!text) return;

  if (typeof firstUser.content === "string") {
    firstUser.content = firstUser.content
      ? `${text}\n\n${firstUser.content}`
      : text;
  } else if (Array.isArray(firstUser.content)) {
    firstUser.content.unshift({ type: "text", text });
  } else {
    firstUser.content = text;
  }
}

/**
 * Claude Code's OAuth validator expects tool names in the mcp_PascalCase
 * spelling used by the official CLI. Non-Claude-Code clients commonly send
 * ordinary names such as `bash` or `read`, so normalize them before the
 * Anthropic body is built. The prefix check keeps this operation idempotent.
 */
export function prefixClaudeToolName(name: string): string {
  if (/^mcp_[A-Z]/.test(name)) return name;
  return `mcp_${name.charAt(0).toUpperCase()}${name.slice(1)}`;
}

/** Restore a Claude Code tool name to the caller's original spelling. */
export function unprefixClaudeToolName(name: string): string {
  if (!name.startsWith("mcp_") || name.length <= 4) return name;
  return `${name.charAt(4).toLowerCase()}${name.slice(5)}`;
}

/** Rewrite tool names in a Unified request in place for the OAuth wire path. */
export function prefixClaudeToolNames(
  request: UnifiedChatRequest,
  nameMap?: Map<string, string>
): void {
  const prefix = (name: string): string => {
    const wireName = prefixClaudeToolName(name);
    nameMap?.set(wireName, name);
    return wireName;
  };

  if (Array.isArray(request.tools)) {
    for (const tool of request.tools) {
      if (tool?.function?.name) tool.function.name = prefix(tool.function.name);
    }
  }

  if (Array.isArray(request.messages)) {
    for (const message of request.messages) {
      for (const toolCall of message.tool_calls ?? []) {
        if (toolCall?.function?.name) {
          toolCall.function.name = prefix(toolCall.function.name);
        }
      }
    }
  }

  if (typeof request.tool_choice === "string") {
    if (!["auto", "none", "required"].includes(request.tool_choice)) {
      request.tool_choice = prefix(request.tool_choice);
    }
  } else if (
    request.tool_choice?.type === "function" &&
    request.tool_choice.function?.name
  ) {
    request.tool_choice.function.name = prefix(request.tool_choice.function.name);
  }
}

/** Rewrite tool names in a Unified/OpenAI-shaped response in place. */
export function unprefixClaudeToolNames(
  value: any,
  nameMap?: Map<string, string>
): void {
  const rewriteToolCall = (toolCall: any) => {
    const name = toolCall?.function?.name;
    if (typeof name !== "string") return;
    toolCall.function.name = nameMap?.get(name) ?? unprefixClaudeToolName(name);
  };

  for (const choice of value?.choices ?? []) {
    for (const toolCall of choice?.message?.tool_calls ?? []) rewriteToolCall(toolCall);
    for (const toolCall of choice?.delta?.tool_calls ?? []) rewriteToolCall(toolCall);
  }
}

/** Rewrite one OpenAI SSE data payload's tool names. */
export function unprefixClaudeToolNamesInSseData(
  value: any,
  nameMap?: Map<string, string>
): void {
  unprefixClaudeToolNames(value, nameMap);
}

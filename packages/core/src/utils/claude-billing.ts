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
 * Insert SYSTEM_IDENTITY as its own entry at system[1]. The caller's other
 * system entries are left exactly where they are — see the plan's rationale
 * for not hoisting them into the first user message (Findings 8 and 11).
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

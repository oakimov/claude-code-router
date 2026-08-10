import { readFile } from "fs/promises";
import { UnifiedChatRequest } from "@/types/llm";
import {
  applyClaudeBillingSystemBlock,
  applyClaudeSystemIdentity,
  normalizeSystemToArray,
  prefixClaudeToolNames,
  relocateForeignSystemContent,
} from "./claude-billing";

export type AnthropicClientKind = "claude_desktop" | "claude_code" | "other";
export type AnthropicProviderMode = "api_key" | "claude_oauth" | "out_of_scope";

export interface AnthropicClientFingerprintSignals {
  desktopMarker: boolean;
  desktopUserAgent: boolean;
  desktopAgentSdkUserAgent: boolean;
  cliUserAgent: boolean;
  cliApp: boolean;
  cliSession: boolean;
  stainlessPackage: boolean;
  billingSystem: boolean;
  identitySystem: boolean;
}

export interface AnthropicClientPolicyContext {
  anthropicClientKind?: AnthropicClientKind;
  anthropicProviderMode?: AnthropicProviderMode;
  anthropicDestinationInScope?: boolean;
  anthropicNativeWire?: boolean;
  anthropicPolicyApplied?: boolean;
  anthropicSystemTransformed?: boolean;
  claudeAuthToolNameMap?: Map<string, string>;
}

export function readHeaderValue(
  headers: Record<string, unknown> | undefined,
  name: string
): string | undefined {
  if (!headers) return undefined;
  const expected = name.toLowerCase();
  for (const [rawName, rawValue] of Object.entries(headers)) {
    if (rawName.toLowerCase() !== expected || rawValue == null) continue;
    if (Array.isArray(rawValue)) {
      const values = rawValue.filter((value) => value != null && String(value));
      return values.length ? values.map(String).join(", ") : undefined;
    }
    const value = String(rawValue);
    return value || undefined;
  }
  return undefined;
}

function hasNativeClaudeBillingBody(body: any): boolean {
  const system = Array.isArray(body?.system) ? body.system : [];
  return system.some(
    (block: any) =>
      typeof block?.text === "string" &&
      block.text.startsWith("x-anthropic-billing-header")
  );
}

function hasClaudeCodeIdentityBody(body: any): boolean {
  const system = Array.isArray(body?.system) ? body.system : [];
  return system.some(
    (block: any) =>
      typeof block?.text === "string" &&
      (block.text.startsWith(
        "You are Claude Code, Anthropic's official CLI for Claude."
      ) ||
        block.text.startsWith(
          "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK."
        ) ||
        block.text.startsWith(
          "You are a Claude agent, built on Anthropic's Claude Agent SDK."
        ))
  );
}

/**
 * Classify the original Anthropic wire request. A generic SDK UA is never
 * enough to grant native pass-through; incomplete fingerprints fail closed to
 * the third-party emulation path.
 */
export function classifyAnthropicClient(
  headers: Record<string, unknown> | undefined,
  body: any
): AnthropicClientKind {
  const signals = inspectAnthropicClientFingerprint(headers, body);
  const desktopTopbarCandidate =
    signals.desktopMarker &&
    signals.desktopUserAgent &&
    Array.isArray(body?.messages);
  const nativeCliShape =
    signals.cliUserAgent &&
    signals.cliApp &&
    signals.cliSession &&
    signals.stainlessPackage &&
    signals.billingSystem &&
    signals.identitySystem &&
    Array.isArray(body?.messages);
  const conflictingCliMarkers =
    signals.cliUserAgent ||
    signals.cliApp ||
    signals.cliSession;
  if (desktopTopbarCandidate && conflictingCliMarkers) {
    return "other";
  }
  if (desktopTopbarCandidate) {
    return "claude_desktop";
  }

  // Current Desktop 3P inference runs through the bundled Agent SDK and
  // Claude Code binary. Its wire request intentionally has the complete CLI
  // shape, but the UA's entrypoint identifies Desktop as the host client.
  if (signals.desktopAgentSdkUserAgent && nativeCliShape) {
    return "claude_desktop";
  }

  return nativeCliShape ? "claude_code" : "other";
}

/** Return only non-sensitive boolean fingerprint signals for debug logging. */
export function inspectAnthropicClientFingerprint(
  headers: Record<string, unknown> | undefined,
  body: any
): AnthropicClientFingerprintSignals {
  const userAgent = readHeaderValue(headers, "user-agent") || "";
  const desktopMarker = readHeaderValue(headers, "anthropic-desktop-topbar");
  const cliUserAgentMatch = /^claude-cli\/[^\s()]+\s+\(([^)]*)\)$/i.exec(
    userAgent
  );
  const cliUserAgentParts = (cliUserAgentMatch?.[1] || "")
    .split(",")
    .map((part) => part.trim().toLowerCase());
  const desktopAgentSdkUserAgent =
    cliUserAgentParts.some((part) =>
      ["claude-desktop", "claude-desktop-3p"].includes(part)
    ) && cliUserAgentParts.some((part) => /^agent-sdk\/[^\s,]+$/.test(part));
  return {
    desktopMarker: desktopMarker === "1",
    desktopUserAgent: /^Anthropic\/JS\s+/i.test(userAgent),
    desktopAgentSdkUserAgent,
    cliUserAgent: /^claude-cli\//i.test(userAgent),
    cliApp: ["cli", "cli-bg"].includes(
      readHeaderValue(headers, "x-app") || ""
    ),
    cliSession: Boolean(readHeaderValue(headers, "x-claude-code-session-id")),
    stainlessPackage: Boolean(
      readHeaderValue(headers, "x-stainless-package-version")
    ),
    billingSystem: hasNativeClaudeBillingBody(body),
    identitySystem: hasClaudeCodeIdentityBody(body),
  };
}

/**
 * The feature is intentionally scoped to CCR's real Anthropic provider, with
 * either the exact Anthropic API-key chain or the exact claude-auth + Anthropic
 * OAuth chain. Adjacent middleware makes a provider out of scope.
 */
export function getAnthropicProviderMode(
  provider: any,
  endpointTransformerName = "Anthropic"
): AnthropicProviderMode {
  const use = Array.isArray(provider?.transformer?.use)
    ? provider.transformer.use
    : [];
  const names = use.map((transformer: any) => transformer?.name);
  if (names.length === 1 && names[0] === endpointTransformerName) {
    return "api_key";
  }
  if (
    names.length === 2 &&
    names[0] === "claude-auth" &&
    names[1] === endpointTransformerName
  ) {
    return "claude_oauth";
  }
  return "out_of_scope";
}

export function isNativeAnthropicClient(kind: AnthropicClientKind): boolean {
  return kind === "claude_desktop" || kind === "claude_code";
}

/**
 * Apply the one and only system transformation allowed by the gateway policy.
 * This runs after routing has identified an in-scope Anthropic destination and
 * before the provider's Anthropic body builder runs.
 */
export async function applyThirdPartyAnthropicPolicy(
  request: UnifiedChatRequest,
  context: AnthropicClientPolicyContext,
  configService: any
): Promise<void> {
  if (
    context.anthropicClientKind !== "other" ||
    context.anthropicDestinationInScope !== true
  ) {
    return;
  }
  const system = normalizeSystemToArray(request);
  applyClaudeBillingSystemBlock(system, request.messages);
  applyClaudeSystemIdentity(system);

  const rewritePrompt = configService?.get?.("REWRITE_SYSTEM_PROMPT");
  if (rewritePrompt && Array.isArray(request.system)) {
    for (const block of request.system as any[]) {
      if (typeof block?.text !== "string" || !block.text.includes("<env>")) {
        continue;
      }
      const prompt = await readFile(rewritePrompt, "utf-8");
      block.text = `${prompt}<env>${block.text.split("<env>").pop()}`;
    }
  }

  relocateForeignSystemContent(system, request.messages);
  const toolNameMap = new Map<string, string>();
  prefixClaudeToolNames(request, toolNameMap);
  applyClaudeCodeCacheProfile(request, context.anthropicProviderMode);
  context.claudeAuthToolNameMap = toolNameMap;
  context.anthropicPolicyApplied = true;
  context.anthropicSystemTransformed = true;
}

function applyClaudeCodeCacheProfile(
  request: UnifiedChatRequest,
  providerMode: AnthropicProviderMode | undefined
): void {
  const cacheControl: { type: "ephemeral"; ttl?: "1h" } = {
    type: "ephemeral",
    ...(providerMode === "claude_oauth" ? { ttl: "1h" } : {}),
  };

  delete (request as any).cache_control;
  for (const tool of request.tools || []) delete (tool as any).cache_control;

  for (const block of (request.system as any[]) || []) {
    if (
      block &&
      typeof block.text === "string" &&
      !block.text.startsWith("x-anthropic-billing-header")
    ) {
      block.cache_control = { ...cacheControl };
    } else if (block) {
      delete block.cache_control;
    }
  }

  for (const message of request.messages || []) {
    delete (message as any).cache_control;
    for (const toolCall of message.tool_calls || []) {
      delete (toolCall as any).cache_control;
    }
    if (Array.isArray(message.content)) {
      for (const part of message.content as any[]) delete part.cache_control;
    }
  }

  const messages = request.messages || [];
  let cacheMessageIndex = -1;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "assistant") {
      cacheMessageIndex = i;
      break;
    }
    if (typeof message.content === "string") {
      cacheMessageIndex = i;
      break;
    }
    if (Array.isArray(message.content)) {
      const lastPart = message.content.at(-1) as any;
      if (
        lastPart &&
        !["thinking", "redacted_thinking", "fallback"].includes(lastPart.type)
      ) {
        cacheMessageIndex = i;
        break;
      }
    }
  }
  const lastMessage = messages[cacheMessageIndex];
  if (!lastMessage) return;
  if (typeof lastMessage.content === "string") {
    lastMessage.content = [{
      type: "text",
      text: lastMessage.content,
      cache_control: { ...cacheControl },
    }];
    return;
  }
  if (!Array.isArray(lastMessage.content)) return;
  for (let i = lastMessage.content.length - 1; i >= 0; i -= 1) {
    const part = lastMessage.content[i] as any;
    // Claude Code marks the last eligible content block, not necessarily a
    // text block: user tool results and assistant tool_use blocks can be the
    // cache breakpoint. Assistant thinking/redacted/fallback blocks are
    // explicitly skipped by the current `a8b`/`zGb` path.
    if (
      lastMessage.role === "assistant" &&
      ["thinking", "redacted_thinking", "fallback"].includes(part?.type)
    ) {
      continue;
    }
    if (part && typeof part === "object") {
      part.cache_control = { ...cacheControl };
      return;
    }
  }
}

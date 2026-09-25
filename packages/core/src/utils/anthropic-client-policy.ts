import { readFile } from "fs/promises";
import { TextContent, UnifiedChatRequest } from "@/types/llm";
import {
  applyClaudeSystemIdentity,
  fillClaudeBillingSystemBlock,
  normalizeSystemToArray,
  prefixClaudeToolNames,
  relocateForeignSystemContent,
  reserveClaudeBillingSystemBlock,
} from "./claude-billing";
import { unifiedUserPartToAnthropic } from "./tool-content";

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
 * Owners of ephemeral breakpoints at the places the Messages API accepts them:
 * top level, tool definitions, system blocks, message content blocks and
 * tool_result content blocks. Tool inputs and other payloads are user data and
 * are never scanned.
 */
function ephemeralMarkerOwners(body: any): any[] {
  const owners: any[] = [];
  const add = (owner: any) => {
    const cc = owner?.cache_control;
    if (cc && typeof cc === "object" && cc.type === "ephemeral") owners.push(owner);
  };
  const blocks = (value: unknown) => (Array.isArray(value) ? value : []);
  add(body);
  blocks(body.tools).forEach(add);
  blocks(body.system).forEach(add);
  for (const message of blocks(body.messages)) {
    for (const block of blocks(message?.content)) {
      add(block);
      if (block?.type === "tool_result") blocks(block.content).forEach(add);
    }
  }
  return owners;
}

/** `CLAUDE_AUTH_NATIVE_CACHE_TTL` values: `1h` (default) or `client`. */
export type NativeClaudeOAuthCacheTtlMode = "1h" | "client";

export function resolveNativeClaudeOAuthCacheTtlMode(
  value: unknown
): NativeClaudeOAuthCacheTtlMode {
  return typeof value === "string" && value.trim().toLowerCase() === "client"
    ? "client"
    : "1h";
}

/**
 * Native Desktop/CLI on claude-auth OAuth: extend the client's default-TTL
 * (5m) ephemeral breakpoints to 1h, the TTL the third-party profile already
 * uses on this route. Breakpoint placement is untouched. A body that already
 * sets a TTL on any breakpoint is left exactly as sent (an explicit client
 * choice), as is every body when the mode is `client`. Returns whether the
 * body changed.
 */
export function applyNativeClaudeOAuthCacheTtl(
  body: any,
  context: AnthropicClientPolicyContext | undefined,
  mode: NativeClaudeOAuthCacheTtlMode = "1h"
): boolean {
  if (
    mode !== "1h" ||
    !context?.anthropicNativeWire ||
    context.anthropicProviderMode !== "claude_oauth" ||
    !body ||
    typeof body !== "object"
  ) {
    return false;
  }
  const markers = ephemeralMarkerOwners(body);
  if (!markers.length || markers.some((owner) => owner.cache_control.ttl !== undefined)) {
    return false;
  }
  for (const owner of markers) {
    owner.cache_control = { ...owner.cache_control, ttl: "1h" };
  }
  return true;
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
  const billingBlock = reserveClaudeBillingSystemBlock(system);
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

  // Give user/assistant text one block shape regardless of position: the
  // relocated prompt then stays a standalone block that can carry its own
  // breakpoint, and the tail breakpoint no longer flips a string into a block
  // array only while that message is last (which read as a rewritten history
  // in the client-stage cache diff). The Anthropic wire is identical.
  for (const message of request.messages) {
    if (
      (message.role === "user" || message.role === "assistant") &&
      typeof message.content === "string" &&
      message.content
    ) {
      message.content = [{ type: "text", text: message.content }];
    }
  }
  const relocatedSystemBlock = relocateForeignSystemContent(
    system,
    request.messages
  );
  // The billing suffix samples the first user text, so compute it from the
  // body actually sent: after relocation that text is the caller's stable
  // prompt, and system[0] survives clients dropping or summarizing early turns.
  fillClaudeBillingSystemBlock(system, billingBlock, request.messages);
  const toolNameMap = new Map<string, string>();
  prefixClaudeToolNames(request, toolNameMap);
  applyClaudeCodeCacheProfile(
    request,
    context.anthropicProviderMode,
    relocatedSystemBlock
  );
  context.claudeAuthToolNameMap = toolNameMap;
  context.anthropicPolicyApplied = true;
  context.anthropicSystemTransformed = true;
}

function applyClaudeCodeCacheProfile(
  request: UnifiedChatRequest,
  providerMode: AnthropicProviderMode | undefined,
  relocatedSystemBlock: TextContent | undefined
): void {
  const cacheControl: { type: "ephemeral"; ttl?: "1h" } = {
    type: "ephemeral",
    ...(providerMode === "claude_oauth" ? { ttl: "1h" } : {}),
  };

  delete (request as any).cache_control;
  for (const tool of request.tools || []) delete (tool as any).cache_control;

  // Normally system[] is billing + identity. Without a user message to relocate
  // into, the caller's blocks stay here too; mark only the first and last
  // cacheable block so identity + system + tail stay within Anthropic's 4.
  const systemBlocks = ((request.system as any[]) || []).filter(Boolean);
  const cacheable = systemBlocks.filter(
    (block) =>
      typeof block.text === "string" &&
      !block.text.startsWith("x-anthropic-billing-header")
  );
  for (const block of systemBlocks) {
    if (block === cacheable[0] || block === cacheable.at(-1)) {
      block.cache_control = { ...cacheControl };
    } else {
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

  // Claude Code caches its system prompt as a system block. Relocation moves
  // the caller's prompt into the first user message, so mark it there: tools +
  // system + instructions stay cached even when the client later rewrites its
  // history (memory windows, summarization). Identity + this + tail = 3 of 4.
  if (relocatedSystemBlock) {
    (relocatedSystemBlock as any).cache_control = { ...cacheControl };
  }

  // The body builder omits a turn with nothing to send, so a tail that cannot
  // take the marker hands it to the previous eligible message.
  const messages = request.messages || [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if (markCacheTail(messages[i], cacheControl)) return;
  }
}

const ASSISTANT_REASONING_PARTS = ["thinking", "redacted_thinking", "fallback"];

/** Place the tail breakpoint on `message`; false when it cannot carry one. */
function markCacheTail(
  message: UnifiedChatRequest["messages"][number],
  cacheControl: { type: "ephemeral"; ttl?: "1h" }
): boolean {
  if (message.role === "tool") {
    // A Unified tool message becomes one Anthropic tool_result block, and the
    // body builder only carries the message-level marker onto that block.
    // Claude Code marks the tool_result itself; a marker on a content part is
    // lost when single-text tool content collapses to a string.
    (message as any).cache_control = { ...cacheControl };
    return true;
  }
  if (typeof message.content === "string") {
    // The body builder drops empty text blocks, and a marker with them.
    if (!message.content) return false;
    message.content = [{
      type: "text",
      text: message.content,
      cache_control: { ...cacheControl },
    }];
    return true;
  }
  if (!Array.isArray(message.content)) return false;
  // An assistant turn ending in reasoning is skipped as a whole, as in Claude
  // Code's current `a8b`/`zGb` path.
  if (
    message.role === "assistant" &&
    ASSISTANT_REASONING_PARTS.includes((message.content.at(-1) as any)?.type)
  ) {
    return false;
  }
  for (let i = message.content.length - 1; i >= 0; i -= 1) {
    const part = message.content[i] as any;
    // Claude Code marks the last eligible content block, not necessarily a
    // text block: user tool results and assistant tool_use blocks can be the
    // cache breakpoint. Parts the body builder does not send (empty text,
    // image without URL, ...) would drop the marker with them, so skip those.
    if (
      message.role === "assistant" &&
      ASSISTANT_REASONING_PARTS.includes(part?.type)
    ) {
      continue;
    }
    if (
      message.role === "user"
        ? unifiedUserPartToAnthropic(part).length === 0
        : part?.type === "text" && !part.text
    ) {
      continue;
    }
    if (part && typeof part === "object") {
      part.cache_control = { ...cacheControl };
      return true;
    }
  }
  return false;
}

import type { RouterScenarioType } from "@/utils/router";
import type {
  AnthropicClientKind,
  AnthropicProviderMode,
} from "@/utils/anthropic-client-policy";

/**
 * Inbound client protocols supported by CCR's gateway lifecycle.
 */
export type ClientProtocol =
  | "anthropic_messages"
  | "openai_chat_completions"
  | "openai_responses";

export interface AnthropicSourceRequestFields {
  metadata?: Record<string, unknown>;
  thinking?: Record<string, unknown>;
  outputConfig?: Record<string, unknown>;
  stopSequences?: string[];
}

export interface ClientProtocolContext {
  protocol: ClientProtocol;
  pathname: string;
  /** Canonical Fastify route path (without preset prefix), e.g. /v1/responses */
  canonicalPath: string;
  /** Alias path that matched, if different from canonical */
  matchedPath: string;
  originalModel?: string;
  /** Client selected the gateway's trailing `[1m]` context variant. */
  requestedOneMillion?: boolean;
  stream: boolean;
  scenarioType?: RouterScenarioType;
  /** Source-only Anthropic semantics retained before destination routing. */
  anthropicSource?: AnthropicSourceRequestFields;
  /** Client fingerprint captured before Anthropic normalization. */
  anthropicClientKind?: AnthropicClientKind;
  /** In-scope Anthropic destination/auth variant selected after routing. */
  anthropicProviderMode?: AnthropicProviderMode;
  anthropicDestinationInScope?: boolean;
  /** Native Desktop/CLI requests must bypass body and response conversion. */
  anthropicNativeWire?: boolean;
  /** Third-party emulation has already modified the Unified projection. */
  anthropicPolicyApplied?: boolean;
  anthropicSystemTransformed?: boolean;
  claudeAuthToolNameMap?: Map<string, string>;
  /** Claude Code routing metadata extracted without mutating the source billing block. */
  claudeCodeSubagent?: boolean;
  taggedSubagentModel?: string;
  /** Transformer that owns this client protocol */
  ownerTransformerName: string;
  /**
   * Client conversation id captured from the original wire (never harness
   * version or system text). Used for prompt-cache affinity and Codex headers.
   */
  sessionId?: string;
}

export interface ProtocolRouteMatch {
  protocol: ClientProtocol;
  canonicalPath: string;
  matchedPath: string;
  ownerTransformerName: string;
  /** True when matchedPath is an alias of canonicalPath */
  isAlias: boolean;
  stream: boolean;
  /** Preset namespace prefix without trailing slash, e.g. /preset/foo */
  presetPrefix?: string;
}

interface ProtocolRouteSpec {
  protocol: ClientProtocol;
  canonicalPath: string;
  aliases: string[];
  ownerTransformerName: string;
  /** Default stream intent when not encoded in the path */
  defaultStream?: boolean;
}

/**
 * Client-facing route table. Aliases are first-class; registration must not
 * rely on transformer endPoint first-wins.
 */
export const PROTOCOL_ROUTE_SPECS: ProtocolRouteSpec[] = [
  {
    protocol: "anthropic_messages",
    canonicalPath: "/v1/messages",
    aliases: [],
    ownerTransformerName: "Anthropic",
  },
  {
    protocol: "openai_chat_completions",
    canonicalPath: "/v1/chat/completions",
    aliases: ["/chat/completions"],
    ownerTransformerName: "OpenAI",
  },
  {
    protocol: "openai_responses",
    canonicalPath: "/v1/responses",
    aliases: ["/responses"],
    ownerTransformerName: "openai-responses",
  },
];

/** Paths that must never be classified as routed LLM posts. */
const EXCLUDED_PATH_SUFFIXES = [
  "/v1/completions",
  "/completions",
  "/v1beta/interactions",
  "/v1/interactions",
];

const PRESET_PREFIX_RE = /^\/preset\/[^/]+/;

function stripQueryAndNormalize(pathname: string): string {
  const withoutQuery = pathname.split("?")[0] || "/";
  if (withoutQuery.length > 1 && withoutQuery.endsWith("/")) {
    return withoutQuery.slice(0, -1);
  }
  return withoutQuery || "/";
}

function stripPresetPrefix(pathname: string): {
  suffix: string;
  presetPrefix?: string;
} {
  const match = pathname.match(PRESET_PREFIX_RE);
  if (!match) {
    return { suffix: pathname };
  }
  const presetPrefix = match[0];
  const suffix = pathname.slice(presetPrefix.length) || "/";
  return { suffix, presetPrefix };
}

function isExcludedSuffix(suffix: string): boolean {
  // Exact match only. endsWith("/completions") would falsely exclude
  // "/v1/chat/completions".
  return EXCLUDED_PATH_SUFFIXES.some((excluded) => suffix === excluded);
}

function matchStaticProtocol(suffix: string): ProtocolRouteMatch | null {
  for (const spec of PROTOCOL_ROUTE_SPECS) {
    const paths = [spec.canonicalPath, ...spec.aliases];
    for (const path of paths) {
      if (suffix === path) {
        return {
          protocol: spec.protocol,
          canonicalPath: spec.canonicalPath,
          matchedPath: path,
          ownerTransformerName: spec.ownerTransformerName,
          isAlias: path !== spec.canonicalPath,
          stream: Boolean(spec.defaultStream),
        };
      }
    }
  }
  return null;
}

/**
 * Match a client LLM POST path to a protocol. Works with preset prefixes,
 * trailing-slash normalization, and query stripping.
 */
export function matchClientProtocol(
  method: string,
  pathnameOrUrl: string
): ProtocolRouteMatch | null {
  if (method.toUpperCase() !== "POST") {
    return null;
  }

  const normalized = stripQueryAndNormalize(pathnameOrUrl);
  const { suffix, presetPrefix } = stripPresetPrefix(normalized);

  if (isExcludedSuffix(suffix)) {
    return null;
  }

  const staticMatch = matchStaticProtocol(suffix);
  if (!staticMatch) {
    return null;
  }

  return { ...staticMatch, presetPrefix };
}

/** True when method+path is an in-scope routed LLM POST. */
export function isRoutedLlmPost(method: string, pathnameOrUrl: string): boolean {
  return matchClientProtocol(method, pathnameOrUrl) !== null;
}

/** All Fastify paths that should be registered for a protocol (canonical + aliases). */
export function getRegisteredPathsForProtocol(
  protocol: ClientProtocol
): string[] {
  const spec = PROTOCOL_ROUTE_SPECS.find((s) => s.protocol === protocol);
  if (!spec) return [];
  return [spec.canonicalPath, ...spec.aliases];
}

/** Flat list of registered client routes and their owner transformers. */
export function listClientRouteRegistrations(): Array<{
  path: string;
  protocol: ClientProtocol;
  ownerTransformerName: string;
  isCanonical: boolean;
}> {
  const result: Array<{
    path: string;
    protocol: ClientProtocol;
    ownerTransformerName: string;
    isCanonical: boolean;
  }> = [];

  for (const spec of PROTOCOL_ROUTE_SPECS) {
    result.push({
      path: spec.canonicalPath,
      protocol: spec.protocol,
      ownerTransformerName: spec.ownerTransformerName,
      isCanonical: true,
    });
    for (const alias of spec.aliases) {
      result.push({
        path: alias,
        protocol: spec.protocol,
        ownerTransformerName: spec.ownerTransformerName,
        isCanonical: false,
      });
    }
  }

  return result;
}

export function createClientProtocolContext(
  match: ProtocolRouteMatch,
  options?: {
    originalModel?: string;
    stream?: boolean;
    scenarioType?: RouterScenarioType;
  }
): ClientProtocolContext {
  return {
    protocol: match.protocol,
    pathname: match.matchedPath,
    canonicalPath: match.canonicalPath,
    matchedPath: match.matchedPath,
    originalModel: options?.originalModel,
    stream: options?.stream ?? match.stream,
    scenarioType: options?.scenarioType,
    ownerTransformerName: match.ownerTransformerName,
  };
}

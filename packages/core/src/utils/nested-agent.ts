type MessageLike = { role?: string; content?: unknown };

function contentFromResponsesItem(item: any): unknown {
  if (!item || typeof item !== "object") return undefined;
  if (item.content !== undefined) return item.content;
  return undefined;
}

function messagesOf(request: unknown): MessageLike[] {
  if (!request || typeof request !== "object") return [];
  const body = request as { messages?: unknown; input?: unknown };
  if (Array.isArray(body.messages)) return body.messages as MessageLike[];
  // Responses inbound / wire-keep uses `input[]` (EasyInput or typed message items).
  if (Array.isArray(body.input)) {
    const out: MessageLike[] = [];
    for (const item of body.input) {
      if (!item || typeof item !== "object") continue;
      const role = (item as MessageLike).role;
      const type = (item as { type?: string }).type;
      if (role === "user" || type === "message") {
        out.push({
          role: role || "user",
          content: contentFromResponsesItem(item),
        });
      }
    }
    return out;
  }
  return [];
}

/** First user-turn text only — never system (billing / harness version). */
export function firstUserText(request: unknown): string {
  for (const msg of messagesOf(request)) {
    if (msg.role !== "user") continue;
    if (typeof msg.content === "string") return msg.content;
    if (Array.isArray(msg.content)) {
      return msg.content
        .map((p: any) => (typeof p === "string" ? p : p?.text || ""))
        .join("");
    }
    return "";
  }
  return "";
}

export function isHarnessUserNoise(text: string): boolean {
  const trimmed = text.trimStart();
  return (
    !trimmed ||
    trimmed.startsWith("<system-reminder>") ||
    trimmed.startsWith("<local-command-caveat>")
  );
}

export function userMessageTextParts(content: unknown): string[] {
  if (typeof content === "string") return content ? [content] : [];
  if (!Array.isArray(content)) return [];
  const parts: string[] = [];
  for (const part of content) {
    if (typeof part === "string") {
      if (part) parts.push(part);
      continue;
    }
    if (
      (part?.type === "text" || part?.type === "input_text") &&
      typeof part.text === "string" &&
      part.text
    ) {
      parts.push(part.text);
    }
  }
  return parts;
}

/**
 * First user text that distinguishes a worker transcript.
 * Shared reminder/caveat preambles are skipped so parallel Tasks do not collide.
 */
export function firstSubstantiveUserText(request: unknown): string {
  for (const msg of messagesOf(request)) {
    if (msg.role !== "user") continue;
    const texts = userMessageTextParts(msg.content).filter(
      (text) => !isHarnessUserNoise(text)
    );
    if (texts.length) return texts.join("");
  }
  return firstUserText(request);
}

const STATUSLINE_USER_RE =
  /Describe your most recent action in 3-5 words/i;

function lastUserText(request: unknown): string {
  const messages = messagesOf(request);
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== "user") continue;
    return userMessageTextParts(msg.content).join("");
  }
  return "";
}

/** Statusline / spinner polls — must not supersede or become a cache baseline. */
export function isStatuslinePollTurn(request: unknown): boolean {
  return STATUSLINE_USER_RE.test(lastUserText(request));
}

const FORK_OPENING_RE =
  /<fork-boilerplate>|You are a worker fork/i;

/** Parent-lineage headers. Never used as the conversation id. */
const PARENT_SESSION_HEADERS = [
  "x-parent-session-id",
  "x-kilocode-parent-taskid",
];

/** Nested-agent labels (Codex and similar). */
const NESTED_AGENT_HEADERS = ["x-openai-subagent"];

function headerValue(
  headers: unknown,
  name: string
): string | undefined {
  if (!headers || typeof headers !== "object") return undefined;
  const want = name.toLowerCase();
  for (const [rawName, rawValue] of Object.entries(
    headers as Record<string, unknown>
  )) {
    if (rawName.toLowerCase() !== want || rawValue == null) continue;
    const value = Array.isArray(rawValue)
      ? rawValue.find((item) => item != null && String(item))
      : rawValue;
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return undefined;
}

/**
 * Nested/worker agent on any inbound protocol.
 *
 * Claude Code: `cc_is_subagent`.
 * OpenCode / Kilocode / MiMo: child session + `x-parent-session-id`.
 * Kilocode gateway: `X-KILOCODE-PARENT-TASKID`.
 * Codex: `x-openai-subagent`.
 * Cursor/Claude forks: `<fork-boilerplate>` / worker-fork opening text.
 * Grok CLI: child `x-grok-session-id` (unique) plus opening-text mix.
 */
export function detectNestedAgent(input: {
  headers?: unknown;
  body?: unknown;
  claudeCodeSubagent?: boolean;
}): boolean {
  if (input.claudeCodeSubagent === true) return true;
  for (const name of PARENT_SESSION_HEADERS) {
    if (headerValue(input.headers, name)) return true;
  }
  for (const name of NESTED_AGENT_HEADERS) {
    if (headerValue(input.headers, name)) return true;
  }
  const opening = firstSubstantiveUserText(input.body);
  return FORK_OPENING_RE.test(opening);
}

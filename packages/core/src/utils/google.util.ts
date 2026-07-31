import { ImageContent, MessageContent } from "../types/llm";
// @ts-expect-error - latex-to-unicode is a plain JS library without type definitions
import latexToUnicode from "latex-to-unicode";

// ---------------------------------------------------------------------------
// Finish reasons
// ---------------------------------------------------------------------------

/**
 * Google finish reasons that have a Unified (OpenAI) equivalent. Anything not
 * listed here is passed through lowercased and lands on Anthropic's `end_turn`
 * default — correct for genuine completions, and harmless for upstream failures
 * because those also carry a visible notice (see buildAbnormalFinishNotice).
 *
 * Without this map, MAX_TOKENS reached Anthropic as `end_turn`, so a response
 * truncated mid-answer was indistinguishable from a finished one.
 */
const GOOGLE_FINISH_REASON_MAP: Record<string, string> = {
  STOP: "stop",
  MAX_TOKENS: "length",
  SAFETY: "content_filter",
  RECITATION: "content_filter",
  BLOCKLIST: "content_filter",
  PROHIBITED_CONTENT: "content_filter",
  SPII: "content_filter",
  IMAGE_SAFETY: "content_filter",
};

/** Marker that prefixes CCR's upstream-failure notice; used to strip it on replay. */
export const UPSTREAM_STOP_NOTICE = "⚠️ Upstream ended the turn without a reply";

export function normalizeGoogleFinishReason(
  raw?: string | null
): string | null {
  if (!raw) return null;
  return GOOGLE_FINISH_REASON_MAP[raw.toUpperCase()] || raw.toLowerCase();
}

/**
 * Text to show when the upstream ends a turn abnormally with nothing to show:
 * no text, no tool calls. Returns null for normal completions.
 *
 * Gemini 3 / Antigravity return e.g. `MALFORMED_FUNCTION_CALL` ("Function call
 * is empty - no input to parse") after streaming only thinking. Without this the
 * turn reached Claude Code as a silent, successful `end_turn`, so the user saw
 * an assistant turn that said nothing and had to prompt again.
 */
export function buildAbnormalFinishNotice(candidate?: any): string | null {
  const raw = candidate?.finishReason;
  if (!raw || String(raw).toUpperCase() === "STOP") return null;
  const detail = String(candidate?.finishMessage || "").trim();
  return (
    `${UPSTREAM_STOP_NOTICE} (${String(raw).toUpperCase()})` +
    (detail ? `: ${detail}` : "")
  );
}

// ---------------------------------------------------------------------------
// ThinkingSequencer
// ---------------------------------------------------------------------------

/**
 * Callbacks used by ThinkingSequencer to emit SSE chunks.
 * The caller provides implementations that handle actual SSE serialization.
 */
export interface ThinkingSequencerEmit {
  thinking: (content: string, chunk?: any) => void;
  signature: (sig: string, chunk?: any) => void;
  content: (
    text: string,
    meta?: {
      chunk?: any;
      candidate?: any;
      mode?: "direct" | "buffered" | "placeholder" | "finish";
      finishReason?: string | null;
    }
  ) => void;
}

/**
 * State machine that enforces Anthropic-safe emission order for Gemini thinking:
 *   Thinking Content → Thinking Signature → Final Content
 *
 * Supports two upstream stream dialects:
 *
 * 1) Public Gemini API (and Vertex):
 *    - Happy path: thought:true → thoughtSignature → text
 *    - Gemini 3 out-of-order: thought:true → text (buffer) → signature (flush)
 *    - Empty thinking: signature first → emit "(no content)" thinking → text
 *
 * 2) Antigravity:
 *    - Visible text first (often with no thought:true parts)
 *    - Final event: thoughtSignature + empty text + STOP
 *    - Must NOT open a late thinking block after text (Claude Code drops the turn)
 *
 * Rule: if visible content was already emitted when a signature arrives, record
 * the signature as handled and skip thinking/signature emission.
 */
export class ThinkingSequencer {
  private _hasThinking = false;
  private _sigSent = false;
  private _contentSent = false;
  private _buffer = "";

  constructor(private emit: ThinkingSequencerEmit) {}

  /** Called when thinking text arrives. Emits immediately. */
  processThinking(text: string, chunk?: any): void {
    this._hasThinking = true;
    this.emit.thinking(text, chunk);
  }

  /**
   * Called when a signature arrives.
   * - Public Gemini: emit thinking placeholder if needed, emit signature, flush buffer
   * - Antigravity trailer after content: no-op emission (see class docs)
   */
  processSignature(sig: string, chunk?: any): void {
    this.processSignatureWithMeta(sig, chunk);
  }

  processSignatureWithMeta(
    sig: string,
    chunk?: any,
    meta?: {
      beforeFlush?: () => void;
      flushMeta?: {
        chunk?: any;
        candidate?: any;
        mode?: "direct" | "buffered" | "placeholder" | "finish";
        finishReason?: string | null;
      };
    }
  ): void {
    if (this._sigSent) return;
    // Antigravity: content already streamed — do not emit late thinking.
    if (this._contentSent) {
      this._sigSent = true;
      return;
    }
    if (!this._hasThinking) {
      this._hasThinking = true;
      this.emit.thinking("(no content)", chunk);
    }
    this._sigSent = true;
    this.emit.signature(sig, chunk);
    meta?.beforeFlush?.();
    this.flushBufferedContent(meta?.flushMeta);
  }

  /**
   * Called when content text arrives.
   * - Signature already sent or no thinking at all: emit immediately
   * - Thinking seen but no signature yet: buffer (public Gemini / Gemini 3)
   */
  processContent(text: string, chunk?: any, candidate?: any): void {
    if (this._sigSent || !this._hasThinking) {
      this._contentSent = true;
      this.emit.content(text, { chunk, candidate, mode: "direct" });
    } else {
      this._buffer += text;
    }
  }

  emitContentPlaceholder(
    text: string,
    meta?: {
      chunk?: any;
      candidate?: any;
      finishReason?: string | null;
    }
  ): void {
    this._contentSent = true;
    this.emit.content(text, {
      chunk: meta?.chunk,
      candidate: meta?.candidate,
      mode: "placeholder",
      finishReason: meta?.finishReason,
    });
  }

  /** Explicitly buffer content (for Gemini 3 out-of-order delivery). */
  bufferContent(text: string): void {
    this._buffer += text;
  }

  /**
   * Finalize the stream:
   * - Emits fallback signature if thinking was seen but no signature arrived
   * - Flushes any remaining buffered content
   */
  finalize(
    chunk?: any,
    candidate?: any,
    options?: { beforeFlush?: () => void }
  ): void {
    if (this._hasThinking && !this._sigSent) {
      this.processSignatureWithMeta(`ccr_${Date.now()}`, chunk, {
        beforeFlush: options?.beforeFlush,
        flushMeta: {
          chunk,
          candidate,
          mode: candidate?.finishReason ? "finish" : "buffered",
          finishReason: normalizeGoogleFinishReason(candidate?.finishReason),
        },
      });
      return;
    }
    this.flushBufferedContent({
      chunk,
      candidate,
      mode: candidate?.finishReason ? "finish" : "buffered",
      finishReason: normalizeGoogleFinishReason(candidate?.finishReason),
    });
  }

  flushBufferedContent(meta?: {
    chunk?: any;
    candidate?: any;
    mode?: "direct" | "buffered" | "placeholder" | "finish";
    finishReason?: string | null;
  }): void {
    if (this._buffer) {
      this._contentSent = true;
      this.emit.content(this._buffer, {
        chunk: meta?.chunk,
        candidate: meta?.candidate,
        mode: meta?.mode || "buffered",
        finishReason: meta?.finishReason,
      });
      this._buffer = "";
    }
  }

  /**
   * Whether Gemini 3 content should be deferred (signature not yet seen,
   * not finishing, no tool calls). Public Gemini 3 out-of-order path.
   */
  shouldDeferContent(isFinish: boolean, hasToolCalls: boolean): boolean {
    return this._hasThinking && !this._sigSent && !isFinish && !hasToolCalls;
  }

  get hasBufferedContent(): boolean { return this._buffer.length > 0; }
  get hasThinkingContent(): boolean { return this._hasThinking; }
  get signatureSent(): boolean { return this._sigSent; }
  get contentSent(): boolean { return this._contentSent; }
  get needsContentPlaceholder(): boolean { return this._sigSent && !this._contentSent; }

  /**
   * Mark a thoughtSignature as handled without emitting thinking/signature/content.
   * Used when the signature belongs on functionCall parts (tool turns) and must
   * not invent "(no content)" thinking/text placeholders.
   */
  acknowledgeSignature(): void {
    this._sigSent = true;
  }
}


/**
 * Interface for normalized image data before provider-specific wrapping
 */
export interface NormalizedImage {
  url: string;
  mediaType: string;
  isBase64: boolean;
}

/**
 * Maps a role based on a provided mapping object.
 * Example mapping: { assistant: 'model' }
 */
export function mapRole(role: string, mapping: Record<string, string>): string {
  return mapping[role] || role;
}

/**
 * Identifies image_url content and normalizes it into a consistent format.
 * Returns an array of NormalizedImage for any images found in the content.
 */
export function extractImageParts(content: string | null | MessageContent[]): NormalizedImage[] {
  if (!content || typeof content === "string") {
    return [];
  }

  return content
    .filter((item): item is ImageContent => item.type === "image_url")
    .map((item) => ({
      url: item.image_url.url,
      mediaType: item.media_type || "image/jpeg",
      isBase64: item.image_url.url.startsWith("data:"),
    }));
}

/**
 * Merges consecutive messages with the same role into a single message
 * by combining their contents in the specified field.
 */
export function consolidateMessages<T extends { role: string;[key: string]: any }>(
  messages: T[],
  contentField: keyof T
): T[] {
  if (messages.length === 0) return [];

  const consolidated: T[] = [];
  for (const msg of messages) {
    const lastMsg = consolidated[consolidated.length - 1];
    if (lastMsg && lastMsg.role === msg.role) {
      const lastContent = lastMsg[contentField];
      const currentContent = msg[contentField];

      if (Array.isArray(lastContent) && Array.isArray(currentContent)) {
        (lastContent as any[]).push(...(currentContent as any[]));
      }
    } else {
      consolidated.push({ ...msg });
    }
  }
  return consolidated;
}

/**
 * Normalizes a tool definition to a standard { name, description, parameters } shape.
 * Handles both OpenAI-style and Anthropic-style tool definitions.
 */
export function normalizeTool(tool: any): { name: string; description: string; parameters: any } {
  // OpenAI shape: { function: { name, description, parameters } }
  if (tool.function?.name) {
    return {
      name: tool.function.name,
      description: tool.function.description,
      parameters: tool.function.parameters,
    };
  }
  // Anthropic shape: { name, description, input_schema }
  return {
    name: tool.name,
    description: tool.description,
    parameters: tool.input_schema,
  };
}

/**
 * Wraps normalized image data into provider-specific JSON structures.
 */
export function processImageContent(
  normalizedImage: NormalizedImage,
  provider: "gemini" | "claude"
): any {
  if (provider === "gemini") {
    // camelCase keys: v1internal (Antigravity) rejects the snake_case aliases
    // that the public REST API tolerates.
    if (normalizedImage.isBase64) {
      return {
        inlineData: {
          mimeType: normalizedImage.mediaType,
          data: normalizedImage.url.split(",").pop() || normalizedImage.url,
        },
      };
    }
    return {
      fileData: {
        mimeType: normalizedImage.mediaType,
        fileUri: normalizedImage.url,
      },
    };
  }

  if (provider === "claude") {
    return {
      type: "image",
      source: {
        type: "base64",
        media_type: normalizedImage.mediaType,
        data: normalizedImage.url.startsWith("data:")
          ? normalizedImage.url.split(",").pop() || normalizedImage.url
          : normalizedImage.url,
      },
    };
  }

  throw new Error(`Unsupported provider for image processing: ${provider}`);
}

export function replaceLatexSymbols(text: string): string {
  if (!text) return text;

  // Guard: if no backslash and no dollar sign, it's likely not LaTeX we want to touch
  if (!text.includes("\\") && !text.includes("$")) return text;

  try {
    const converter =
      typeof latexToUnicode === "function"
        ? latexToUnicode
        : (latexToUnicode as any).default;

    if (typeof converter !== "function") return text;

    // Pattern to match LaTeX math blocks and commands:
    // 1. $$...$$ or $...$
    // 2. \(...\) or \[...\]
    // 3. \command (backslash followed by letters)
    const latexPattern =
      /(\$\$?[\s\S]+?\$\$?|\\\(?:[\s\S]+?\\\)|\\\[[\s\S]+?\\\]|\\(?:[a-zA-Z]+))/g;

    return text.replace(latexPattern, (match) => {
      let converted = match;
      let prev;
      do {
        prev = converted;
        converted = converter(converted);
      } while (converted !== prev);

      // Some models wrap symbols in $, e.g. $\rightarrow$. 
      // The library converts it to $→$, so we clean up the $ signs if they surround converted chars
      return converted.replace(/\$([^$]+)\$/g, "$1");
    });
  } catch {
    return text;
  }
}

/**
 * Sanitize a function name for Gemini's naming rules:
 * Must start with a letter or underscore, contain only [a-zA-Z0-9_.:\-], max 128 chars 
 */
export function sanitizeGeminiFunctionName(name: string): string {
  if (!name) return "unnamed_function";
  let sanitized = name.replace(/[^a-zA-Z0-9_.:-]/g, "_");
  if (/^[^a-zA-Z_]/.test(sanitized)) {
    sanitized = "_" + sanitized;
  }
  return sanitized.substring(0, 128);
}

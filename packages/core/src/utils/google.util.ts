import { UnifiedMessage, UnifiedTool, ImageContent, MessageContent } from "../types/llm";
// @ts-ignore - latex-to-unicode is a plain JS library without type definitions
import latexToUnicode from "latex-to-unicode";

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
 * State machine that enforces the emission order for Gemini thinking blocks:
 *   Thinking Content -> Thinking Signature -> Final Content
 *
 * Handles:
 * - Happy path: thinking, signature, content arrive in order
 * - Gemini 3: content arrives before signature (buffered until signature)
 * - Missing signature: fallback signature generated at finalize
 * - Empty thinking: signature arrives with no prior thinking content
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
   * - Emits "(no content)" thinking if no thinking was seen
   * - Emits the signature
   * - Flushes any buffered content
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
   * - Thinking seen but no signature yet: buffer
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
          finishReason: candidate?.finishReason?.toLowerCase() || null,
        },
      });
      return;
    }
    this.flushBufferedContent({
      chunk,
      candidate,
      mode: candidate?.finishReason ? "finish" : "buffered",
      finishReason: candidate?.finishReason?.toLowerCase() || null,
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

  /** Whether Gemini 3 content should be deferred (signature not yet seen, not finishing, no tool calls). */
  shouldDeferContent(isFinish: boolean, hasToolCalls: boolean): boolean {
    return this._hasThinking && !this._sigSent && !isFinish && !hasToolCalls;
  }

  get hasBufferedContent(): boolean { return this._buffer.length > 0; }
  get hasThinkingContent(): boolean { return this._hasThinking; }
  get signatureSent(): boolean { return this._sigSent; }
  get contentSent(): boolean { return this._contentSent; }
  get needsContentPlaceholder(): boolean { return this._sigSent && !this._contentSent; }
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
    if (normalizedImage.isBase64) {
      return {
        inlineData: {
          mime_type: normalizedImage.mediaType,
          data: normalizedImage.url.split(",").pop() || normalizedImage.url,
        },
      };
    }
    return {
      file_data: {
        mime_type: normalizedImage.mediaType,
        file_uri: normalizedImage.url,
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

/** 
 * Replace LaTeX math symbols that some models generate with standard unicode 
 * using the latex-to-unicode library.
 */
export function replaceLatexSymbols(text: string): string {
  if (!text) return text;
  try {
    // The library only replaces the first occurrence of each symbol
    // So we loop until the string stops changing to ensure all occurrences are handled
    let converted = text;
    let prev;
    const converter = typeof latexToUnicode === "function"
      ? latexToUnicode
      : (latexToUnicode as any).default;

    if (typeof converter !== "function") return text;

    do {
      prev = converted;
      converted = converter(converted);
    } while (converted !== prev);

    // Some models wrap symbols in $, e.g. $\rightarrow$. 
    // The library converts it to $→$, so we clean up the $ signs if they surround converted chars
    return converted.replace(/\$([^\$]+)\$/g, '$1');
  } catch (e) {
    return text;
  }
}

/**
 * Sanitize a function name for Gemini's naming rules:
 * Must start with a letter or underscore, contain only [a-zA-Z0-9_.:\-], max 128 chars 
 */
export function sanitizeGeminiFunctionName(name: string): string {
  if (!name) return "unnamed_function";
  let sanitized = name.replace(/[^a-zA-Z0-9_.:\-]/g, "_");
  if (/^[^a-zA-Z_]/.test(sanitized)) {
    sanitized = "_" + sanitized;
  }
  return sanitized.substring(0, 128);
}

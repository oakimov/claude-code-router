import { UnifiedMessage, UnifiedTool, ImageContent, MessageContent } from "../types/llm";
// @ts-ignore - latex-to-unicode is a plain JS library without type definitions
import latexToUnicode from "latex-to-unicode";


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


/**
 * Multimodal tool-result helpers.
 *
 * Responses / OpenCode attach images (and files) on function_call_output as
 * structured parts. Unified carries them as text / image_url / file parts on
 * role:"tool". Destinations differ:
 *   - Responses/Codex: re-emit input_image / input_file in output[]
 *   - Anthropic: image / document blocks inside tool_result.content
 *   - Gemini: text in functionResponse + sibling inlineData parts
 *   - Chat Completions / Mistral: string tool content only — extract media
 *     into a follow-up user message (OpenCode's pattern for non-supporting APIs)
 */

export type UnifiedToolPart =
  | {
      type: "text";
      text: string;
      cache_control?: any;
    }
  | {
      type: "image_url";
      image_url: { url: string; detail?: string };
      media_type?: string;
      cache_control?: any;
    }
  | {
      type: "file";
      filename?: string;
      file_data?: string;
      file_url?: string;
      media_type?: string;
      cache_control?: any;
    };

const SYNTHETIC_TOOL_MEDIA_PROMPT =
  "The previous tool result included the following media attachment(s).";

export function isUnifiedToolMediaPart(
  part: any
): part is Extract<UnifiedToolPart, { type: "image_url" | "file" }> {
  return (
    !!part &&
    typeof part === "object" &&
    (part.type === "image_url" || part.type === "file")
  );
}

/** Normalize string | part[] tool content into a part list. */
export function normalizeUnifiedToolParts(content: unknown): UnifiedToolPart[] {
  if (typeof content === "string") {
    return content ? [{ type: "text", text: content }] : [];
  }
  if (!Array.isArray(content)) {
    if (content == null) return [];
    return [{ type: "text", text: JSON.stringify(content) }];
  }
  const parts: UnifiedToolPart[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    if (part.type === "text") {
      parts.push({
        type: "text",
        text: String(part.text ?? ""),
        ...(part.cache_control ? { cache_control: part.cache_control } : {}),
      });
    } else if (part.type === "image_url" && part.image_url?.url) {
      parts.push({
        type: "image_url",
        image_url: {
          url: String(part.image_url.url),
          ...(typeof part.image_url.detail === "string" && part.image_url.detail
            ? { detail: part.image_url.detail }
            : {}),
        },
        ...(part.media_type ? { media_type: part.media_type } : {}),
        ...(part.cache_control ? { cache_control: part.cache_control } : {}),
      });
    } else if (part.type === "file") {
      parts.push({
        type: "file",
        ...(part.filename ? { filename: String(part.filename) } : {}),
        ...(part.file_data ? { file_data: String(part.file_data) } : {}),
        ...(part.file_url ? { file_url: String(part.file_url) } : {}),
        ...(part.media_type ? { media_type: String(part.media_type) } : {}),
        ...(part.cache_control ? { cache_control: part.cache_control } : {}),
      });
    }
  }
  return parts;
}

export function unifiedToolTextOnly(content: unknown): string {
  const texts = normalizeUnifiedToolParts(content)
    .filter((p): p is Extract<UnifiedToolPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .filter(Boolean);
  return texts.join("\n");
}

export function unifiedToolHasMedia(content: unknown): boolean {
  return normalizeUnifiedToolParts(content).some(isUnifiedToolMediaPart);
}

/** Anthropic tool_result.content: string or (text|image|document)[]. */
export function unifiedToolContentToAnthropic(content: unknown): string | any[] {
  const parts = normalizeUnifiedToolParts(content);
  if (parts.length === 0) return "";
  if (parts.length === 1 && parts[0].type === "text") {
    return parts[0].text;
  }

  const blocks: any[] = [];
  for (const part of parts) {
    if (part.type === "text") {
      blocks.push({
        type: "text",
        text: part.text,
        ...(part.cache_control ? { cache_control: part.cache_control } : {}),
      });
      continue;
    }
    if (part.type === "image_url") {
      const url = part.image_url.url;
      if (url.startsWith("data:")) {
        const [meta, data] = url.split(",");
        const mediaType =
          part.media_type ||
          meta.split(":")[1]?.split(";")[0] ||
          "image/jpeg";
        blocks.push({
          type: "image",
          source: { type: "base64", media_type: mediaType, data },
          ...(part.cache_control ? { cache_control: part.cache_control } : {}),
        });
      } else {
        blocks.push({
          type: "image",
          source: { type: "url", url },
          ...(part.cache_control ? { cache_control: part.cache_control } : {}),
        });
      }
      continue;
    }
    // file → Anthropic document (PDF) or image when media is image/*
    const mediaType =
      part.media_type ||
      mediaTypeFromDataUrl(part.file_data) ||
      "application/pdf";
    if (mediaType.startsWith("image/") && part.file_data) {
      const data = part.file_data.includes(",")
        ? part.file_data.split(",").pop()!
        : part.file_data;
      blocks.push({
        type: "image",
        source: { type: "base64", media_type: mediaType, data },
        ...(part.cache_control ? { cache_control: part.cache_control } : {}),
      });
    } else if (part.file_url) {
      blocks.push({
        type: "document",
        source: { type: "url", url: part.file_url },
        ...(part.cache_control ? { cache_control: part.cache_control } : {}),
      });
    } else if (part.file_data) {
      const data = part.file_data.includes(",")
        ? part.file_data.split(",").pop()!
        : part.file_data;
      blocks.push({
        type: "document",
        source: {
          type: "base64",
          media_type: mediaType,
          data,
        },
        ...(part.filename ? { title: part.filename } : {}),
        ...(part.cache_control ? { cache_control: part.cache_control } : {}),
      });
    }
  }
  return blocks.length === 1 && blocks[0].type === "text"
    ? blocks[0].text
    : blocks.length > 0
      ? blocks
      : "";
}

/**
 * Anthropic inbound tool_result.content → Unified tool content
 * (string or text/image_url/file parts).
 */
export function anthropicToolResultToUnified(content: unknown): string | any[] {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) {
    return content == null ? "" : JSON.stringify(content);
  }
  const parts: any[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    if (block.type === "text" && block.text) {
      parts.push({
        type: "text",
        text: block.text,
        ...(block.cache_control ? { cache_control: block.cache_control } : {}),
      });
    } else if (block.type === "image" && block.source) {
      const url =
        block.source.type === "base64"
          ? `data:${block.source.media_type || "image/jpeg"};base64,${block.source.data}`
          : block.source.url;
      if (typeof url === "string" && url) {
        parts.push({
          type: "image_url",
          image_url: { url },
          media_type: block.source.media_type,
          ...(block.cache_control ? { cache_control: block.cache_control } : {}),
        });
      }
    } else if (block.type === "document" && block.source) {
      if (block.source.type === "base64" && block.source.data) {
        const mediaType = block.source.media_type || "application/pdf";
        parts.push({
          type: "file",
          filename: block.title || block.source.media_type || "document",
          file_data: `data:${mediaType};base64,${block.source.data}`,
          media_type: mediaType,
          ...(block.cache_control ? { cache_control: block.cache_control } : {}),
        });
      } else if (block.source.type === "url" && block.source.url) {
        parts.push({
          type: "file",
          filename: block.title || "document",
          file_url: block.source.url,
          media_type: block.source.media_type || "application/pdf",
          ...(block.cache_control ? { cache_control: block.cache_control } : {}),
        });
      }
    }
  }
  if (parts.length === 1 && parts[0].type === "text") return parts[0].text;
  return parts.length > 0 ? parts : "";
}

/** Gemini sibling inlineData / fileData parts for tool media. */
export function unifiedToolMediaToGeminiParts(content: unknown): any[] {
  const parts: any[] = [];
  for (const part of normalizeUnifiedToolParts(content)) {
    if (part.type === "image_url") {
      const url = part.image_url.url;
      const mediaType =
        part.media_type ||
        mediaTypeFromDataUrl(url) ||
        "image/jpeg";
      if (url.startsWith("data:")) {
        parts.push({
          inlineData: {
            mimeType: mediaType,
            data: url.split(",").pop() || url,
          },
        });
      } else {
        parts.push({
          fileData: { mimeType: mediaType, fileUri: url },
        });
      }
    } else if (part.type === "file") {
      const mediaType =
        part.media_type ||
        mediaTypeFromDataUrl(part.file_data) ||
        "application/pdf";
      if (part.file_data) {
        parts.push({
          inlineData: {
            mimeType: mediaType,
            data: part.file_data.includes(",")
              ? part.file_data.split(",").pop()!
              : part.file_data,
          },
        });
      } else if (part.file_url) {
        parts.push({
          fileData: { mimeType: mediaType, fileUri: part.file_url },
        });
      }
    }
  }
  return parts;
}

/**
 * Chat Completions / Mistral: string-only tool content. Pull media out of
 * tool messages and insert a synthetic user message after each contiguous
 * tool-result group so vision still reaches the model.
 */
export function extractToolMediaForStringToolApis(messages: any[]): any[] {
  if (!Array.isArray(messages) || messages.length === 0) return messages;

  const out: any[] = [];
  let pendingMedia: any[] = [];

  const flushMedia = () => {
    if (pendingMedia.length === 0) return;
    out.push({
      role: "user",
      content: [
        { type: "text", text: SYNTHETIC_TOOL_MEDIA_PROMPT },
        ...pendingMedia,
      ],
    });
    pendingMedia = [];
  };

  for (const msg of messages) {
    if (msg?.role !== "tool") {
      flushMedia();
      out.push(msg);
      continue;
    }

    if (!unifiedToolHasMedia(msg.content)) {
      out.push(msg);
      continue;
    }

    const parts = normalizeUnifiedToolParts(msg.content);
    const text = parts
      .filter((p): p is Extract<UnifiedToolPart, { type: "text" }> => p.type === "text")
      .map((p) => p.text)
      .filter(Boolean)
      .join("\n");
    out.push({ ...msg, content: text });

    for (const part of parts) {
      if (part.type === "image_url") {
        pendingMedia.push({
          type: "image_url",
          image_url: part.image_url,
          ...(part.media_type ? { media_type: part.media_type } : {}),
        });
      } else if (part.type === "file") {
        // Chat Completions has no portable file part. Image data-URLs can ride
        // as image_url; other files become a short notice (PDF bytes stay on
        // Responses/Anthropic/Gemini paths that understand documents).
        if (
          part.file_data?.startsWith("data:image/") ||
          (part.media_type?.startsWith("image/") && part.file_data)
        ) {
          pendingMedia.push({
            type: "image_url",
            image_url: { url: part.file_data! },
            media_type: part.media_type,
          });
        } else {
          pendingMedia.push({
            type: "text",
            text: `[Attached file${part.filename ? `: ${part.filename}` : ""}${
              part.media_type ? ` (${part.media_type})` : ""
            }${part.file_url ? `: ${part.file_url}` : ""}]`,
          });
        }
      }
    }
  }
  flushMedia();
  return out;
}

function mediaTypeFromDataUrl(dataUrl?: string): string | undefined {
  if (!dataUrl || !dataUrl.startsWith("data:")) return undefined;
  return dataUrl.slice(5).split(";")[0] || undefined;
}

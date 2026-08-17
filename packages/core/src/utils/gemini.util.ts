import { UnifiedChatRequest, UnifiedMessage } from "../types/llm";
import { Content, ContentListUnion, Part, ToolListUnion } from "@google/genai";
import { collapseTypelessUnions, sanitizeJsonSchema } from "./schema";
import { buildGeminiThinkingConfig } from "./gemini-thinking";
import {
  anthropicThinkingSignatureFrom,
  thinkingFromUnifiedAssistant,
} from "./openai.responses.util";

/**
 * Models that Antigravity serves from an Anthropic backend while still speaking
 * the Gemini wire format to us.
 */
function isAnthropicBackedModel(model: string): boolean {
  return String(model || "")
    .toLowerCase()
    .replace(/^models\//, "")
    .startsWith("claude");
}
import { createSSEStreamReader, StreamContext } from "./stream";
import {
  mapRole,
  extractImageParts,
  processImageContent,
  consolidateMessages,
  normalizeTool,
  replaceLatexSymbols,
  sanitizeGeminiFunctionName,
  ThinkingSequencer,
  normalizeGoogleFinishReason,
  buildAbnormalFinishNotice,
  UPSTREAM_STOP_NOTICE,
} from "./google.util";
import {
  recallThoughtSignature,
  rememberThoughtSignature,
} from "./thought-signature-cache";

// Type enum equivalent in JavaScript
const Type = {
  TYPE_UNSPECIFIED: "TYPE_UNSPECIFIED",
  STRING: "STRING",
  NUMBER: "NUMBER",
  INTEGER: "INTEGER",
  BOOLEAN: "BOOLEAN",
  ARRAY: "ARRAY",
  OBJECT: "OBJECT",
  NULL: "NULL",
};

/**
 * Transform the type field from an array of types to an array of anyOf fields.
 * @param {string[]} typeList - List of types
 * @param {Object} resultingSchema - The schema object to modify
 */
function flattenTypeArrayToAnyOf(
  typeList: Array<string>,
  resultingSchema: any
): void {
  if (typeList.includes("null")) {
    resultingSchema["nullable"] = true;
  }
  const listWithoutNull = typeList.filter((type) => type !== "null");

  if (listWithoutNull.length === 1) {
    const upperCaseType = listWithoutNull[0].toUpperCase();
    resultingSchema["type"] = Object.values(Type).includes(upperCaseType)
      ? upperCaseType
      : Type.TYPE_UNSPECIFIED;
  } else {
    resultingSchema["anyOf"] = [];
    for (const i of listWithoutNull) {
      const upperCaseType = i.toUpperCase();
      resultingSchema["anyOf"].push({
        type: Object.values(Type).includes(upperCaseType)
          ? upperCaseType
          : Type.TYPE_UNSPECIFIED,
      });
    }
  }
}

/**
 * Process a JSON schema to make it compatible with the GenAI API
 * @param {Object} _jsonSchema - The JSON schema to process
 * @returns {Object} - The processed schema
 */
function processJsonSchema(_jsonSchema: any): any {
  const genAISchema: Record<string, any> = {};
  const schemaFieldNames = ["items"];
  const listSchemaFieldNames = ["anyOf"];
  const dictSchemaFieldNames = ["properties"];

  if (_jsonSchema["type"] && _jsonSchema["anyOf"]) {
    throw new Error("type and anyOf cannot be both populated.");
  }

  /*
  This is to handle the nullable array or object. The _jsonSchema will
  be in the format of {anyOf: [{type: 'null'}, {type: 'object'}]}. The
  logic is to check if anyOf has 2 elements and one of the element is null,
  if so, the anyOf field is unnecessary, so we need to get rid of the anyOf
  field and make the schema nullable. Then use the other element as the new
  _jsonSchema for processing. This is because the backend doesn't have a null
  type.
  */
  const incomingAnyOf = _jsonSchema["anyOf"];
  if (
    incomingAnyOf != null &&
    Array.isArray(incomingAnyOf) &&
    incomingAnyOf.length == 2
  ) {
    if (incomingAnyOf[0] && incomingAnyOf[0]["type"] === "null") {
      genAISchema["nullable"] = true;
      _jsonSchema = incomingAnyOf[1];
    } else if (incomingAnyOf[1] && incomingAnyOf[1]["type"] === "null") {
      genAISchema["nullable"] = true;
      _jsonSchema = incomingAnyOf[0];
    }
  }

  if (_jsonSchema["type"] && Array.isArray(_jsonSchema["type"])) {
    flattenTypeArrayToAnyOf(_jsonSchema["type"], genAISchema);
  }

  for (const [fieldName, fieldValue] of Object.entries(_jsonSchema)) {
    // Skip if the fieldValue is undefined or null.
    if (fieldValue == null) {
      continue;
    }

    if (fieldName == "type") {
      if (fieldValue === "null") {
        throw new Error(
          "type: null can not be the only possible type for the field."
        );
      }
      if (Array.isArray(fieldValue)) {
        // we have already handled the type field with array of types in the
        // beginning of this function.
        continue;
      }
      const upperCaseValue = String(fieldValue).toUpperCase();
      genAISchema["type"] = Object.values(Type).includes(upperCaseValue)
        ? upperCaseValue
        : Type.TYPE_UNSPECIFIED;
    } else if (schemaFieldNames.includes(fieldName)) {
      genAISchema[fieldName] = processJsonSchema(fieldValue);
    } else if (listSchemaFieldNames.includes(fieldName)) {
      const listSchemaFieldValue = [];
      for (const item of Array.isArray(fieldValue) ? fieldValue : []) {
        if (item["type"] == "null") {
          genAISchema["nullable"] = true;
          continue;
        }
        listSchemaFieldValue.push(processJsonSchema(item));
      }
      genAISchema[fieldName] = listSchemaFieldValue;
    } else if (dictSchemaFieldNames.includes(fieldName)) {
      const dictSchemaFieldValue: Record<string, any> = {};
      for (const [key, value] of Object.entries(fieldValue as Record<string, any>)) {
        dictSchemaFieldValue[key] = processJsonSchema(value);
      }
      genAISchema[fieldName] = dictSchemaFieldValue;
    } else {
      // additionalProperties is not included in JSONSchema, skipping it.
      if (fieldName === "additionalProperties") {
        continue;
      }
      genAISchema[fieldName] = fieldValue;
    }
  }

  // Antigravity / strict Gemini reject ARRAY without items.
  if (genAISchema.type === "ARRAY" && !genAISchema.items) {
    genAISchema.items = { type: "STRING" };
  }

  // Drop required entries that no longer exist after schema sanitization.
  if (Array.isArray(genAISchema.required) && genAISchema.properties) {
    const keys = new Set(Object.keys(genAISchema.properties));
    genAISchema.required = genAISchema.required.filter(
      (name: string) => keys.has(name)
    );
    if (genAISchema.required.length === 0) {
      delete genAISchema.required;
    }
  }

  return genAISchema;
}

/**
 * Transform a tool object
 * @param {Object} tool - The tool object to transform
 * @returns {Object} - The transformed tool object
 */
export function tTool(tool: any, opts?: { collapseUnions?: boolean }): any {
  const finish = (schema: any) =>
    opts?.collapseUnions ? collapseTypelessUnions(schema) : schema;

  if (tool.functionDeclarations) {
    for (const functionDeclaration of tool.functionDeclarations) {
      if (!functionDeclaration.parameters) {
        functionDeclaration.parameters = { type: "OBJECT", properties: {} };
      } else {
        const sanitized = sanitizeJsonSchema(functionDeclaration.parameters);
        functionDeclaration.parameters = finish(processJsonSchema(sanitized));
      }
      if (functionDeclaration.response) {
        const sanitized = sanitizeJsonSchema(functionDeclaration.response);
        functionDeclaration.response = finish(processJsonSchema(sanitized));
      }
    }
  }
  return tool;
}

export const SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator";

export type GeminiBuildOptions = {
  /**
   * How to replay functionCall parts whose thought signature is missing (Claude
   * Code strips the field from tool_use blocks).
   *
   * - "skip" (default): stamp the documented `skip_thought_signature_validator`
   *   sentinel on the first functionCall part of the turn. Gemini 3 and
   *   Antigravity return 400 without it.
   * - "none": never stamp the sentinel — escape hatch for endpoints that reject
   *   it, at the cost of 400s on unsigned tool replays.
   *
   * The sentinel is only reached when the real signature is unavailable: CCR
   * caches signatures per tool-call id (see thought-signature-cache) precisely
   * so that unsigned replays stop degrading the model.
   */
  thoughtSignatureFallback?: "skip" | "none";

  /**
   * Cache scope for remembered thought signatures — the provider name. A
   * signature is only valid at the upstream that minted it, so a mismatch must
   * miss the cache and fall back to the sentinel.
   */
  signatureScope?: string;
};

/**
 * Build a Gemini generateContent body from UnifiedChatRequest.
 *
 * Whitelist-by-construction: messages/tools/system are rebuilt from named fields,
 * so Anthropic `cache_control` markers never reach upstream. Do not refactor toward
 * pass-through of content objects without stripping cache_control first.
 */
export function buildRequestBody(
  request: UnifiedChatRequest,
  opts?: GeminiBuildOptions
): Record<string, any> {
  const tools = [];
  const requestTools = request.tools || [];
  const functionDeclarations = requestTools
    .filter((tool) => normalizeTool(tool).name !== "web_search")
    .map((tool) => {
      const { name, description, parameters } = normalizeTool(tool);
      return {
        name: sanitizeGeminiFunctionName(name),
        description,
        parameters: parameters ?? { type: "object", properties: {} },
      };
    });
  if (functionDeclarations?.length) {
    tools.push(
      tTool(
        { functionDeclarations },
        // Claude on Antigravity is reached through this same Gemini body, but
        // its tool schemas are re-emitted for Anthropic, which rejects the
        // typeless unions Gemini accepts. See collapseTypelessUnions.
        { collapseUnions: isAnthropicBackedModel(request.model) }
      )
    );
  }
  const webSearch = requestTools.find(
    (tool) => normalizeTool(tool).name === "web_search"
  );
  if (webSearch) {
    tools.push({
      googleSearch: {},
    });
  }

  const rawContents: any[] = [];
  const rawMessages = request.messages || [];



  // Collect system instructions from request.system and system role messages
  const systemTexts: string[] = [];
  const extractText = (content: any): void => {
    if (typeof content === "string") {
      if (content) systemTexts.push(content);
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (part?.type === "text" && part.text) systemTexts.push(part.text);
        else if (typeof part === "string" && part) systemTexts.push(part);
      }
    }
  };
  if (request.system) extractText(request.system);
  for (const msg of rawMessages) {
    if (msg.role === "system") extractText(msg.content);
  }

  const toolResponses = rawMessages.filter((item) => item.role === "tool");
  const filteredMessages = rawMessages.filter((msg) => msg.role !== "tool" && msg.role !== "system");

  const skipFallbackEnabled = opts?.thoughtSignatureFallback !== "none";

  filteredMessages.forEach((message: UnifiedMessage) => {
    const role = mapRole(message.role, { assistant: "model" });
    const parts = [];

    // Chat `reasoning_content` and Unified `thinking` are the same history.
    // Unsigned thought parts are still omitted — Gemini 3 / Antigravity 400
    // when a thought part has no thoughtSignature. The sentinel is not
    // substituted here: it belongs on functionCall parts only.
    const unifiedThinking = thinkingFromUnifiedAssistant(message);
    const rawSignature = anthropicThinkingSignatureFrom(unifiedThinking);
    const realSignature = rawSignature?.startsWith("ccr_")
      ? undefined
      : rawSignature;

    // CCR emits "(no content)" as a placeholder for empty model turns (see
    // ThinkingSequencer), and an upstream-failure notice when a turn ends
    // abnormally with nothing to show. Strip both when replaying model turns so
    // Gemini does not start echoing them back as an empty final answer. User and
    // tool text is left alone — it can legitimately contain those strings.
    const isModelPlaceholder = (text: string): boolean =>
      role === "model" &&
      (text === "(no content)" || text.startsWith(UPSTREAM_STOP_NOTICE));

    if (realSignature && role === "model") {
      parts.push({
        thought: true,
        text: unifiedThinking?.content || "",
        thoughtSignature: realSignature,
      });
    }

    if (typeof message.content === "string") {
      if (message.content && !isModelPlaceholder(message.content)) {
        parts.push({ text: message.content });
      }
    } else if (Array.isArray(message.content)) {
      // Text parts
      message.content.forEach((item) => {
        if (item.type === "text") {
          const text = item.text || "";
          if (text && !isModelPlaceholder(text)) {
            parts.push({ text });
          }
        }
      });
      // Image parts
      const images = extractImageParts(message.content);
      images.forEach((img) => {
        parts.push(processImageContent(img, "gemini"));
      });
    } else if (message.content && typeof message.content === "object") {
      if ((message.content as any).text) {
        const text = (message.content as any).text;
        if (!isModelPlaceholder(text)) {
          parts.push({ text });
        }
      } else {
        parts.push({ text: JSON.stringify(message.content) });
      }
    }

    // Shared call ids so functionCall.id === functionResponse.id. Gemini needs
    // that match for parallel tools; Claude-on-Antigravity remaps
    // functionResponse.id → Anthropic tool_result.tool_use_id and 400s when
    // the id is missing ("tool_use_id: Field required").
    const preparedCalls = Array.isArray(message.tool_calls)
      ? message.tool_calls.map((toolCall, toolIndex) => {
          const id =
            toolCall.id || `tool_${Math.random().toString(36).substring(2, 15)}`;

          // A per-tool signature (round-tripped via Anthropic tool_use) always
          // wins, then the one CCR remembered when the model returned this call
          // — Claude Code cannot carry it, and replaying the real signature is
          // what keeps the model's reasoning chain intact across tool turns.
          // Each is restored exactly where the model produced it, including
          // sibling parallel calls.
          //
          // Only when no real signature is available does the fallback apply:
          // Gemini 3 / Antigravity validate the *first* functionCall part of
          // each step and 400 when its thoughtSignature is missing, so that part
          // gets the turn-level thinking signature, else the documented sentinel
          // (which costs model quality — see thought-signature-cache).
          let signature = ((toolCall as any).thought_signature ||
            recallThoughtSignature(opts?.signatureScope, id)) as
            | string
            | undefined;
          if (!signature && toolIndex === 0) {
            signature =
              realSignature ||
              (skipFallbackEnabled ? SKIP_THOUGHT_SIGNATURE : undefined);
          }
          return { toolCall, id, signature };
        })
      : [];

    if (preparedCalls.length > 0) {
      parts.push(
        ...preparedCalls.map(({ toolCall, id, signature }) => ({
          functionCall: {
            id,
            name: toolCall.function.name,
            args: JSON.parse(toolCall.function.arguments || "{}"),
          },
          ...(signature && { thoughtSignature: signature }),
        }))
      );
    }

    if (parts.length === 0) {
      parts.push({ text: "" });
    }

    rawContents.push({
      role,
      parts,
    });

    if (role === "model" && preparedCalls.length > 0) {
      const functionResponses = preparedCalls.map(({ toolCall, id }) => {
        const response = toolResponses.find(
          (item) => item.tool_call_id === toolCall.id
        );

        let resultText = response?.content;
        if (Array.isArray(resultText)) {
          resultText = resultText
            .filter((part: any) => part.type === "text")
            .map((part: any) => part.text)
            .join("\n");
        } else if (typeof resultText === "object" && resultText !== null) {
          resultText = JSON.stringify(resultText);
        }

        return {
          functionResponse: {
            id,
            name: toolCall?.function?.name,
            response: { result: resultText },
          },
        };
      });
      rawContents.push({
        role: "user",
        parts: functionResponses,
      });
    }
  });

  const contents = consolidateMessages(rawContents, 'parts');

  const generationConfig: any = {};

  // Claude Code's effort decides how much the model thinks; the model id is
  // never rewritten. Each family gets the dialect it accepts — see
  // gemini-thinking.ts.
  const thinkingConfig = buildGeminiThinkingConfig(request);
  if (thinkingConfig) {
    generationConfig.thinkingConfig = thinkingConfig;
  }

  // Map other generation config fields
  if (request.max_tokens) generationConfig.maxOutputTokens = request.max_tokens;
  if (request.temperature !== undefined) generationConfig.temperature = request.temperature;

  const body: Record<string, any> = {
    contents: contents.length ? contents : [{ role: "user", parts: [{ text: "" }] }],
    tools: tools.length ? tools : undefined,
    generationConfig,
  };
  if (systemTexts.length) {
    body.systemInstruction = {
      parts: [{ text: systemTexts.join("\n\n") }],
    };
  }

  if (request.tool_choice) {
    const toolConfig: {
      functionCallingConfig: {
        mode?: string;
        allowedFunctionNames?: string[];
      };
    } = {
      functionCallingConfig: {},
    };
    if (request.tool_choice === "auto") {
      toolConfig.functionCallingConfig.mode = "auto";
    } else if (request.tool_choice === "none") {
      toolConfig.functionCallingConfig.mode = "none";
    } else if (request.tool_choice === "required") {
      toolConfig.functionCallingConfig.mode = "any";
    } else if (
      typeof request.tool_choice === "object" &&
      request.tool_choice.function?.name
    ) {
      toolConfig.functionCallingConfig.mode = "any";
      toolConfig.functionCallingConfig.allowedFunctionNames = [
        request.tool_choice.function.name,
      ];
    }
    body.toolConfig = toolConfig;
  }

  return body;
}

export function transformRequestOut(
  request: Record<string, any>
): UnifiedChatRequest {
  const contents: ContentListUnion = request.contents;
  const tools: ToolListUnion = request.tools;
  const model: string = request.model;
  const max_tokens: number | undefined = request.max_tokens;
  const temperature: number | undefined = request.temperature;
  const stream: boolean | undefined = request.stream;
  const tool_choice: "auto" | "none" | string | undefined = request.tool_choice;

  const unifiedChatRequest: UnifiedChatRequest = {
    messages: [],
    model,
    max_tokens,
    temperature,
    stream,
    tool_choice,
  };

  if (Array.isArray(contents)) {
    contents.forEach((content) => {
      if (typeof content === "string") {
        unifiedChatRequest.messages.push({
          role: "user",
          content,
        });
      } else if (typeof (content as Part).text === "string") {
        unifiedChatRequest.messages.push({
          role: "user",
          content: (content as Part).text || null,
        });
      } else if ((content as Content).role === "user") {
        unifiedChatRequest.messages.push({
          role: "user",
          content:
            (content as Content)?.parts?.map((part: Part) => ({
              type: "text",
              text: part.text || "",
            })) || [],
        });
      } else if ((content as Content).role === "model") {
        unifiedChatRequest.messages.push({
          role: "assistant",
          content:
            (content as Content)?.parts?.map((part: Part) => ({
              type: "text",
              text: part.text || "",
            })) || [],
        });
      }
    });
  }

  if (Array.isArray(tools)) {
    unifiedChatRequest.tools = [];
    tools.forEach((tool) => {
      const functionDeclarations = (tool as any).functionDeclarations;
      if (Array.isArray(functionDeclarations)) {
        functionDeclarations.forEach((tool: any) => {
          unifiedChatRequest.tools!.push({
            type: "function",
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.parameters,
            },
          });
        });
      }
    });
  }

  return unifiedChatRequest;
}

/**
 * Record the signatures upstream returned with this turn's tool calls, so the
 * next request can replay them instead of the validator-skip sentinel. CCR's own
 * placeholders are never cached — they are not upstream reasoning state.
 */
function rememberToolSignatures(
  scope: string | undefined,
  tool_calls: Array<{ id?: string; thought_signature?: string }>
): void {
  tool_calls.forEach((tool) => {
    const sig = tool.thought_signature;
    if (!sig || sig === SKIP_THOUGHT_SIGNATURE || sig.startsWith("ccr_")) return;
    rememberThoughtSignature(scope, tool.id, sig);
  });
}

export async function transformResponseOut(
  response: Response,
  providerName: string,
  logger?: any,
  signatureScope?: string
): Promise<Response> {
  const scope = signatureScope || providerName;
  if (response.headers.get("Content-Type")?.includes("application/json")) {
    const jsonResponse: any = await response.json();
    logger?.debug({ response: jsonResponse }, `${providerName} response:`);

    if (response.status >= 400) {
      const errorMessage: string = jsonResponse.error?.message || "";
      const lowerMessage = errorMessage.toLowerCase();
      const isContextExceeded = [
        "user input too long",
        "input too long",
        "prompt is too long",
        "exceeds the token limit",
        "request payload size exceeds",
        "context_length_exceeded",
      ].some((phrase) => lowerMessage.includes(phrase));

      if (isContextExceeded) {
        const res = {
          id: `ctxexceeded_${Date.now()}`,
          choices: [
            {
              finish_reason: "model_context_window_exceeded",
              index: 0,
              message: { content: "", role: "assistant" },
            },
          ],
          created: Math.floor(Date.now() / 1000),
          model: "",
          object: "chat.completion",
          usage: { completion_tokens: 0, prompt_tokens: 0, total_tokens: 0 },
        };
        return new Response(JSON.stringify(res), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }

      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    if (!jsonResponse.candidates || jsonResponse.candidates.length === 0) {
      return new Response(JSON.stringify(jsonResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    // Extract thinking content from parts with thought: true
    let thinkingContent = "";
    let thinkingSignature: string | undefined;

    const parts = jsonResponse.candidates[0]?.content?.parts || [];
    const nonThinkingParts: Part[] = [];

    for (const part of parts) {
      if (part.text && part.thought === true) {
        thinkingContent += replaceLatexSymbols(part.text);
      } else {
        nonThinkingParts.push(part);
      }
    }

    // Get thoughtSignature from functionCall args or usageMetadata
    thinkingSignature = parts.find(
      (part: any) => part.thoughtSignature
    )?.thoughtSignature;

    if (thinkingContent && !thinkingSignature) {
      thinkingSignature = `ccr_${+new Date()}`;
    }

    const tool_calls =
      nonThinkingParts
        ?.filter((part: Part) => part.functionCall)
        ?.map((part: Part) => ({
          id:
            part.functionCall?.id ||
            `tool_${Math.random().toString(36).substring(2, 15)}`,
          type: "function",
          function: {
            name: part.functionCall?.name,
            arguments: JSON.stringify(part.functionCall?.args || {}),
          },
          // Same-part sibling thoughtSignature (Antigravity / Gemini generateContent).
          thought_signature:
            (part as any).thoughtSignature ||
            (part as any).thought_signature ||
            undefined,
        })) || [];

    // Parallel calls: only the first functionCall carries the signature upstream.
    // If it arrived as a sibling part, attach it to the first tool that lacks one.
    if (thinkingSignature && tool_calls.length > 0 && !tool_calls[0].thought_signature) {
      tool_calls[0].thought_signature = thinkingSignature;
    }

    rememberToolSignatures(scope, tool_calls);

    const textContent =
      nonThinkingParts
        ?.filter((part: Part) => part.text)
        ?.map((part: Part) => replaceLatexSymbols(part.text || ""))
        ?.join("\n") || "";

    const candidate0 = jsonResponse.candidates[0];

    const visibleText =
      textContent && textContent !== "(no content)" ? textContent : "";

    // Same abnormal-finish rule as the streaming path.
    const finishNotice =
      !visibleText && tool_calls.length === 0
        ? buildAbnormalFinishNotice(candidate0)
        : null;

    const rawFinish = normalizeGoogleFinishReason(candidate0?.finishReason);

    const res = {
      id: jsonResponse.responseId,
      choices: [
        {
          finish_reason: tool_calls.length > 0 ? "tool_calls" : rawFinish,
          index: 0,
          message: {
            content: finishNotice || visibleText,
            role: "assistant",
            tool_calls: tool_calls.length > 0 ? tool_calls : undefined,
            // Only surface a thinking block when there was real thought text.
            // Signature-only tool parts must not invent "(no content)" thinking.
            ...(thinkingSignature &&
              thinkingContent && {
                thinking: {
                  content: thinkingContent,
                  signature: thinkingSignature,
                },
              }),
          },
        },
      ],
      created: parseInt(new Date().getTime() / 1000 + "", 10),
      model: jsonResponse.modelVersion,
      object: "chat.completion",
      usage: {
        completion_tokens:
          jsonResponse.usageMetadata?.candidatesTokenCount || 0,
        prompt_tokens: jsonResponse.usageMetadata?.promptTokenCount || 0,
        prompt_tokens_details: {
          cached_tokens:
            jsonResponse.usageMetadata?.cachedContentTokenCount || 0,
        },
        total_tokens: jsonResponse.usageMetadata?.totalTokenCount || 0,
        output_tokens_details: {
          reasoning_tokens: jsonResponse.usageMetadata?.thoughtsTokenCount || 0,
        },
      },
    };
    return new Response(JSON.stringify(res), {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    });
  } else if (response.headers.get("Content-Type")?.includes("stream")) {
    if (!response.body) {
      return response;
    }

    let contentIndex = 0;
    let toolCallIndex = -1;
    let activeStreamContext: StreamContext | null = null;
    let lastChunk: any = null;
    let lastCandidate: any = null;

    const enqueueChunk = (ctx: StreamContext, payload: any) => {
      ctx.controller.enqueue(
        ctx.encoder.encode(`data: ${JSON.stringify(payload)}\n\n`)
      );
    };

    const buildUsage = (chunk?: any) => ({
      completion_tokens: chunk?.usageMetadata?.candidatesTokenCount || 0,
      prompt_tokens: chunk?.usageMetadata?.promptTokenCount || 0,
      prompt_tokens_details: {
        cached_tokens: chunk?.usageMetadata?.cachedContentTokenCount || 0,
      },
      total_tokens: chunk?.usageMetadata?.totalTokenCount || 0,
      output_tokens_details: {
        reasoning_tokens: chunk?.usageMetadata?.thoughtsTokenCount || 0,
      },
    });

    const buildAnnotations = (candidate?: any) =>
      candidate?.groundingMetadata?.groundingChunks?.map(
        (groundingChunk: any, index: number) => {
          const support = candidate?.groundingMetadata?.groundingSupports?.filter(
            (item: any) => item.groundingChunkIndices?.includes(index)
          );
          return {
            type: "url_citation",
            url_citation: {
              url: groundingChunk?.web?.uri || "",
              title: groundingChunk?.web?.title || "",
              content: support?.[0]?.segment?.text || "",
              start_index: support?.[0]?.segment?.startIndex || 0,
              end_index: support?.[0]?.segment?.endIndex || 0,
            },
          };
        }
      ) || [];

    const emitContentChunk = (
      ctx: StreamContext,
      text: string,
      meta?: {
        chunk?: any;
        candidate?: any;
        mode?: "direct" | "buffered" | "placeholder" | "finish";
        finishReason?: string | null;
      }
    ) => {
      const mode = meta?.mode || "direct";
      if (mode === "direct") {
        contentIndex++;
      }

      const res: any = {
        choices: [
          {
            delta: {
              role: "assistant",
              content: text,
            },
            finish_reason:
              meta?.finishReason ??
              (mode === "direct" || mode === "finish"
                ? normalizeGoogleFinishReason(meta?.candidate?.finishReason)
                : null),
            index: contentIndex,
            logprobs: null,
          },
        ],
        created: parseInt(new Date().getTime() / 1000 + "", 10),
        id: meta?.chunk?.responseId || "",
        model: meta?.chunk?.modelVersion || "",
        object: "chat.completion.chunk",
        system_fingerprint: "fp_a49d71b8a1",
      };

      if (mode === "direct" || mode === "finish") {
        res.usage = buildUsage(meta?.chunk);
        const annotations = buildAnnotations(meta?.candidate);
        if (annotations.length > 0) {
          (res.choices[0].delta as any).annotations = annotations;
        }
      }

      enqueueChunk(ctx, res);
    };

    const sequencer = new ThinkingSequencer({
      thinking: (content: string, chunk?: any) => {
        if (!activeStreamContext) return;
        enqueueChunk(activeStreamContext, {
          choices: [
            {
              delta: { role: "assistant", content: null, thinking: { content } },
              finish_reason: null,
              index: contentIndex,
              logprobs: null,
            },
          ],
          created: parseInt(new Date().getTime() / 1000 + "", 10),
          id: chunk?.responseId || "",
          model: chunk?.modelVersion || "",
          object: "chat.completion.chunk",
          system_fingerprint: "fp_a49d71b8a1",
        });
      },
      signature: (sig: string, chunk?: any) => {
        if (!activeStreamContext) return;
        enqueueChunk(activeStreamContext, {
          choices: [
            {
              delta: {
                role: "assistant",
                content: null,
                thinking: { signature: sig },
              },
              finish_reason: null,
              index: contentIndex,
              logprobs: null,
            },
          ],
          created: parseInt(new Date().getTime() / 1000 + "", 10),
          id: chunk?.responseId || "",
          model: chunk?.modelVersion || "",
          object: "chat.completion.chunk",
          system_fingerprint: "fp_a49d71b8a1",
        });
      },
      content: (text: string, meta) => {
        if (!activeStreamContext) return;
        emitContentChunk(activeStreamContext, text, meta);
      },
    });

    const processLine = (line: string, ctx: StreamContext) => {
      activeStreamContext = ctx;
      if (line.startsWith("data: ")) {
        const chunkStr = line.slice(6).trim();
        if (chunkStr) {
          logger?.debug({ chunkStr }, `${providerName} chunk:`);
          try {
            const chunk = JSON.parse(chunkStr);

            if (!chunk.candidates || !chunk.candidates[0]) {
              logger?.debug({ chunkStr }, "Invalid chunk structure");
              return;
            }

            const candidate = chunk.candidates[0];
            lastChunk = chunk;
            lastCandidate = candidate;
            const parts = candidate.content?.parts || [];

            // Dialect-aware within-chunk order:
            // 1) thought:true (public Gemini thinking)
            // 2) visible text (buffer for Gemini 3 out-of-order, or emit for Antigravity)
            // 3) thoughtSignature (flush buffer / or no-op if content already emitted)
            // 4) functionCall tools
            // This supports both public Gemini (think→sig→text) and Antigravity
            // (text then signature trailer) without late thinking-after-text.

            parts
              .filter((part: any) => part.text && part.thought === true)
              .forEach((part: any) => {
                sequencer.processThinking(replaceLatexSymbols(part.text), chunk);
              });

            const tool_calls = parts
              .filter((part: Part) => part.functionCall)
              .map((part: Part) => ({
                id:
                  part.functionCall?.id ||
                  `ccr_tool_${Math.random().toString(36).substring(2, 15)}`,
                type: "function",
                function: {
                  name: part.functionCall?.name,
                  arguments: JSON.stringify(part.functionCall?.args || {}),
                },
                // Same-part sibling thoughtSignature (both public Gemini and Antigravity).
                thought_signature:
                  (part as any).thoughtSignature ||
                  (part as any).thought_signature ||
                  undefined,
              }));

            const textContent = parts
              .filter((part: Part) => part.text && part.thought !== true)
              .map((part: Part) => replaceLatexSymbols(part.text || ""))
              .join("\n");

            const signature = parts.find(
              (part: Part) => part.thoughtSignature
            )?.thoughtSignature;

            // Chunk-/sibling-level signature for tools that lack a per-part one.
            // Do not emit late thinking after content (Antigravity dialect); the
            // signature still must ride on tool_calls for the next-turn replay.
            if (signature && tool_calls.length > 0 && !tool_calls[0].thought_signature) {
              tool_calls[0].thought_signature = signature;
            }

            rememberToolSignatures(scope, tool_calls);

            const hasFinalEvent =
              Boolean(textContent && textContent !== "(no content)") ||
              tool_calls.length > 0 ||
              Boolean(candidate.finishReason);

            if (
              textContent &&
              textContent !== "(no content)" &&
              chunk.modelVersion?.includes("3") &&
              sequencer.shouldDeferContent(
                Boolean(candidate.finishReason),
                tool_calls.length > 0
              )
            ) {
              sequencer.bufferContent(textContent);
            } else if (textContent && textContent !== "(no content)") {
              // Fallback synthetic signature before emitting content when
              // thinking was seen but this final-ish event has no signature.
              if (
                sequencer.hasThinkingContent &&
                !sequencer.signatureSent &&
                hasFinalEvent &&
                !signature
              ) {
                const hasBufferedContent = sequencer.hasBufferedContent;
                sequencer.processSignatureWithMeta(`ccr_${+new Date()}`, chunk, {
                  beforeFlush: hasBufferedContent
                    ? () => {
                        contentIndex++;
                      }
                    : undefined,
                  flushMeta: hasBufferedContent
                    ? {
                        chunk,
                        candidate,
                        mode: "buffered",
                        finishReason: null,
                      }
                    : undefined,
                });
              }
              sequencer.processContent(textContent, chunk, candidate);
            } else if (textContent === "(no content)") {
              // Drop echoed CCR placeholders so they are not re-sent to Claude Code.
            }

            if (signature) {
              // Tool turns: thoughtSignature often rides on the functionCall part
              // (Antigravity / Gemini). Acknowledge it without synthesizing empty
              // thinking or text — those placeholders become "(no content)" in
              // Claude Code history and the model starts echoing them.
              if (
                tool_calls.length > 0 &&
                !sequencer.hasThinkingContent &&
                !textContent &&
                !sequencer.contentSent
              ) {
                sequencer.acknowledgeSignature();
              } else {
                sequencer.processSignatureWithMeta(signature, chunk, {
                  beforeFlush: () => {
                    contentIndex++;
                  },
                  flushMeta: {
                    chunk,
                    candidate,
                    mode: "buffered",
                  },
                });
              }
            }

            // An abnormal finish (MALFORMED_FUNCTION_CALL, MAX_TOKENS, SAFETY,
            // OTHER…) with nothing to show must not look like a normal empty
            // turn: report the upstream reason instead of "(no content)". Only
            // when no tools went out — text after tool_use is illegal Anthropic
            // block order, and the tool loop carries the turn anyway.
            const finishNotice =
              toolCallIndex < 0 && tool_calls.length === 0 && !textContent
                ? buildAbnormalFinishNotice(candidate)
                : null;

            // Never invent visible text on tool turns. Also wait until finish
            // before emitting a placeholder — a later chunk may bring tools or
            // real text (public Gemini think→sig→tools).
            // toolCallIndex >= 0 means tools already went out earlier in this
            // stream: Antigravity splits tool turns into a functionCall chunk
            // plus a bare `text: ""` + STOP trailer, and the trailer alone would
            // otherwise pass this guard and emit text after tool_use.
            if (
              !textContent &&
              tool_calls.length === 0 &&
              toolCallIndex < 0 &&
              candidate.finishReason &&
              !finishNotice &&
              sequencer.needsContentPlaceholder
            ) {
              sequencer.emitContentPlaceholder("(no content)", {
                chunk,
                candidate,
              });
            }

            // Thinking without signature on a finishing event (e.g. tool-only).
            if (
              !signature &&
              !textContent &&
              sequencer.hasThinkingContent &&
              !sequencer.signatureSent &&
              hasFinalEvent
            ) {
              const hasBufferedContent = sequencer.hasBufferedContent;
              sequencer.processSignatureWithMeta(`ccr_${+new Date()}`, chunk, {
                beforeFlush: hasBufferedContent
                  ? () => {
                      contentIndex++;
                    }
                  : undefined,
                flushMeta: hasBufferedContent
                  ? {
                      chunk,
                      candidate,
                      mode: "buffered",
                      finishReason: null,
                    }
                  : undefined,
              });
            }

            // Gemini 3: content was buffered earlier this chunk; do not also
            // fall through to tools/finish until signature on a later chunk —
            // unless this chunk also carries tools/finish/signature.
            if (
              textContent &&
              chunk.modelVersion?.includes("3") &&
              sequencer.hasBufferedContent &&
              !signature &&
              tool_calls.length === 0 &&
              !candidate.finishReason
            ) {
              return;
            }

            if (tool_calls.length > 0) {
              tool_calls.forEach((tool: any, toolIdx: number) => {
                contentIndex++;
                toolCallIndex++;
                // Gemini/Antigravity finish with STOP even when returning
                // functionCall parts. Claude Code's agent loop needs tool_calls
                // → Anthropic stop_reason tool_use, not end_turn.
                const finishReason =
                  candidate.finishReason != null &&
                  toolIdx === tool_calls.length - 1
                    ? "tool_calls"
                    : null;
                const res: any = {
                  choices: [
                    {
                      delta: {
                        role: "assistant",
                        tool_calls: [
                          {
                            ...tool,
                            index: toolCallIndex,
                          },
                        ],
                      },
                      finish_reason: finishReason,
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                  usage: buildUsage(chunk),
                };

                const annotations = buildAnnotations(candidate);
                if (annotations.length > 0) {
                  (res.choices[0].delta as any).annotations = annotations;
                }

                enqueueChunk(ctx, res);
              });
            }

            if (candidate.finishReason) {
              if (sequencer.hasBufferedContent) {
                sequencer.finalize(chunk, candidate, {
                  beforeFlush: () => {
                    if (!sequencer.signatureSent) {
                      contentIndex++;
                    }
                  },
                });
              } else if (!textContent && tool_calls.length === 0) {
                contentIndex++;
                emitContentChunk(ctx, finishNotice || "", {
                  chunk,
                  candidate,
                  mode: "finish",
                  // Antigravity's STOP trailer is the only finish-bearing chunk
                  // when tools arrived in an earlier chunk (that one carries
                  // finish_reason null). Report tool_calls so Anthropic does not
                  // depend on the stop_reason safety net alone.
                  ...(toolCallIndex >= 0 ? { finishReason: "tool_calls" } : {}),
                });
              }
            }
          } catch (error: any) {
            logger?.error(
              `Error parsing ${providerName} stream chunk`,
              chunkStr,
              error.message
            );
          }
        }
      }
    };

    return createSSEStreamReader(response, processLine, {
      onComplete: (ctx) => {
        activeStreamContext = ctx;
        if (
          sequencer.hasBufferedContent ||
          (sequencer.hasThinkingContent && !sequencer.signatureSent)
        ) {
          sequencer.finalize(lastChunk || undefined, lastCandidate || undefined, {
            beforeFlush: () => {
              if (!sequencer.signatureSent) {
                contentIndex++;
              }
            },
          });
        }
      },
    });
  }

  return response;
}

import { UnifiedChatRequest, UnifiedMessage } from "../types/llm";
import { Content, ContentListUnion, Part, ToolListUnion } from "@google/genai";
import { sanitizeJsonSchema } from "./schema";
import { createSSEStreamReader } from "./stream";

declare module 'latex-to-unicode' {
  function latexToUnicode(str: string): string;
  export default latexToUnicode;
}

import latexToUnicode from "latex-to-unicode";

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
  const genAISchema = {};
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
      const upperCaseValue = fieldValue.toUpperCase();
      genAISchema["type"] = Object.values(Type).includes(upperCaseValue)
        ? upperCaseValue
        : Type.TYPE_UNSPECIFIED;
    } else if (schemaFieldNames.includes(fieldName)) {
      genAISchema[fieldName] = processJsonSchema(fieldValue);
    } else if (listSchemaFieldNames.includes(fieldName)) {
      const listSchemaFieldValue = [];
      for (const item of fieldValue) {
        if (item["type"] == "null") {
          genAISchema["nullable"] = true;
          continue;
        }
        listSchemaFieldValue.push(processJsonSchema(item));
      }
      genAISchema[fieldName] = listSchemaFieldValue;
    } else if (dictSchemaFieldNames.includes(fieldName)) {
      const dictSchemaFieldValue = {};
      for (const [key, value] of Object.entries(fieldValue)) {
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
  return genAISchema;
}

/**
 * Transform a tool object
 * @param {Object} tool - The tool object to transform
 * @returns {Object} - The transformed tool object
 */
export function tTool(tool: any): any {
  if (tool.functionDeclarations) {
    for (const functionDeclaration of tool.functionDeclarations) {
      if (functionDeclaration.parameters) {
        const sanitized = sanitizeJsonSchema(functionDeclaration.parameters);
        functionDeclaration.parameters = processJsonSchema(sanitized);
      }
      if (functionDeclaration.response) {
        const sanitized = sanitizeJsonSchema(functionDeclaration.response);
        functionDeclaration.response = processJsonSchema(sanitized);
      }
    }
  }
  return tool;
}

/** Normalize a tool to unified format (handles both OpenAI and Anthropic tool shapes) */
function normalizeTool(tool: any): { name: string; description: string; parameters: any } {
  if (tool.function?.name) {
    return { name: tool.function.name, description: tool.function.description, parameters: tool.function.parameters };
  }
  return { name: tool.name, description: tool.description, parameters: tool.input_schema };
}

/** Sanitize a function name for Gemini's naming rules:
 * Must start with a letter or underscore, contain only [a-zA-Z0-9_.:\-], max 128 chars */
function sanitizeGeminiFunctionName(name: string): string {
  if (!name) return "unnamed_function";
  let sanitized = name.replace(/[^a-zA-Z0-9_.:\-]/g, "_");
  if (/^[^a-zA-Z_]/.test(sanitized)) {
    sanitized = "_" + sanitized;
  }
  return sanitized.substring(0, 128);
}

/** Replace LaTeX math symbols that some models generate with standard unicode using latex-to-unicode library */
function replaceLatexSymbols(text: string): string {
  if (!text) return text;
  try {
    // The library only replaces the first occurrence of each symbol (str.replace(key, val))
    // So we loop until the string stops changing to ensure all occurrences are handled
    let converted = text;
    let prev;
    do {
      prev = converted;
      converted = latexToUnicode(converted);
    } while (converted !== prev);

    // Some models wrap symbols in $, e.g. $\rightarrow$. 
    // The library converts it to $→$, so we clean up the $ signs if they surround a single unicode char
    return converted.replace(/\$([^\$])\$/g, '$1');
  } catch (e) {
    return text;
  }
}

export function buildRequestBody(
  request: UnifiedChatRequest
): Record<string, any> {
  const tools = [];
  const requestTools = request.tools || [];
  const functionDeclarations = requestTools
    .filter((tool) => normalizeTool(tool).name !== "web_search")
    .map((tool) => {
      const { name, description, parameters } = normalizeTool(tool);
      return { name: sanitizeGeminiFunctionName(name), description, parameters };
    });
  if (functionDeclarations?.length) {
    tools.push(
      tTool({
        functionDeclarations,
      })
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

  const messages: UnifiedMessage[] = [];

  for (const msg of rawMessages) {
    if (msg.role === "tool" || msg.role === "system") continue;

    const role = msg.role === "assistant" ? "assistant" : "user";
    messages.push({ ...msg, role });
  }

  const toolResponses = rawMessages.filter((item) => item.role === "tool");
  messages.forEach((message: UnifiedMessage) => {
    let role: "user" | "model";
    if (message.role === "assistant") {
      role = "model";
    } else {
      role = "user";
    }
    const parts = [];

    const realSignature =
      message.thinking?.signature &&
        !message.thinking.signature.startsWith("ccr_")
        ? message.thinking.signature
        : undefined;

    if (realSignature && role === "model") {
      parts.push({
        thought: true,
        text: message.thinking?.content || "",
        thought_signature: realSignature,
      });
    }

    if (typeof message.content === "string") {
      parts.push({ text: message.content });
    } else if (Array.isArray(message.content)) {
      parts.push(
        ...message.content.map((content) => {
          if (content.type === "text") {
            return {
              text: content.text || "",
            };
          }
          if (content.type === "image_url") {
            if (content.image_url.url.startsWith("http")) {
              return {
                file_data: {
                  mime_type: content.media_type,
                  file_uri: content.image_url.url,
                },
              };
            } else {
              return {
                inlineData: {
                  mime_type: content.media_type,
                  data:
                    content.image_url.url?.split(",")?.pop() ||
                    content.image_url.url,
                },
              };
            }
          }
          return null;
        }).filter(Boolean)
      );
    } else if (message.content && typeof message.content === "object") {
      if ((message.content as any).text) {
        parts.push({ text: (message.content as any).text });
      } else {
        parts.push({ text: JSON.stringify(message.content) });
      }
    }

    if (Array.isArray(message.tool_calls)) {
      parts.push(
        ...message.tool_calls.map((toolCall) => {
          const signature = (toolCall as any).thought_signature || realSignature;
          return {
            functionCall: {
              id:
                toolCall.id ||
                `tool_${Math.random().toString(36).substring(2, 15)}`,
              name: toolCall.function.name,
              args: JSON.parse(toolCall.function.arguments || "{}"),
            },
            ...(signature && { thought_signature: signature }),
          };
        })
      );
    }

    if (parts.length === 0) {
      parts.push({ text: "" });
    }

    rawContents.push({
      role,
      parts,
    });

    if (role === "model" && message.tool_calls) {
      const functionResponses = message.tool_calls.map((tool) => {
        const response = toolResponses.find(
          (item) => item.tool_call_id === tool.id
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
            name: tool?.function?.name,
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

  const contents: any[] = [];
  for (const item of rawContents) {
    const lastItem = contents[contents.length - 1];
    if (lastItem && lastItem.role === item.role) {
      lastItem.parts.push(...item.parts);
    } else {
      contents.push({ ...item, parts: [...item.parts] });
    }
  }

  const generationConfig: any = {};

  if (
    request.reasoning &&
    request.reasoning.effort &&
    request.reasoning.effort !== "none"
  ) {
    generationConfig.thinkingConfig = {
      includeThoughts: true,
    };
    if (request.model.includes("gemini-3")) {
      generationConfig.thinkingConfig.thinkingLevel = request.reasoning.effort;
    } else {
      const thinkingBudgets = request.model.includes("pro")
        ? [128, 32768]
        : [0, 24576];
      let thinkingBudget;
      const max_tokens = request.reasoning.max_tokens;
      if (typeof max_tokens !== "undefined") {
        if (
          max_tokens >= thinkingBudgets[0] &&
          max_tokens <= thinkingBudgets[1]
        ) {
          thinkingBudget = max_tokens;
        } else if (max_tokens < thinkingBudgets[0]) {
          thinkingBudget = thinkingBudgets[0];
        } else if (max_tokens > thinkingBudgets[1]) {
          thinkingBudget = thinkingBudgets[1];
        }
        generationConfig.thinkingConfig.thinkingBudget = thinkingBudget;
      }
    }
  }

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
    const toolConfig = {
      functionCallingConfig: {},
    };
    if (request.tool_choice === "auto") {
      toolConfig.functionCallingConfig.mode = "auto";
    } else if (request.tool_choice === "none") {
      toolConfig.functionCallingConfig.mode = "none";
    } else if (request.tool_choice === "required") {
      toolConfig.functionCallingConfig.mode = "any";
    } else if (request.tool_choice?.function?.name) {
      toolConfig.functionCallingConfig.mode = "any";
      toolConfig.functionCallingConfig.allowedFunctionNames = [
        request.tool_choice?.function?.name,
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
      if (Array.isArray(tool.functionDeclarations)) {
        tool.functionDeclarations.forEach((tool) => {
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

export async function transformResponseOut(
  response: Response,
  providerName: string,
  logger?: any
): Promise<Response> {
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
    let thinkingSignature = "";

    const parts = jsonResponse.candidates[0]?.content?.parts || [];
    const nonThinkingParts: Part[] = [];

    for (const part of parts) {
      if (part.text && part.thought === true) {
        thinkingContent += part.text;
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
          thought_signature: (part as any).thoughtSignature || (part as any).thought_signature,
        })) || [];

    const textContent =
      nonThinkingParts
        ?.filter((part: Part) => part.text)
        ?.map((part: Part) => replaceLatexSymbols(part.text))
        ?.join("\n") || "";

    const res = {
      id: jsonResponse.responseId,
      choices: [
        {
          finish_reason:
            (
              jsonResponse.candidates[0]?.finishReason as string
            )?.toLowerCase() || null,
          index: 0,
          message: {
            content: textContent,
            role: "assistant",
            tool_calls: tool_calls.length > 0 ? tool_calls : undefined,
            // Add thinking as separate field if available
            ...(thinkingSignature && {
              thinking: {
                content: thinkingContent || "(no content)",
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

    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    let signatureSent = false;
    let contentSent = false;
    let hasThinkingContent = false;
    let pendingContent = "";
    let contentIndex = 0;
    let toolCallIndex = -1;

    const processLine = (
      line: string,
      ctx: { controller: ReadableStreamDefaultController, encoder: TextEncoder }
    ) => {
      const { controller, encoder } = ctx;
      if (line.startsWith("data: ")) {
        const chunkStr = line.slice(6).trim();
        if (chunkStr) {
          logger?.debug({ chunkStr }, `${providerName} chunk:`);
          try {
            const chunk = JSON.parse(chunkStr);

            // Check if chunk has valid structure
            if (!chunk.candidates || !chunk.candidates[0]) {
              logger?.debug({ chunkStr }, `Invalid chunk structure`);
              return;
            }

            const candidate = chunk.candidates[0];
            const parts = candidate.content?.parts || [];

            parts
              .filter((part: any) => part.text && part.thought === true)
              .forEach((part: any) => {
                if (!hasThinkingContent) {
                  hasThinkingContent = true;
                }
                const thinkingChunk = {
                  choices: [
                    {
                      delta: {
                        role: "assistant",
                        content: null,
                        thinking: {
                          content: part.text,
                        },
                      },
                      finish_reason: null,
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                };
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify(thinkingChunk)}\n\n`
                  )
                );
              });

            let signature = parts.find(
              (part: Part) => part.thoughtSignature
            )?.thoughtSignature;
            if (signature && !signatureSent) {
              if (!hasThinkingContent) {
                const thinkingChunk = {
                  choices: [
                    {
                      delta: {
                        role: "assistant",
                        content: null,
                        thinking: {
                          content: "(no content)",
                        },
                      },
                      finish_reason: null,
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                };
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify(thinkingChunk)}\n\n`
                  )
                );
              }
              const signatureChunk = {
                choices: [
                  {
                    delta: {
                      role: "assistant",
                      content: null,
                      thinking: {
                        signature,
                      },
                    },
                    finish_reason: null,
                    index: contentIndex,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.responseId || "",
                model: chunk.modelVersion || "",
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
              };
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify(signatureChunk)}\n\n`
                )
              );
              signatureSent = true;
              contentIndex++;
              if (pendingContent) {
                const res = {
                  choices: [
                    {
                      delta: {
                        role: "assistant",
                        content: pendingContent,
                      },
                      finish_reason: null,
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                };

                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
                );

                pendingContent = "";
                if (!contentSent) {
                  contentSent = true;
                }
              }
            }

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
                thought_signature: (part as any).thoughtSignature || (part as any).thought_signature,
              }));

            const textContent = parts
              .filter((part: Part) => part.text && part.thought !== true)
              .map((part: Part) => replaceLatexSymbols(part.text))
              .join("\n");

            if (!textContent && signatureSent && !contentSent) {
              const emptyContentChunk = {
                choices: [
                  {
                    delta: {
                      role: "assistant",
                      content: "(no content)",
                    },
                    index: contentIndex,
                    finish_reason: null,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.responseId || "",
                model: chunk.modelVersion || "",
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
              };
              controller.enqueue(
                encoder.encode(
                  `data: ${JSON.stringify(emptyContentChunk)}\n\n`
                )
              );

              if (!contentSent) {
                contentSent = true;
              }
            }

            const hasFinalEvent = textContent || tool_calls.length > 0 || candidate.finishReason;
            if (hasThinkingContent && !signatureSent && hasFinalEvent) {
              if (chunk.modelVersion.includes("3") && !candidate.finishReason && tool_calls.length === 0) {
                if (textContent) {
                  pendingContent += textContent;
                }
                return;
              } else {
                const signatureChunk = {
                  choices: [
                    {
                      delta: {
                        role: "assistant",
                        content: null,
                        thinking: {
                          signature: `ccr_${+new Date()}`,
                        },
                      },
                      finish_reason: null,
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                };
                controller.enqueue(
                  encoder.encode(
                    `data: ${JSON.stringify(signatureChunk)}\n\n`
                  )
                );
                signatureSent = true;

                if (pendingContent) {
                  contentIndex++;
                  const res = {
                    choices: [
                      {
                        delta: {
                          role: "assistant",
                          content: pendingContent,
                        },
                        finish_reason: null,
                        index: contentIndex,
                        logprobs: null,
                      },
                    ],
                    created: parseInt(new Date().getTime() / 1000 + "", 10),
                    id: chunk.responseId || "",
                    model: chunk.modelVersion || "",
                    object: "chat.completion.chunk",
                    system_fingerprint: "fp_a49d71b8a1",
                  };
                  controller.enqueue(
                    encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
                  );
                  pendingContent = "";
                  contentSent = true;
                }
              }
            }

            if (textContent) {
              if (!pendingContent) contentIndex++;
              const res = {
                choices: [
                  {
                    delta: {
                      role: "assistant",
                      content: textContent,
                    },
                    finish_reason:
                      candidate.finishReason?.toLowerCase() || null,
                    index: contentIndex,
                    logprobs: null,
                  },
                ],
                created: parseInt(new Date().getTime() / 1000 + "", 10),
                id: chunk.responseId || "",
                model: chunk.modelVersion || "",
                object: "chat.completion.chunk",
                system_fingerprint: "fp_a49d71b8a1",
                usage: {
                  completion_tokens:
                    chunk.usageMetadata?.candidatesTokenCount || 0,
                  prompt_tokens: chunk.usageMetadata?.promptTokenCount || 0,
                  prompt_tokens_details: {
                    cached_tokens:
                      chunk.usageMetadata?.cachedContentTokenCount || 0,
                  },
                  total_tokens: chunk.usageMetadata?.totalTokenCount || 0,
                  output_tokens_details: {
                    reasoning_tokens:
                      chunk.usageMetadata?.thoughtsTokenCount || 0,
                  },
                },
              };

              if (candidate?.groundingMetadata?.groundingChunks?.length) {
                (res.choices[0].delta as any).annotations =
                  candidate.groundingMetadata.groundingChunks.map(
                    (groundingChunk: any, index: number) => {
                      const support =
                        candidate?.groundingMetadata?.groundingSupports?.filter(
                          (item: any) =>
                            item.groundingChunkIndices?.includes(index)
                        );
                      return {
                        type: "url_citation",
                        url_citation: {
                          url: groundingChunk?.web?.uri || "",
                          title: groundingChunk?.web?.title || "",
                          content: support?.[0]?.segment?.text || "",
                          start_index:
                            support?.[0]?.segment?.startIndex || 0,
                          end_index: support?.[0]?.segment?.endIndex || 0,
                        },
                      };
                    }
                  );
              }
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
              );

              if (!contentSent && textContent) {
                contentSent = true;
              }
            }

            if (tool_calls.length > 0) {
              tool_calls.forEach((tool) => {
                contentIndex++;
                toolCallIndex++;
                const res = {
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
                      finish_reason:
                        candidate.finishReason?.toLowerCase() || null,
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                  usage: {
                    completion_tokens:
                      chunk.usageMetadata?.candidatesTokenCount || 0,
                    prompt_tokens:
                      chunk.usageMetadata?.promptTokenCount || 0,
                    prompt_tokens_details: {
                      cached_tokens:
                        chunk.usageMetadata?.cachedContentTokenCount || 0,
                    },
                    total_tokens: chunk.usageMetadata?.totalTokenCount || 0,
                    output_tokens_details: {
                      reasoning_tokens:
                        chunk.usageMetadata?.thoughtsTokenCount || 0,
                    },
                  },
                };

                if (candidate?.groundingMetadata?.groundingChunks?.length) {
                  (res.choices[0].delta as any).annotations =
                    candidate.groundingMetadata.groundingChunks.map(
                      (groundingChunk: any, index: number) => {
                        const support =
                          candidate?.groundingMetadata?.groundingSupports?.filter(
                            (item: any) =>
                              item.groundingChunkIndices?.includes(index)
                          );
                        return {
                          type: "url_citation",
                          url_citation: {
                            url: groundingChunk?.web?.uri || "",
                            title: groundingChunk?.web?.title || "",
                            content: support?.[0]?.segment?.text || "",
                            start_index:
                              support?.[0]?.segment?.startIndex || 0,
                            end_index: support?.[0]?.segment?.endIndex || 0,
                          },
                        };
                      }
                    );
                }
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(res)}\n\n`)
                );
              });

              if (!contentSent && textContent) {
                contentSent = true;
              }
            }

            // Flush buffered text or send final finish_reason on stream end
            if (candidate.finishReason) {
              if (pendingContent) {
                if (!signatureSent && hasThinkingContent) {
                  const signatureChunk = {
                    choices: [
                      {
                        delta: {
                          role: "assistant",
                          content: null,
                          thinking: { signature: `ccr_${+new Date()}` },
                        },
                        finish_reason: null,
                        index: contentIndex,
                        logprobs: null,
                      },
                    ],
                    created: parseInt(new Date().getTime() / 1000 + "", 10),
                    id: chunk.responseId || "",
                    model: chunk.modelVersion || "",
                    object: "chat.completion.chunk",
                    system_fingerprint: "fp_a49d71b8a1",
                  };
                  controller.enqueue(
                    encoder.encode(
                      `data: ${JSON.stringify(signatureChunk)}\n\n`
                    )
                  );
                  signatureSent = true;
                  contentIndex++;
                }
                const flushRes = {
                  choices: [
                    {
                      delta: {
                        role: "assistant",
                        content: pendingContent,
                      },
                      finish_reason: candidate.finishReason.toLowerCase(),
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                  usage: {
                    completion_tokens:
                      chunk.usageMetadata?.candidatesTokenCount || 0,
                    prompt_tokens: chunk.usageMetadata?.promptTokenCount || 0,
                    prompt_tokens_details: {
                      cached_tokens:
                        chunk.usageMetadata?.cachedContentTokenCount || 0,
                    },
                    total_tokens: chunk.usageMetadata?.totalTokenCount || 0,
                    output_tokens_details: {
                      reasoning_tokens:
                        chunk.usageMetadata?.thoughtsTokenCount || 0,
                    },
                  },
                };
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(flushRes)}\n\n`)
                );
                pendingContent = "";
                contentSent = true;
              } else if (!textContent && tool_calls.length === 0) {
                contentIndex++;
                const flushRes = {
                  choices: [
                    {
                      delta: {
                        role: "assistant",
                        content: "",
                      },
                      finish_reason: candidate.finishReason.toLowerCase(),
                      index: contentIndex,
                      logprobs: null,
                    },
                  ],
                  created: parseInt(new Date().getTime() / 1000 + "", 10),
                  id: chunk.responseId || "",
                  model: chunk.modelVersion || "",
                  object: "chat.completion.chunk",
                  system_fingerprint: "fp_a49d71b8a1",
                  usage: {
                    completion_tokens:
                      chunk.usageMetadata?.candidatesTokenCount || 0,
                    prompt_tokens: chunk.usageMetadata?.promptTokenCount || 0,
                    prompt_tokens_details: {
                      cached_tokens:
                        chunk.usageMetadata?.cachedContentTokenCount || 0,
                    },
                    total_tokens: chunk.usageMetadata?.totalTokenCount || 0,
                    output_tokens_details: {
                      reasoning_tokens:
                        chunk.usageMetadata?.thoughtsTokenCount || 0,
                    },
                  },
                };
                controller.enqueue(
                  encoder.encode(`data: ${JSON.stringify(flushRes)}\n\n`)
                );
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

    return createSSEStreamReader(response, processLine);
  }
}

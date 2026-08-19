import { Readable } from "node:stream";
import type { FastifyReply } from "fastify";
import {
  DEFAULT_DEBUG_INSTRUCTIONS,
  applyReasoningEffortToBody,
  parseDebugChatBody,
  resolveDebugModel,
} from "./model";
import { parseOpenAiTools, stubToolExecute } from "./tools";
import {
  installLlmCaptureFetch,
  resetLlmCaptureFetchForTests,
  runWithLlmCapture,
} from "./llm-capture";
import { errorExchangeFromMessage, type CapturedLlmExchange, type DebugChatInput } from "./types";

function ensureLlmCaptureInstalled(): void {
  installLlmCaptureFetch();
}

export function resetDebugCaptureForTests(): void {
  resetLlmCaptureFetchForTests();
}

export type StreamDebugChat = (
  input: DebugChatInput,
  config: any,
  signal?: AbortSignal
) => Promise<Response>;

function createClaudeOAuthFetch(apiKey: string): typeof fetch {
  return async (input, init) => {
    const headers = new Headers(init?.headers);
    headers.delete("x-api-key");
    headers.set("authorization", `Bearer ${apiKey}`);
    const betas = (headers.get("anthropic-beta") || "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean);
    if (!betas.some((value) => value.toLowerCase() === "oauth-2025-04-20")) {
      betas.push("oauth-2025-04-20");
    }
    headers.set("anthropic-beta", betas.join(","));
    return globalThis.fetch(input, { ...init, headers });
  };
}

function createCodexFetch(): typeof fetch {
  return async (input, init) => {
    const rawUrl =
      input instanceof Request
        ? input.url
        : input instanceof URL
          ? input.toString()
          : input;
    const url = new URL(rawUrl);
    if (!url.searchParams.has("client_version")) {
      url.searchParams.set("client_version", "0.145.0");
    }
    const nextInput =
      input instanceof Request ? new Request(url, input) : url;
    return globalThis.fetch(nextInput, init);
  };
}

async function defaultStreamDebugChat(
  input: DebugChatInput,
  config: any,
  signal?: AbortSignal
): Promise<Response> {
  ensureLlmCaptureInstalled();
  const [ai, { createOpenAI }, { createAnthropic }] = await Promise.all([
    import("ai"),
    import("@ai-sdk/openai"),
    import("@ai-sdk/anthropic"),
  ]);
  const {
    convertToModelMessages,
    createUIMessageStream,
    createUIMessageStreamResponse,
    jsonSchema,
    stepCountIs,
    streamText,
    tool,
  } = ai;

  const model = await resolveDebugModel(input, config);
  const instructions = input.system.trim() || DEFAULT_DEBUG_INSTRUCTIONS;
  const specs = parseOpenAiTools(input.tools);
  const tools: Record<string, unknown> = {};
  for (const spec of specs) {
    const parameters =
      spec.parameters && typeof spec.parameters === "object"
        ? spec.parameters
        : { type: "object", properties: {} };
    tools[spec.id] = tool({
      description: spec.description,
      inputSchema: jsonSchema(parameters),
      execute: async (args: unknown) => stubToolExecute(args),
    });
  }

  const [, ...modelIdParts] = model.id.split("/");
  const modelId = modelIdParts.join("/") || model.id;
  const languageModel =
    input.protocol === "messages"
      ? createAnthropic({
          baseURL: model.url,
          apiKey: model.apiKey,
          headers: model.headers,
          ...(model.authKind === "claude-auth"
            ? { fetch: createClaudeOAuthFetch(model.apiKey) }
            : {}),
        })(modelId)
      : input.protocol === "responses"
        ? createOpenAI({
            baseURL: model.url,
            apiKey: model.apiKey,
            headers: model.headers,
            ...(model.authKind === "codex"
              ? { fetch: createCodexFetch() }
              : {}),
          }).responses(modelId)
        : createOpenAI({
            baseURL: model.url,
            apiKey: model.apiKey,
            headers: model.headers,
          }).chat(modelId);

  const messages = input.messages.length > 0 ? input.messages : [];
  const streamOptions: Record<string, unknown> = {
    model: languageModel,
    system: instructions,
    messages: await convertToModelMessages(messages as any, {
      ...(Object.keys(tools).length > 0 ? { tools: tools as any } : {}),
    }),
    stopWhen: stepCountIs(1),
    maxRetries: 0,
    ...(signal ? { abortSignal: signal } : {}),
    ...(Object.keys(tools).length > 0 ? { tools: tools as any } : {}),
  };
  if (input.protocol === "chat_completions" && input.stream) {
    streamOptions.providerOptions = {
      openai: {
        stream_options: { include_usage: true },
        ...(input.reasoningEffort ? { reasoningEffort: input.reasoningEffort } : {}),
      },
    };
  } else if (input.reasoningEffort) {
    if (input.protocol === "messages") {
      streamOptions.providerOptions = {
        anthropic: {
          thinking:
            input.reasoningEffort === "none"
              ? { type: "disabled" }
              : { type: "adaptive" },
        },
      };
    } else {
      streamOptions.providerOptions = {
        openai: { reasoningEffort: input.reasoningEffort },
      };
    }
  }

  const uiMessageStream = createUIMessageStream({
    originalMessages: messages as any,
    onError: (err) => (err instanceof Error ? err.message : String(err)),
    execute: async ({ writer }) => {
      const writeExchange = async (exchange?: CapturedLlmExchange) => {
        if (!exchange) return;
        await writer.write({
          type: "data-llm-exchange",
          data: exchange,
        } as any);
      };

      let streamError: unknown;
      const { result, last, finalize, error } = await runWithLlmCapture(
        async () => {
          const streamResult = await streamText(streamOptions as any);
          try {
            for await (const part of streamResult.toUIMessageStream({
              originalMessages: messages as any,
              sendReasoning: true,
            }) as any) {
              // AI SDK emits generic error parts on provider failures — skip them;
              // the debug panel reads the captured upstream exchange instead.
              if (part?.type === "error") {
                streamError =
                  streamError ??
                  new Error(
                    typeof part.errorText === "string" ? part.errorText : "Stream error"
                  );
                continue;
              }
              await writer.write(part);
            }
          } catch (err) {
            streamError = streamError ?? err;
          }
          return streamResult;
        },
        {
          patchRequestBody: (body) =>
            applyReasoningEffortToBody(body, input.protocol, input.reasoningEffort),
          signal,
        }
      );

      if (error || !result) {
        const captured = await finalize();
        const message =
          error instanceof Error ? error.message : String(error || "Debug chat failed");
        await writeExchange(errorExchangeFromMessage(message, captured || last));
        return;
      }

      if (streamError) {
        const captured = await finalize();
        const message =
          streamError instanceof Error ? streamError.message : String(streamError);
        await writeExchange(errorExchangeFromMessage(message, captured || last));
        return;
      }

      const captured = await finalize();
      if (captured) {
        await writeExchange(captured);
        if (captured.usage) {
          await writer.write({
            type: "data-usage",
            data: captured.usage,
          } as any);
        }
      }
    },
  });

  return createUIMessageStreamResponse({ stream: uiMessageStream });
}

export let streamDebugChat: StreamDebugChat = defaultStreamDebugChat;

export function setStreamDebugChatForTests(fn: StreamDebugChat | null): void {
  streamDebugChat = fn || defaultStreamDebugChat;
}

export async function sendWebResponse(
  reply: FastifyReply,
  webResponse: Response
): Promise<void> {
  reply.status(webResponse.status);
  webResponse.headers.forEach((value, key) => {
    reply.header(key, value);
  });
  if (!webResponse.body) {
    return reply.send();
  }
  return reply.send(Readable.fromWeb(webResponse.body as any));
}

export { parseDebugChatBody, resolveDebugModel };

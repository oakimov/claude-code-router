import {
  buildEndpointBody,
  copyCurlCommand,
  exchangeFromChatError,
  formatDebugResponseNotice,
  guessInboundProtocol,
  isDebugResponseError,
  parseDebugResponseError,
  parseHeadersJson,
  readDebugExchangeFromSseLine,
  tapDebugExchangeStream,
  type CapturedExchange,
} from "../lib/debugChat";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const PROVIDER_GONE_BODY = JSON.stringify({
  error: {
    message:
      'Error from provider(nvidia,deepseek-ai/deepseek-v4-pro: 410): {"type":"about:blank","title":"Gone","status":410,"detail":"The model \'deepseek-ai/deepseek-v4-pro\' has reached its end of life on 2026-08-07T09:00:00Z and is no longer available"}',
    type: "api_error",
    param: null,
    code: "provider_response_error",
  },
});

const PROVIDER_RATE_LIMIT_BODY =
  '{"error":{"message":"Error from provider(nvidia,z-ai/glm-5.2: 429): {\\"status\\":429,\\"title\\":\\"Too Many Requests\\"}","type":"api_error","param":null,"code":"provider_response_error"}}\n';

function testParseGenericHttpError(): void {
  const parsed = parseDebugResponseError('{"message":"Bad gateway","code":"upstream_error"}', 502);
  assert(parsed != null, "generic 502 body should parse");
  assert(parsed!.status === 502, "HTTP status should be preserved");
  assert(parsed!.message === "Bad gateway", "message should be shown");
}

function testParseStringErrorField(): void {
  const parsed = parseDebugResponseError('{"error":"something went wrong"}', 400);
  assert(parsed != null, "string error field should parse");
  assert(parsed!.message === "something went wrong", "string error should be shown");
}

function testParseProviderRateLimitError(): void {
  const parsed = parseDebugResponseError(PROVIDER_RATE_LIMIT_BODY, 429);
  assert(parsed != null, "429 provider error body should parse");
  assert(parsed!.status === 429, "status should be 429");
  assert(parsed!.code === "provider_response_error", "error code should be preserved");
  assert(parsed!.message === "Too Many Requests", "title should be used as summary");
}

function testParseProviderResponseError(): void {
  const parsed = parseDebugResponseError(PROVIDER_GONE_BODY, 410);
  assert(parsed != null, "provider error body should parse");
  assert(parsed!.status === 410, "status should be extracted from provider message");
  assert(parsed!.code === "provider_response_error", "error code should be preserved");
  assert(
    parsed!.message.includes("end of life"),
    "detail should be extracted from nested provider JSON"
  );
  assert(
    formatDebugResponseNotice(parsed!).includes("410"),
    "notice should include HTTP status"
  );
}

function testLineHeaders(): void {
  const parsed = parseHeadersJson("X-Test: one\nAuthorization: Bearer secret");
  assert(
    parsed["X-Test"] === "one" &&
      parsed.Authorization === "Bearer secret",
    "line-based headers should be parsed"
  );
}

function testCurlUsesPosixSafeQuoting(): void {
  const command = copyCurlCommand({
    url: "https://example.test/'$(touch unsafe)",
    method: "POST",
    headers: {
      "X-Test": "`echo unsafe` $HOME 'quoted'",
      Authorization: "Bearer secret",
    },
    body: { prompt: "$(echo unsafe)\nsecond line", quote: "it's safe" },
  });

  assert(command.includes("Bearer PLACEHOLDER"), "auth values should be redacted");
  assert(!command.includes("Bearer secret"), "auth secrets should not be copied");
  assert(command.includes("'\\''"), "single quotes should use POSIX escaping");
  assert(
    command.includes("--data-raw '{") &&
      command.includes('"prompt": "$(echo unsafe)'),
    "command substitutions should remain inside single quotes"
  );
}

function testParseAnthropicStyleErrorEnvelope(): void {
  const parsed = parseDebugResponseError(
    JSON.stringify({
      type: "error",
      error: { type: "overloaded_error", message: "Overloaded" },
    }),
    529
  );
  assert(parsed != null, "anthropic error envelope should parse");
  assert(parsed!.message === "Overloaded", "nested error message should be shown");
  assert(parsed!.code === "overloaded_error", "nested error type should be the code");
}

function testParseTypeErrorMessage(): void {
  const parsed = parseDebugResponseError(
    JSON.stringify({ type: "error", message: "stream aborted", code: "aborted" }),
    0
  );
  assert(parsed != null, "type=error body should parse without an error object");
  assert(parsed!.message === "stream aborted", "top-level message should be shown");
  assert(parsed!.code === "aborted", "top-level code should be preserved");
}

function testSuccessBodyIsNotAnError(): void {
  const ok = '{"id":"c","object":"chat.completion","choices":[{"message":{"content":"hi"}}]}';
  assert(!isDebugResponseError(ok, 200), "HTTP 200 completion is not an error");
  assert(!isDebugResponseError(ok, 0), "status-0 completion JSON is not an error");
  assert(isDebugResponseError(PROVIDER_RATE_LIMIT_BODY, 0), "error JSON is an error even without HTTP status");
}

function testExchangeFromChatError(): void {
  const exchange = exchangeFromChatError("network down");
  assert(exchange.status === 0, "fallback exchange has no HTTP status");
  assert(exchange.responseBody.includes("network down"), "fallback body should include the chat error");
}

function testReadDebugExchangeFromSseLine(): void {
  assert(readDebugExchangeFromSseLine("data: [DONE]") === null, "[DONE] is not an exchange");
  assert(readDebugExchangeFromSseLine("data: not-json") === null, "non-JSON data lines are ignored");
  assert(
    readDebugExchangeFromSseLine('data: {"type":"text-delta","delta":"hi"}') === null,
    "text deltas are not exchanges"
  );
  const parsed = readDebugExchangeFromSseLine(
    `data: ${JSON.stringify({ type: "data-llm-exchange", data: { status: 429, responseBody: "err" } })}`
  );
  assert(parsed != null && parsed.status === 429, "llm-exchange data parts should parse");
}

async function testTapDebugExchangeStreamSplitsChunks(): Promise<void> {
  const exchange: CapturedExchange = {
    url: "https://example.test/v1/chat/completions",
    method: "POST",
    requestHeaders: {},
    requestBody: {},
    status: 429,
    responseHeaders: { "retry-after": "2" },
    responseBody: '{"error":"Too Many Requests"}',
    streaming: false,
  };
  const dataLine = `data: ${JSON.stringify({ type: "data-llm-exchange", data: exchange })}`;
  const sse = [
    'data: {"type":"text-delta","delta":"hi"}',
    "",
    dataLine,
    "data: not-json",
    "data: [DONE]",
    "",
  ].join("\n");
  const bytes = new TextEncoder().encode(sse);
  const splitAt = sse.indexOf(dataLine) + Math.floor(dataLine.length / 2);
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes.slice(0, splitAt));
      controller.enqueue(bytes.slice(splitAt));
      controller.close();
    },
  });
  const seen: CapturedExchange[] = [];
  await tapDebugExchangeStream(stream, (item) => seen.push(item));
  assert(seen.length === 1, `expected one exchange, got ${seen.length}`);
  assert(seen[0].status === 429, "chunk-split JSON should still parse");
  assert(seen[0].responseBody.includes("Too Many Requests"), "exchange body should survive the split");
}

function testResponsesProtocolOmitsStreamOptions(): void {
  const responsesBody = buildEndpointBody({
    protocol: "responses",
    model: "codex,gpt-5.4-mini",
    system: "",
    messages: [{ role: "user", content: "ping" }],
    toolsJson: "[]",
    stream: true,
  });
  assert(responsesBody.stream_options === undefined, "responses body should omit stream_options");
  assert(
    guessInboundProtocol({ transformer: { use: ["codex", "openai-responses"] } } as any) ===
      "responses",
    "codex chained with openai-responses is a Responses provider"
  );
  assert(
    guessInboundProtocol({ transformer: { use: ["openai-responses"] } } as any) === "responses",
    "openai-responses transformer is a Responses backend"
  );
  const openAiBody = buildEndpointBody({
    protocol: "chat_completions",
    model: "openai,gpt-4o",
    system: "",
    messages: [{ role: "user", content: "ping" }],
    toolsJson: "[]",
    stream: true,
  });
  assert(
    (openAiBody.stream_options as { include_usage?: boolean } | undefined)?.include_usage === true,
    "chat completions body keeps stream_options.include_usage"
  );
}

async function main(): Promise<void> {
  testParseGenericHttpError();
  testParseStringErrorField();
  testParseProviderRateLimitError();
  testParseProviderResponseError();
  testParseAnthropicStyleErrorEnvelope();
  testParseTypeErrorMessage();
  testSuccessBodyIsNotAnError();
  testExchangeFromChatError();
  testReadDebugExchangeFromSseLine();
  await testTapDebugExchangeStreamSplitsChunks();
  testLineHeaders();
  testCurlUsesPosixSafeQuoting();
  testResponsesProtocolOmitsStreamOptions();
  console.log("debug-chat ui helpers: PASS");
}

main().catch((error) => {
  console.error(error);
  throw error;
});

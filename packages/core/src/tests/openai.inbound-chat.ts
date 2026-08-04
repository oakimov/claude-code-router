/**
 * Inbound OpenAI Chat Completions: validation, unsupported-field errors,
 * text/tool JSON + stream, usage, finish reason, [DONE], error envelopes.
 */
import assert from "node:assert/strict";
import { OpenAITransformer } from "../transformer/openai.transformer";

const logger = { debug() {}, info() {}, warn() {}, error() {} } as any;

async function expectReject(
  fn: () => Promise<unknown>,
  code: string
): Promise<void> {
  let caught: any;
  try {
    await fn();
  } catch (e) {
    caught = e;
  }
  assert.ok(caught, `expected reject with code ${code}`);
  assert.equal(caught.code, code);
  assert.equal(caught.statusCode, 400);
}

async function testSupportedFields() {
  const tf = new OpenAITransformer();
  (tf as any).logger = logger;

  const unified = await tf.transformRequestOut({
    model: "openai,gpt-4o",
    messages: [
      { role: "developer", content: "sys" },
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "Read", arguments: "{}" },
          },
        ],
      },
      { role: "tool", tool_call_id: "call_1", content: "ok" },
      {
        role: "user",
        content: [
          { type: "text", text: "img" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,aa" },
          },
        ],
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "Read",
          description: "read",
          parameters: { type: "object", properties: {} },
        },
      },
    ],
    tool_choice: "auto",
    temperature: 0.5,
    top_p: 0.9,
    stop: ["\n"],
    max_completion_tokens: 256,
    stream: true,
    stream_options: { include_usage: true },
    parallel_tool_calls: true,
  });

  assert.equal(unified.model, "openai,gpt-4o");
  assert.equal(unified.messages[0].role, "system");
  assert.equal(unified.max_tokens, 256);
  assert.equal(unified.max_completion_tokens, 256);
  assert.equal(unified.stream, true);
  assert.equal(unified.stream_options?.include_usage, true);
  assert.equal(unified.tool_choice, "auto");
}

async function testUnsupportedFields() {
  const tf = new OpenAITransformer();

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        n: 2,
      }),
    "unsupported_n"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        logprobs: true,
      }),
    "unsupported_logprobs"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        modalities: ["text", "audio"],
      }),
    "unsupported_modalities"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        audio: { voice: "alloy" },
      }),
    "unsupported_audio"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        response_format: { type: "json_schema", json_schema: {} },
      }),
    "unsupported_response_format"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        max_tokens: 10,
        max_completion_tokens: 20,
      }),
    "conflicting_token_limits"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [
          {
            role: "user",
            content: [{ type: "input_audio", input_audio: {} }],
          },
        ],
      }),
    "unsupported_audio"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        stream_options: { include_usage: true },
      }),
    "invalid_stream_options"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        store: true,
      }),
    "unsupported_state"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: 42 }],
      }),
    "invalid_message_content"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "tool", content: "result" }],
      }),
    "invalid_tool_call_id"
  );

  await expectReject(
    () =>
      tf.transformRequestOut({
        model: "m",
        messages: [{ role: "user", content: "x" }],
        frequency_penalty: 0.5,
      }),
    "unsupported_field"
  );
}

async function testMatchingTokenLimitsOk() {
  const tf = new OpenAITransformer();
  const unified = await tf.transformRequestOut({
    model: "m",
    messages: [{ role: "user", content: "x" }],
    max_tokens: 10,
    max_completion_tokens: 10,
  });
  assert.equal(unified.max_tokens, 10);
}

async function testJsonResponsePassthrough() {
  const tf = new OpenAITransformer();
  const payload = {
    id: "chatcmpl-1",
    object: "chat.completion",
    model: "gpt-4o",
    choices: [
      {
        index: 0,
        finish_reason: "stop",
        message: { role: "assistant", content: "hello" },
      },
    ],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
  const out = await tf.transformResponseIn(
    new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    })
  );
  const json = await out.json();
  assert.equal(json.object, "chat.completion");
  assert.equal(json.choices[0].message.content, "hello");
  assert.equal(json.choices[0].finish_reason, "stop");
  assert.equal(json.usage.total_tokens, 2);
}

async function testToolJsonResponse() {
  const tf = new OpenAITransformer();
  const payload = {
    id: "chatcmpl-2",
    object: "chat.completion",
    model: "gpt-4o",
    choices: [
      {
        index: 0,
        finish_reason: "tool_calls",
        message: {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: { name: "Read", arguments: '{"path":"a.ts"}' },
            },
          ],
        },
      },
    ],
    usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
  };
  const out = await tf.transformResponseIn(
    new Response(JSON.stringify(payload), {
      headers: { "Content-Type": "application/json" },
    })
  );
  const json = await out.json();
  assert.equal(json.choices[0].finish_reason, "tool_calls");
  assert.equal(json.choices[0].message.tool_calls[0].function.name, "Read");
}

async function testStreamAddsDone() {
  const tf = new OpenAITransformer();
  const sse = [
    'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}',
    "",
    'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
    "",
  ].join("\n");

  const out = await tf.transformResponseIn(
    new Response(sse, {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const text = await out.text();
  assert.ok(text.includes("chat.completion.chunk"));
  assert.ok(text.includes("finish_reason"));
  assert.ok(text.includes("[DONE]"));
}

async function testStreamPreservesExistingDone() {
  const tf = new OpenAITransformer();
  const sse = [
    'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}',
    "",
    "data: [DONE]",
    "",
  ].join("\n");

  const out = await tf.transformResponseIn(
    new Response(sse, {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const text = await out.text();
  const doneCount = (text.match(/\[DONE\]/g) || []).length;
  assert.equal(doneCount, 1);
}

async function testDoneTextDoesNotSuppressTerminator() {
  const tf = new OpenAITransformer();
  const sse = [
    'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"literal [DONE] text"},"finish_reason":"stop"}]}',
    "",
  ].join("\n");
  const out = await tf.transformResponseIn(
    new Response(sse, {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const text = await out.text();
  assert.ok(text.endsWith("data: [DONE]\n\n"));
}

async function testStreamFailureEmitsErrorAndDone() {
  const tf = new OpenAITransformer();
  const encoder = new TextEncoder();
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(
        encoder.encode(
          'data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
        )
      );
      controller.error(new Error("socket broke with secret details"));
    },
  });
  const out = await tf.transformResponseIn(
    new Response(upstream, {
      headers: { "Content-Type": "text/event-stream" },
    })
  );
  const text = await out.text();
  assert.ok(text.includes('"code":"provider_response_error"'));
  assert.ok(!text.includes("secret details"));
  assert.equal((text.match(/data: \[DONE\]/g) || []).length, 1);
}

async function testMissingMessages() {
  const tf = new OpenAITransformer();
  await expectReject(
    () => tf.transformRequestOut({ model: "m" }),
    "invalid_body"
  );
}

async function main() {
  await testSupportedFields();
  await testUnsupportedFields();
  await testMatchingTokenLimitsOk();
  await testJsonResponsePassthrough();
  await testToolJsonResponse();
  await testStreamAddsDone();
  await testStreamPreservesExistingDone();
  await testDoneTextDoesNotSuppressTerminator();
  await testStreamFailureEmitsErrorAndDone();
  await testMissingMessages();
  console.log("openai.inbound-chat: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

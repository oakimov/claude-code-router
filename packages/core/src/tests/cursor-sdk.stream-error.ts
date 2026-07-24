import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

async function main() {
  const upstreamError = Object.assign(
    new Error("Cursor usage limit exceeded"),
    {
      type: "api_error",
      code: "provider_response_error",
    }
  );
  const upstream = new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.error(upstreamError);
      },
    }),
    {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }
  );

  const transformer = new AnthropicTransformer();
  transformer.logger = {
    debug() {},
    error() {},
  };
  const response = await transformer.transformResponseIn(upstream, {
    req: { id: "cursor-stream-error" },
  } as any);
  const body = await response.text();

  assert.match(body, /event: error/);
  assert.match(body, /"type":"api_error"/);
  assert.match(body, /Cursor usage limit exceeded/);
  assert.doesNotMatch(body, /event: message_stop/);
  assert.doesNotMatch(body, /"stop_reason":"end_turn"/);

  console.log("cursor-sdk.stream-error: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

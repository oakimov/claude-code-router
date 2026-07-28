/**
 * Client abort must cancel the upstream OpenAI/provider ReadableStream.
 * Otherwise owns-fetch providers (Cursor SDK) keep an active run and the next
 * Claude Code turn hangs after interrupt.
 */
import assert from "node:assert/strict";
import { AnthropicTransformer } from "../transformer/anthropic.transformer";

async function testCancelPropagatesToUpstreamReader() {
  let upstreamCancelReason: unknown = "not-called";
  let upstreamPulls = 0;
  let releaseUpstreamCancel!: () => void;
  const upstreamCancelGate = new Promise<void>((resolve) => {
    releaseUpstreamCancel = resolve;
  });

  const upstream = new ReadableStream<Uint8Array>({
    pull(controller) {
      upstreamPulls += 1;
      // Keep the stream open until cancel — never enqueue a terminal chunk.
      if (upstreamPulls > 1) {
        return new Promise(() => undefined);
      }
      const chunk = {
        id: "chatcmpl_test",
        model: "test-model",
        choices: [{ index: 0, delta: { content: "partial" }, finish_reason: null }],
      };
      controller.enqueue(
        new TextEncoder().encode(`data: ${JSON.stringify(chunk)}\n\n`)
      );
    },
    async cancel(reason) {
      upstreamCancelReason = reason;
      await upstreamCancelGate;
    },
  });

  const logs: string[] = [];
  const transformer = new AnthropicTransformer();
  (transformer as any).logger = {
    debug: (_meta: unknown, msg?: string) => {
      if (typeof msg === "string") logs.push(msg);
      else if (typeof _meta === "string") logs.push(_meta);
    },
    info: () => {},
    warn: () => {},
    error: () => {},
  };

  const out = await transformer.transformResponseIn(
    new Response(upstream, {
      headers: { "Content-Type": "text/event-stream" },
    }),
    { req: { id: "cancel-test" } } as any
  );

  assert.ok(out.body, "expected streaming body");
  const reader = out.body.getReader();
  // Allow the Anthropic start() loop to acquire the upstream reader.
  await new Promise((r) => setTimeout(r, 20));
  let cancellationSettled = false;
  const cancellation = reader.cancel("client_aborted").then(() => {
    cancellationSettled = true;
  });

  // Give cancel() microtasks a chance to reach the upstream barrier.
  await new Promise((r) => setTimeout(r, 20));

  assert.equal(
    upstreamCancelReason,
    "client_aborted",
    "upstream ReadableStream.cancel must receive the client abort reason"
  );
  assert.equal(
    cancellationSettled,
    false,
    "downstream cancellation must await upstream teardown"
  );
  releaseUpstreamCancel();
  await cancellation;
  assert.equal(cancellationSettled, true);
  assert.ok(
    logs.some((msg) => msg.startsWith("cancel stream:")),
    `expected "cancel stream" log, got: ${JSON.stringify(logs)}`
  );
  assert.ok(
    !logs.some((msg) => msg.includes("cancle stream")),
    "typo cancle stream must not appear in logs"
  );
}

await testCancelPropagatesToUpstreamReader();
console.log("anthropic.stream-cancel: PASS");

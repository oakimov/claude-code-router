import assert from "node:assert/strict";
import { createSSEStreamReader } from "../utils/stream";
import { isFallbackEligibleError, isProviderNetworkError } from "../utils/retry";

const unhandled: unknown[] = [];
process.on("unhandledRejection", (reason) => unhandled.push(reason));

/** The exact shape undici raises when the peer closes the socket mid-stream. */
function terminatedError(): TypeError {
  return Object.assign(new TypeError("terminated"), {
    cause: Object.assign(new Error("other side closed"), {
      code: "UND_ERR_SOCKET",
    }),
  });
}

function sseResponse(
  start: (controller: ReadableStreamDefaultController<Uint8Array>) => void,
  hooks: { cancel?: (reason: any) => void } = {}
): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start,
      cancel(reason) {
        hooks.cancel?.(reason);
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );
}

const passthrough = (line: string, ctx: any) =>
  ctx.controller.enqueue(ctx.encoder.encode(line + "\n"));

const silentLogger = { error() {} };

async function midStreamTerminationIsClassified() {
  let logged: any;
  const upstream = sseResponse((controller) => {
    controller.enqueue(new TextEncoder().encode('data: {"a":1}\n\n'));
    controller.error(terminatedError());
  });

  const out = createSSEStreamReader(upstream, passthrough, {
    logger: {
      error(error: unknown) {
        logged = error;
      },
    },
  });

  let thrown: any;
  await assert.rejects(
    () => out.text(),
    (error) => {
      thrown = error;
      return true;
    }
  );

  // A bare "terminated" is indistinguishable from a client abort downstream.
  // It must carry a provider-network code so fallback/retry can classify it.
  assert.equal((logged as any)?.code, "provider_network_error");
  assert.equal((logged as any)?.type, "api_error");
  assert.match(String((logged as any)?.message), /Upstream stream terminated/);
  assert.match(String((logged as any)?.message), /other side closed/);
  assert.ok(
    isProviderNetworkError(logged),
    "terminated stream must classify as a provider network error"
  );
  assert.ok(
    isFallbackEligibleError(logged),
    "terminated stream must be eligible for provider fallback"
  );
  // The thrown error retains the original cause; logs expose only redacted text.
  assert.equal(thrown?.cause?.cause?.code, "UND_ERR_SOCKET");
  assert.equal((logged as any)?.cause, "terminated");
}

async function clientCancelPropagatesUpstream() {
  let cancelledWith: any = "<never>";
  const upstream = sseResponse(
    (controller) => {
      controller.enqueue(new TextEncoder().encode('data: {"a":1}\n\n'));
      // Deliberately left open: only an explicit cancel can end this stream.
    },
    { cancel: (reason) => (cancelledWith = reason) }
  );

  const out = createSSEStreamReader(upstream, passthrough, {
    logger: silentLogger,
  });

  const reader = out.body!.getReader();
  await reader.read();
  await reader.cancel("client gone");
  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.equal(
    cancelledWith,
    "client gone",
    "client disconnect must tear down the upstream provider stream"
  );
}

async function cancelDoesNotEmitUnhandledRejection() {
  const upstream = sseResponse((controller) => {
    controller.enqueue(new TextEncoder().encode('data: {"a":1}\n\n'));
  });

  const out = createSSEStreamReader(upstream, passthrough, {
    logger: silentLogger,
  });

  const reader = out.body!.getReader();
  await reader.read();
  await reader.cancel("client gone");
  await new Promise((resolve) => setTimeout(resolve, 20));

  assert.deepEqual(
    unhandled.map((r) => (r as any)?.message ?? String(r)),
    [],
    "cancel path must not orphan a rejected read"
  );
}

async function healthyStreamStillCompletes() {
  const upstream = sseResponse((controller) => {
    controller.enqueue(new TextEncoder().encode('data: {"a":1}\n\n'));
    controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
    controller.close();
  });

  const body = await createSSEStreamReader(upstream, passthrough, {
    logger: silentLogger,
  }).text();

  assert.match(body, /"a":1/);
  assert.match(body, /\[DONE\]/);
}

async function malformedLineLoggingIsBoundedAndRedacted() {
  let logged: any;
  const secret = `sk-${"a".repeat(32)}`;
  const line = `data: {"authorization":"Bearer ${secret}","content":"${"x".repeat(500)}"}`;
  const upstream = sseResponse((controller) => {
    controller.enqueue(new TextEncoder().encode(`${line}\n\n`));
    controller.close();
  });

  await createSSEStreamReader(
    upstream,
    () => {
      throw new Error("malformed provider event");
    },
    {
      logger: {
        error(details: unknown) {
          logged = details;
        },
      },
    }
  ).text();

  assert.equal(typeof logged?.line, "string");
  assert.ok(logged.line.length <= 240);
  assert.ok(!logged.line.includes(secret));
  assert.match(logged.line, /redacted/);
}

async function main() {
  await midStreamTerminationIsClassified();
  await clientCancelPropagatesUpstream();
  await cancelDoesNotEmitUnhandledRejection();
  await healthyStreamStillCompletes();
  await malformedLineLoggingIsBoundedAndRedacted();

  console.log("stream.upstream-terminated: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

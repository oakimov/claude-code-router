import assert from "node:assert/strict";
import { withSSEClientKeepalive } from "../utils/sse/client-keepalive";

function encode(text: string): Uint8Array {
  return new TextEncoder().encode(text);
}

function decode(chunks: Uint8Array[]): string {
  return new TextDecoder().decode(Buffer.concat(chunks.map((c) => Buffer.from(c))));
}

async function readAll(
  stream: ReadableStream<Uint8Array>,
  timeoutMs = 5000
): Promise<Uint8Array[]> {
  const reader = stream.getReader();
  const out: Uint8Array[] = [];
  const deadline = Date.now() + timeoutMs;
  while (true) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error("readAll timed out");
    const result = await Promise.race([
      reader.read(),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error("readAll timed out")), remaining)
      ),
    ]);
    if (result.done) break;
    if (result.value) out.push(result.value);
  }
  return out;
}

async function forwardsUpstreamBytesUnchanged() {
  const payload = 'event: ping\ndata: {"type":"ping"}\n\n';
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encode(payload));
      controller.close();
    },
  });
  const tapped = withSSEClientKeepalive(upstream, { idleMs: 50 });
  const chunks = await readAll(tapped);
  assert.equal(decode(chunks), payload);
}

async function injectsCommentAfterIdle() {
  let pull = 0;
  const upstream = new ReadableStream<Uint8Array>({
    async pull(controller) {
      pull += 1;
      if (pull === 1) {
        controller.enqueue(encode("data: one\n\n"));
        return;
      }
      if (pull === 2) {
        // Sit idle long enough for keepalive, then send more bytes.
        await new Promise((r) => setTimeout(r, 80));
        controller.enqueue(encode("data: two\n\n"));
        controller.close();
      }
    },
  });

  const tapped = withSSEClientKeepalive(upstream, { idleMs: 30 });
  const chunks = await readAll(tapped, 2000);
  const text = decode(chunks);
  assert.match(text, /data: one/);
  assert.match(text, /: keepalive\n\n/);
  assert.match(text, /data: two/);
  // Keepalive must land between the two data frames.
  const iOne = text.indexOf("data: one");
  const iKeep = text.indexOf(": keepalive");
  const iTwo = text.indexOf("data: two");
  assert.ok(iOne < iKeep && iKeep < iTwo, "keepalive should sit between upstream frames");
}

async function cancelPropagatesToUpstream() {
  let cancelled = false;
  const upstream = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encode("data: hi\n\n"));
      // leave open until cancel
    },
    cancel() {
      cancelled = true;
    },
  });
  const tapped = withSSEClientKeepalive(upstream, { idleMs: 10_000 });
  await tapped.cancel("client gone");
  assert.equal(cancelled, true);
}

async function main() {
  await forwardsUpstreamBytesUnchanged();
  await injectsCommentAfterIdle();
  await cancelPropagatesToUpstream();
  console.log("sse.client-keepalive: ok");
}

main().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});

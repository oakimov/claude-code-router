import assert from "node:assert/strict";
import {
  isClientAbortError,
  CLIENT_DISCONNECT_REASON,
  toClientAbortError,
} from "../utils/retry";

function testDisconnectMessageShapes(): void {
  assert.equal(isClientAbortError("client closed connection"), true);
  assert.equal(isClientAbortError("Client Disconnect"), true);
  assert.equal(isClientAbortError("socket hang up"), true);
  assert.equal(isClientAbortError("SOCKET HANG UP"), true);
  assert.equal(
    isClientAbortError(`${CLIENT_DISCONNECT_REASON} (response close)`),
    true
  );

  // Non-disconnect errors must stay false
  assert.equal(isClientAbortError("ENOTFOUND upstream"), false);
  assert.equal(isClientAbortError("provider 500"), false);
  assert.equal(isClientAbortError(""), false);
  assert.equal(isClientAbortError(null), false);
  assert.equal(isClientAbortError(undefined), false);
}

function testAbortErrorObjects(): void {
  const abortErr = Object.assign(new Error("aborted"), { name: "AbortError" });
  assert.equal(isClientAbortError(abortErr), true);

  const abortCode = Object.assign(new Error("gone"), { code: "ABORT_ERR" });
  assert.equal(isClientAbortError(abortCode), true);

  const premature = Object.assign(new Error("stream closed"), {
    code: "ERR_STREAM_PREMATURE_CLOSE",
  });
  assert.equal(isClientAbortError(premature), true);

  // Timeouts must not be treated as client disconnects
  const timeoutErr = Object.assign(
    new Error("The operation was aborted due to timeout"),
    { name: "TimeoutError", code: 23 }
  );
  assert.equal(isClientAbortError(timeoutErr), false);
}

function testReDoSResistantDisconnectCheck(): void {
  // Pathological "client" repetition previously stressed
  // /client.*(closed|disconnect)/. Linear helper must stay fast and correct.
  const pathological = `client${"client".repeat(50_000)} closed`;
  const started = process.hrtime.bigint();
  assert.equal(isClientAbortError(pathological), true);
  const elapsedMs = Number(process.hrtime.bigint() - started) / 1e6;
  assert.ok(
    elapsedMs < 500,
    `disconnect classification took ${elapsedMs.toFixed(1)}ms (expected <500ms)`
  );

  const noMatch = `client${"client".repeat(20_000)} still connected`;
  assert.equal(isClientAbortError(noMatch), false);
}

function testNormalizedAbortError(): void {
  const normalized = toClientAbortError(
    `${CLIENT_DISCONNECT_REASON} (already destroyed)`
  );
  assert.equal(normalized.name, "AbortError");
  assert.equal((normalized as any).code, "ABORT_ERR");
  assert.equal(isClientAbortError(normalized), true);
}

function main(): void {
  testDisconnectMessageShapes();
  testAbortErrorObjects();
  testReDoSResistantDisconnectCheck();
  testNormalizedAbortError();
  console.log("client-abort-classification tests passed.");
}

main();

import assert from "node:assert/strict";
import { isCursorTransientProviderError } from "../cursor-sdk/shared";

// Observed shape: Error "[resource_exhausted] Error", 502, provider_response_error.
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("[resource_exhausted] Error"), {
      statusCode: 502,
      code: "provider_response_error",
    })
  ),
  true
);
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("too many requests"), { statusCode: 429 })
  ),
  true
);
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("overloaded"), { status: 503 })
  ),
  true
);
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("gateway timeout"), { statusCode: 504 })
  ),
  true
);
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("boom"), { code: "resource_exhausted" })
  ),
  true
);

// Auth errors and aborts must keep the old retire-the-session behavior.
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("authentication error"), { statusCode: 401 })
  ),
  false
);
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("aborted"), { name: "AbortError" })
  ),
  false
);
// Non-transient failures still retire.
assert.equal(isCursorTransientProviderError(new Error("boom")), false);
assert.equal(
  isCursorTransientProviderError(
    Object.assign(new Error("bad request"), { statusCode: 400 })
  ),
  false
);
assert.equal(isCursorTransientProviderError(null), false);
assert.equal(isCursorTransientProviderError(undefined), false);

console.log("cursor-sdk.shared: ok");

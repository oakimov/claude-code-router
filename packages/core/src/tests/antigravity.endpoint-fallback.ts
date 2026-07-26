/**
 * Antigravity endpoint fallback.
 *
 * Reproduces the failure seen in ~/.claude-code-router/logs: Claude Code's
 * /model validation probe (non-stream, max_tokens: 1) hit a transport failure on
 * the daily sandbox host, the walk moved to the autopush (staging) host, and that
 * host answered 403 SERVICE_DISABLED because the account's project is not
 * entitled to the staging deployment. The 403 body was then returned as if it
 * were a normal reply, so the response converter failed on the missing choices
 * and the client received an opaque 500 with Google's error text inside.
 *
 * Required behaviour: a 403 from a host that is not entitled keeps walking to the
 * next host (prod is last), that host is not probed again this session, and if
 * the walk runs out the upstream status is reported as itself.
 */
import assert from "node:assert/strict";
import {
  ANTIGRAVITY_ENDPOINT_AUTOPUSH,
  ANTIGRAVITY_ENDPOINT_DAILY,
  ANTIGRAVITY_ENDPOINT_PROD,
  antigravityEndpointCandidates,
  clearPreferredEndpoint,
  clearUnusableEndpoints,
  isEndpointDisabledError,
  isEndpointUnusable,
  markEndpointUnusable,
  shouldWalkEndpoint,
} from "../utils/antigravity-auth";
import { AntigravityAuthTransformer } from "../transformer/antigravity-auth.transformer";

/** Verbatim from the log (project id kept, it is not a secret). */
const DISABLED_403 = JSON.stringify({
  error: {
    code: 403,
    message:
      "Gemini for Google Cloud API (Staging) has not been used in project nodal-particle-jzftk before or it is disabled.",
    status: "PERMISSION_DENIED",
    details: [
      {
        "@type": "type.googleapis.com/google.rpc.ErrorInfo",
        reason: "SERVICE_DISABLED",
        domain: "googleapis.com",
      },
    ],
  },
});

function noopLogger() {
  const noop = () => {};
  return { debug: noop, info: noop, warn: noop, error: noop };
}

function testClassifiers() {
  assert.equal(isEndpointDisabledError(DISABLED_403), true);
  assert.equal(isEndpointDisabledError('{"error":{"code":403}}'), false);

  // Endpoint-level failures walk while another host is untried.
  for (const status of [403, 404, 500, 503]) {
    assert.equal(shouldWalkEndpoint(status, true), true, `walk on ${status}`);
    assert.equal(
      shouldWalkEndpoint(status, false),
      false,
      `surface ${status} on the last endpoint`
    );
  }
  // A malformed request is the caller's problem, not the host's.
  assert.equal(shouldWalkEndpoint(400, true), false);
}

function testUnusableEndpointsAreSkipped() {
  clearUnusableEndpoints();
  clearPreferredEndpoint();

  const before = antigravityEndpointCandidates(ANTIGRAVITY_ENDPOINT_DAILY);
  assert.deepEqual(before, [
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_AUTOPUSH,
    ANTIGRAVITY_ENDPOINT_PROD,
  ]);

  markEndpointUnusable(`${ANTIGRAVITY_ENDPOINT_AUTOPUSH}/`);
  assert.equal(isEndpointUnusable(ANTIGRAVITY_ENDPOINT_AUTOPUSH), true, "trailing slash normalized");
  assert.deepEqual(antigravityEndpointCandidates(ANTIGRAVITY_ENDPOINT_DAILY), [
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_PROD,
  ]);

  // Never strand the provider with an empty candidate list.
  markEndpointUnusable(ANTIGRAVITY_ENDPOINT_DAILY);
  markEndpointUnusable(ANTIGRAVITY_ENDPOINT_PROD);
  assert.deepEqual(antigravityEndpointCandidates(ANTIGRAVITY_ENDPOINT_DAILY), before);
  clearUnusableEndpoints();
}

/** Drive the real fallback loop with a stubbed fetch. */
async function withStubbedFetch<T>(
  handler: (url: string) => Promise<Response>,
  run: () => Promise<T>
): Promise<{ result?: T; error?: any; urls: string[] }> {
  const original = globalThis.fetch;
  const urls: string[] = [];
  globalThis.fetch = (async (input: any, _init?: any) => {
    const url = typeof input === "string" ? input : input?.url ?? String(input);
    urls.push(url);
    return handler(url);
  }) as any;
  try {
    return { result: await run(), urls };
  } catch (error) {
    return { error, urls };
  } finally {
    globalThis.fetch = original;
  }
}

function callSendWithFallback(transformer: AntigravityAuthTransformer) {
  const provider = {
    name: "antigravity",
    baseUrl: ANTIGRAVITY_ENDPOINT_DAILY,
    models: ["gemini-3.6-flash-tiered"],
  };
  // The probe Claude Code sends on /model: non-stream, max_tokens 1, one message.
  const body = {
    project: "nodal-particle-jzftk",
    model: "gemini-3.6-flash-tiered",
    request: {
      contents: [{ role: "user", parts: [{ text: "Hi" }] }],
      generationConfig: { maxOutputTokens: 1 },
    },
    userAgent: "antigravity",
    requestId: "test-request-id",
  };
  return (transformer as any).sendWithFallback(
    body,
    "test-access-token",
    false,
    provider,
    { req: { id: "req-1e" } }
  ) as Promise<Response>;
}

/** Transport failure on daily → staging 403 → prod serves the request. */
async function testWalksPastDisabledStagingToProd() {
  clearUnusableEndpoints();
  clearPreferredEndpoint();
  const transformer = new AntigravityAuthTransformer();
  transformer.logger = noopLogger();

  const { result, error, urls } = await withStubbedFetch(async (url) => {
    if (url.includes("daily-cloudcode-pa")) throw new TypeError("fetch failed");
    if (url.includes("autopush-cloudcode-pa")) {
      return new Response(DISABLED_403, {
        status: 403,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ response: { candidates: [] } }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }, () => callSendWithFallback(transformer));

  assert.equal(error, undefined, `unexpected error: ${error?.message}`);
  assert.equal(result?.status, 200, "prod must serve the request");
  assert.equal(urls.length, 3, "all three hosts tried in order");
  assert.ok(urls[2].includes("cloudcode-pa.googleapis.com"));

  // The staging host is out for the rest of the session.
  assert.equal(isEndpointUnusable(ANTIGRAVITY_ENDPOINT_AUTOPUSH), true);
  assert.deepEqual(antigravityEndpointCandidates(ANTIGRAVITY_ENDPOINT_DAILY), [
    ANTIGRAVITY_ENDPOINT_DAILY,
    ANTIGRAVITY_ENDPOINT_PROD,
  ]);
  clearUnusableEndpoints();
}

/** Every host disabled → report 403, never a 500 with the body embedded. */
async function testExhaustedWalkSurfacesUpstreamStatus() {
  clearUnusableEndpoints();
  clearPreferredEndpoint();
  const transformer = new AntigravityAuthTransformer();
  transformer.logger = noopLogger();

  const { result, error } = await withStubbedFetch(
    async () =>
      new Response(DISABLED_403, {
        status: 403,
        headers: { "Content-Type": "application/json" },
      }),
    () => callSendWithFallback(transformer)
  );

  assert.equal(result, undefined, "must not return the error body as a reply");
  assert.equal(error?.statusCode, 403, "upstream status is reported as itself");
  assert.match(String(error?.message), /has not been used in project/);
  clearUnusableEndpoints();
}

/** A client hanging up must not re-send the prompt to the remaining hosts. */
async function testClientAbortDoesNotWalk() {
  clearUnusableEndpoints();
  clearPreferredEndpoint();
  const transformer = new AntigravityAuthTransformer();
  transformer.logger = noopLogger();

  const abort = Object.assign(new Error("This operation was aborted"), {
    name: "AbortError",
  });
  const { error, urls } = await withStubbedFetch(
    async () => {
      throw abort;
    },
    () => callSendWithFallback(transformer)
  );

  assert.equal(urls.length, 1, "only the first host may be contacted");
  assert.equal(error?.name, "AbortError");
  clearUnusableEndpoints();
}

async function main() {
  testClassifiers();
  testUnusableEndpointsAreSkipped();
  await testWalksPastDisabledStagingToProd();
  await testExhaustedWalkSurfacesUpstreamStatus();
  await testClientAbortDoesNotWalk();
  console.log("antigravity.endpoint-fallback: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

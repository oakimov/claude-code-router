import assert from "node:assert/strict";
import {
  getProxyDispatcher,
  closeProxyDispatchers,
  __getProxyDispatcherCacheForTests,
  resolveFetchProxyOption,
  isBunRuntime,
} from "../utils/request";

function sameUrlReusesNormalizedKey() {
  closeProxyDispatchers();
  const a = getProxyDispatcher("http://127.0.0.1:8888");
  const b = getProxyDispatcher("http://127.0.0.1:8888");
  assert.strictEqual(a, b, "same proxy URL should normalize identically");
  assert.equal(__getProxyDispatcherCacheForTests().size, 1);
}

function normalizedUrlReusesKey() {
  closeProxyDispatchers();
  const a = getProxyDispatcher("http://proxy.example:8080");
  const b = getProxyDispatcher(new URL("http://proxy.example:8080").toString());
  assert.strictEqual(a, b, "URL-normalized proxy strings should match");
}

function differentUrlsGetDistinctKeys() {
  closeProxyDispatchers();
  const a = getProxyDispatcher("http://127.0.0.1:8888");
  const b = getProxyDispatcher("http://127.0.0.1:9999");
  assert.notStrictEqual(a, b);
  assert.equal(__getProxyDispatcherCacheForTests().size, 2);
}

function closeClearsCache() {
  closeProxyDispatchers();
  getProxyDispatcher("http://127.0.0.1:8888");
  getProxyDispatcher("http://127.0.0.1:9999");
  assert.equal(__getProxyDispatcherCacheForTests().size, 2);
  closeProxyDispatchers();
  assert.equal(__getProxyDispatcherCacheForTests().size, 0);
  const again = getProxyDispatcher("http://127.0.0.1:8888");
  assert.ok(again);
  assert.equal(__getProxyDispatcherCacheForTests().size, 1);
  closeProxyDispatchers();
}

function resolveProxyOptionMatchesConfiguredSemantics() {
  closeProxyDispatchers();
  const configured = resolveFetchProxyOption(
    "https://opencode.ai/inference/openai/v1/chat/completions",
    "http://127.0.0.1:8888"
  );
  assert.equal(configured, "http://127.0.0.1:8888/");

  const loopback = resolveFetchProxyOption(
    "http://127.0.0.1:3456/v1/messages",
    "http://127.0.0.1:8888"
  );
  // Bypass → Bun forces proxy:false so env HTTPS_PROXY cannot leak through.
  assert.equal(loopback, isBunRuntime() ? false : undefined);

  const unset = resolveFetchProxyOption(
    "https://api.anthropic.com/v1/messages",
    undefined
  );
  assert.equal(unset, isBunRuntime() ? false : undefined);
  closeProxyDispatchers();
}

async function main() {
  sameUrlReusesNormalizedKey();
  normalizedUrlReusesKey();
  differentUrlsGetDistinctKeys();
  closeClearsCache();
  resolveProxyOptionMatchesConfiguredSemantics();
  console.log("proxy-dispatcher-cache: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

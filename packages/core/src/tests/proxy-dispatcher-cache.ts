import assert from "node:assert/strict";
import {
  getProxyDispatcher,
  closeProxyDispatchers,
  __getProxyDispatcherCacheForTests,
} from "../utils/request";

function sameUrlReusesAgent() {
  closeProxyDispatchers();
  const a = getProxyDispatcher("http://127.0.0.1:8888");
  const b = getProxyDispatcher("http://127.0.0.1:8888");
  assert.strictEqual(a, b, "same proxy URL should reuse ProxyAgent");
  assert.equal(__getProxyDispatcherCacheForTests().size, 1);
}

function normalizedUrlReusesAgent() {
  closeProxyDispatchers();
  const a = getProxyDispatcher("http://proxy.example:8080");
  const b = getProxyDispatcher(new URL("http://proxy.example:8080").toString());
  assert.strictEqual(a, b, "URL-normalized proxy strings should share an agent");
}

function differentUrlsGetDistinctAgents() {
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

async function main() {
  sameUrlReusesAgent();
  normalizedUrlReusesAgent();
  differentUrlsGetDistinctAgents();
  closeClearsCache();
  console.log("proxy-dispatcher-cache: ok");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

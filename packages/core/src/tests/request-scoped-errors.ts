/**
 * Request-scoped error rules: validation, status/body matching, raw
 * classification text, provider over global precedence, decision kinds, and
 * the scoped cooldown registry.
 */
import assert from "node:assert/strict";
import { inspect } from "node:util";
import {
  attachScopedErrorClassificationText,
  compiledScopedErrorRules,
  decideScopedError,
  errorStatusForClassification,
  errorTextForClassification,
  matchScopedErrorRule,
  MAX_SCOPED_ERROR_CLASSIFICATION_CHARS,
  normalizeRequestScopedErrorRules,
  readGlobalScopedErrorRules,
  readScopedErrorClassificationText,
  readScopedErrorRulesFromCarrier,
  ScopedErrorCooldownRegistry,
  scopedErrorCooldownKey,
  scopedErrorCooldownsFor,
  type ScopedErrorRuleIssue,
} from "../utils/request-scoped-errors";
import { createApiError } from "../api/middleware";
import { sanitizeErrorForLog } from "../utils/redact";

function rules(input: unknown) {
  return normalizeRequestScopedErrorRules(input);
}

function rulesWithIssues(input: unknown) {
  const issues: ScopedErrorRuleIssue[] = [];
  const parsed = normalizeRequestScopedErrorRules(input, (issue) =>
    issues.push(issue)
  );
  return { parsed, issues };
}

async function testNormalization() {
  assert.deepEqual(rules(undefined), []);
  assert.deepEqual(rules(null), []);
  const parsed = rules([
    { status: 400, match: ["abc"], action: "stop" },
    { match: ["x"], action: "continue-and-cooldown", cooldown_seconds: 5 },
  ]);
  assert.equal(parsed.length, 2);
  assert.equal(parsed[0].status, 400);
  assert.deepEqual(parsed[0].match, ["abc"]);
  assert.equal(parsed[0].cooldownSeconds, 60);
  assert.equal(parsed[1].status, undefined);
  assert.equal(parsed[1].action, "continue-and-cooldown");
  assert.equal(parsed[1].cooldownSeconds, 5);
  // Kebab-case and camelCase regex keys are accepted.
  const regexes = rules([
    { matchRegex: ["foo\\d+"], action: "stop" },
    { "match-regex": ["^bar"], action: "stop" },
  ]);
  assert.equal(regexes.length, 2);
  assert.ok(regexes[0].matchRegex[0].test("foo123"));
  assert.ok(regexes[1].matchRegex[0].test("barstool"));
}

async function testValidationRejectsAndReports() {
  // Every malformed rule is dropped with a reason; none is widened into a
  // broader match. Only the last entry survives.
  const { parsed, issues } = rulesWithIssues([
    { status: 429, action: "bogus" },
    { status: 500, match_regex: ["(["], action: "continue" },
    { status: "429", match: ["x"], action: "continue" },
    { status: 42, action: "stop" },
    { matches: ["typo"], action: "stop" },
    { match: "not-a-list", action: "stop" },
    { match: [""], action: "stop" },
    { match: ["x"], action: "stop-and-cooldown", cooldown_seconds: 0 },
    { match: ["x"], action: "stop-and-cooldown", cooldownSeconds: "5" },
    "junk",
    null,
    { status: 400, match: ["ok"], action: "stop" },
  ]);
  assert.equal(parsed.length, 1);
  assert.deepEqual(parsed[0].match, ["ok"]);
  assert.deepEqual(
    issues.map((issue) => issue.index),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  );
  assert.match(issues[0].reason, /"action"/);
  assert.match(issues[1].reason, /invalid "match_regex" pattern/);
  assert.match(issues[2].reason, /"status"/);
  assert.match(issues[3].reason, /"status"/);
  assert.match(issues[4].reason, /unknown key\(s\): "matches"/);
  assert.match(issues[5].reason, /"match" must be an array/);
  assert.match(issues[6].reason, /"match" must be an array/);
  assert.match(issues[7].reason, /"cooldown_seconds"/);
  assert.match(issues[8].reason, /"cooldownSeconds"/);

  const notList = rulesWithIssues({ action: "stop" });
  assert.deepEqual(notList.parsed, []);
  assert.deepEqual(notList.issues, [
    { index: -1, reason: "rule list must be an array" },
  ]);
}

async function testCompiledRulesAreCachedPerList() {
  const list = [{ status: 429, action: "continue" }, { matches: [], action: "stop" }];
  let reports = 0;
  const first = compiledScopedErrorRules(list, () => (reports += 1));
  const second = compiledScopedErrorRules(list, () => (reports += 1));
  assert.equal(first, second, "same list object reuses the compiled rules");
  assert.equal(first.length, 1);
  assert.equal(reports, 1, "an invalid rule is reported once per list");
  // A new list (config update) compiles and reports again.
  compiledScopedErrorRules([...list], () => (reports += 1));
  assert.equal(reports, 2);
}

async function testCarrierKeys() {
  for (const key of [
    "request_scoped_errors",
    "requestScopedErrors",
    "request-scoped-errors",
  ]) {
    const list = [{ action: "stop" }];
    assert.equal(readScopedErrorRulesFromCarrier({ [key]: list }), list, key);
  }
  assert.equal(readScopedErrorRulesFromCarrier({}), undefined);
  assert.equal(readScopedErrorRulesFromCarrier(null), undefined);
  // An array is never a carrier (the old global-rules bug fed one in).
  assert.equal(readScopedErrorRulesFromCarrier([{ action: "stop" }]), undefined);
}

async function testGlobalRuleLookup() {
  const list = [{ action: "stop" }];
  const config = (values: Record<string, unknown>) => ({
    get: (key: string) => values[key],
  });
  assert.equal(readGlobalScopedErrorRules(config({ request_scoped_errors: list })), list);
  assert.equal(readGlobalScopedErrorRules(config({ requestScopedErrors: list })), list);
  assert.equal(
    readGlobalScopedErrorRules(config({ "request-scoped-errors": list })),
    list
  );
  assert.equal(readGlobalScopedErrorRules(config({})), undefined);
}

async function testStatusExtraction() {
  assert.equal(
    errorStatusForClassification(createApiError("x", 429, "c")),
    429
  );
  assert.equal(
    errorStatusForClassification({ upstream: { status: 503 } }),
    503
  );
  assert.equal(errorStatusForClassification(new Error("plain")), undefined);
}

async function testStatusOnlyMatch() {
  const parsed = rules([{ status: 429, action: "continue" }]);
  assert.ok(matchScopedErrorRule(createApiError("busy", 429, "c"), parsed));
  assert.equal(
    matchScopedErrorRule(createApiError("busy", 500, "c"), parsed),
    undefined
  );
}

async function testSubstringMatchIsCaseInsensitive() {
  const parsed = rules([
    { status: 400, match: ["maximum_context_length"], action: "stop" },
  ]);
  const hit = createApiError(
    "Error from provider(x: 400): MAXIMUM_CONTEXT_LENGTH exceeded",
    400,
    "context_length_exceeded"
  );
  assert.ok(matchScopedErrorRule(hit, parsed));
  const miss = createApiError("Error from provider(x: 400): bad request", 400, "c");
  assert.equal(matchScopedErrorRule(miss, parsed), undefined);
  // Upstream body participates in matching when no raw text is attached.
  const upstreamHit = createApiError("Error", 400, "c") as any;
  upstreamHit.upstream = {
    body: { error: { message: "maximum_context_length" } },
  };
  assert.ok(matchScopedErrorRule(upstreamHit, parsed));
}

async function testRegexMatch() {
  const parsed = rules([
    { match_regex: ["context_length_\\w+"], action: "stop" },
  ]);
  assert.ok(
    matchScopedErrorRule(createApiError("context_length_exceeded!", 400, "c"), parsed)
  );
  assert.equal(
    matchScopedErrorRule(createApiError("all good", 400, "c"), parsed),
    undefined
  );
}

async function testBodyOnlyRuleMatchesAnyStatus() {
  const parsed = rules([{ match: ["server_is_overloaded"], action: "continue" }]);
  assert.ok(
    matchScopedErrorRule(createApiError("server_is_overloaded", 200, "c"), parsed)
  );
}

async function testGenericFieldsNeverMatch() {
  // `api_error` (the type of every provider failure) and CCR's catch-all
  // codes must not make a broad pattern match every error.
  const parsed = rules([{ match: ["api_error"], action: "stop" }]);
  assert.equal(
    matchScopedErrorRule(createApiError("bad gateway", 502, "c", "api_error"), parsed),
    undefined
  );
  const catchAll = rules([{ match: ["provider_response_error"], action: "stop" }]);
  assert.equal(
    matchScopedErrorRule(
      createApiError("bad gateway", 502, "provider_response_error"),
      catchAll
    ),
    undefined
  );
  // A specific code still matches (e.g. Codex `usage_limit_reached`).
  const quota = rules([{ match: ["usage_limit_reached"], action: "continue" }]);
  assert.ok(
    matchScopedErrorRule(createApiError("quota", 429, "usage_limit_reached"), quota)
  );
}

async function testRawClassificationText() {
  const marker = "server_is_overloaded";
  const raw = `${"x".repeat(1_000)} ${marker}`;
  const error = createApiError(
    `Error from provider(codex,gpt-5: 400): ${raw}`,
    400,
    "provider_response_error"
  );
  // The sanitized message is cut to 240 chars, so the marker is gone.
  assert.ok(!error.message.includes(marker));
  const parsed = rules([{ status: 400, match: [marker], action: "continue" }]);
  assert.equal(matchScopedErrorRule(error, parsed), undefined);

  attachScopedErrorClassificationText(error, raw);
  assert.equal(readScopedErrorClassificationText(error), raw);
  assert.ok(matchScopedErrorRule(error, parsed));

  // Only the raw text (plus a specific code) is matched: the provider/model
  // message prefix no longer is.
  const byProvider = rules([{ match: ["codex"], action: "stop" }]);
  assert.equal(matchScopedErrorRule(error, byProvider), undefined);

  // Never serialized, spread, inspected or logged.
  assert.ok(!JSON.stringify(error).includes(marker));
  assert.ok(!JSON.stringify({ ...error }).includes(marker));
  assert.ok(!inspect(error).includes(marker));
  assert.ok(!JSON.stringify(sanitizeErrorForLog(error)).includes(marker));
  for (const symbol of Object.getOwnPropertySymbols(error)) {
    assert.equal(
      Object.getOwnPropertyDescriptor(error, symbol)?.enumerable,
      false
    );
  }

  // Bounded.
  const big = createApiError("big", 500, "c");
  attachScopedErrorClassificationText(big, "y".repeat(MAX_SCOPED_ERROR_CLASSIFICATION_CHARS + 50));
  assert.equal(
    readScopedErrorClassificationText(big)?.length,
    MAX_SCOPED_ERROR_CLASSIFICATION_CHARS
  );
  // No-ops.
  const plain = createApiError("plain", 500, "c");
  attachScopedErrorClassificationText(plain, "");
  attachScopedErrorClassificationText(plain, undefined);
  assert.equal(readScopedErrorClassificationText(plain), undefined);
  attachScopedErrorClassificationText(undefined, "x");
  assert.equal(readScopedErrorClassificationText("str"), undefined);
  assert.equal(
    errorTextForClassification(createApiError("m", 500, "specific_code", "api_error")),
    "m\nspecific_code"
  );
}

async function testDecisionPrecedence() {
  const provider = rules([{ status: 429, action: "stop" }]);
  const global = rules([{ status: 429, action: "continue" }]);
  const decided = decideScopedError(
    createApiError("slow down", 429, "c"),
    provider,
    global
  );
  assert.equal(decided.kind, "stop");
  if (decided.kind === "stop") {
    assert.equal(decided.rule.action, "stop");
    assert.equal(decided.cooldown, false);
  } else {
    assert.fail("expected stop decision");
  }
  const fallback = decideScopedError(
    createApiError("slow down", 429, "c"),
    [],
    global
  );
  assert.equal(fallback.kind, "continue");
  assert.equal(
    decideScopedError(createApiError("ok?", 200, "c"), provider, global).kind,
    "default"
  );
  const cooling = decideScopedError(
    createApiError("slow down", 429, "c"),
    rules([{ status: 429, action: "continue-and-cooldown" }]),
    []
  );
  assert.equal(cooling.kind, "continue");
  if (cooling.kind === "continue") {
    assert.equal(cooling.cooldown, true);
  } else {
    assert.fail("expected continue decision");
  }
}

async function testCooldownRegistry() {
  const registry = new ScopedErrorCooldownRegistry();
  const t0 = 1_000_000;
  assert.equal(registry.isCooledDown("p", "m", t0), false);
  registry.put("p", "m", 60, t0);
  assert.equal(registry.isCooledDown("p", "m", t0), true);
  assert.equal(registry.isCooledDown("p", "other", t0), false);
  // Active until the window lapses, cold (and dropped) at expiry.
  assert.equal(registry.isCooledDown("p", "m", t0 + 59_999), true);
  assert.equal(registry.isCooledDown("p", "m", t0 + 60_000), false);
  assert.equal(registry.size, 0, "expired entry deleted on read");

  // The `[1m]` marker names the same upstream model.
  assert.equal(scopedErrorCooldownKey("p", "m[1m]"), scopedErrorCooldownKey("p", "m"));
  registry.put("p", "m[1m]", 10, t0);
  assert.equal(registry.isCooledDown("p", "m", t0), true);

  // Expired keys that are never read again are pruned on the next put.
  registry.put("p", "never-read", 1, t0);
  assert.equal(registry.size, 2);
  registry.put("p", "fresh", 60, t0 + 20_000);
  assert.equal(registry.size, 1, "both expired entries pruned");
  assert.equal(registry.isCooledDown("p", "fresh", t0 + 20_000), true);

  // Registries are scoped: namespaces with same-named providers do not share.
  const scopeA = {};
  const scopeB = {};
  scopedErrorCooldownsFor(scopeA).put("shared", "m", 60);
  assert.equal(scopedErrorCooldownsFor(scopeA).isCooledDown("shared", "m"), true);
  assert.equal(scopedErrorCooldownsFor(scopeB).isCooledDown("shared", "m"), false);
  assert.equal(scopedErrorCooldownsFor(scopeA), scopedErrorCooldownsFor(scopeA));
}

async function main() {
  await testNormalization();
  await testValidationRejectsAndReports();
  await testCompiledRulesAreCachedPerList();
  await testCarrierKeys();
  await testGlobalRuleLookup();
  await testStatusExtraction();
  await testStatusOnlyMatch();
  await testSubstringMatchIsCaseInsensitive();
  await testRegexMatch();
  await testBodyOnlyRuleMatchesAnyStatus();
  await testGenericFieldsNeverMatch();
  await testRawClassificationText();
  await testDecisionPrecedence();
  await testCooldownRegistry();
  console.log("request-scoped-errors: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

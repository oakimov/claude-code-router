/**
 * Protocol registry path/method matching.
 */
import assert from "node:assert/strict";
import {
  isRoutedLlmPost,
  listClientRouteRegistrations,
  matchClientProtocol,
} from "../routing/protocol-endpoints";

function main() {
  // Anthropic Messages
  {
    const m = matchClientProtocol("POST", "/v1/messages");
    assert.ok(m);
    assert.equal(m!.protocol, "anthropic_messages");
    assert.equal(m!.canonicalPath, "/v1/messages");
    assert.equal(m!.ownerTransformerName, "Anthropic");
    assert.equal(m!.isAlias, false);
  }

  // OpenAI Chat — canonical + alias
  {
    const canonical = matchClientProtocol("POST", "/v1/chat/completions");
    assert.ok(canonical);
    assert.equal(canonical!.protocol, "openai_chat_completions");
    assert.equal(canonical!.ownerTransformerName, "OpenAI");

    const alias = matchClientProtocol("POST", "/chat/completions");
    assert.ok(alias);
    assert.equal(alias!.protocol, "openai_chat_completions");
    assert.equal(alias!.isAlias, true);
    assert.equal(alias!.canonicalPath, "/v1/chat/completions");
  }

  // OpenAI Responses — canonical + alias
  {
    const canonical = matchClientProtocol("POST", "/v1/responses");
    assert.ok(canonical);
    assert.equal(canonical!.protocol, "openai_responses");
    assert.equal(canonical!.ownerTransformerName, "openai-responses");

    const alias = matchClientProtocol("POST", "/responses");
    assert.ok(alias);
    assert.equal(alias!.isAlias, true);
  }

  // Preset prefix
  {
    const m = matchClientProtocol(
      "POST",
      "/preset/my-preset/v1/responses"
    );
    assert.ok(m);
    assert.equal(m!.protocol, "openai_responses");
    assert.equal(m!.presetPrefix, "/preset/my-preset");
    assert.equal(m!.matchedPath, "/v1/responses");
  }

  // Query stripping + trailing slash
  {
    const q = matchClientProtocol(
      "POST",
      "/v1/chat/completions?foo=1"
    );
    assert.ok(q);
    assert.equal(q!.protocol, "openai_chat_completions");

    const slash = matchClientProtocol("POST", "/v1/messages/");
    assert.ok(slash);
    assert.equal(slash!.protocol, "anthropic_messages");
  }

  // Exclusions: unimplemented protocols and legacy/stateful APIs.
  assert.equal(
    matchClientProtocol(
      "POST",
      "/v1beta/models/gemini-2.5-flash:generateContent"
    ),
    null
  );
  assert.equal(
    matchClientProtocol(
      "POST",
      "/v1/models/foo:streamGenerateContent?alt=sse"
    ),
    null
  );
  assert.equal(matchClientProtocol("POST", "/v1/completions"), null);
  assert.equal(matchClientProtocol("POST", "/completions"), null);
  assert.equal(
    matchClientProtocol("POST", "/v1beta/interactions"),
    null
  );
  assert.equal(matchClientProtocol("POST", "/v1/interactions"), null);
  assert.equal(matchClientProtocol("GET", "/v1/messages"), null);

  assert.equal(isRoutedLlmPost("POST", "/v1/messages"), true);
  assert.equal(isRoutedLlmPost("POST", "/v1/completions"), false);

  // Registration table contains only Anthropic Messages and OpenAI routes.
  const regs = listClientRouteRegistrations();
  assert.ok(regs.some((r) => r.path === "/v1/messages" && r.isCanonical));
  assert.ok(
    regs.some(
      (r) => r.path === "/chat/completions" && r.isCanonical === false
    )
  );
  assert.ok(regs.some((r) => r.path === "/responses" && !r.isCanonical));
  assert.deepEqual(
    new Set(regs.map((r) => r.protocol)),
    new Set([
      "anthropic_messages",
      "openai_chat_completions",
      "openai_responses",
    ])
  );

  console.log("protocol-endpoints: PASS");
}

main();

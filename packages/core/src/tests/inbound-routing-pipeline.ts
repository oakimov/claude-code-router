/**
 * Inbound routing pipeline: destination resolution, protocol errors,
 * Unified scenario helpers, and protocol-aware bypass.
 */
import assert from "node:assert/strict";
import {
  resolveDestination,
  protocolAwareBypass,
} from "../routing/inbound-pipeline";
import { protocolErrorBody } from "../routing/protocol-errors";
import {
  adaptClientRequest,
  cloneProtocolBody,
  normalizeClientToUnified,
  sanitizePassthroughHeaders,
  shouldBypassTransformersProtocolAware,
} from "../routing/protocol-adapter";
import { matchClientProtocol } from "../routing/protocol-endpoints";
import {
  canonicalizeOutboundHeaders,
  mergeHeadersCaseInsensitive,
} from "../utils/headers";

function expectThrow(
  fn: () => unknown,
  code: string,
  statusCode: number
): void {
  let caught: any;
  try {
    fn();
  } catch (e) {
    caught = e;
  }
  assert.ok(caught, `expected throw with code ${code}`);
  assert.equal(caught.code, code);
  assert.equal(caught.statusCode, statusCode);
}

async function main() {
  // Explicit provider,model
  {
    const d = resolveDestination("openai,gpt-4o", "openai_chat_completions");
    assert.equal(d.providerName, "openai");
    assert.equal(d.modelName, "gpt-4o");
  }

  // Bare model → 400 unresolved_model
  expectThrow(
    () => resolveDestination("gpt-4o", "openai_chat_completions"),
    "unresolved_model",
    400
  );

  // Missing model → 400
  expectThrow(
    () => resolveDestination(undefined, "anthropic_messages"),
    "missing_model",
    400
  );

  // Protocol error envelopes
  {
    const openai = protocolErrorBody(
      "openai_chat_completions",
      "bad",
      400,
      "unresolved_model"
    );
    assert.equal((openai.body as any).error.code, "unresolved_model");
    assert.equal((openai.body as any).error.type, "invalid_request_error");

    const anthropic = protocolErrorBody(
      "anthropic_messages",
      "bad",
      400,
      "unresolved_model"
    );
    assert.equal((anthropic.body as any).type, "error");
    assert.equal((anthropic.body as any).error.code, "unresolved_model");

  }

  // Chat adapt + normalize (identity)
  {
    const match = matchClientProtocol("POST", "/v1/chat/completions")!;
    const { normalizationInput, context } = adaptClientRequest(match, {
      model: "openai,gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      stream: true,
    });
    assert.equal(context.stream, true);
    assert.equal(context.protocol, "openai_chat_completions");

    const unified = await normalizeClientToUnified(
      "openai_chat_completions",
      normalizationInput,
      {} as any,
      {}
    );
    assert.equal(unified.model, "openai,gpt-4o");
    assert.equal(unified.messages.length, 1);
  }

  // Responses adapt: reject store:true; project instructions+input
  {
    const match = matchClientProtocol("POST", "/v1/responses")!;
    await assert.rejects(
      async () => {
        const { normalizationInput } = adaptClientRequest(match, {
          model: "openai,gpt-4o",
          input: "hi",
          store: true,
        });
        await normalizeClientToUnified(
          "openai_responses",
          normalizationInput,
          {} as any,
          {}
        );
      },
      (err: any) => err?.code === "unsupported_store"
    );

    const { normalizationInput, context } = adaptClientRequest(match, {
      model: "openai,gpt-4o",
      instructions: "be brief",
      input: "hello",
      stream: false,
      reasoning: { effort: "high" },
      tools: [{ type: "web_search" }],
    });
    assert.equal(context.stream, false);
    const unified = await normalizeClientToUnified(
      "openai_responses",
      normalizationInput,
      {} as any,
      {}
    );
    assert.ok(unified.messages.some((m: any) => m.role === "system"));
    assert.ok(unified.messages.some((m: any) => m.role === "user"));
    assert.equal(unified.reasoning?.effort, "high");
    assert.ok(
      Array.isArray(unified.tools) &&
        unified.tools.some((t: any) => t.type === "web_search")
    );
  }

  // Protocol-aware bypass: same-protocol OpenAI yes; cross-protocol no
  {
    const openaiTf = { name: "OpenAI" };
    const responsesTf = { name: "openai-responses" };
    const providerSame = {
      transformer: { use: [{ name: "OpenAI" }] },
    };
    assert.equal(
      shouldBypassTransformersProtocolAware(
        providerSame,
        openaiTf as any,
        "openai_chat_completions",
        "gpt-4o"
      ),
      true
    );
    assert.equal(
      shouldBypassTransformersProtocolAware(
        providerSame,
        responsesTf as any,
        "openai_responses",
        "gpt-4o"
      ),
      false
    );
    assert.equal(
      protocolAwareBypass(
        providerSame,
        openaiTf as any,
        {
          protocol: "openai_chat_completions",
          pathname: "/v1/chat/completions",
          canonicalPath: "/v1/chat/completions",
          matchedPath: "/v1/chat/completions",
          stream: false,
          ownerTransformerName: "OpenAI",
        },
        "gpt-4o"
      ),
      true
    );
  }

  // Stream intent from Responses body
  {
    const match = matchClientProtocol("POST", "/responses")!;
    const { context } = adaptClientRequest(match, {
      model: "x,y",
      input: "hi",
      stream: true,
    });
    assert.equal(context.stream, true);
    assert.equal(match.isAlias, true);
  }

  // Provider attempts receive independent bodies and never inherit mutation.
  {
    const original = { messages: [{ role: "user", content: "hi" }] };
    const cloned = cloneProtocolBody(original);
    cloned.messages[0].content = "changed";
    assert.equal(original.messages[0].content, "hi");
  }

  // Exact-wire passthrough preserves protocol metadata, never CCR secrets.
  {
    const headers = sanitizePassthroughHeaders({
      authorization: "Bearer ccr-secret",
      "x-api-key": "ccr-secret",
      cookie: "ccr_session=secret",
      "content-length": "123",
      "x-auth-token": "must-not-cross-boundary",
      "openai-secret": "must-not-cross-boundary",
      "anthropic-version": "2023-06-01",
      "openai-beta": "responses=v1",
      "x-app": "cli",
      "x-claude-code-session-id": "session-id",
      "x-client-request-id": "request-id",
    });
    assert.equal(headers.authorization, undefined);
    assert.equal(headers["x-api-key"], undefined);
    assert.equal(headers.cookie, undefined);
    assert.equal(headers["content-length"], undefined);
    assert.equal(headers["x-auth-token"], undefined);
    assert.equal(headers["openai-secret"], undefined);
    assert.equal(headers["anthropic-version"], "2023-06-01");
    assert.equal(headers["openai-beta"], "responses=v1");
    assert.equal(headers["x-app"], "cli");
    assert.equal(headers["x-claude-code-session-id"], "session-id");
    assert.equal(headers["x-client-request-id"], "request-id");
  }

  // Transformer headers compose without casing duplicates or auth leakage.
  {
    const merged = mergeHeadersCaseInsensitive(
      {
        Authorization: "Bearer oauth",
        "anthropic-beta": "oauth-2025-04-20",
        "x-app": "cli",
      },
      {
        authorization: "Bearer rotated",
        "Anthropic-Version": "2023-06-01",
        "x-app": undefined,
      }
    );
    assert.equal(merged.Authorization, undefined);
    assert.equal(merged.authorization, "Bearer rotated");
    assert.equal(merged["x-app"], undefined);
    assert.equal(merged["anthropic-beta"], "oauth-2025-04-20");

    const bearer = canonicalizeOutboundHeaders(merged, "no-key");
    assert.equal(bearer.Authorization, "Bearer rotated");
    assert.equal(bearer.authorization, undefined);
    assert.equal(bearer["x-api-key"], undefined);

    const apiKey = canonicalizeOutboundHeaders(
      {
        Authorization: "Bearer no-key",
        "X-API-Key": "sk-provider",
      },
      "no-key"
    );
    assert.equal(apiKey["x-api-key"], "sk-provider");
    assert.equal(apiKey.Authorization, undefined);
  }

  console.log("inbound-routing-pipeline: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

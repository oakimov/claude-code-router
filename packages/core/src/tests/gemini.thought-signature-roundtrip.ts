/**
 * Real thought signatures must survive a Claude Code round trip.
 *
 * Gemini 3 / Antigravity sign the reasoning behind every functionCall. Anthropic
 * tool_use cannot carry that signature (and must not grow the field), so CCR used
 * to replay every tool call with `skip_thought_signature_validator` — documented
 * by Google as a last resort that "will negatively impact model performance".
 * In practice a long tool session degraded until the model emitted an empty
 * functionCall and the turn died with MALFORMED_FUNCTION_CALL.
 *
 * CCR now caches the signature under the tool-call id, which does survive the
 * round trip (functionCall.id → tool_use.id → tool_result.tool_use_id →
 * tool_calls[].id), and restores it on replay. The sentinel stays as the
 * fallback for genuine cache misses.
 */
import assert from "node:assert/strict";
import {
  buildRequestBody,
  SKIP_THOUGHT_SIGNATURE,
  transformResponseOut,
} from "../utils/gemini.util";
import { clearThoughtSignatureCache } from "../utils/thought-signature-cache";
import type { UnifiedChatRequest } from "../types/llm";

const REAL_SIG = "EvkBCvYBARFNMg+fwHa6uGuHVsbAXCp0yypxvsSmuOIBvYhciXRW";
const OTHER_SIG = "Cq4BAWZNZ2FudGlncmF2aXR5c2lnbmF0dXJlMg";

/** Stream one Antigravity-shaped tool turn and return the Unified deltas. */
async function streamToolTurn(
  calls: Array<{ id: string; name: string; signature?: string }>,
  scope: string
): Promise<any[]> {
  const chunks = [
    {
      responseId: "r1",
      modelVersion: "gemini-3.6-flash-tiered",
      candidates: [
        {
          content: {
            role: "model",
            parts: calls.map((c) => ({
              ...(c.signature ? { thoughtSignature: c.signature } : {}),
              functionCall: { id: c.id, name: c.name, args: { command: "ls" } },
            })),
          },
        },
      ],
    },
    {
      responseId: "r1",
      modelVersion: "gemini-3.6-flash-tiered",
      candidates: [
        {
          content: { role: "model", parts: [{ text: "" }] },
          finishReason: "STOP",
        },
      ],
    },
  ];

  const sse = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("");
  const out = await transformResponseOut(
    new Response(sse, {
      status: 200,
      headers: { "content-type": "text/event-stream" },
    }),
    "gemini",
    undefined,
    scope
  );
  return (await out.text())
    .split("\n")
    .filter((l) => l.startsWith("data: "))
    .map((l) => JSON.parse(l.slice(6)));
}

/**
 * Claude Code's replay shape: the tool call comes back with its id but no
 * signature anywhere — not on the tool call, not as a turn-level thinking
 * signature.
 */
function replayRequest(
  calls: Array<{ id: string; name: string }>
): UnifiedChatRequest {
  return {
    model: "gemini-3.6-flash-tiered",
    messages: [
      { role: "user", content: "list the files" },
      {
        role: "assistant",
        content: null,
        tool_calls: calls.map((c) => ({
          id: c.id,
          type: "function",
          function: { name: c.name, arguments: '{"command":"ls"}' },
        })),
      },
      ...calls.map((c) => ({
        role: "tool" as const,
        content: "ok",
        tool_call_id: c.id,
      })),
    ],
  } as UnifiedChatRequest;
}

function functionCallParts(body: any): any[] {
  const model = body.contents.find((c: any) => c.role === "model");
  assert.ok(model, "expected a model turn");
  return model.parts.filter((p: any) => p.functionCall);
}

async function testSignatureRestoredOnReplay() {
  clearThoughtSignatureCache();
  await streamToolTurn([{ id: "WsVKVkU3", name: "Bash", signature: REAL_SIG }], "antigravity");

  const body = buildRequestBody(replayRequest([{ id: "WsVKVkU3", name: "Bash" }]), {
    signatureScope: "antigravity",
  });
  const parts = functionCallParts(body);
  assert.equal(parts.length, 1);
  assert.equal(
    parts[0].thoughtSignature,
    REAL_SIG,
    "the real signature must be replayed, not the validator-skip sentinel"
  );
}

/** Each parallel call carries its own signature back to where it was produced. */
async function testParallelCallsKeepTheirOwnSignatures() {
  clearThoughtSignatureCache();
  await streamToolTurn(
    [
      { id: "aaa11111", name: "Bash", signature: REAL_SIG },
      { id: "bbb22222", name: "Read", signature: OTHER_SIG },
    ],
    "antigravity"
  );

  const body = buildRequestBody(
    replayRequest([
      { id: "aaa11111", name: "Bash" },
      { id: "bbb22222", name: "Read" },
    ]),
    { signatureScope: "antigravity" }
  );
  const parts = functionCallParts(body);
  assert.deepEqual(
    parts.map((p: any) => p.thoughtSignature),
    [REAL_SIG, OTHER_SIG]
  );
}

/**
 * A sibling call that upstream left unsigned stays unsigned — the sentinel is
 * only ever stamped on the first part, and never over a real signature.
 */
async function testUnsignedSiblingStaysUnsigned() {
  clearThoughtSignatureCache();
  await streamToolTurn(
    [
      { id: "ccc33333", name: "Bash", signature: REAL_SIG },
      { id: "ddd44444", name: "Read" },
    ],
    "antigravity"
  );

  const body = buildRequestBody(
    replayRequest([
      { id: "ccc33333", name: "Bash" },
      { id: "ddd44444", name: "Read" },
    ]),
    { signatureScope: "antigravity" }
  );
  const parts = functionCallParts(body);
  assert.equal(parts[0].thoughtSignature, REAL_SIG);
  assert.equal(parts[1].thoughtSignature, undefined);
}

/** A signature is only valid at the upstream that minted it. */
async function testScopeIsolation() {
  clearThoughtSignatureCache();
  await streamToolTurn([{ id: "eee55555", name: "Bash", signature: REAL_SIG }], "antigravity");

  const body = buildRequestBody(replayRequest([{ id: "eee55555", name: "Bash" }]), {
    signatureScope: "gemini-public",
  });
  const parts = functionCallParts(body);
  assert.equal(
    parts[0].thoughtSignature,
    SKIP_THOUGHT_SIGNATURE,
    "a foreign provider must fall back to the sentinel, never replay the signature"
  );
}

/** Cache miss keeps the previous behaviour exactly. */
async function testMissFallsBackToSentinel() {
  clearThoughtSignatureCache();
  const body = buildRequestBody(replayRequest([{ id: "unknown1", name: "Bash" }]), {
    signatureScope: "antigravity",
  });
  assert.equal(functionCallParts(body)[0].thoughtSignature, SKIP_THOUGHT_SIGNATURE);
}

/** "none" still disables the sentinel, and a cached signature is unaffected. */
async function testNoneFallbackStillHonoured() {
  clearThoughtSignatureCache();
  const miss = buildRequestBody(replayRequest([{ id: "unknown2", name: "Bash" }]), {
    signatureScope: "antigravity",
    thoughtSignatureFallback: "none",
  });
  assert.equal(functionCallParts(miss)[0].thoughtSignature, undefined);

  await streamToolTurn([{ id: "fff66666", name: "Bash", signature: REAL_SIG }], "antigravity");
  const hit = buildRequestBody(replayRequest([{ id: "fff66666", name: "Bash" }]), {
    signatureScope: "antigravity",
    thoughtSignatureFallback: "none",
  });
  assert.equal(functionCallParts(hit)[0].thoughtSignature, REAL_SIG);
}

/** CCR's own placeholder signatures are not upstream state — never cache them. */
async function testCcrPlaceholderSignaturesAreNotCached() {
  clearThoughtSignatureCache();
  await streamToolTurn([{ id: "ggg77777", name: "Bash", signature: "ccr_1785089918" }], "antigravity");

  const body = buildRequestBody(replayRequest([{ id: "ggg77777", name: "Bash" }]), {
    signatureScope: "antigravity",
  });
  assert.equal(functionCallParts(body)[0].thoughtSignature, SKIP_THOUGHT_SIGNATURE);
}

/** Non-streaming replies feed the cache too. */
async function testNonStreamingRemembersSignatures() {
  clearThoughtSignatureCache();
  const payload = {
    responseId: "r2",
    modelVersion: "gemini-3.6-flash",
    candidates: [
      {
        content: {
          role: "model",
          parts: [
            {
              thoughtSignature: REAL_SIG,
              functionCall: { id: "hhh88888", name: "Bash", args: {} },
            },
          ],
        },
        finishReason: "STOP",
      },
    ],
  };
  await transformResponseOut(
    new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }),
    "gemini",
    undefined,
    "antigravity"
  );

  const body = buildRequestBody(replayRequest([{ id: "hhh88888", name: "Bash" }]), {
    signatureScope: "antigravity",
  });
  assert.equal(functionCallParts(body)[0].thoughtSignature, REAL_SIG);
}

/** An explicit per-tool signature on the Unified request still wins. */
async function testExplicitSignatureWins() {
  clearThoughtSignatureCache();
  await streamToolTurn([{ id: "iii99999", name: "Bash", signature: REAL_SIG }], "antigravity");

  const req = replayRequest([{ id: "iii99999", name: "Bash" }]);
  (req.messages[1] as any).tool_calls[0].thought_signature = OTHER_SIG;
  const body = buildRequestBody(req, { signatureScope: "antigravity" });
  assert.equal(functionCallParts(body)[0].thoughtSignature, OTHER_SIG);
}

async function main() {
  await testSignatureRestoredOnReplay();
  await testParallelCallsKeepTheirOwnSignatures();
  await testUnsignedSiblingStaysUnsigned();
  await testScopeIsolation();
  await testMissFallsBackToSentinel();
  await testNoneFallbackStillHonoured();
  await testCcrPlaceholderSignaturesAreNotCached();
  await testNonStreamingRemembersSignatures();
  await testExplicitSignatureWins();
  clearThoughtSignatureCache();
  console.log("gemini.thought-signature-roundtrip: PASS");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

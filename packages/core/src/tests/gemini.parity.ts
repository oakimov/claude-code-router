/**
 * Gemini Parity Test Suite
 *
 * Captures current behavior of buildRequestBody and transformResponseOut
 * (JSON + Stream) as "golden" baseline. Any refactor must produce
 * 100% identical output.
 *
 * Usage:
 *   npx tsx packages/core/src/tests/gemini.parity.ts            # compare vs golden
 *   npx tsx packages/core/src/tests/gemini.parity.ts --update   # regenerate golden
 */
import { writeFileSync, readFileSync, mkdirSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

import {
  buildRequestBody,
  transformResponseOut,
} from "../utils/gemini.util";
import type { UnifiedChatRequest } from "../types/llm";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const GOLDEN_DIR = join(__dirname, "__golden__");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ensureGoldenDir() {
  if (!existsSync(GOLDEN_DIR)) mkdirSync(GOLDEN_DIR, { recursive: true });
}

function sortKeysDeep(obj: unknown): unknown {
  if (obj === null || obj === undefined || typeof obj !== "object") return obj;
  if (Array.isArray(obj)) return obj.map(sortKeysDeep);
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(obj as object).sort()) {
    sorted[key] = sortKeysDeep((obj as Record<string, unknown>)[key]);
  }
  return sorted;
}

function stableStringify(obj: unknown): string {
  return JSON.stringify(sortKeysDeep(obj), null, 2);
}

function deepDiff(
  actual: unknown,
  expected: unknown,
  path: string = ""
): string[] {
  if (actual === expected) return [];
  if (typeof actual !== typeof expected)
    return [`${path}: type mismatch ${typeof actual} vs ${typeof expected}`];
  if (actual === null || expected === null) {
    if (actual !== expected) return [`${path}: ${actual} !== ${expected}`];
    return [];
  }
  if (Array.isArray(actual) && Array.isArray(expected)) {
    const diffs: string[] = [];
    const len = Math.max(actual.length, expected.length);
    for (let i = 0; i < len; i++) {
      diffs.push(...deepDiff(actual[i], expected[i], `${path}[${i}]`));
    }
    return diffs;
  }
  if (typeof actual === "object") {
    const diffs: string[] = [];
    const allKeys = new Set([
      ...Object.keys(actual as object),
      ...Object.keys(expected as object),
    ]);
    for (const key of allKeys) {
      const a = (actual as any)[key];
      const e = (expected as any)[key];
      if (a === undefined && e === undefined) continue;
      diffs.push(...deepDiff(a, e, path ? `${path}.${key}` : key));
    }
    return diffs;
  }
  if (actual !== expected) return [`${path}: ${JSON.stringify(actual)} !== ${JSON.stringify(expected)}`];
  return [];
}

// ---------------------------------------------------------------------------
// Test Case Definitions
// ---------------------------------------------------------------------------

function makeRequestCases(): Record<string, UnifiedChatRequest> {
  return {
    "simple-text": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "Hello, world!" },
      ],
    },

    "multi-turn": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "Hi" },
        { role: "assistant", content: "Hello!" },
        { role: "user", content: "How are you?" },
      ],
    },

    "system-instruction": {
      model: "gemini-2.5-pro",
      system: "You are a helpful assistant.",
      messages: [
        { role: "user", content: "Hello" },
      ],
    },

    "system-role-message": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "system", content: "System prompt here" },
        { role: "user", content: "Hello" },
      ],
    },

    "thinking-block": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "Think about this" },
        {
          role: "assistant",
          content: "The answer is 42.",
          thinking: {
            content: "Let me think step by step...",
            signature: "sig_abc123",
          },
        },
      ],
    },

    "thinking-no-signature": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "Think" },
        {
          role: "assistant",
          content: "Answer",
          thinking: {
            content: "Hmm...",
            signature: "ccr_fallback",
          },
        },
      ],
    },

    "tool-calls": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "What is the weather?" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_1",
              type: "function",
              function: {
                name: "get_weather",
                arguments: '{"location":"NYC"}',
              },
            },
          ],
        },
        {
          role: "tool",
          content: "Sunny, 72F",
          tool_call_id: "call_1",
        },
      ],
      tools: [
        {
          type: "function",
          function: {
            name: "get_weather",
            description: "Get weather",
            parameters: {
              type: "object",
              properties: {
                location: { type: "string" },
              },
              required: ["location"],
            },
          },
        },
      ],
    },

    "tool-call-with-thought-signature": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "Search for X" },
        {
          role: "assistant",
          content: null,
          thinking: {
            content: "I need to search",
            signature: "sig_xyz789",
          },
          tool_calls: [
            {
              id: "call_2",
              type: "function",
              function: {
                name: "web_search",
                arguments: '{"query":"X"}',
              },
            },
          ],
        },
        {
          role: "tool",
          content: "Results for X",
          tool_call_id: "call_2",
        },
      ],
    },

    "multimodal-image-url": {
      model: "gemini-2.5-pro",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "What is in this image?" },
            {
              type: "image_url",
              image_url: { url: "https://example.com/image.jpg" },
              media_type: "image/jpeg",
            },
          ],
        },
      ],
    },

    "multimodal-image-base64": {
      model: "gemini-2.5-pro",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "Describe" },
            {
              type: "image_url",
              image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
              media_type: "image/png",
            },
          ],
        },
      ],
    },

    "tool-response-array": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "Do something" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_3",
              type: "function",
              function: { name: "my_tool", arguments: "{}" },
            },
          ],
        },
        {
          role: "tool",
          content: [
            { type: "text", text: "Result part 1" },
            { type: "text", text: "Result part 2" },
          ],
          tool_call_id: "call_3",
        },
      ],
    },

    "tool-response-object": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "Do something" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: "call_4",
              type: "function",
              function: { name: "my_tool", arguments: "{}" },
            },
          ],
        },
        {
          role: "tool",
          content: { key: "value" } as any,
          tool_call_id: "call_4",
        },
      ],
    },

    "generation-config-thinking": {
      model: "gemini-2.5-pro",
      messages: [{ role: "user", content: "Think hard" }],
      reasoning: {
        effort: "high",
        max_tokens: 16384,
      },
    },

    "generation-config-thinking-gemini3": {
      model: "gemini-3-flash",
      messages: [{ role: "user", content: "Think" }],
      reasoning: {
        effort: "medium",
      },
    },

    "generation-config-max-tokens": {
      model: "gemini-2.5-pro",
      messages: [{ role: "user", content: "Short answer" }],
      max_tokens: 100,
      temperature: 0.5,
    },

    "tool-choice-auto": {
      model: "gemini-2.5-pro",
      messages: [{ role: "user", content: "Use tools" }],
      tool_choice: "auto",
      tools: [
        {
          type: "function",
          function: {
            name: "my_func",
            description: "A function",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    },

    "tool-choice-none": {
      model: "gemini-2.5-pro",
      messages: [{ role: "user", content: "No tools" }],
      tool_choice: "none",
      tools: [
        {
          type: "function",
          function: {
            name: "my_func",
            description: "A function",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    },

    "tool-choice-required": {
      model: "gemini-2.5-pro",
      messages: [{ role: "user", content: "Must use tool" }],
      tool_choice: "required",
      tools: [
        {
          type: "function",
          function: {
            name: "my_func",
            description: "A function",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    },

    "tool-choice-specific": {
      model: "gemini-2.5-pro",
      messages: [{ role: "user", content: "Use specific tool" }],
      tool_choice: {
        type: "function",
        function: { name: "specific_func" },
      },
      tools: [
        {
          type: "function",
          function: {
            name: "specific_func",
            description: "Specific",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    },

    "web-search-tool": {
      model: "gemini-2.5-pro",
      messages: [{ role: "user", content: "Search for X" }],
      tools: [
        {
          type: "function",
          function: {
            name: "web_search",
            description: "Search the web",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    },

    "consolidate-roles": {
      model: "gemini-2.5-pro",
      messages: [
        { role: "user", content: "First" },
        { role: "user", content: "Second" },
        { role: "assistant", content: "Reply" },
      ],
    },
  };
}

// ---------------------------------------------------------------------------
// Stream Test Helpers
// ---------------------------------------------------------------------------

interface GeminiChunk {
  candidates: Array<{
    content: { parts: any[]; role?: string };
    finishReason?: string;
    groundingMetadata?: any;
  }>;
  responseId?: string;
  modelVersion?: string;
  usageMetadata?: any;
}

function makeGeminiSSEBody(chunks: GeminiChunk[]): string {
  return chunks
    .map((c) => `data: ${JSON.stringify(c)}`)
    .join("\n\n") + "\n\n";
}

function makeStreamResponse(chunks: GeminiChunk[]): Response {
  const body = makeGeminiSSEBody(chunks);
  return new Response(body, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

function makeJSONResponse(data: any): Response {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

async function collectStreamChunks(response: Response): Promise<any[]> {
  const results: any[] = [];
  if (!response.body) return results;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed.startsWith("data: ")) {
        const jsonStr = trimmed.slice(6);
        if (jsonStr) {
          try {
            results.push(JSON.parse(jsonStr));
          } catch {
            // skip malformed
          }
        }
      }
    }
  }
  return results;
}

// ---------------------------------------------------------------------------
// Stream Test Case Definitions
// ---------------------------------------------------------------------------

function makeStreamCases(): Record<string, GeminiChunk[]> {
  const baseMeta = { responseId: "resp_123", modelVersion: "gemini-2.5-pro" };

  return {
    "happy-path-thinking-sig-content": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Let me think...", thought: true }],
            role: "model",
          },
        }],
      },
        {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ thoughtSignature: "sig_real_abc" }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "The answer is 42." }],
            role: "model",
          },
          finishReason: "STOP",
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 20,
            totalTokenCount: 30,
            thoughtsTokenCount: 5,
          },
        }],
      },
    ],

    "gemini3-out-of-order": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Thinking...", thought: true }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        modelVersion: "gemini-3-flash",
        candidates: [{
          content: {
            parts: [{ text: "Final answer text" }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        modelVersion: "gemini-3-flash",
        candidates: [{
          content: {
            parts: [{ thoughtSignature: "sig_delayed" }],
            role: "model",
          },
          finishReason: "STOP",
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 20,
            totalTokenCount: 30,
          },
        }],
      },
    ],

    "missing-signature-fallback": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Some thinking", thought: true }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Final text without signature" }],
            role: "model",
          },
          finishReason: "STOP",
          usageMetadata: {
            promptTokenCount: 5,
            candidatesTokenCount: 15,
            totalTokenCount: 20,
          },
        }],
      },
    ],

    "empty-thinking-sig-only": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ thoughtSignature: "sig_empty_thinking" }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Content after empty thinking" }],
            role: "model",
          },
          finishReason: "STOP",
        }],
      },
    ],

    "tool-calls-with-thinking": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Planning tool use...", thought: true }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ thoughtSignature: "sig_tool_123" }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{
              functionCall: {
                id: "tool_abc",
                name: "get_weather",
                args: { location: "NYC" },
              },
            }],
            role: "model",
          },
          finishReason: "STOP",
          usageMetadata: {
            promptTokenCount: 10,
            candidatesTokenCount: 25,
            totalTokenCount: 35,
          },
        }],
      },
    ],

    "grounding-annotations": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Based on search results..." }],
            role: "model",
          },
          finishReason: "STOP",
          groundingMetadata: {
            groundingChunks: [
              { web: { uri: "https://example.com/1", title: "Result 1" } },
            ],
            groundingSupports: [
              {
                groundingChunkIndices: [0],
                segment: { text: "search results", startIndex: 0, endIndex: 15 },
              },
            ],
          },
        }],
      },
    ],

    "no-thinking-simple-content": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Just a simple response." }],
            role: "model",
          },
          finishReason: "STOP",
        }],
      },
    ],

    "multiple-tool-calls": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [
              {
                functionCall: {
                  id: "tool_1",
                  name: "func_a",
                  args: { x: 1 },
                },
              },
              {
                functionCall: {
                  id: "tool_2",
                  name: "func_b",
                  args: { y: 2 },
                },
              },
            ],
            role: "model",
          },
          finishReason: "STOP",
        }],
      },
    ],

    "gemini3-buffered-content-final-tool": [
      {
        ...baseMeta,
        modelVersion: "gemini-3-flash",
        candidates: [{
          content: {
            parts: [{ text: "Hmm let me think", thought: true }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        modelVersion: "gemini-3-flash",
        candidates: [{
          content: {
            parts: [{ text: "Intermediate content" }],
            role: "model",
          },
        }],
      },
      {
        ...baseMeta,
        modelVersion: "gemini-3-flash",
        candidates: [{
          content: {
            parts: [{
              functionCall: {
                id: "tool_final",
                name: "final_tool",
                args: {},
              },
            }],
            role: "model",
          },
          finishReason: "STOP",
        }],
      },
    ],

    "finish-reason-only": [
      {
        ...baseMeta,
        candidates: [{
          content: {
            parts: [{ text: "Hello" }],
            role: "model",
          },
          finishReason: "STOP",
          usageMetadata: {
            promptTokenCount: 5,
            candidatesTokenCount: 1,
            totalTokenCount: 6,
            thoughtsTokenCount: 0,
            cachedContentTokenCount: 2,
          },
        }],
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// JSON Response Test Cases
// ---------------------------------------------------------------------------

function makeJSONResponseCases(): Record<string, any> {
  return {
    "simple-response": {
      responseId: "resp_json_1",
      modelVersion: "gemini-2.5-pro",
      candidates: [{
        content: {
          parts: [{ text: "Hello!" }],
          role: "model",
        },
        finishReason: "STOP",
      }],
      usageMetadata: {
        promptTokenCount: 5,
        candidatesTokenCount: 2,
        totalTokenCount: 7,
      },
    },

    "thinking-response": {
      responseId: "resp_json_2",
      modelVersion: "gemini-2.5-pro",
      candidates: [{
        content: {
          parts: [
            { text: "Deep thoughts...", thought: true },
            { thoughtSignature: "sig_json_abc" },
            { text: "The answer." },
          ],
          role: "model",
        },
        finishReason: "STOP",
      }],
      usageMetadata: {
        promptTokenCount: 10,
        candidatesTokenCount: 30,
        totalTokenCount: 40,
        thoughtsTokenCount: 15,
      },
    },

    "tool-call-response": {
      responseId: "resp_json_3",
      modelVersion: "gemini-2.5-pro",
      candidates: [{
        content: {
          parts: [{
            functionCall: {
              id: "fc_1",
              name: "my_function",
              args: { param: "value" },
            },
          }],
          role: "model",
        },
        finishReason: "STOP",
      }],
      usageMetadata: {
        promptTokenCount: 8,
        candidatesTokenCount: 20,
        totalTokenCount: 28,
      },
    },

    "thinking-no-sig-response": {
      responseId: "resp_json_4",
      modelVersion: "gemini-2.5-pro",
      candidates: [{
        content: {
          parts: [
            { text: "Thinking without sig", thought: true },
            { text: "Answer" },
          ],
          role: "model",
        },
        finishReason: "STOP",
      }],
      usageMetadata: {
        promptTokenCount: 5,
        candidatesTokenCount: 10,
        totalTokenCount: 15,
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Test Runner
// ---------------------------------------------------------------------------

function makeDeterministic(obj: unknown): unknown {
  // Replace timestamps and random IDs with stable placeholders for comparison
  const s = JSON.stringify(obj);
  const stabilized = s
    .replace(/"created":\s*\d+/g, '"created":0')
    .replace(/"id":\s*"ctxexceeded_\d+"/g, '"id":"ctxexceeded_0"')
    .replace(/"id":\s*"tool_[a-z0-9]+"/g, '"id":"tool_STABLE"')
    .replace(/"id":\s*"ccr_tool_[a-z0-9]+"/g, '"id":"ccr_tool_STABLE"')
    .replace(/ccr_\d+/g, "ccr_STABLE")
    .replace(/"signature":\s*"ccr_STABLE"/g, '"signature":"ccr_STABLE"');
  return JSON.parse(stabilized);
}

async function runTests(isUpdate: boolean) {
  const errors: string[] = [];
  let totalChecks = 0;
  let passedChecks = 0;

  // ---- Request Body Tests ----
  console.log("\n=== Request Body Tests ===\n");
  const requestCases = makeRequestCases();
  const requestResults: Record<string, any> = {};

  for (const [name, req] of Object.entries(requestCases)) {
    totalChecks++;
    try {
      const result = buildRequestBody(req);
      requestResults[name] = result;
      passedChecks++;
      console.log(`  PASS: ${name}`);
    } catch (err: any) {
      errors.push(`Request[${name}]: ${err.message}`);
      console.log(`  FAIL: ${name} - ${err.message}`);
    }
  }

  // ---- JSON Response Tests ----
  console.log("\n=== JSON Response Tests ===\n");
  const jsonResponseCases = makeJSONResponseCases();
  const jsonResponseResults: Record<string, any> = {};

  for (const [name, geminiResp] of Object.entries(jsonResponseCases)) {
    totalChecks++;
    try {
      const mockResp = makeJSONResponse(geminiResp);
      const transformed = await transformResponseOut(mockResp, "test-provider");
      const body = await transformed.json();
      jsonResponseResults[name] = body;
      passedChecks++;
      console.log(`  PASS: ${name}`);
    } catch (err: any) {
      errors.push(`JSON[${name}]: ${err.message}`);
      console.log(`  FAIL: ${name} - ${err.message}`);
    }
  }

  // ---- Stream Response Tests ----
  console.log("\n=== Stream Response Tests ===\n");
  const streamCases = makeStreamCases();
  const streamResults: Record<string, any[]> = {};

  for (const [name, chunks] of Object.entries(streamCases)) {
    totalChecks++;
    try {
      const mockResp = makeStreamResponse(chunks);
      const transformed = await transformResponseOut(mockResp, "test-provider");
      const collected = await collectStreamChunks(transformed);
      streamResults[name] = collected;
      passedChecks++;
      console.log(`  PASS: ${name} (${collected.length} chunks)`);
    } catch (err: any) {
      errors.push(`Stream[${name}]: ${err.message}`);
      console.log(`  FAIL: ${name} - ${err.message}`);
    }
  }

  // ---- Save or Compare ----
  ensureGoldenDir();

  const goldenFiles = {
    "gemini-request.json": requestResults,
    "gemini-response-json.json": jsonResponseResults,
    "gemini-response-stream.json": streamResults,
  };

  for (const [filename, results] of Object.entries(goldenFiles)) {
    const filepath = join(GOLDEN_DIR, filename);

    if (isUpdate) {
      writeFileSync(filepath, stableStringify(results) + "\n", "utf-8");
      console.log(`\n  Updated golden: ${filename}`);
    } else {
      if (!existsSync(filepath)) {
        errors.push(`Golden file missing: ${filename}. Run with --update first.`);
        console.log(`\n  MISSING: ${filename}`);
        continue;
      }

      const golden = JSON.parse(readFileSync(filepath, "utf-8"));
      const actualStable = makeDeterministic(results);
      const goldenStable = makeDeterministic(golden);

      const diffs = deepDiff(actualStable, goldenStable);
      if (diffs.length > 0) {
        errors.push(`Golden mismatch in ${filename}:`);
        diffs.slice(0, 20).forEach((d) => errors.push(`  ${d}`));
        if (diffs.length > 20) errors.push(`  ... and ${diffs.length - 20} more`);
        console.log(`\n  MISMATCH: ${filename} (${diffs.length} diffs)`);
        // Show first few diffs
        diffs.slice(0, 5).forEach((d) => console.log(`    ${d}`));
      } else {
        console.log(`\n  MATCH: ${filename}`);
      }
    }
  }

  // ---- Summary ----
  console.log("\n" + "=".repeat(50));
  console.log(`Results: ${passedChecks}/${totalChecks} tests passed`);
  if (errors.length > 0) {
    console.log(`\nErrors:`);
    errors.forEach((e) => console.log(`  - ${e}`));
    process.exit(1);
  } else {
    console.log("\nAll parity checks passed!");
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const isUpdate = process.argv.includes("--update");
runTests(isUpdate).catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});

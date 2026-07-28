import assert from "node:assert/strict";
import type {
  UnifiedChatRequest,
  UnifiedMessage,
} from "../types/llm";
import {
  createCursorCompatibilityStamp,
  createCursorTranscriptCommit,
  fingerprintCursorTurn,
  isStrictCursorTranscriptExtension,
} from "../cursor-sdk/turn-identity";

function baseRequest(): UnifiedChatRequest {
  return {
    model: "composer-2.5",
    stream: true,
    max_tokens: 1024,
    temperature: 0.2,
    system: [
      {
        type: "text",
        text: "Use the host tools.",
        cache_control: { type: "ephemeral" },
      },
    ],
    messages: [
      {
        role: "user",
        content: [{ type: "text", text: "Inspect the project." }],
        cache_control: { type: "ephemeral" },
      },
    ],
    tools: [
      {
        type: "function",
        function: {
          name: "Read",
          description: "Read a host file",
          parameters: {
            type: "object",
            properties: {
              path: { type: "string" },
            },
            required: ["path"],
          },
        },
        cache_control: { type: "ephemeral" },
      },
      {
        type: "function",
        function: {
          name: "Bash",
          description: "Run a host command",
          parameters: {
            type: "object",
            properties: {
              command: { type: "string" },
            },
            required: ["command"],
          },
        },
      },
    ],
  };
}

const first = baseRequest() as UnifiedChatRequest & Record<string, unknown>;
first.request_id = "transport-a";
first.metadata = { user_id: "transport-user-a" };

const equivalent = baseRequest() as UnifiedChatRequest &
  Record<string, unknown>;
equivalent.stream = false;
equivalent.stream_options = { include_usage: true };
equivalent.request_id = "transport-b";
equivalent.metadata = { user_id: "transport-user-b" };
equivalent.system = "Use the host tools.";
equivalent.messages = [
  {
    cache_control: { ttl: "1h", type: "ephemeral" },
    content: "Inspect the project.",
    role: "user",
  },
];
equivalent.tools = [...(equivalent.tools || [])].reverse();

assert.equal(
  fingerprintCursorTurn(first),
  fingerprintCursorTurn(equivalent),
  "transport fields, cache hints, equivalent text forms, and tool order must not change identity"
);
assert.equal(fingerprintCursorTurn(first).length, 64);

const changedTurn = baseRequest();
changedTurn.messages[0] = {
  role: "user",
  content: "Inspect a different project.",
};
assert.notEqual(fingerprintCursorTurn(first), fingerprintCursorTurn(changedTurn));
assert.notEqual(
  fingerprintCursorTurn(first, {
    turnIntent: {
      source: "anthropic",
      trailingToolResults: [
        { toolCallId: "tool-1", content: "denied", isError: false },
      ],
      interruption: "none",
      steering: "none",
    },
  }),
  fingerprintCursorTurn(first, {
    turnIntent: {
      source: "anthropic",
      trailingToolResults: [
        { toolCallId: "tool-1", content: "denied", isError: true },
      ],
      interruption: "none",
      steering: "none",
    },
  }),
  "tool-result error provenance is part of logical turn identity"
);
assert.notEqual(
  fingerprintCursorTurn(first, {
    turnIntent: {
      source: "anthropic",
      trailingToolResults: [
        { toolCallId: "tool-1", content: "denied", isError: true },
      ],
      interruption: "synthetic_client_interrupt",
      steering: "none",
    },
  }),
  fingerprintCursorTurn(first, {
    turnIntent: {
      source: "anthropic",
      trailingToolResults: [
        { toolCallId: "tool-1", content: "denied", isError: true },
      ],
      interruption: "none",
      steering: "meaningful",
    },
  }),
  "lifecycle-relevant interruption and steering intent must change identity"
);

const compatibilityA = createCursorCompatibilityStamp({
  model: {
    params: [
      { value: "high", id: "reasoning_effort" },
    ],
    id: "composer-2.5",
  },
  mode: "bridge",
  workspaceDir: "/tmp/cursor-workspace",
  guidanceFingerprint: "guidance-a",
  credentialFingerprint: "account-a",
});
const compatibilityEquivalent = createCursorCompatibilityStamp({
  credentialFingerprint: "account-a",
  guidanceFingerprint: "guidance-a",
  workspaceDir: "/tmp/cursor-workspace",
  mode: "bridge",
  model: {
    id: "composer-2.5",
    params: [
      { id: "reasoning_effort", value: "high" },
    ],
  },
  sandboxEnabled: false,
});
assert.equal(compatibilityA, compatibilityEquivalent);

for (const incompatible of [
  createCursorCompatibilityStamp({
    model: { id: "composer-2.5" },
    mode: "agent",
    workspaceDir: "/tmp/cursor-workspace",
    guidanceFingerprint: "guidance-a",
    credentialFingerprint: "account-a",
  }),
  createCursorCompatibilityStamp({
    model: { id: "composer-2.5" },
    mode: "bridge",
    workspaceDir: "/tmp/another-workspace",
    guidanceFingerprint: "guidance-a",
    credentialFingerprint: "account-a",
  }),
  createCursorCompatibilityStamp({
    model: { id: "composer-2.5" },
    mode: "bridge",
    workspaceDir: "/tmp/cursor-workspace",
    guidanceFingerprint: "guidance-b",
    credentialFingerprint: "account-a",
  }),
  createCursorCompatibilityStamp({
    model: {
      params: [{ value: "high", id: "reasoning_effort" }],
      id: "composer-2.5",
    },
    mode: "bridge",
    workspaceDir: "/tmp/cursor-workspace",
    guidanceFingerprint: "guidance-a",
    credentialFingerprint: "account-a",
    tools: baseRequest().tools?.slice(0, 1),
  }),
]) {
  assert.notEqual(compatibilityA, incompatible);
}

const assistant: UnifiedMessage = {
  role: "assistant",
  content: "I need to inspect package.json.",
  thinking: {
    content: "Choose the smallest useful read.",
    signature: "opaque-signature-a",
  },
  tool_calls: [
    {
      id: "tool-read",
      type: "function",
      function: {
        name: "Read",
        arguments: '{"path":"package.json","limit":200}',
      },
    },
  ],
};
const commit = createCursorTranscriptCommit(baseRequest(), assistant);
assert.equal(commit.messageCount, 2);
assert.equal(commit.transcriptHash.length, 64);
assert.equal(Object.isFrozen(commit), true);

const aligned = baseRequest();
aligned.system = "Use the host tools.";
aligned.messages.push({
  ...assistant,
  cache_control: { type: "ephemeral" },
  thinking: {
    content: "Choose the smallest useful read.",
    signature: "different-opaque-signature",
  },
  tool_calls: [
    {
      id: "tool-read",
      type: "function",
      function: {
        name: "Read",
        arguments: '{ "limit": 200, "path": "package.json" }',
      },
    },
  ],
});
aligned.messages.push({
  role: "tool",
  tool_call_id: "tool-read",
  content: '{"name":"claude-code-router"}',
});
assert.equal(isStrictCursorTranscriptExtension(commit, aligned), true);

const equalButNotExtended = baseRequest();
equalButNotExtended.messages.push(assistant);
assert.equal(
  isStrictCursorTranscriptExtension(commit, equalButNotExtended),
  false,
  "an identical prefix without a new message is not a new turn"
);

const divergent = structuredClone(aligned);
divergent.messages[1] = {
  ...divergent.messages[1],
  tool_calls: [
    {
      id: "different-tool-id",
      type: "function",
      function: {
        name: "Read",
        arguments: '{"path":"package.json","limit":200}',
      },
    },
  ],
};
assert.equal(isStrictCursorTranscriptExtension(commit, divergent), false);

assert.throws(
  () =>
    createCursorTranscriptCommit(baseRequest(), {
      role: "user",
      content: "not an assistant message",
    }),
  TypeError
);

console.log("cursor-sdk.turn-identity: ok");

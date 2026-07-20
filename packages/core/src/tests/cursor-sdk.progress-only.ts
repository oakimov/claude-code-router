import assert from "node:assert/strict";
import {
  buildBridgeSystemGuidance,
  isProgressOnlyAssistantText,
  progressOnlyContinuationPrompt,
  shouldContinueProgressOnlyTurn,
} from "../cursor-sdk/prompt";

const observedFailure =
  "Checking the commit that moved project metadata out of the workspace.";

assert.equal(isProgressOnlyAssistantText(observedFailure), true);
assert.equal(isProgressOnlyAssistantText("Inspecting how the path is constructed."), true);
assert.equal(isProgressOnlyAssistantText("Let me check the latest commit."), true);
assert.equal(isProgressOnlyAssistantText("Looking into the failing test."), true);

// Avoid retrying legitimate short answers or messages which already continued
// past the progress preamble.
assert.equal(isProgressOnlyAssistantText("Checking is disabled by default."), false);
assert.equal(isProgressOnlyAssistantText("Looking good."), false);
assert.equal(
  isProgressOnlyAssistantText(
    "Checking the commit.\n\nThe change was introduced in f64ff71."
  ),
  false
);
assert.equal(
  isProgressOnlyAssistantText(
    "The change was introduced in f64ff71 and stores metadata outside the workspace."
  ),
  false
);

const baseDecision = {
  mode: "bridge" as const,
  assistantText: observedFailure,
  emittedHostTools: 0,
  continuationAttempts: 0,
};
assert.equal(shouldContinueProgressOnlyTurn(baseDecision), true);
assert.equal(
  shouldContinueProgressOnlyTurn({ ...baseDecision, mode: "plan" }),
  false
);
assert.equal(
  shouldContinueProgressOnlyTurn({ ...baseDecision, emittedHostTools: 1 }),
  false
);
assert.equal(
  shouldContinueProgressOnlyTurn({ ...baseDecision, continuationAttempts: 1 }),
  false
);

const guidance = buildBridgeSystemGuidance(
  { messages: [], tools: [] } as any,
  "/tmp/cursor-bridge"
);
assert.match(guidance, /Never end a turn with progress narration alone\./);
assert.match(guidance, /provide the complete user-facing answer before finishing/);

const continuation = progressOnlyContinuationPrompt();
assert.match(continuation.text, /Do not repeat the progress update\./);
assert.match(continuation.text, /call an available host tool immediately/);


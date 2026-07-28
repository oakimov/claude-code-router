import assert from "node:assert/strict";
import {
  matchParkedToolResultsExactly,
  planCursorLifecycle,
  type CursorLifecycleInput,
  type CursorLifecyclePlan,
} from "../cursor-sdk/lifecycle-planner";

type Parked = {
  id: string;
  name: string;
};

type Result = {
  toolCallId: string;
  content: string;
};

const parkedBuild: Parked = { id: "tool-build", name: "Bash" };
const parkedTest: Parked = { id: "tool-test", name: "Bash" };
const buildResult: Result = {
  toolCallId: "tool-build",
  content: "build ok",
};
const testResult: Result = {
  toolCallId: "tool-test",
  content: "tests ok",
};

const frozenParked = Object.freeze([
  Object.freeze(parkedBuild),
  Object.freeze(parkedTest),
]);
const frozenResults = Object.freeze([
  Object.freeze(testResult),
  Object.freeze(buildResult),
]);
const parkedBefore = JSON.stringify(frozenParked);
const resultsBefore = JSON.stringify(frozenResults);
const exact = matchParkedToolResultsExactly(frozenParked, frozenResults);

assert.deepEqual(
  exact?.map(({ parked, result }) => [parked.id, result.toolCallId]),
  [
    ["tool-build", "tool-build"],
    ["tool-test", "tool-test"],
  ]
);
assert.equal(JSON.stringify(frozenParked), parkedBefore);
assert.equal(JSON.stringify(frozenResults), resultsBefore);
assert.equal(exact?.[0]?.parked, parkedBuild);
assert.equal(exact?.[0]?.result, buildResult);

const mismatchCases: Array<{
  name: string;
  parked: readonly Parked[];
  results: readonly Result[];
}> = [
  { name: "both empty", parked: [], results: [] },
  { name: "missing result", parked: frozenParked, results: [buildResult] },
  {
    name: "extra result",
    parked: [parkedBuild],
    results: [buildResult, testResult],
  },
  {
    name: "wrong id",
    parked: [parkedBuild],
    results: [{ toolCallId: "tool-other", content: "other" }],
  },
  {
    name: "duplicate parked id",
    parked: [parkedBuild, { ...parkedBuild }],
    results: [buildResult, testResult],
  },
  {
    name: "duplicate result id",
    parked: frozenParked,
    results: [buildResult, { ...buildResult }],
  },
  {
    name: "blank parked id",
    parked: [{ id: "", name: "Bash" }],
    results: [{ toolCallId: "", content: "ambiguous" }],
  },
  {
    name: "blank result id",
    parked: [parkedBuild],
    results: [{ toolCallId: "", content: "ambiguous" }],
  },
];

for (const testCase of mismatchCases) {
  assert.equal(
    matchParkedToolResultsExactly(testCase.parked, testCase.results),
    undefined,
    testCase.name
  );
}

const baseInput: CursorLifecycleInput<Parked, Result> = {
  session: {
    hasSentPrompt: true,
    poisoned: false,
    alignment: "strict",
    run: { kind: "idle" },
  },
  turn: {
    hasMeaningfulSteering: false,
    toolResults: [],
  },
};

const lifecycleCases: Array<{
  name: string;
  input: CursorLifecycleInput<Parked, Result>;
  action: CursorLifecyclePlan["action"];
  reason: CursorLifecyclePlan["reason"];
}> = [
  {
    name: "unused idle session receives a full prompt",
    input: {
      ...baseInput,
      session: { ...baseInput.session, hasSentPrompt: false },
    },
    action: "send-full",
    reason: "unused-session",
  },
  {
    name: "strictly aligned idle session receives an incremental prompt",
    input: baseInput,
    action: "send-incremental",
    reason: "strictly-aligned-idle-session",
  },
  {
    name: "strictly aligned idle user steering remains incremental",
    input: {
      ...baseInput,
      turn: { ...baseInput.turn, hasMeaningfulSteering: true },
    },
    action: "send-incremental",
    reason: "strictly-aligned-idle-session",
  },
  {
    name: "live parked run resumes for all exact results",
    input: {
      ...baseInput,
      session: {
        ...baseInput.session,
        run: { kind: "parked", live: true, tools: frozenParked },
      },
      turn: {
        hasMeaningfulSteering: false,
        toolResults: frozenResults,
      },
    },
    action: "resume-parked",
    reason: "exact-parked-tool-results",
  },
  {
    name: "parked result plus steering retires and replays",
    input: {
      ...baseInput,
      session: {
        ...baseInput.session,
        run: { kind: "parked", live: true, tools: [parkedBuild] },
      },
      turn: {
        hasMeaningfulSteering: true,
        toolResults: [buildResult],
      },
    },
    action: "retire-and-replay-full",
    reason: "parked-turn-has-steering",
  },
  {
    name: "missing parked result retires and replays",
    input: {
      ...baseInput,
      session: {
        ...baseInput.session,
        run: { kind: "parked", live: true, tools: frozenParked },
      },
      turn: {
        hasMeaningfulSteering: false,
        toolResults: [buildResult],
      },
    },
    action: "retire-and-replay-full",
    reason: "parked-tool-results-mismatch",
  },
  {
    name: "extra parked result retires and replays",
    input: {
      ...baseInput,
      session: {
        ...baseInput.session,
        run: { kind: "parked", live: true, tools: [parkedBuild] },
      },
      turn: {
        hasMeaningfulSteering: false,
        toolResults: [buildResult, testResult],
      },
    },
    action: "retire-and-replay-full",
    reason: "parked-tool-results-mismatch",
  },
  {
    name: "dead parked run retires even with exact results",
    input: {
      ...baseInput,
      session: {
        ...baseInput.session,
        run: { kind: "parked", live: false, tools: [parkedBuild] },
      },
      turn: {
        hasMeaningfulSteering: false,
        toolResults: [buildResult],
      },
    },
    action: "retire-and-replay-full",
    reason: "dead-parked-run",
  },
  {
    name: "different turn while active retires and replays",
    input: {
      ...baseInput,
      session: {
        ...baseInput.session,
        run: { kind: "active-different-turn" },
      },
    },
    action: "retire-and-replay-full",
    reason: "active-different-turn",
  },
  {
    name: "poisoned session retires and replays",
    input: {
      ...baseInput,
      session: { ...baseInput.session, poisoned: true },
    },
    action: "retire-and-replay-full",
    reason: "poisoned-session",
  },
  {
    name: "unknown context alignment retires and replays",
    input: {
      ...baseInput,
      session: { ...baseInput.session, alignment: "unknown" },
    },
    action: "retire-and-replay-full",
    reason: "unknown-context-alignment",
  },
  {
    name: "divergent context alignment retires and replays",
    input: {
      ...baseInput,
      session: { ...baseInput.session, alignment: "divergent" },
    },
    action: "retire-and-replay-full",
    reason: "divergent-context-alignment",
  },
  {
    name: "tool result without a parked run retires and replays",
    input: {
      ...baseInput,
      turn: {
        hasMeaningfulSteering: false,
        toolResults: [buildResult],
      },
    },
    action: "retire-and-replay-full",
    reason: "orphaned-tool-results",
  },
  {
    name: "unused session with a live run is retired as inconsistent",
    input: {
      ...baseInput,
      session: {
        ...baseInput.session,
        hasSentPrompt: false,
        run: { kind: "active-different-turn" },
      },
    },
    action: "retire-and-replay-full",
    reason: "inconsistent-unused-session",
  },
];

for (const testCase of lifecycleCases) {
  const actual = planCursorLifecycle(testCase.input);
  assert.equal(actual.action, testCase.action, `${testCase.name}: action`);
  assert.equal(actual.reason, testCase.reason, `${testCase.name}: reason`);
}

const resumePlan = planCursorLifecycle({
  ...baseInput,
  session: {
    ...baseInput.session,
    run: { kind: "parked", live: true, tools: frozenParked },
  },
  turn: {
    hasMeaningfulSteering: false,
    toolResults: frozenResults,
  },
});
assert.equal(resumePlan.action, "resume-parked");
if (resumePlan.action === "resume-parked") {
  assert.deepEqual(
    resumePlan.matches.map(({ parked, result }) => [
      parked.id,
      result.toolCallId,
    ]),
    [
      ["tool-build", "tool-build"],
      ["tool-test", "tool-test"],
    ]
  );
}

console.log("cursor-sdk.lifecycle-planner: ok");

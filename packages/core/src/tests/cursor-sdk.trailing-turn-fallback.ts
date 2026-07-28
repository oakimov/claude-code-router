/**
 * `analyzeTrailingCursorToolTurn` has two paths: the structured one driven by
 * `turnIntent`, and a fallback that re-derives intent from already-flattened
 * Unified messages. Claude Code's interruption marker is protocol metadata, so
 * both paths must agree it is not steering — otherwise the fallback degrades a
 * resume-parked plan into a full retire-and-replay.
 */
import assert from "node:assert/strict";
import { analyzeTrailingCursorToolTurn } from "../cursor-sdk/prompt";
import { ANTHROPIC_SYNTHETIC_INTERRUPT_MARKERS } from "../types/turn-intent";

const toolMessage = {
  role: "tool" as const,
  tool_call_id: "tool-build",
  content: "build ok",
};

function requestWithTrailingUser(content: any) {
  return {
    messages: [
      { role: "user", content: "run the build" },
      {
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: "tool-build",
            type: "function",
            function: { name: "Bash", arguments: "{}" },
          },
        ],
      },
      toolMessage,
      { role: "user", content },
    ],
  } as any;
}

async function main() {
  for (const marker of ANTHROPIC_SYNTHETIC_INTERRUPT_MARKERS) {
    // String content and block content both reach the fallback path.
    for (const content of [marker, [{ type: "text", text: `  ${marker}\n` }]]) {
      const turn = analyzeTrailingCursorToolTurn(
        requestWithTrailingUser(content)
      );
      assert.equal(
        turn.hasTrailingUserInput,
        false,
        `marker must not count as steering: ${JSON.stringify(content)}`
      );
      assert.deepEqual(turn.toolResults, [
        { toolCallId: "tool-build", content: "build ok" },
      ]);
    }
  }

  // Real steering still registers, including alongside the marker.
  const steering = analyzeTrailingCursorToolTurn(
    requestWithTrailingUser("stop, do the tests instead")
  );
  assert.equal(steering.hasTrailingUserInput, true);

  const markerPlusSteering = analyzeTrailingCursorToolTurn(
    requestWithTrailingUser([
      { type: "text", text: ANTHROPIC_SYNTHETIC_INTERRUPT_MARKERS[0] },
      { type: "text", text: "do the tests instead" },
    ])
  );
  assert.equal(markerPlusSteering.hasTrailingUserInput, true);

  // Marker text embedded in a larger sentence is ordinary user input.
  const embedded = analyzeTrailingCursorToolTurn(
    requestWithTrailingUser(
      `why did you print ${ANTHROPIC_SYNTHETIC_INTERRUPT_MARKERS[0]} there?`
    )
  );
  assert.equal(embedded.hasTrailingUserInput, true);

  // The structured path is unchanged and remains the source of truth.
  const structured = analyzeTrailingCursorToolTurn(
    requestWithTrailingUser(ANTHROPIC_SYNTHETIC_INTERRUPT_MARKERS[0]),
    {
      source: "anthropic",
      trailingToolResults: [
        { toolCallId: "tool-build", content: "build ok", isError: false },
      ],
      interruption: "synthetic_client_interrupt",
      steering: "none",
    }
  );
  assert.equal(structured.hasTrailingUserInput, false);
  assert.deepEqual(structured.toolResults, [
    { toolCallId: "tool-build", content: "build ok", isError: false },
  ]);

  console.log("cursor-sdk.trailing-turn-fallback: ok");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

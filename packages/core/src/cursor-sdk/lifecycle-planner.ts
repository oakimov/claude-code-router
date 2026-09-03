export type CursorContextAlignment = "strict" | "unknown" | "divergent";

export type CursorParkedToolRef = {
  id: string;
};

export type CursorToolResultRef = {
  toolCallId: string;
};

export type ExactParkedResultMatch<
  TParked extends CursorParkedToolRef = CursorParkedToolRef,
  TResult extends CursorToolResultRef = CursorToolResultRef,
> = ReadonlyArray<{
  parked: TParked;
  result: TResult;
}>;

/**
 * Pair every parked tool with exactly one result by id.
 *
 * Empty, missing, extra, blank, or duplicate ids are not exact matches. The
 * returned pairs follow parked-tool order, and neither input array is mutated.
 */
export function matchParkedToolResultsExactly<
  TParked extends CursorParkedToolRef,
  TResult extends CursorToolResultRef,
>(
  parkedTools: readonly TParked[],
  toolResults: readonly TResult[]
): ExactParkedResultMatch<TParked, TResult> | undefined {
  if (
    parkedTools.length === 0 ||
    parkedTools.length !== toolResults.length
  ) {
    return undefined;
  }

  const resultsById = new Map<string, TResult>();
  for (const result of toolResults) {
    const id = result.toolCallId;
    if (!id || resultsById.has(id)) return undefined;
    resultsById.set(id, result);
  }

  const seenParkedIds = new Set<string>();
  const matches: Array<{ parked: TParked; result: TResult }> = [];
  for (const parked of parkedTools) {
    if (!parked.id || seenParkedIds.has(parked.id)) return undefined;
    seenParkedIds.add(parked.id);

    const result = resultsById.get(parked.id);
    if (!result) return undefined;
    matches.push({ parked, result });
  }

  return matches;
}

export type CursorRunSnapshot<
  TParked extends CursorParkedToolRef = CursorParkedToolRef,
> =
  | { kind: "idle" }
  /**
   * Identical active turns must be coalesced before invoking this planner.
   * This state therefore always represents a different, superseding turn.
   */
  | { kind: "active-different-turn" }
  | {
      kind: "parked";
      /** False when the prior iterator/run can no longer be continued. */
      live: boolean;
      tools: readonly TParked[];
    };

export type CursorLifecycleInput<
  TParked extends CursorParkedToolRef = CursorParkedToolRef,
  TResult extends CursorToolResultRef = CursorToolResultRef,
> = {
  session: {
    hasSentPrompt: boolean;
    poisoned: boolean;
    alignment: CursorContextAlignment;
    /**
     * False when agent/model/workspace/tool configuration changed since the
     * transcript commit. Undefined preserves the legacy behavior (match).
     */
    compatibilityMatch?: boolean;
    run: CursorRunSnapshot<TParked>;
  };
  turn: {
    hasMeaningfulSteering: boolean;
    toolResults: readonly TResult[];
  };
};

/** Hard remint only — soft alignment cases stay on the same agent. */
export type CursorLifecycleRetirementReason = "poisoned-session";

export type CursorLifecycleIncrementalReason =
  | "strictly-aligned-idle-session"
  | "unknown-context-alignment"
  | "divergent-context-alignment"
  | "active-different-turn"
  | "dead-parked-run"
  | "parked-turn-has-steering"
  | "parked-tool-results-mismatch"
  | "orphaned-tool-results";

export type CursorLifecycleRetirementPlan = {
  action: "retire-and-replay-full";
  reason: CursorLifecycleRetirementReason;
};

export type CursorLifecyclePlan<
  TParked extends CursorParkedToolRef = CursorParkedToolRef,
  TResult extends CursorToolResultRef = CursorToolResultRef,
> =
  | {
      action: "resume-parked";
      reason: "exact-parked-tool-results";
      matches: ExactParkedResultMatch<TParked, TResult>;
    }
  | {
      action: "send-full";
      reason: "unused-session" | "inconsistent-unused-session";
    }
  | {
      action: "send-incremental";
      reason: CursorLifecycleIncrementalReason;
    }
  | CursorLifecycleRetirementPlan;

const retireAndReplay = (
  reason: CursorLifecycleRetirementReason
): CursorLifecycleRetirementPlan => ({
  action: "retire-and-replay-full",
  reason,
});

const sendIncremental = (
  reason: CursorLifecycleIncrementalReason
): {
  action: "send-incremental";
  reason: CursorLifecycleIncrementalReason;
} => ({
  action: "send-incremental",
  reason,
});

/**
 * Soft incremental reasons where the runner must cancel/abandon any prior
 * active or parked run before `agent.send` — without reminting the agent.
 */
export const CURSOR_SOFT_CANCEL_THEN_INCREMENTAL = new Set<
  CursorLifecycleIncrementalReason
>([
  "active-different-turn",
  "dead-parked-run",
  "parked-turn-has-steering",
  "parked-tool-results-mismatch",
]);

/**
 * Choose the lifecycle action for a new, non-coalesced host turn.
 *
 * Soft transcript-alignment mismatches stay on the same Cursor agent via
 * `send-incremental` (caller cancels parked/active runs when needed). Hard
 * remint (`retire-and-replay-full`) is reserved for poisoned sessions; send/
 * stream failure recovery remints in the runner, not here.
 *
 * This function is intentionally pure. It neither resolves parked tools nor
 * changes session state; the caller executes the returned plan under its
 * per-session coordinator.
 */
export function planCursorLifecycle<
  TParked extends CursorParkedToolRef,
  TResult extends CursorToolResultRef,
>(
  input: CursorLifecycleInput<TParked, TResult>
): CursorLifecyclePlan<TParked, TResult> {
  const { session, turn } = input;

  if (session.poisoned) {
    return retireAndReplay("poisoned-session");
  }

  if (!session.hasSentPrompt) {
    // Unused session with a stray run: clear in the runner, then first send.
    // Do not remint — there is no agent cache to preserve yet.
    return session.run.kind === "idle"
      ? { action: "send-full", reason: "unused-session" }
      : { action: "send-full", reason: "inconsistent-unused-session" };
  }

  // Full-history replay clients (Claude Code) re-send the whole transcript,
  // so transcript-hash alignment is almost always "divergent" even when the
  // new tool results exactly match the parked tools. Prioritize the parked
  // ID match — the IDs are the correlation contract — so these turns resume
  // instead of cancelling into a cache-busting incremental send. Config
  // changes since the commit still force the safe incremental path.
  if (
    session.run.kind === "parked" &&
    session.run.live &&
    session.compatibilityMatch !== false
  ) {
    if (!turn.hasMeaningfulSteering) {
      const matches = matchParkedToolResultsExactly(
        session.run.tools,
        turn.toolResults
      );
      if (matches) {
        return {
          action: "resume-parked",
          reason: "exact-parked-tool-results",
          matches,
        };
      }
    }
  }

  if (session.alignment === "unknown") {
    return sendIncremental("unknown-context-alignment");
  }
  if (session.alignment === "divergent") {
    return sendIncremental("divergent-context-alignment");
  }

  if (session.run.kind === "active-different-turn") {
    return sendIncremental("active-different-turn");
  }

  if (session.run.kind === "parked") {
    if (!session.run.live) {
      return sendIncremental("dead-parked-run");
    }
    if (turn.hasMeaningfulSteering) {
      return sendIncremental("parked-turn-has-steering");
    }

    const matches = matchParkedToolResultsExactly(
      session.run.tools,
      turn.toolResults
    );
    return matches
      ? {
          action: "resume-parked",
          reason: "exact-parked-tool-results",
          matches,
        }
      : sendIncremental("parked-tool-results-mismatch");
  }

  if (turn.toolResults.length > 0) {
    return sendIncremental("orphaned-tool-results");
  }

  return sendIncremental("strictly-aligned-idle-session");
}

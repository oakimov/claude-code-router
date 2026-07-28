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
    run: CursorRunSnapshot<TParked>;
  };
  turn: {
    hasMeaningfulSteering: boolean;
    toolResults: readonly TResult[];
  };
};

export type CursorLifecycleRetirementReason =
  | "poisoned-session"
  | "inconsistent-unused-session"
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
      reason: "unused-session";
    }
  | {
      action: "send-incremental";
      reason: "strictly-aligned-idle-session";
    }
  | CursorLifecycleRetirementPlan;

const retireAndReplay = (
  reason: CursorLifecycleRetirementReason
): CursorLifecycleRetirementPlan => ({
  action: "retire-and-replay-full",
  reason,
});

/**
 * Choose the only safe lifecycle action for a new, non-coalesced host turn.
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
    return session.run.kind === "idle"
      ? { action: "send-full", reason: "unused-session" }
      : retireAndReplay("inconsistent-unused-session");
  }

  if (session.alignment === "unknown") {
    return retireAndReplay("unknown-context-alignment");
  }
  if (session.alignment === "divergent") {
    return retireAndReplay("divergent-context-alignment");
  }

  if (session.run.kind === "active-different-turn") {
    return retireAndReplay("active-different-turn");
  }

  if (session.run.kind === "parked") {
    if (!session.run.live) {
      return retireAndReplay("dead-parked-run");
    }
    if (turn.hasMeaningfulSteering) {
      return retireAndReplay("parked-turn-has-steering");
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
      : retireAndReplay("parked-tool-results-mismatch");
  }

  if (turn.toolResults.length > 0) {
    return retireAndReplay("orphaned-tool-results");
  }

  return {
    action: "send-incremental",
    reason: "strictly-aligned-idle-session",
  };
}

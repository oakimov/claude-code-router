const COMPLETED_TURN_TTL_MS = 60_000;
const SUPERSEDED_TURN_TTL_MS = 15 * 60_000;
const MAX_SUPERSEDED_FINGERPRINTS = 16;
const MAX_REPLAY_BYTES = 8 * 1024 * 1024;
const SUPERSEDE_WAIT_MS = 3_000;

type ResponseMetadata = {
  status: number;
  statusText: string;
  headers: Array<[string, string]>;
};

type Subscriber = {
  controller: ReadableStreamDefaultController<Uint8Array>;
  removeAbortListener?: () => void;
};

type TurnPhase =
  | "awaiting_response"
  | "streaming"
  | "completed"
  | "aborting"
  | "failed";

function abortError(reason?: unknown): Error {
  const message =
    reason instanceof Error
      ? reason.message
      : typeof reason === "string" && reason
        ? reason
        : "Cursor turn subscriber aborted";
  return Object.assign(new Error(message), { name: "AbortError" });
}

async function waitAtMost(promise: Promise<void>, timeoutMs: number): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      promise,
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

class SharedCursorTurn {
  readonly abortController = new AbortController();

  private phase: TurnPhase = "awaiting_response";
  private responseMetadata?: ResponseMetadata;
  private sourceReader?: ReadableStreamDefaultReader<Uint8Array>;
  private chunks: Uint8Array[] = [];
  private bufferedBytes = 0;
  private replayable = true;
  private readySettled = false;
  private completedAt?: number;
  private nextInterestId = 1;
  private nextSubscriberId = 1;
  private interests = new Set<number>();
  private subscribers = new Map<number, Subscriber>();
  private failure?: Error;
  private readyResolve!: () => void;
  private readyReject!: (error: Error) => void;
  private settledResolve!: () => void;

  private readonly ready = new Promise<void>((resolve, reject) => {
    this.readyResolve = resolve;
    this.readyReject = reject;
  });

  private readonly settled = new Promise<void>((resolve) => {
    this.settledResolve = resolve;
  });

  constructor(
    readonly sessionKey: string,
    readonly fingerprint: string,
    readonly responseKind: "stream" | "json",
    private readonly onTerminal: (
      turn: SharedCursorTurn,
      outcome: "completed" | "failed"
    ) => void
  ) {
    // A leader can fail before any joiner awaits readiness.
    void this.ready.catch(() => undefined);
  }

  get canReplay(): boolean {
    return this.phase === "completed" && this.replayable;
  }

  get replayAgeMs(): number {
    return this.completedAt === undefined
      ? Number.POSITIVE_INFINITY
      : Date.now() - this.completedAt;
  }

  get isJoinable(): boolean {
    return (
      this.replayable &&
      (this.phase === "awaiting_response" || this.phase === "streaming")
    );
  }

  get isProducing(): boolean {
    return this.phase === "awaiting_response" || this.phase === "streaming";
  }

  reserve(
    signal: AbortSignal | undefined,
    kind: "leader" | "join" | "replay"
  ): CursorTurnLease {
    const interestId = this.nextInterestId++;
    this.interests.add(interestId);
    return new CursorTurnLease(this, interestId, signal, kind);
  }

  releaseInterest(interestId: number): Promise<void> | undefined {
    if (!this.interests.delete(interestId)) return undefined;
    return this.abortIfUnobserved("Cursor turn has no remaining subscribers");
  }

  async attach(
    response: Response,
    leaderInterestId: number,
    leaderSignal?: AbortSignal
  ): Promise<Response | undefined> {
    if (
      this.phase === "aborting" ||
      this.phase === "failed" ||
      this.abortController.signal.aborted
    ) {
      try {
        await response.body?.cancel(this.abortController.signal.reason);
      } catch {
        // The turn is already stale; cleanup is best effort here.
      }
      this.finishFailed(
        abortError(this.abortController.signal.reason ?? "Cursor turn superseded")
      );
      return undefined;
    }

    const headers: Array<[string, string]> = [];
    response.headers.forEach((value, key) => {
      headers.push([key, value]);
    });
    this.responseMetadata = {
      status: response.status,
      statusText: response.statusText,
      headers,
    };
    this.phase = "streaming";

    // Install the leader's subscriber before making the response ready to
    // joiners or pumping the source. Otherwise an oversized first chunk cannot
    // be replayed and could be lost before the original caller subscribes.
    const leaderResponse = this.interests.has(leaderInterestId)
      ? this.responseForReadyInterest(leaderInterestId, leaderSignal)
      : undefined;
    this.readySettled = true;
    this.readyResolve();

    if (!response.body) {
      this.finishCompleted();
      return leaderResponse;
    }

    this.sourceReader = response.body.getReader();
    if (leaderResponse) this.ensurePumpStarted();
    return leaderResponse;
  }

  fail(error: unknown): void {
    this.finishFailed(
      error instanceof Error ? error : new Error(String(error))
    );
  }

  async supersede(reason: string): Promise<void> {
    if (
      this.phase === "completed" ||
      this.phase === "failed" ||
      this.phase === "aborting"
    ) {
      await waitAtMost(this.settled, SUPERSEDE_WAIT_MS);
      return;
    }

    this.phase = "aborting";
    const error = abortError(reason);
    if (!this.readySettled) {
      this.readySettled = true;
      this.readyReject(error);
    }
    this.abortController.abort(error);
    this.errorSubscribers(error);

    const reader = this.sourceReader;
    if (reader) {
      const sourceCancel = Promise.resolve(reader.cancel(error))
        .catch(() => undefined)
        .then(() => undefined);
      await waitAtMost(sourceCancel, SUPERSEDE_WAIT_MS);
      this.finishFailed(error);
    }

    // Before attach(), the leader observes abortController.signal and calls
    // fail(). Bound this wait so one broken SDK send cannot block all re-entry.
    await waitAtMost(this.settled, SUPERSEDE_WAIT_MS);
  }

  async responseFor(
    interestId: number,
    signal?: AbortSignal
  ): Promise<Response> {
    await this.ready;
    if (!this.interests.has(interestId)) {
      throw abortError(signal?.reason);
    }
    if (!this.responseMetadata) {
      throw this.failure || new Error("Cursor turn response metadata is missing");
    }
    if (!this.replayable && this.phase !== "streaming") {
      this.releaseInterest(interestId);
      throw Object.assign(
        new Error("Cursor turn result exceeded the bounded replay buffer"),
        { statusCode: 409, code: "cursor_turn_not_replayable" }
      );
    }

    const response = this.responseForReadyInterest(interestId, signal);
    this.ensurePumpStarted();
    return response;
  }

  private responseForReadyInterest(
    interestId: number,
    signal?: AbortSignal
  ): Response {
    const stream = this.openStream(interestId, signal);
    return new Response(stream, {
      status: this.responseMetadata!.status,
      statusText: this.responseMetadata!.statusText,
      headers: this.responseMetadata!.headers,
    });
  }

  private openStream(
    interestId: number,
    signal?: AbortSignal
  ): ReadableStream<Uint8Array> {
    let subscriberId: number | undefined;
    return new ReadableStream<Uint8Array>({
      start: (controller) => {
        if (!this.interests.has(interestId)) {
          controller.error(abortError(signal?.reason));
          return;
        }

        subscriberId = this.nextSubscriberId++;
        const subscriber: Subscriber = { controller };
        this.subscribers.set(subscriberId, subscriber);

        // Transfer the pre-response reservation to the actual stream before
        // checking cancellation, so there is never a zero-observer gap.
        this.interests.delete(interestId);

        if (signal) {
          const onAbort = () => {
            if (subscriberId !== undefined) {
              void this.removeSubscriber(
                subscriberId,
                abortError(signal.reason)
              );
            }
          };
          if (signal.aborted) {
            onAbort();
            return;
          }
          signal.addEventListener("abort", onAbort, { once: true });
          subscriber.removeAbortListener = () =>
            signal.removeEventListener("abort", onAbort);
        }

        if (!this.replayable) {
          void this.removeSubscriber(
            subscriberId,
            Object.assign(
              new Error("Cursor turn cannot replay an oversized active result"),
              { statusCode: 409, code: "cursor_turn_not_replayable" }
            )
          );
          return;
        }

        for (const chunk of this.chunks) {
          controller.enqueue(chunk);
        }

        if (this.phase === "completed") {
          this.closeSubscriber(subscriberId);
        } else if (this.phase === "failed" || this.phase === "aborting") {
          void this.removeSubscriber(
            subscriberId,
            this.failure || abortError("Cursor turn ended before subscription")
          );
        }
      },
      cancel: (reason) => {
        if (subscriberId !== undefined) {
          return this.removeSubscriber(
            subscriberId,
            abortError(reason),
            false
          );
        } else {
          return this.releaseInterest(interestId);
        }
      },
    });
  }

  private pumpStarted = false;

  private ensurePumpStarted(): void {
    if (this.pumpStarted || !this.sourceReader || this.phase !== "streaming") {
      return;
    }
    this.pumpStarted = true;
    void this.pump();
  }

  private async pump(): Promise<void> {
    const reader = this.sourceReader;
    if (!reader) return;

    try {
      while (this.phase === "streaming") {
        const { done, value } = await reader.read();
        if (done) {
          this.finishCompleted();
          return;
        }
        if (!value) continue;

        if (this.replayable) {
          if (this.bufferedBytes + value.byteLength <= MAX_REPLAY_BYTES) {
            const retained = value.slice();
            this.chunks.push(retained);
            this.bufferedBytes += retained.byteLength;
          } else {
            this.replayable = false;
            this.chunks = [];
          }
        }

        for (const subscriber of this.subscribers.values()) {
          subscriber.controller.enqueue(value);
        }
      }
    } catch (error) {
      this.finishFailed(error instanceof Error ? error : new Error(String(error)));
    }
  }

  private abortIfUnobserved(reason: string): Promise<void> | undefined {
    if (
      this.interests.size > 0 ||
      this.subscribers.size > 0 ||
      this.phase === "completed" ||
      this.phase === "failed" ||
      this.phase === "aborting"
    ) {
      return undefined;
    }
    return this.supersede(reason);
  }

  private removeSubscriber(
    subscriberId: number,
    error: Error,
    surfaceError = true
  ): Promise<void> | undefined {
    const subscriber = this.subscribers.get(subscriberId);
    if (!subscriber) return undefined;
    this.subscribers.delete(subscriberId);
    subscriber.removeAbortListener?.();
    if (surfaceError) {
      try {
        subscriber.controller.error(error);
      } catch {
        // Reader may already be cancelled.
      }
    }
    return this.abortIfUnobserved(error.message);
  }

  private closeSubscriber(subscriberId: number): void {
    const subscriber = this.subscribers.get(subscriberId);
    if (!subscriber) return;
    this.subscribers.delete(subscriberId);
    subscriber.removeAbortListener?.();
    try {
      subscriber.controller.close();
    } catch {
      // Reader may already be closed.
    }
  }

  private errorSubscribers(error: Error): void {
    for (const subscriberId of [...this.subscribers.keys()]) {
      void this.removeSubscriber(subscriberId, error);
    }
  }

  private releaseSourceReader(): void {
    const reader = this.sourceReader;
    this.sourceReader = undefined;
    if (!reader) return;
    try {
      reader.releaseLock();
    } catch {
      // A cancellation/read may still be unwinding; dropping our reference is
      // sufficient and the stream owns the remaining cleanup.
    }
  }

  private finishCompleted(): void {
    if (this.phase === "completed" || this.phase === "failed") return;
    if (this.phase === "aborting") {
      this.finishFailed(abortError(this.abortController.signal.reason));
      return;
    }
    this.phase = "completed";
    this.completedAt = Date.now();
    this.releaseSourceReader();
    for (const subscriberId of [...this.subscribers.keys()]) {
      this.closeSubscriber(subscriberId);
    }
    this.settledResolve();
    this.onTerminal(this, "completed");
  }

  private finishFailed(error: Error): void {
    if (this.phase === "completed" || this.phase === "failed") return;
    this.phase = "failed";
    this.failure = error;
    this.releaseSourceReader();
    if (!this.readySettled) {
      this.readySettled = true;
      this.readyReject(error);
    }
    this.errorSubscribers(error);
    this.interests.clear();
    this.settledResolve();
    this.onTerminal(this, "failed");
  }
}

export class CursorTurnLease {
  private attachedResponse?: Response;
  private removeReservationAbortListener?: () => void;

  constructor(
    private readonly turn: SharedCursorTurn,
    private readonly interestId: number,
    private readonly signal?: AbortSignal,
    readonly kind: "leader" | "join" | "replay" = "join"
  ) {
    if (signal) {
      const onAbort = () => {
        void this.turn.releaseInterest(this.interestId);
      };
      if (signal.aborted) {
        onAbort();
      } else {
        signal.addEventListener("abort", onAbort, { once: true });
        this.removeReservationAbortListener = () =>
          signal.removeEventListener("abort", onAbort);
      }
    }
  }

  get producerSignal(): AbortSignal {
    return this.turn.abortController.signal;
  }

  attach(response: Response): Promise<void> {
    if (this.kind !== "leader") {
      throw new Error("Only the Cursor turn leader may attach a response");
    }
    return this.turn
      .attach(response, this.interestId, this.signal)
      .then((attachedResponse) => {
        this.attachedResponse = attachedResponse;
      });
  }

  fail(error: unknown): void {
    if (this.kind === "leader") this.turn.fail(error);
    this.removeReservationAbortListener?.();
    this.removeReservationAbortListener = undefined;
  }

  async response(): Promise<Response> {
    try {
      if (this.attachedResponse) {
        const response = this.attachedResponse;
        this.attachedResponse = undefined;
        return response;
      }
      return await this.turn.responseFor(this.interestId, this.signal);
    } finally {
      this.removeReservationAbortListener?.();
      this.removeReservationAbortListener = undefined;
    }
  }
}

type SessionTurns = {
  active?: SharedCursorTurn;
  completed?: SharedCursorTurn;
  superseded: Map<string, number>;
};

export class CursorTurnRegistry {
  private sessions = new Map<string, SessionTurns>();
  /** FIFO admission barrier; lifecycle supersession contains awaited cleanup. */
  private admissionGates = new Map<string, Promise<void>>();

  private forgetExpiredSuperseded(
    sessionKey: string,
    turns: SessionTurns,
    deleteEmpty = true
  ): void {
    const now = Date.now();
    for (const [fingerprint, expiresAt] of turns.superseded) {
      if (expiresAt <= now) turns.superseded.delete(fingerprint);
    }
    if (
      deleteEmpty &&
      !turns.active &&
      !turns.completed &&
      turns.superseded.size === 0
    ) {
      this.sessions.delete(sessionKey);
    }
  }

  private rememberSuperseded(
    sessionKey: string,
    turns: SessionTurns,
    fingerprint: string
  ): void {
    turns.superseded.delete(fingerprint);
    turns.superseded.set(
      fingerprint,
      Date.now() + SUPERSEDED_TURN_TTL_MS
    );
    while (turns.superseded.size > MAX_SUPERSEDED_FINGERPRINTS) {
      const oldest = turns.superseded.keys().next().value;
      if (typeof oldest !== "string") break;
      turns.superseded.delete(oldest);
    }

    const expiry = setTimeout(() => {
      const current = this.sessions.get(sessionKey);
      if (!current) return;
      this.forgetExpiredSuperseded(sessionKey, current);
    }, SUPERSEDED_TURN_TTL_MS);
    if (typeof expiry.unref === "function") expiry.unref();
  }

  async admit(input: {
    sessionKey: string;
    fingerprint: string;
    responseKind: "stream" | "json";
    signal?: AbortSignal;
  }): Promise<CursorTurnLease> {
    if (input.signal?.aborted) throw abortError(input.signal.reason);
    const previousAdmission = this.admissionGates.get(input.sessionKey);
    let releaseAdmission!: () => void;
    const admissionGate = new Promise<void>((resolve) => {
      releaseAdmission = resolve;
    });
    this.admissionGates.set(input.sessionKey, admissionGate);
    if (previousAdmission) await previousAdmission;

    try {
      if (input.signal?.aborted) throw abortError(input.signal.reason);
      while (true) {
        if (input.signal?.aborted) throw abortError(input.signal.reason);
        const turns = this.sessions.get(input.sessionKey) || {
          superseded: new Map<string, number>(),
        };
        this.sessions.set(input.sessionKey, turns);
        this.forgetExpiredSuperseded(input.sessionKey, turns, false);
        if (turns.superseded.has(input.fingerprint)) {
          throw Object.assign(
            new Error("This Cursor turn was superseded by a newer request"),
            { statusCode: 409, code: "cursor_turn_superseded" }
          );
        }

        const active = turns.active;
        if (active) {
          if (active.fingerprint === input.fingerprint) {
            if (active.responseKind !== input.responseKind) {
              throw Object.assign(
                new Error(
                  "An identical Cursor turn is already using a different response format"
                ),
                { statusCode: 409, code: "cursor_turn_response_format_conflict" }
              );
            }
            if (active.isJoinable) {
              return active.reserve(input.signal, "join");
            }
            if (active.isProducing) {
              throw Object.assign(
                new Error(
                  "The active Cursor turn exceeded the bounded duplicate replay buffer"
                ),
                { statusCode: 409, code: "cursor_turn_not_replayable" }
              );
            }
            // An identical retry arriving while every prior subscriber's abort
            // is still settling is a new generation, not stale work.
            await active.supersede(
              "Waiting for the aborted Cursor turn to finish cleanup"
            );
            if (turns.active === active) turns.active = undefined;
            continue;
          }
          this.rememberSuperseded(
            input.sessionKey,
            turns,
            active.fingerprint
          );
          await active.supersede("Cursor turn superseded by a newer request");
          if (turns.active === active) turns.active = undefined;
          continue;
        }

        const completed = turns.completed;
        if (
          completed &&
          completed.replayAgeMs <= COMPLETED_TURN_TTL_MS &&
          completed.fingerprint === input.fingerprint &&
          completed.canReplay
        ) {
          if (completed.responseKind !== input.responseKind) {
            throw Object.assign(
              new Error(
                "The cached Cursor turn uses a different response format"
              ),
              { statusCode: 409, code: "cursor_turn_response_format_conflict" }
            );
          }
          return completed.reserve(input.signal, "replay");
        }
        if (completed && completed.fingerprint !== input.fingerprint) {
          this.rememberSuperseded(
            input.sessionKey,
            turns,
            completed.fingerprint
          );
        }
        turns.completed = undefined;

        const turn = new SharedCursorTurn(
          input.sessionKey,
          input.fingerprint,
          input.responseKind,
          (terminalTurn, outcome) => {
            const current = this.sessions.get(input.sessionKey);
            if (!current) return;
            if (current.active === terminalTurn) current.active = undefined;
            if (outcome === "completed" && terminalTurn.canReplay) {
              current.completed = terminalTurn;
              const expiry = setTimeout(() => {
                const latest = this.sessions.get(input.sessionKey);
                if (latest?.completed !== terminalTurn) return;
                latest.completed = undefined;
                this.forgetExpiredSuperseded(input.sessionKey, latest);
              }, COMPLETED_TURN_TTL_MS);
              if (typeof expiry.unref === "function") expiry.unref();
            }
            if (
              !current.active &&
              !current.completed &&
              current.superseded.size === 0
            ) {
              this.sessions.delete(input.sessionKey);
            }
          }
        );
        turns.active = turn;
        turns.completed = undefined;
        return turn.reserve(input.signal, "leader");
      }
    } finally {
      releaseAdmission();
      if (this.admissionGates.get(input.sessionKey) === admissionGate) {
        this.admissionGates.delete(input.sessionKey);
      }
    }
  }

  peekActive(sessionKey: string): { fingerprint: string } | undefined {
    const active = this.sessions.get(sessionKey)?.active;
    if (!active) return undefined;
    return { fingerprint: active.fingerprint };
  }

  clear(): void {
    for (const turns of this.sessions.values()) {
      if (turns.active) {
        void turns.active.supersede("Cursor turn registry cleared");
      }
    }
    this.sessions.clear();
  }
}

export const globalCursorTurnRegistry = new CursorTurnRegistry();

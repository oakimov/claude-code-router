/**
 * Optional process vitals for the `/health` probe.
 *
 * The API routes are registered inside encapsulated Fastify plugins, which
 * snapshot the instance they inherit from at registration time — a decorator
 * added to the root afterwards is invisible to them. Whoever owns the process
 * (the server package, which builds its health heartbeat after the routes are
 * up) therefore publishes the reporter here instead.
 */
export type HealthReporter = () => unknown;

let reporter: HealthReporter | undefined;

export function setHealthReporter(fn: HealthReporter | undefined): void {
  reporter = fn;
}

/** Vitals for the current instant, or `undefined` when none are published. */
export function readHealthVitals(): unknown {
  if (!reporter) return undefined;
  try {
    return reporter();
  } catch {
    // A liveness probe must answer even when the reporter is broken.
    return undefined;
  }
}

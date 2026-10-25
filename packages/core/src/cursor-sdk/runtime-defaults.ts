import { Cursor } from "@cursor/sdk";

let configured = false;

/**
 * Apply Cursor SDK defaults once per process.
 *
 * Under Bun, local agent streams must use HTTP/1.1 + SSE — HTTP/2 via
 * `node:http2` / connect-node hits NGHTTP2_FRAME_SIZE_ERROR (Bun#31499).
 * Docs: Cursor.configure({ local: { useHttp1ForAgent: true } }). Bun already
 * defaults to HTTP/1.1; set it explicitly so Node transitional runs and tests
 * share the same path. Bun 1.4+ provides `node:sqlite`, so the default SQLite
 * local store works; JsonlLocalAgentStore is only needed when sqlite is absent.
 */
export function ensureCursorSdkRuntimeDefaults(): void {
  if (configured) return;
  configured = true;
  Cursor.configure({
    local: {
      useHttp1ForAgent: true,
    },
  });
}

/** Test-only: allow re-running configure after a reset. */
export function __resetCursorSdkRuntimeDefaultsForTests(): void {
  configured = false;
}

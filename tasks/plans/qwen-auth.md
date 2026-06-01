# Qwen Auth — Implementation Plan

## Goal

Add Qwen Chat authentication to CCR, parallel to the existing `codex-auth` flow:

1. A `ccr qwen-auth` CLI command that lets the user paste a JWT copied from
   `chat.qwen.ai` → `localStorage.getItem('token')`.
2. A new `QwenAuthTransformer` that loads/refreshes the stored JWT, sets the
   `Authorization: Bearer <jwt>` header, and (optionally) strips the trailing
   `<details>...</details>` metadata block Qwen injects into responses.
3. Server endpoints to manage the token (`GET /qwen/auth`, `POST /qwen/auth`,
   `GET /qwen/forget`, `GET /qwen/status`).
4. Public path whitelisting for `/qwen/*` in the auth middleware so the
   browser-based paste flow works without an `APIKEY`.

The OpenAI endpoint registration and request/response body conversion for
`/v1/chat/completions` is **already implemented** in the existing
`OpenAITransformer`. Qwen's new `QwenAuthTransformer` plugs into the chain
alongside it as a peer in `provider.transformer.use[]`.

---

## Architecture

### Qwen Auth Flow (reference: `qwen-proxy.mjs`)

```
┌──────────┐   localStorage.token   ┌──────────────────┐
│  user    │ ─────────────────────► │  ccr qwen-auth   │
│ (browser)│                        │  CLI / Web UI    │
└──────────┘                        └────────┬─────────┘
                                              │ POST /qwen/auth
                                              ▼
                                    ┌──────────────────┐
                                    │ CCR server route │
                                    │ validate() with  │  ──POST──► qwen.aikit.club/v1/validate
                                    │ upstream, save   │  ◄──access_token──
                                    │ qwen_auth.json   │
                                    └────────┬─────────┘
                                             │
                                             ▼
                              ~/.claude-code-router/qwen_auth.json
                              (mode 0600, { token, expiresAt, updatedAt })
```

### Request Flow (per LLM call)

The framework registers `POST /v1/chat/completions` from the `OpenAI`
transformer's `endPoint` property. Per the existing chain in
`packages/core/src/api/routes.ts`:

1. `transformRequestOut` runs on the endpoint transformer (`OpenAI`) if
   defined. The `OpenAI` stub has no body conversion, so the request passes
   through as-is.
2. `provider.transformer.use[]` runs in order. `QwenAuthTransformer`
   (registered first) injects the `Authorization` header and base URL.
3. `OpenAI` transformer does no body work — its only role is endpoint
   registration.
4. Response flows back through the same chain in reverse: provider
   transformers run their `transformResponseOut` in reverse order, with
   `QwenAuthTransformer` running last (i.e. applied first in the chain).
   `QwenAuthTransformer.transformResponseOut` strips the trailing
   `<details>...</details>` block.

```
Claude Code ──POST /v1/messages──► anthropic.transformer
                                       │ (request format conversion)
                                       ▼
                       ┌─ /v1/chat/completions endpoint ─┐
                       │  OpenAI.endPoint registers route │
                       └────────────────┬────────────────┘
                                        │
                                        ▼
                              QwenAuthTransformer.transformRequestIn
                                  • getValidAccessToken()
                                  • if expiring < 6h → refreshToken()
                                  • sets headers.Authorization = Bearer <jwt>
                                  • sets config.url = https://qwen.aikit.club
                                  (no body conversion — passes through)
                                       │
                                       ▼
                              qwen.aikit.club/v1/chat/completions
                                       │
                                       ▼
                              QwenAuthTransformer.transformResponseOut
                                  • strip <details>...</details> blocks
                                       │
                                       ▼
                              anthropic.transformer → Claude Code
```

### Config Example

```json
{
  "Providers": [
    {
      "name": "qwen",
      "api_base_url": "https://qwen.aikit.club/v1/chat/completions",
      "api_key": "qwen-placeholder",
      "models": ["qwen3-max", "qwen3-coder-plus"],
      "transformer": {
        "use": ["qwen-auth", "OpenAI"]
      }
    }
  ]
}
```

- `qwen-auth` is the new auth + response-strip transformer (this work).
- `OpenAI` is the existing transformer that registers `POST /v1/chat/completions`.
  The order matters: `qwen-auth` first (so it sets the Authorization header
  on the way out), `OpenAI` second (it just provides the endpoint; it has
  no body/response logic to disturb the chain).
- `api_key` stays a placeholder (per CLAUDE.md security rule) — actual
  auth comes from the JWT in `qwen_auth.json`.

---

## File-by-File Changes

### 1. `packages/core/src/utils/qwen-auth.ts` (NEW)

Mirror of `packages/core/src/utils/codex-auth.ts`, adapted to Qwen's two-step
validate/refresh flow.

- `QWEN_AUTH_FILE = ~/.claude-code-router/qwen_auth.json` (mode 0600)
- `QWEN_TARGET = "https://qwen.aikit.club"`
- `QwenTokens` interface: `{ token: string; expiresAt: number | null; updatedAt: number }`
  - Note: only one token (Qwen uses a single JWT, no separate refresh)
- `loadTokens()`, `saveTokens(tokens)` — file I/O
- `extractExpFromJwt(token)` — base64url decode `token.split('.')[1]`, return
  `payload.exp * 1000` or `null` (matches `qwen-proxy.mjs` lines 30–36)
- `isTokenExpired(tokens, leewaySeconds=120)` — `Date.now() + leeway >= expiresAt`
  (note: 120s leeway matches `qwen-proxy.mjs` line 14)
- `validateToken(token)` — POST `qwen.aikit.club/v1/validate` with
  `{ token }`, return `data.access_token || token`; 10s timeout
- `refreshToken(token)` — POST `qwen.aikit.club/v1/refresh` with `{ token }`,
  return `data.access_token || null`; 10s timeout
- `getValidAccessToken()` — load → if expired, call `refreshToken()` and
  re-save → if refresh fails, throw a clear error message
  (`"Run 'ccr qwen-auth' to re-authenticate"`)
- `clearTokens()` — delete `qwen_auth.json` (for `ccr qwen-forget`)

All comments in English (per CLAUDE.md rule).

### 2. `packages/core/src/transformer/qwen-auth.transformer.ts` (NEW)

Provider-level transformer (no `endPoint` property — runs in the chain, does
not register a route). Sits alongside `codex.transformer.ts` in style.

```ts
export class QwenAuthTransformer implements Transformer {
  name = "qwen-auth";
  logger?: any;
  // no endPoint — this is a request/response shim, not a route handler

  async transformRequestIn(request, provider, context): Promise<{ body, config }> {
    const tokens = await getValidAccessToken();  // refreshes if expiring < 6h
    const baseUrl = provider?.api_base_url || provider?.baseUrl || "https://qwen.aikit.club";
    return {
      body: request,
      config: {
        url: baseUrl,
        headers: {
          Authorization: `Bearer ${tokens.token}`,
        },
      },
    };
  }

  async auth(_request, provider): Promise<{ config }> {
    // passthrough-mode path: identical headers/url, no body change
    const tokens = await getValidAccessToken();
    const baseUrl = provider?.api_base_url || provider?.baseUrl || "https://qwen.aikit.club";
    return {
      config: {
        url: baseUrl,
        headers: { Authorization: `Bearer ${tokens.token}` },
      },
    };
  }

  async transformResponseOut(response): Promise<Response> {
    // strip trailing <details> metadata block from JSON bodies and SSE chunks
    const META_RE = /(?:\\n|\n)?<details>[\s\S]*?<\/details>\s*/g;
    const contentType = response.headers.get("Content-Type") || "";
    if (contentType.includes("text/event-stream")) {
      // tee the stream and strip matches per chunk
      // (reuses the createSSEStreamReader pattern from codex.transformer.ts)
    } else {
      const text = await response.text();
      const cleaned = text.replace(META_RE, "");
      return new Response(cleaned, { status: response.status, headers: response.headers });
    }
  }
}
```

The transformer **does not** alter the body. The existing `OpenAI`
transformer (the `endPoint` stub) handles endpoint registration only.

### 3. `packages/core/src/transformer/index.ts`

Add `QwenAuthTransformer` to the registry (mirror `CodexTransformer` import
and export lines 21, 48).

### 4. `packages/server/src/routes/qwen-auth.ts` (NEW)

Mirror of `codex-auth.ts`, with these routes:

- `GET /qwen/auth` → returns HTML page (use the same stylesheet/structure as
  `qwen-proxy.mjs` HTML() helper). Shows status: "Authenticated" if
  `loadTokens()` succeeds, "Not authenticated" otherwise. Includes a
  `<form method="post">` for pasting the JWT, and a "Forget token" link.
- `POST /qwen/auth` → reads body, calls `validateToken(raw)`, if good
  `saveTokens()` and return success HTML; if not, return 400 with the
  rejection HTML. **This route is the natural integration point** — user
  visits the page, sees the form, pastes the JWT, hits submit. No bookmarklet
  / localStorage dance needed.
- `GET /qwen/forget` → `clearTokens()` + return 200 HTML redirect back to
  `/qwen/auth`.
- `GET /qwen/status` → JSON `{ ok: boolean, expiresAt: number|null }` for
  programmatic checks.

The HTML form is the auth UX: no need to replicate the bookmarklet.

### 5. `packages/server/src/server.ts`

Add `import { registerQwenAuthRoutes }` and call `await registerQwenAuthRoutes(app)`
right after `registerCodexAuthRoutes(app)` (line 86).

### 6. `packages/server/src/middleware/auth.ts`

Update `publicPaths` so the browser flow works without `APIKEY`:

```ts
const publicPaths = ["/", "/health", "/auth/callback", "/qwen/auth", "/qwen/forget", "/qwen/status"];
if (publicPaths.includes(req.url) || req.url.startsWith("/ui") || req.url.startsWith("/auth") || req.url.startsWith("/qwen/")) {
  return done();
}
```

This is safe because the routes only **store/serve status of the local
token file** — they do not call upstream with the user's data, only the
local server.

### 7. `packages/cli/src/utils/qwen-cli-auth.ts` (NEW)

Mirror of `codex-cli-auth.ts`, but no OAuth handshake — just print the URL
for the auth page and wait for the user to confirm:

```ts
export async function runQwenAuth(): Promise<void> {
  const port = /* read from CCR config or default 3456 */;
  const url = `http://127.0.0.1:${port}/qwen/auth`;
  console.log(`Open this URL in your browser and paste your Qwen token:\n\n${url}\n`);
  console.log("To get a token: sign in at https://chat.qwen.ai, open dev tools (F12),");
  console.log("and run: copy(localStorage.getItem('token'))");
  console.log();
  console.log("After saving the token in the browser, press Enter here to verify...");
  // wait for Enter, then read qwen_auth.json and print expiry
}
```

The CLI does **not** call the server with an API key — it simply tells the
user where to go and reads the token file directly to confirm success
(matches the codex CLI pattern of reading `codex_auth.json` after Enter).

### 8. `packages/cli/src/cli.ts`

- Add `"qwen-auth"` to `KNOWN_COMMANDS` (around line 42)
- Add help text line: `qwen-auth   Authenticate with Qwen Chat (paste JWT)`
- Add `case "qwen-auth": await runQwenAuth(); break;` (around line 317)
- Import `runQwenAuth` at top of file

### 9. `README.md`

Add a new "Qwen Provider Authentication" section right after the existing
"Codex Provider Authentication" section (around line 392). Mirror the
codex-auth section's structure with these specifics:

- 1. Run `ccr qwen-auth`, open the printed URL in browser
- 2. Sign in at chat.qwen.ai in another tab, open dev tools console,
     run `copy(localStorage.getItem('token'))`
- 3. Paste into the form, submit
- 4. Token saved to `~/.claude-code-router/qwen_auth.json` (mode 0600)
- 5. Transformer auto-refreshes within 6h of expiry
- 6. Note about Docker: no callback port needed (no OAuth redirect) —
     the auth page is served by the running CCR server on its regular port

Update line 25 (`Codex (ChatGPT) Integration` bullet) — add a parallel
bullet for Qwen mentioning the auth flow.

Add a new "Qwen Provider Configuration" example block parallel to the
"Codex Provider Configuration" block at line 651. The Qwen config uses
`"use": ["qwen-auth", "OpenAI"]`.

### 10. `tasks/lessons.md`

Append a new section "LLM Provider Integration (Qwen)" capturing hard-won
knowledge:

- Token is a single JWT (no separate refresh), expires typically ~30 days
- 120s leeway for expiry (matches upstream behavior)
- Refresh threshold = 6h before expiry
- Trailing `<details>...</details>` block must be stripped from responses
  (regex: `/(?:\\n|\n)?<details>[\s\S]*?<\/details>\s*/g`)
- Token is read from `localStorage.getItem('token')` at `chat.qwen.ai`,
  not from an OAuth flow
- Validation via `qwen.aikit.club/v1/validate` may return a rotated token
  — always use the returned `access_token`, not the input
- Endpoint pairing: `provider.transformer.use[]` must include both
  `qwen-auth` (for the Authorization header + response strip) and
  `OpenAI` (which registers the `/v1/chat/completions` endpoint)

### 11. `tasks/TODO.md`

Add a single line: "Qwen auth: complete (cli + transformer + routes + docs)".

---

## Authentication Considerations

1. **Server ↔ local browser**: The `/qwen/*` routes are public (whitelisted
   in `auth.ts`). The browser's POST to `/qwen/auth` is unauthenticated —
   acceptable because the route only writes a local file the user already
   controls.

2. **CLI ↔ server**: The CLI does **not** call the CCR server. It points
   the user to `http://127.0.0.1:${PORT}/qwen/auth` (using the local port)
   and after the user submits the form + presses Enter, the CLI reads
   `~/.claude-code-router/qwen_auth.json` directly. This matches the codex
   CLI pattern and avoids any need for an API key during auth.

3. **CCR server ↔ qwen.aikit.club**: The `QwenAuthTransformer` injects
   `Authorization: Bearer <jwt>` per request. The Qwen `api_key` field in
   `config.json` stays a placeholder string (per CLAUDE.md security rule:
   "never resolve environment variables, keep placeholders").

4. **Token on disk**: Same security posture as codex — `qwen_auth.json`
   written with `mode: 0o600`, in `~/.claude-code-router/`.

5. **Refresh flow**: When `getValidAccessToken()` is called and the token
   is within 6h of expiry, it transparently calls `refreshToken()` and
   re-saves. If refresh fails (e.g. user revoked the token), throws with a
   message instructing the user to re-run `ccr qwen-auth`. This is the same
   pattern as `codex-auth.ts`.

6. **Upstream 401 retry**: The qwen-proxy handles 401-from-upstream by
   attempting one refresh + retry. We can implement this in the
   transformer as well, but it's also handled implicitly by the next call
   (since `getValidAccessToken()` will refresh on the next invocation).
   For now, rely on the proactive refresh + next-call retry. If this proves
   insufficient, add a one-shot retry in `sendRequestToProvider` later.

---

## Implementation Order (checkable items)

- [ ] 1. Create `packages/core/src/utils/qwen-auth.ts` (token load/save/validate/refresh)
- [ ] 2. Create `packages/core/src/transformer/qwen-auth.transformer.ts`
- [ ] 3. Register `QwenAuthTransformer` in `packages/core/src/transformer/index.ts`
- [ ] 4. Create `packages/server/src/routes/qwen-auth.ts` (HTML auth page + POST + status + forget)
- [ ] 5. Whitelist `/qwen/*` in `packages/server/src/middleware/auth.ts`
- [ ] 6. Register `registerQwenAuthRoutes` in `packages/server/src/server.ts`
- [ ] 7. Create `packages/cli/src/utils/qwen-cli-auth.ts`
- [ ] 8. Wire `qwen-auth` command into `packages/cli/src/cli.ts`
- [ ] 9. Update `README.md` (new section + Qwen config example)
- [ ] 10. Update `tasks/lessons.md` (Qwen integration notes)
- [ ] 11. Update `tasks/TODO.md`
- [ ] 12. Verify: `pnpm build` succeeds, server starts, `ccr qwen-auth`
        runs, `/qwen/auth` renders, paste flow saves token, `qwen-auth`
        transformer loads it on a test request to a qwen model.

---

## Out of Scope

- Token rotation on 401 retry inside the transformer (relying on proactive
  6h-window refresh + next-call refresh is sufficient for v1)
- A `qwen-cli` transformer (the existing `qwen-cli` model selector entry
  at `packages/cli/src/utils/modelSelector.ts:66` is unrelated and stays)
- A web UI panel for Qwen auth (the HTML form at `/qwen/auth` is enough;
  UI can be added later by calling the same JSON endpoints)
- The existing `OpenAI` transformer (endpoint registration only) and
  `openai-responses` chain are already implemented and **not** part of
  this work — Qwen plugs into the chain as a peer in
  `provider.transformer.use[]`

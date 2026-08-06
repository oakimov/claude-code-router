# codex-statsig-unlock

Make the **ChatGPT / Codex desktop app** show CCR-routed models in its native
model picker, by overriding the server-delivered Statsig allowlist that hides
custom slugs.

## Background

The desktop picker is filtered client-side by a Statsig dynamic config
(`107580212`) delivered through the app's post-login bootstrap. When that
config sets `use_hidden_models = true`, the picker only shows models whose slug
is in a server-delivered `available_models` allowlist — which contains native
GPT slugs only. Custom slugs (e.g. `codex/gpt-5.4`,
`opencode-openai/deepseek-v4-flash-free`) are filtered out, even though the
app's own `codex` binary serves them from `model_catalog_json` with
`visibility: "list"`.

There is no config, env var, hosts entry, or SDK override for this: the
allowlist is fetched every launch from
`https://chatgpt.com/backend-api/wham/statsig/bootstrap` and the request goes
renderer → IPC → Electron main process, so `window.fetch` and `og.safePost`
are not interceptable. The one remaining lever is a Chrome DevTools Protocol
injection into the renderer, where `window.__STATSIG__` (the Statsig SDK's
page-global) exposes the active client.

## How it works

`launch.sh` quits ChatGPT, relaunches it with
`--remote-debugging-port=9222 --remote-allow-origins='*'`, then runs the
injector. `inject-statsig.mjs` connects to the CDP endpoint, finds the renderer
targets, and evaluates a patch that:

1. grabs every `StatsigClient` from `window.__STATSIG__.instances` /
   `firstInstance` (the app creates a pre-login and a post-login client);
2. wraps `getDynamicConfig` so config `107580212` returns
   `use_hidden_models: false` and `available_models: []`;
3. calls `updateUserAsync(...)` to force the picker store to re-read it.

With `use_hidden_models` off, the picker's filter falls through to
`!model.hidden`, so every `visibility: "list"` catalog model appears.

## Usage

```bash
scripts/codex-statsig-unlock/launch.sh
```

Or in Automator: add a **Run Shell Script** action (shell `/bin/bash`,
"Pass input: to stdin") running that script.

Injector options:

```bash
node scripts/codex-statsig-unlock/inject-statsig.mjs [--port 9222] [--config <statsig-id>] [--reload]
```

- default: patches the already-loaded page in place (no reload)
- `--reload`: installs via `Page.addScriptToEvaluateOnNewDocument` and reloads
- `--config`: override the Statsig dynamic-config id if an app update changes it

## Caveats

- **Per-launch**: the patch lasts for the session; re-run `launch.sh` after each
  app restart/update.
- **Fragile**: `107580212` is an internal Statsig config id and the minified
  bundle can change. Re-verify with the `--config`/`--reload` flags when the
  app updates.
- **Requires Node >= 22** (global `fetch` + `WebSocket`) and `osascript`
  (macOS). `launch.sh` resolves `node` from PATH, falling back to nvm.
- **Local debugging access**: the launch enables a loopback-only Chrome
  debugging endpoint for the authenticated app. Other processes running as
  your user can connect to it while ChatGPT is open; quit the app when done.
- The app must be fully quit first (Electron's single-instance lock drops
  `argv`, which would silently disable the debug port) — `launch.sh` handles
  this.

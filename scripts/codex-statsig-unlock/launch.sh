#!/usr/bin/env bash
# Quit ChatGPT, relaunch it with a CDP debug port, then inject the Statsig
# model-picker patch (see inject-statsig.mjs).
#
# Use as-is from a terminal, or from Automator as a "Run Shell Script" action
# (shell: /bin/bash, "Pass input: to stdin"). Automator's PATH is minimal, so
# NODE_BIN falls back to the latest nvm install when `node` isn't on PATH.
set -euo pipefail

APP="/Applications/ChatGPT.app"
BIN="$APP/Contents/MacOS/ChatGPT"
PORT="${CDP_PORT:-9222}"
HERE="$(cd "$(dirname "$0")" && pwd)"

# --- resolve node (Automator has a minimal PATH) ----------------------------
NODE_BIN="${NODE_BIN:-$(command -v node || true)}"
if [[ -z "$NODE_BIN" && -d "$HOME/.nvm/versions/node" ]]; then
  NODE_BIN="$(
    for candidate in "$HOME"/.nvm/versions/node/*/bin/node; do
      [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
    done | sort -V | tail -n 1
  )"
fi
if [[ ! -x "$NODE_BIN" ]]; then
  echo "node not found. Install Node >= 22 or set NODE_BIN=/path/to/node." >&2
  exit 1
fi
if ! "$NODE_BIN" -e 'process.exit(Number(process.versions.node.split(".")[0]) >= 22 ? 0 : 1)'; then
  echo "Node >= 22 is required; found $("$NODE_BIN" --version)." >&2
  exit 1
fi

# --- 1. fully quit the app -----------------------------------------------
# Electron's single-instance lock drops argv on a second launch, so the CDP
# flag would never apply unless the old instance is gone first.
osascript -e 'quit app "ChatGPT"' >/dev/null 2>&1 || true
for _ in $(seq 1 15); do
  pgrep -f "$APP/Contents/MacOS/ChatGPT" >/dev/null 2>&1 || break
  sleep 1
done
if pgrep -f "$APP/Contents/MacOS/ChatGPT" >/dev/null 2>&1; then
  echo "ChatGPT did not quit cleanly; aborting (no relaunch)." >&2
  exit 1
fi

# --- 2. relaunch with CDP enabled ----------------------------------------
# --remote-allow-origins=* lets the Node CDP websocket connect on newer
# Chromium builds (which otherwise reject non-DevTools origins).
"$BIN" --remote-debugging-address=127.0.0.1 --remote-debugging-port="$PORT" --remote-allow-origins='*' >/dev/null 2>&1 &
echo "[launch] ChatGPT starting with --remote-debugging-port=$PORT"

# --- 3. wait for the debug port, then inject ------------------------------
"$NODE_BIN" "$HERE/inject-statsig.mjs"

#!/bin/sh
# Bun runtime entrypoint. Docker Compose `restart: always` covers hard crashes;
# this loop recovers soft exits without pm2.
set -eu
# Derive the default from $HOME so the server can run as a non-root user:
# /root is mode 0700, so a non-root process cannot create anything under it.
LOG_DIR="${CCR_LOG_DIR:-${HOME:-/root}/.claude-code-router/logs}"
mkdir -p "$LOG_DIR"
SERVER_JS="/app/packages/server/dist/index.js"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting ccr under bun $(bun --version)" >>"$LOG_DIR/out.log"
exec bun "$SERVER_JS"

#!/bin/sh
# Bun runtime entrypoint. Docker Compose `restart: always` covers hard crashes;
# this loop recovers soft exits without pm2.
set -eu
LOG_DIR="${CCR_LOG_DIR:-/root/.claude-code-router/logs}"
mkdir -p "$LOG_DIR"
SERVER_JS="/app/packages/server/dist/index.js"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting ccr under bun $(bun --version)" >>"$LOG_DIR/out.log"
exec bun "$SERVER_JS"

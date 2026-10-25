# ===========================
# Build Stage — Bun only
# ===========================
FROM oven/bun:1.4.4-alpine AS builder

WORKDIR /app

# Workspace manifest + lockfile first for install caching
COPY package.json bun.lock tsconfig.base.json ./
COPY scripts ./scripts
COPY packages/shared/package.json ./packages/shared/
COPY packages/core/package.json ./packages/core/
COPY packages/server/package.json ./packages/server/
COPY packages/ui/package.json ./packages/ui/
COPY packages/cli/package.json ./packages/cli/
COPY packages/docusaurus-image-size/package.json ./packages/docusaurus-image-size/
COPY docs/package.json ./docs/

COPY packages/shared ./packages/shared
COPY packages/core ./packages/core
COPY packages/server ./packages/server
COPY packages/ui ./packages/ui
COPY packages/cli ./packages/cli
COPY packages/docusaurus-image-size ./packages/docusaurus-image-size
COPY docs ./docs

RUN bun install --frozen-lockfile

WORKDIR /app/packages/shared
RUN bun run build

WORKDIR /app/packages/core
RUN bun run build

WORKDIR /app/packages/server
RUN bun run build

WORKDIR /app/packages/ui
RUN bun run build

WORKDIR /app/packages/cli
RUN bun run build

# ===========================
# Production Stage — Bun runtime
# ===========================
FROM oven/bun:1.4.4-alpine AS production

RUN apk add --no-cache curl

WORKDIR /app

COPY --from=builder /app/packages/core ./packages/core
COPY --from=builder /app/packages/cli/dist ./packages/cli/dist

RUN chmod +x /app/packages/cli/dist/cli.js && \
    ln -sf /app/packages/cli/dist/cli.js /usr/local/bin/ccr

ENV PATH="/app/packages/cli/dist:$PATH"

COPY --from=builder /app/packages/server/dist ./packages/server/dist
COPY --from=builder /app/packages/ui/dist/. ./packages/server/dist/

# Runtime externals (@cursor/sdk platform bits; AI SDK + zod for debug-chat).
# Install into a clean prefix so workspace:* edges from server package.json
# are not required.
COPY --from=builder /app/packages/server/package.json /tmp/server-package.json
RUN mkdir -p /tmp/runtime-externals /app/packages/server/node_modules && \
    cd /tmp/runtime-externals && \
    bun init -y >/dev/null && \
    CURSOR_SDK="$(bun -e "console.log(require('/tmp/server-package.json').dependencies['@cursor/sdk'])")" && \
    AI_SDK_ANTHROPIC="$(bun -e "console.log(require('/tmp/server-package.json').dependencies['@ai-sdk/anthropic'])")" && \
    AI_SDK_OPENAI="$(bun -e "console.log(require('/tmp/server-package.json').dependencies['@ai-sdk/openai'])")" && \
    AI_SDK="$(bun -e "console.log(require('/tmp/server-package.json').dependencies['ai'])")" && \
    ZOD="$(bun -e "console.log(require('/tmp/server-package.json').dependencies['zod'])")" && \
    test -n "$CURSOR_SDK" && test -n "$AI_SDK_ANTHROPIC" && test -n "$AI_SDK_OPENAI" && test -n "$AI_SDK" && test -n "$ZOD" && \
    bun add "@cursor/sdk@${CURSOR_SDK}" "@ai-sdk/anthropic@${AI_SDK_ANTHROPIC}" "@ai-sdk/openai@${AI_SDK_OPENAI}" "ai@${AI_SDK}" "zod@${ZOD}" && \
    cp -a /tmp/runtime-externals/node_modules/. /app/packages/server/node_modules/ && \
    rm -rf /tmp/runtime-externals /tmp/server-package.json /root/.bun

RUN mkdir -p /app/packages/cli/node_modules /app/node_modules && \
    ln -sfn /app/packages/server/node_modules/@cursor /app/packages/cli/node_modules/@cursor && \
    ln -sfn /app/packages/server/node_modules/@cursor /app/node_modules/@cursor && \
    ln -sfn /app/packages/server/node_modules/@ai-sdk /app/packages/cli/node_modules/@ai-sdk && \
    ln -sfn /app/packages/server/node_modules/@ai-sdk /app/node_modules/@ai-sdk && \
    ln -sfn /app/packages/server/node_modules/ai /app/packages/cli/node_modules/ai && \
    ln -sfn /app/packages/server/node_modules/ai /app/node_modules/ai && \
    ln -sfn /app/packages/server/node_modules/zod /app/packages/cli/node_modules/zod && \
    ln -sfn /app/packages/server/node_modules/zod /app/node_modules/zod

COPY packages/server/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

RUN mkdir -p /root/.claude-code-router/logs

EXPOSE 3456

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:3456/health || exit 1

ENV NODE_ENV=production

CMD ["/app/docker-entrypoint.sh"]

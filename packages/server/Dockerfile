# ===========================
# Build Stage
# ===========================
FROM node:22.23.2-alpine AS builder

WORKDIR /app

# Copy workspace configuration and package files first for caching
COPY pnpm-workspace.yaml pnpm-lock.yaml package.json tsconfig.base.json ./

# Install the exact pnpm version from package.json's "packageManager" field.
# Prefer a direct npm global install over upgrading corepack: under QEMU
# arm64 emulation, `npm install -g corepack@…` has been observed to SIGILL
# (exit 132) on GitHub Actions multi-arch builds.
RUN PNPM_VERSION="$(node -p "require('./package.json').packageManager.replace(/^pnpm@/, '')")" && \
    test -n "$PNPM_VERSION" && \
    npm install -g "pnpm@${PNPM_VERSION}" && \
    rm -rf /root/.npm
COPY scripts ./scripts
COPY packages/shared/package.json ./packages/shared/
COPY packages/core/package.json ./packages/core/
COPY packages/server/package.json ./packages/server/
COPY packages/ui/package.json ./packages/ui/
COPY packages/cli/package.json ./packages/cli/

# Copy all source code before installing dependencies to avoid symlink conflicts
COPY packages/shared ./packages/shared
COPY packages/core ./packages/core
COPY packages/server ./packages/server
COPY packages/ui ./packages/ui
COPY packages/cli ./packages/cli

# Install all dependencies
RUN pnpm install --frozen-lockfile && \
    pnpm store prune

# Build all packages (shared before core — core imports @caeliq/ccr-shared)
WORKDIR /app/packages/shared
RUN pnpm build

WORKDIR /app/packages/core
RUN pnpm build

WORKDIR /app/packages/server
RUN pnpm build && \
    rm -rf node_modules/.cache

WORKDIR /app/packages/ui
RUN pnpm build

WORKDIR /app/packages/cli
RUN pnpm build

# ===========================
# Production Stage
# ===========================
# Pin to the current Node 22 LTS patch. The workspace floor is >=22.19.0
# (undici); @cursor/sdk needs >=22.13. Keep in sync with the builder stage.
FROM node:22.23.2-alpine AS production

# Install PM2, curl, and pm2-logrotate
RUN apk add --no-cache curl && \
    npm install -g pm2 pm2-logrotate --no-scripts && \
    pm2 install pm2-logrotate && \
    pm2 set pm2-logrotate:max_size 100M && \
    pm2 set pm2-logrotate:retain 5 && \
    pm2 set pm2-logrotate:compress true && \
    pm2 set pm2-logrotate:rotateInterval '0 0 * * *'

WORKDIR /app

# Copy core package (workspace dependency)
COPY --from=builder /app/packages/core ./packages/core

# Copy CLI bundle
COPY --from=builder /app/packages/cli/dist ./packages/cli/dist

# Make CLI executable and create symlink for 'ccr' command
RUN chmod +x /app/packages/cli/dist/cli.js && \
    ln -sf /app/packages/cli/dist/cli.js /usr/local/bin/ccr

# Add CLI dist to PATH (for direct access)
ENV PATH="/app/packages/cli/dist:$PATH"

# Copy server bundle from build stage
COPY --from=builder /app/packages/server/dist ./packages/server/dist
# Copy locally pre-built UI assets to the same directory
COPY --from=builder /app/packages/ui/dist/. ./packages/server/dist/

# Bundles externalize a few runtime packages (@cursor/sdk for native
# platform bits; Mastra + ai + zod for the debug-chat playground). Workspace
# pnpm node_modules are symlink farms into /app/node_modules/.pnpm, so copying
# only packages/server/node_modules leaves those links broken in the runtime
# image. Install real copies here instead (into a clean prefix — do not use
# packages/server/package.json, which has workspace:* deps that cannot resolve
# in the runtime stage).
# Versions are read from packages/server/package.json so they cannot silently
# drift from the workspace declaration (still a separate npm install vs pnpm-lock).
COPY --from=builder /app/packages/server/package.json /tmp/server-package.json
RUN mkdir -p /tmp/runtime-externals /app/packages/server/node_modules && \
    cd /tmp/runtime-externals && \
    npm init -y >/dev/null && \
    node -e "const fs=require('fs');const p=require('./package.json');p.overrides={'@ai-sdk/provider-utils@3.0.19':{undici:'^8.10.0'},'@ai-sdk/provider-utils@3.0.25':{undici:'^8.10.0'},'@ai-sdk/provider-utils@3.0.32':{undici:'^8.10.0'}};fs.writeFileSync('package.json',JSON.stringify(p));" && \
    CURSOR_SDK="$(node -p "require('/tmp/server-package.json').dependencies['@cursor/sdk']")" && \
    MASTRA_CORE="$(node -p "require('/tmp/server-package.json').dependencies['@mastra/core']")" && \
    MASTRA_AI_SDK="$(node -p "require('/tmp/server-package.json').dependencies['@mastra/ai-sdk']")" && \
    AI_SDK_ANTHROPIC="$(node -p "require('/tmp/server-package.json').dependencies['@ai-sdk/anthropic']")" && \
    AI_SDK_OPENAI="$(node -p "require('/tmp/server-package.json').dependencies['@ai-sdk/openai']")" && \
    BUF_PROTOBUF="$(node -p "require('/tmp/server-package.json').dependencies['@bufbuild/protobuf']")" && \
    AI_SDK="$(node -p "require('/tmp/server-package.json').dependencies['ai']")" && \
    ZOD="$(node -p "require('/tmp/server-package.json').dependencies['zod']")" && \
    test -n "$CURSOR_SDK" && test -n "$MASTRA_CORE" && test -n "$MASTRA_AI_SDK" && test -n "$AI_SDK_ANTHROPIC" && test -n "$AI_SDK_OPENAI" && test -n "$BUF_PROTOBUF" && test -n "$AI_SDK" && test -n "$ZOD" && \
    npm install "@cursor/sdk@${CURSOR_SDK}" "@mastra/core@${MASTRA_CORE}" "@mastra/ai-sdk@${MASTRA_AI_SDK}" "@ai-sdk/anthropic@${AI_SDK_ANTHROPIC}" "@ai-sdk/openai@${AI_SDK_OPENAI}" "@bufbuild/protobuf@${BUF_PROTOBUF}" "ai@${AI_SDK}" "zod@${ZOD}" --omit=dev && \
    cp -a /tmp/runtime-externals/node_modules/. /app/packages/server/node_modules/ && \
    rm -rf /tmp/runtime-externals /tmp/server-package.json /root/.npm

# Expose the same install to the CLI bundle resolution path
# (/app/packages/cli/dist → … → node_modules).
RUN mkdir -p /app/packages/cli/node_modules /app/node_modules && \
    ln -sfn /app/packages/server/node_modules/@cursor /app/packages/cli/node_modules/@cursor && \
    ln -sfn /app/packages/server/node_modules/@cursor /app/node_modules/@cursor && \
    ln -sfn /app/packages/server/node_modules/@mastra /app/packages/cli/node_modules/@mastra && \
    ln -sfn /app/packages/server/node_modules/@mastra /app/node_modules/@mastra && \
    ln -sfn /app/packages/server/node_modules/@ai-sdk /app/packages/cli/node_modules/@ai-sdk && \
    ln -sfn /app/packages/server/node_modules/@ai-sdk /app/node_modules/@ai-sdk && \
    ln -sfn /app/packages/server/node_modules/ai /app/packages/cli/node_modules/ai && \
    ln -sfn /app/packages/server/node_modules/ai /app/node_modules/ai && \
    ln -sfn /app/packages/server/node_modules/zod /app/packages/cli/node_modules/zod && \
    ln -sfn /app/packages/server/node_modules/zod /app/node_modules/zod

# Copy PM2 configuration file
COPY packages/server/ecosystem.config.cjs /app/

# Create logs directory
RUN mkdir -p /root/.claude-code-router/logs

# Expose API port (host 1455/51121 also map here for OAuth callbacks)
EXPOSE 3456

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:3456/health || exit 1

# Start the application
CMD ["pm2-runtime", "start", "/app/ecosystem.config.cjs"]
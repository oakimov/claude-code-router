#!/bin/bash
set -euo pipefail

# Publish script
# - @caeliq/ccr-shared
# - @caeliq/llms
# - @caeliq/claude-code-router (CLI bundle)
# - Docker image caeliq/claude-code-router (optional)
#
# Usage:
#   ./scripts/release.sh npm
#   ./scripts/release.sh docker
#   ./scripts/release.sh all
#   NPM_DRY_RUN=1 ./scripts/release.sh npm
#
# CI / OIDC trusted publishing:
#   CI=true ./scripts/release.sh npm
#   (requires id-token: write; no NPM token)

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=$(node -p "require('${ROOT_DIR}/packages/cli/package.json').version")
SHARED_VERSION=$(node -p "require('${ROOT_DIR}/packages/shared/package.json').version")
CORE_VERSION=$(node -p "require('${ROOT_DIR}/packages/core/package.json').version")
IMAGE_NAME="${DOCKER_IMAGE_NAME:-caeliq/claude-code-router}"
IMAGE_TAG="${VERSION}"
LATEST_TAG="latest"
DRY_RUN="${NPM_DRY_RUN:-0}"
# Use OIDC trusted publishing in GitHub Actions (no npm token).
TRUSTED_PUBLISHING="${NPM_TRUSTED_PUBLISHING:-${CI:-false}}"
PUBLISH_ARGS=(--access public)
if [[ "${TRUSTED_PUBLISHING}" == "true" || "${TRUSTED_PUBLISHING}" == "1" ]]; then
  PUBLISH_ARGS+=(--provenance)
fi
if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
  PUBLISH_ARGS+=(--dry-run)
fi

REPO_URL="https://github.com/oakimov/claude-code-router.git"
echo "========================================="
echo "Publishing Claude Code Router"
echo "  CLI:    @caeliq/claude-code-router@${VERSION}"
echo "  shared: @caeliq/ccr-shared@${SHARED_VERSION}"
echo "  llms:   @caeliq/llms@${CORE_VERSION}"
echo "  mode:   trusted=${TRUSTED_PUBLISHING} dry_run=${DRY_RUN}"
echo "========================================="

PUBLISH_TYPE="${1:-all}"

case "$PUBLISH_TYPE" in
  npm|docker|all) ;;
  *)
    echo "Usage: $0 [npm|docker|all]"
    exit 1
    ;;
esac

require_npm_login() {
  if [[ "${TRUSTED_PUBLISHING}" == "true" || "${TRUSTED_PUBLISHING}" == "1" ]]; then
    echo "Using npm trusted publishing (OIDC); skipping npm whoami"
    return 0
  fi
  if ! npm whoami &>/dev/null; then
    echo "Error: not logged into npm. Run: npm login"
    exit 1
  fi
  echo "npm user: $(npm whoami)"
}

npm_publish() {
  local pkg_dir="$1"
  (
    cd "$pkg_dir"
    echo "npm publish ${PUBLISH_ARGS[*]}"
    set +e
    local output
    output="$(npm publish "${PUBLISH_ARGS[@]}" 2>&1)"
    local status=$?
    set -e
    printf '%s\n' "$output"
    if [[ $status -eq 0 ]]; then
      exit 0
    fi
    # Dry-run against an already-published version is still a useful CI smoke test.
    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
      if printf '%s\n' "$output" | grep -q "cannot publish over the previously published versions"; then
        echo "Dry-run OK: version already on npm (packaging/auth path exercised)"
        exit 0
      fi
    fi
    exit "$status"
  )
}

# Rewrite workspace:* deps to concrete versions for publish.
prepare_publish_package() {
  local pkg_dir="$1"
  local name="$2"
  local repo_directory="$3"
  local extra_node="$4"
  local enable_provenance="false"
  if [[ "${TRUSTED_PUBLISHING}" == "true" || "${TRUSTED_PUBLISHING}" == "1" ]]; then
    enable_provenance="true"
  fi

  local backup_dir="${pkg_dir}/.backup"
  mkdir -p "$backup_dir"
  cp "${pkg_dir}/package.json" "${backup_dir}/package.json.original"

  node -e "
    const fs = require('fs');
    const path = require('path');
    const pkgPath = path.join('${pkg_dir}', 'package.json');
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    pkg.name = '${name}';
    delete pkg.scripts;
    pkg.author = 'caeliq';
    pkg.repository = {
      type: 'git',
      url: 'git+${REPO_URL}',
      directory: '${repo_directory}'
    };
    pkg.publishConfig = {
      access: 'public',
      registry: 'https://registry.npmjs.org/'
    };
    // Provenance only works on supported CI providers (GitHub Actions, etc.).
    if (${enable_provenance}) {
      pkg.publishConfig.provenance = true;
    }
    const rewrite = (deps = {}) => {
      for (const [key, value] of Object.entries(deps)) {
        if (value === 'workspace:*') {
          if (key === '@caeliq/ccr-shared') deps[key] = '^${SHARED_VERSION}';
          else if (key === '@caeliq/llms') deps[key] = '^${CORE_VERSION}';
          else delete deps[key];
        }
      }
      return deps;
    };
    pkg.dependencies = rewrite(pkg.dependencies || {});
    pkg.devDependencies = rewrite(pkg.devDependencies || {});
    ${extra_node}
    fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n');
  "
}

restore_package() {
  local pkg_dir="$1"
  local backup_dir="${pkg_dir}/.backup"
  local original="${backup_dir}/package.json.original"
  if [ -f "$original" ]; then
    mv "$original" "${pkg_dir}/package.json"
  fi
  rmdir "$backup_dir" 2>/dev/null || true
}

copy_package_meta() {
  local pkg_dir="$1"
  cp "${ROOT_DIR}/README.md" "${pkg_dir}/" 2>/dev/null || true
  cp "${ROOT_DIR}/LICENSE" "${pkg_dir}/" 2>/dev/null || true
}

publish_shared_npm() {
  echo ""
  echo "========================================="
  echo "Publishing @caeliq/ccr-shared@${SHARED_VERSION}"
  echo "========================================="
  require_npm_login

  local pkg_dir="${ROOT_DIR}/packages/shared"
  copy_package_meta "$pkg_dir"
  prepare_publish_package "$pkg_dir" "@caeliq/ccr-shared" "packages/shared" "
    pkg.files = ['dist', 'README.md', 'LICENSE'];
  "

  npm_publish "$pkg_dir"
  restore_package "$pkg_dir"
  echo "✅ @caeliq/ccr-shared@${SHARED_VERSION}"
}

publish_core_npm() {
  echo ""
  echo "========================================="
  echo "Publishing @caeliq/llms@${CORE_VERSION}"
  echo "========================================="
  require_npm_login

  local pkg_dir="${ROOT_DIR}/packages/core"
  copy_package_meta "$pkg_dir"
  prepare_publish_package "$pkg_dir" "@caeliq/llms" "packages/core" "
    pkg.files = ['dist', 'README.md', 'LICENSE'];
  "

  npm_publish "$pkg_dir"
  restore_package "$pkg_dir"
  echo "✅ @caeliq/llms@${CORE_VERSION}"
}

publish_cli_npm() {
  echo ""
  echo "========================================="
  echo "Publishing @caeliq/claude-code-router@${VERSION}"
  echo "========================================="
  require_npm_login

  local pkg_dir="${ROOT_DIR}/packages/cli"
  copy_package_meta "$pkg_dir"

  local cursor_sdk
  cursor_sdk=$(node -p "require('${ROOT_DIR}/packages/cli/package.json').dependencies['@cursor/sdk'] || '^1.0.23'")

  prepare_publish_package "$pkg_dir" "@caeliq/claude-code-router" "packages/cli" "
    pkg.files = ['dist', 'README.md', 'LICENSE'];
    pkg.bin = { ccr: 'dist/cli.js' };
    pkg.dependencies = { '@cursor/sdk': '${cursor_sdk}' };
    pkg.devDependencies = {};
    pkg.engines = { node: '>=22.13.0' };
  "

  npm_publish "$pkg_dir"
  restore_package "$pkg_dir"
  echo "✅ @caeliq/claude-code-router@${VERSION}"
}

publish_docker() {
  echo ""
  echo "========================================="
  echo "Publishing Docker image ${IMAGE_NAME}"
  echo "========================================="

  if ! docker info &>/dev/null; then
    echo "Error: Docker is not running"
    exit 1
  fi

  echo "Building ${IMAGE_NAME}:${IMAGE_TAG}..."
  docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${ROOT_DIR}/packages/server/Dockerfile" "${ROOT_DIR}"
  docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${IMAGE_NAME}:${LATEST_TAG}"

  if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
    echo "Dry run: skipping docker push"
    return 0
  fi

  echo "Pushing ${IMAGE_NAME}:${IMAGE_TAG}..."
  docker push "${IMAGE_NAME}:${IMAGE_TAG}"
  docker push "${IMAGE_NAME}:${LATEST_TAG}"

  echo "✅ ${IMAGE_NAME}:${IMAGE_TAG}"
  echo "✅ ${IMAGE_NAME}:latest"
}

if [ "$PUBLISH_TYPE" = "npm" ] || [ "$PUBLISH_TYPE" = "all" ]; then
  publish_shared_npm
  publish_core_npm
  publish_cli_npm
fi

if [ "$PUBLISH_TYPE" = "docker" ] || [ "$PUBLISH_TYPE" = "all" ]; then
  publish_docker
fi

echo ""
echo "========================================="
echo "Publish complete"
echo "========================================="
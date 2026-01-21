#!/usr/bin/env bash
set -euo pipefail

echo "==> Checking prerequisites"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PKG_JSON="$ROOT_DIR/../package.json"
LOCK_JSON="$ROOT_DIR/../package-lock.json"

if [[ ! -f "$PKG_JSON" ]]; then
    echo "✘ package.json not found: $PKG_JSON"
    exit 1
else
    echo "✔ package.json found: $PKG_JSON"
fi

if [[ ! -f "$LOCK_JSON" ]]; then
    echo "✘ package-lock.json not found: $LOCK_JSON"
    exit 1
else
    echo "✔ package-lock.json found: $LOCK_JSON"
fi

# Read name/version from package.json
PKG_NAME="$(node -p "require('$PKG_JSON').name")"
PKG_VERSION="$(node -p "require('$PKG_JSON').version")"

if [[ -z "$PKG_NAME" || -z "$PKG_VERSION" ]]; then
    echo "✘ Failed to read name/version from package.json"
    exit 1
else
    echo "✔ Read name/version from package.json:"
    echo "  name:    $PKG_NAME"
    echo "  version: $PKG_VERSION"
fi

GIT_TAG="v$PKG_VERSION"

if git tag -l | grep -qx "$GIT_TAG"; then
    echo "✘ Git tag '$GIT_TAG' already exists."
    echo "  Please bump version in package.json before releasing."
    exit 1
else
    echo "✔ Git tag '$GIT_TAG' does not exist."
fi

OUT_VSIX="${PKG_NAME}-v${PKG_VERSION}.vsix"

echo "==> Installing dependencies"
npm install >/dev/null

echo "==> Running tests"
npm test >/dev/null 2>&1

echo "==> Compiling extension"
npm run compile >/dev/null

echo "==> Packaging VSIX: $OUT_VSIX"
npx vsce package -o "$OUT_VSIX" >/dev/null

if [[ -f "$OUT_VSIX" ]]; then
    echo "✔ VSIX file created: $OUT_VSIX"
else
    echo "✘ Failed to create VSIX file: $OUT_VSIX"
    exit 1
fi

# Read version from package-lock.json
LOCK_VERSION="$(node -p "require('$LOCK_JSON').version")"

if [[ -z "$LOCK_VERSION" ]]; then
    echo "✘ Failed to read version from package-lock.json: $LOCK_JSON"
    exit 1
else
    echo "✔ Read version from package-lock.json:"
    echo "  version: $LOCK_VERSION"
fi

# Verify versions match
if [[ "$PKG_VERSION" != "$LOCK_VERSION" ]]; then
    echo "✘ Version mismatch:"
    echo "  package.json      version: $PKG_VERSION"
    echo "  package-lock.json version: $LOCK_VERSION"
    echo "  Please run 'npm install' (or update lockfile) and commit changes."
    exit 1
else
    echo "✔ Version match: $PKG_VERSION"
fi

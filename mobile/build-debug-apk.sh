#!/usr/bin/env bash
# Build a standalone debug APK (embedded JS bundle, no Metro on device).
# Prerequisites: Node 20, JDK 17, Android SDK (API 35, build-tools 35, NDK 27.1.12297006).
# See BUILD.md in this directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

info() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Optional: load mobile/.env so EXPO_PUBLIC_* is set for prebuild and Gradle bundle.
if [[ -f .env ]]; then
  info "Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

command -v node >/dev/null 2>&1 || die "node not found (install Node.js 20)"
command -v npm >/dev/null 2>&1 || die "npm not found"
command -v java >/dev/null 2>&1 || die "java not found (install openjdk-17-jdk and set JAVA_HOME)"

if [[ -z "${JAVA_HOME:-}" ]]; then
  info "Hint: export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 (or your JDK 17 path)"
fi

SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}"
if [[ -z "$SDK" ]]; then
  die "Set ANDROID_SDK_ROOT or ANDROID_HOME to your Android SDK root (contains platforms/, build-tools/)"
fi

info "Using Android SDK: $SDK"

info "npm install"
npm install

info "expo prebuild (android)"
npx expo prebuild --platform android --no-install

# Ensure Gradle can find the SDK if local.properties is missing or has no sdk.dir.
PROP="android/local.properties"
if [[ ! -f "$PROP" ]] || ! grep -q '^sdk.dir=' "$PROP" 2>/dev/null; then
  echo "sdk.dir=${SDK}" >>"$PROP"
  info "Ensured sdk.dir in $PROP"
fi

info "gradle assembleDebug (NODE_ENV=production embeds the JS bundle)"
cd android
chmod +x gradlew
NODE_ENV=production ./gradlew assembleDebug --no-daemon

APK_DIR="app/build/outputs/apk/debug"
info "Done. APK(s):"
ls -la "$APK_DIR"/*.apk 2>/dev/null || die "No APK found under $APK_DIR"

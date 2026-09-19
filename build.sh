#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
BUILDER="${ELYX_BUILDER:-/home/PluginDev/.venv/bin/elyb}"
NDK_BUILD="/home/PluginDev/.android-sdk/ndk/29.0.14206865/ndk-build"
export JAVA_HOME

if [[ ! -x "$BUILDER" ]]; then
    echo "ElyxBuilder not found: $BUILDER" >&2
    exit 1
fi

cd "$ROOT_DIR"
"$NDK_BUILD" -C "$ROOT_DIR" NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk
./gradlew clean buildDex
python3 build.py
"$BUILDER" build --ast --verbose --no-folder

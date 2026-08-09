#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
NDK="${ANDROID_NDK_HOME:-/home/PluginDev/.android-sdk/ndk/29.0.14206865}"

if [[ ! -x "$NDK/ndk-build" ]]; then
    echo "Android NDK not found: $NDK" >&2
    exit 1
fi

# Inject a unique SONAME into Android.mk so each plugin version ships a
# distinct shared library. Android's linker refuses to load two libraries
# with the same SONAME into different ClassLoaders, so a per-version SONAME
# is required for hot-reload (reloading the plugin without restarting the
# Telegram process). The version is read from loader/metadata.py.
SONAME_VERSION="$(python3 -c 'import re,sys; print(re.search(r"__version__ = \"([^\"]+)\"", open("'"$ROOT_DIR"'/loader/metadata.py").read()).group(1).replace(".","_"))')"
echo "SONAME version: $SONAME_VERSION"

TMP_MK="$ROOT_DIR/Android.versioned.mk"
sed "s/@@SONAME_VERSION@@/$SONAME_VERSION/g" "$ROOT_DIR/Android.mk" > "$TMP_MK"

rm -rf "$ROOT_DIR/build/obj" "$ROOT_DIR/build/arm64-v8a"
mkdir -p "$ROOT_DIR/build"

"$NDK/ndk-build" \
    NDK_PROJECT_PATH="$ROOT_DIR" \
    APP_BUILD_SCRIPT="$TMP_MK" \
    NDK_APPLICATION_MK="$ROOT_DIR/Application.mk" \
    NDK_OUT="$ROOT_DIR/build/obj" \
    NDK_LIBS_OUT="$ROOT_DIR/build" \
    V=0

rm -f "$TMP_MK"
python3 "$ROOT_DIR/loader/build.py"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolve JAVA_HOME
if [[ -z "${JAVA_HOME:-}" || ! -d "${JAVA_HOME}" ]]; then
    if [[ -d "/usr/lib/jvm/java-17-openjdk-amd64" ]]; then
        JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
    elif command -v javac >/dev/null 2>&1; then
        JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")"
    fi
fi
if [[ -z "${JAVA_HOME:-}" || ! -d "${JAVA_HOME}" ]]; then
    echo "Error: JAVA_HOME not found. Please install JDK 17+ and export JAVA_HOME." >&2
    exit 1
fi
export JAVA_HOME

# Resolve ElyxBuilder (elyb)
BUILDER=""
for candidate in "${ELYX_BUILDER:-}" "${ELYB_PATH:-}" "$(command -v elyb 2>/dev/null || true)" "/home/PluginDev/.venv/bin/elyb" "$HOME/.venv/bin/elyb"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        BUILDER="$candidate"
        break
    fi
done
if [[ -z "$BUILDER" ]]; then
    echo "Error: ElyxBuilder (elyb) not found. Please set ELYX_BUILDER or ELYB_PATH environment variable." >&2
    exit 1
fi

# Resolve NDK ndk-build
NDK_BUILD=""
for candidate in "${NDK_BUILD:-}" "${ANDROID_NDK_ROOT:-}/ndk-build" "${ANDROID_NDK_HOME:-}/ndk-build" "${NDK_PATH:-}/ndk-build" "$(command -v ndk-build 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        NDK_BUILD="$candidate"
        break
    fi
done
if [[ -z "$NDK_BUILD" ]]; then
    for search_dir in "/home/PluginDev/.android-sdk/ndk" "${ANDROID_HOME:-}/ndk" "${ANDROID_SDK_ROOT:-}/ndk" "$HOME/Android/Sdk/ndk"; do
        if [[ -d "$search_dir" ]]; then
            ndk_found="$(ls -d "$search_dir"/*/ndk-build 2>/dev/null | sort -V | tail -n 1 || true)"
            if [[ -n "$ndk_found" && -x "$ndk_found" ]]; then
                NDK_BUILD="$ndk_found"
                break
            fi
        fi
    done
fi
if [[ -z "$NDK_BUILD" ]]; then
    echo "Error: ndk-build not found. Please set ANDROID_NDK_ROOT, NDK_PATH, or NDK_BUILD environment variable." >&2
    exit 1
fi

cd "$ROOT_DIR"
"$NDK_BUILD" -C "$ROOT_DIR" NDK_PROJECT_PATH=. APP_BUILD_SCRIPT=Android.mk NDK_APPLICATION_MK=Application.mk
./gradlew clean buildDex
python3 build.py
"$BUILDER" build --ast --verbose --no-folder

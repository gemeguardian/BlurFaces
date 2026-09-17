#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
BUILDER="${ELYX_BUILDER:-/home/PluginDev/.venv/bin/elyb}"
export JAVA_HOME

if [[ ! -x "$BUILDER" ]]; then
    echo "ElyxBuilder not found: $BUILDER" >&2
    exit 1
fi

cd "$ROOT_DIR"
./gradlew clean buildDex extractMeshNative
python3 build.py
"$BUILDER" build --ast --verbose --no-folder
python3 tests_elyx_contract.py

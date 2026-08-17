#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export JAVA_HOME

# Official reproducible build: no SCRFD/NDK output and no stale build input.
"$ROOT_DIR/gradlew" clean buildDex extractMeshNative
python3 "$ROOT_DIR/loader/build.py"

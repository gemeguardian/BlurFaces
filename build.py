"""Builds the blur-faces plugin with embedded model and native library."""
import os
import base64
import shutil
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(ROOT_DIR, "build")
PLUGIN_DIR = os.path.join(BUILD_DIR, "plugin")
MODEL_DIR = os.path.join(ROOT_DIR, "model")
NATIVE_LIB = os.path.join(BUILD_DIR, "arm64-v8a", "libblur_faces.so")

# Ensure clean build directory
if os.path.exists(BUILD_DIR):
    shutil.rmtree(BUILD_DIR)
os.makedirs(PLUGIN_DIR, exist_ok=True)

print("[build] Embedding model files...")
# Embed model files
with open(os.path.join(MODEL_DIR, "scrfd_10g-opt2.param"), "rb") as f:
    param_b64 = base64.b64encode(f.read()).decode("ascii")

with open(os.path.join(MODEL_DIR, "scrfd_10g-opt2.bin"), "rb") as f:
    bin_b64 = base64.b64encode(f.read()).decode("ascii")

print(f"[build] Model param: {len(param_b64)} bytes (base64)")
print(f"[build] Model bin: {len(bin_b64)} bytes (base64)")

print("[build] Embedding native library...")
# Embed native library
with open(NATIVE_LIB, "rb") as f:
    native_b64 = base64.b64encode(f.read()).decode("ascii")

print(f"[build] Native library: {len(native_b64)} bytes (base64)")

print("[build] Assembling plugin...")
# Read all loader files
loader_files = [
    "metadata.py",
    "imports.py",
    "constants.py",
    "utils.py",
    "native.py",
    "plugin.py"
]

# Build final plugin file
output = []

# Add embedded data
output.append(f'EMBEDDED_NATIVE_BASE64 = """{native_b64}"""\n')
output.append(f'EMBEDDED_MODEL_PARAM_BASE64 = """{param_b64}"""\n')
output.append(f'EMBEDDED_MODEL_BIN_BASE64 = """{bin_b64}"""\n')

# Add loader files
for fname in loader_files:
    fpath = os.path.join(ROOT_DIR, "loader", fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()
    output.append(f"\n# === {fname} ===\n")
    output.append(content)

# Write final plugin
plugin_path = os.path.join(PLUGIN_DIR, "blur-faces.plugin")
with open(plugin_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print(f"[build] Plugin assembled: {plugin_path}")
print(f"[build] Total size: {os.path.getsize(plugin_path) / 1024 / 1024:.2f} MB")

# Create install script
install_script = f"""#!/bin/bash
# Install blur-faces plugin to exteraGram
PLUGIN_FILE="{plugin_path}"

if [ ! -f "$PLUGIN_FILE" ]; then
    echo "Plugin file not found: $PLUGIN_FILE"
    exit 1
fi

# Detect exteraGram package
PACKAGE="org.telegram.messenger"
if adb shell pm list packages | grep -q "org.telegram.messenger.beta"; then
    PACKAGE="org.telegram.messenger.beta"
fi

echo "Installing to: $PACKAGE"
DEST="/data/data/$PACKAGE/files/plugins/blur_faces"

adb shell "run-as $PACKAGE mkdir -p $DEST"
adb push "$PLUGIN_FILE" "$DEST/blur-faces.plugin"

echo "Plugin installed. Restart exteraGram to load."
"""

install_path = os.path.join(PLUGIN_DIR, "install.sh")
with open(install_path, "w", encoding="utf-8") as f:
    f.write(install_script)
os.chmod(install_path, 0o755)

print(f"[build] Install script: {install_path}")
print("[build] Done!")

#!/usr/bin/env python3
"""Build blur-faces.plugin with embedded .so and model."""

import base64
import hashlib
import os
import sys

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PLUGIN_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, "model")
NATIVE_LIB = os.path.join(ROOT_DIR, "build", "arm64-v8a", "libblur_faces.so")
MESH_NATIVE_LIB = os.path.join(ROOT_DIR, "build", "mesh-runtime", "libmediapipe_tasks_vision_jni.so")
NATIVE_URL = "https://makey.dev/blur-faces/libblur_faces.so"
DEX_PATH = os.path.join(ROOT_DIR, "build", "dex", "classes.dex")
CORE_DEX_PATH = os.path.join(ROOT_DIR, "build", "dex", "core.dex")
MESH_DEX_PATH = os.path.join(ROOT_DIR, "build", "dex", "mesh-runtime.dex")

# Load all source files
def load_file(path):
    with open(path, 'r') as f:
        return f.read()

def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Publish the native library separately so plugin updates stay small.
print("Hashing remote native library...")
with open(NATIVE_LIB, "rb") as f:
    native_sha256 = hashlib.sha256(f.read()).hexdigest()
mesh_native_sha256 = sha256_file(MESH_NATIVE_LIB)
print(f"  root: {os.path.getsize(NATIVE_LIB)} bytes -> SHA-256 {native_sha256}")
print(f"  mesh JNI: {os.path.getsize(MESH_NATIVE_LIB)} bytes -> SHA-256 {mesh_native_sha256}")

# Publish detector and dense-mesh assets separately so the installable plugin stays small.
print("Hashing remote face model files...")
model_param = os.path.join(MODEL_DIR, "retinaface_mnet025_5kps.param")
model_bin = os.path.join(MODEL_DIR, "retinaface_mnet025_5kps.bin")
mesh_task = os.path.join(MODEL_DIR, "face_landmarker.task")
param_sha256 = sha256_file(model_param)
bin_sha256 = sha256_file(model_bin)
mesh_sha256 = sha256_file(mesh_task)
print(f"  param: {os.path.getsize(model_param)} bytes -> SHA-256 {param_sha256}")
print(f"  bin: {os.path.getsize(model_bin)} bytes -> SHA-256 {bin_sha256}")
print(f"  mesh: {os.path.getsize(mesh_task)} bytes -> SHA-256 {mesh_sha256}")

print("Loading compact core DEX...")
with open(CORE_DEX_PATH, "rb") as f:
    dex_b64 = base64.b64encode(f.read()).decode("ascii")
mesh_dex_sha256 = sha256_file(MESH_DEX_PATH)
print(f"  core: {os.path.getsize(CORE_DEX_PATH)} bytes -> {len(dex_b64)} base64 chars")
print(f"  remote mesh DEX: {os.path.getsize(MESH_DEX_PATH)} bytes -> SHA-256 {mesh_dex_sha256}")

# Build plugin
print("Assembling plugin...")
plugin_code = f'''# Auto-generated blur-faces.plugin
# DEX runtime hooks
EMBEDDED_DEX_BASE64 = "{dex_b64}"

# Native library fetched once from the plugin distribution endpoint.
NATIVE_URL = "{NATIVE_URL}"
NATIVE_SHA256 = "{native_sha256}"

# Stable five-point root and optional dense mesh fetched from verified endpoints.
MODEL_PARAM_URL = "https://makey.dev/blur-faces/retinaface_mnet025_5kps.param"
MODEL_PARAM_SHA256 = "{param_sha256}"
MODEL_BIN_URL = "https://makey.dev/blur-faces/retinaface_mnet025_5kps.bin"
MODEL_BIN_SHA256 = "{bin_sha256}"
MESH_MODEL_URL = "https://makey.dev/blur-faces/face_landmarker.task"
MESH_MODEL_SHA256 = "{mesh_sha256}"
MESH_NATIVE_URL = "https://makey.dev/blur-faces/libmediapipe_tasks_vision_jni.so"
MESH_NATIVE_SHA256 = "{mesh_native_sha256}"
MESH_DEX_URL = "https://makey.dev/blur-faces/mediapipe-face-landmarker.dex"
MESH_DEX_SHA256 = "{mesh_dex_sha256}"

# Loader modules
{load_file(os.path.join(PLUGIN_DIR, "metadata.py"))}

{load_file(os.path.join(PLUGIN_DIR, "imports.py"))}

{load_file(os.path.join(PLUGIN_DIR, "constants.py"))}

{load_file(os.path.join(PLUGIN_DIR, "dex.py"))}

{load_file(os.path.join(PLUGIN_DIR, "utils.py"))}

{load_file(os.path.join(PLUGIN_DIR, "native.py"))}

{load_file(os.path.join(PLUGIN_DIR, "plugin.py"))}
'''

# Write plugin
output_path = os.path.join(ROOT_DIR, "build", "plugin", "blur-faces.plugin")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    f.write(plugin_code)

size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"✓ Built: {output_path}")
print(f"  Size: {size_mb:.2f} MB")

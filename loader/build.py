#!/usr/bin/env python3
"""Package the embedded core and SHA-pinned remote MediaPipe assets."""

import base64
import hashlib
import os

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PLUGIN_DIR)
CORE_DEX = os.path.join(ROOT_DIR, "build", "dex", "core.dex")
MP_DEX = os.path.join(ROOT_DIR, "build", "dex", "mesh-runtime.dex")
MP_NATIVE = os.path.join(ROOT_DIR, "build", "mesh-runtime", "libmediapipe_tasks_vision_jni.so")
MP_MODEL = os.path.join(ROOT_DIR, "model", "face_landmarker.task")


def load_file(path):
    with open(path, "r", encoding="utf-8") as source:
        return source.read()


def sha256(path):
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


for required in (CORE_DEX, MP_DEX, MP_NATIVE, MP_MODEL):
    if not os.path.isfile(required):
        raise FileNotFoundError(f"generated build input missing: {required}")

with open(CORE_DEX, "rb") as source:
    core_b64 = base64.b64encode(source.read()).decode("ascii")

plugin_code = f'''# Auto-generated blur-faces.plugin
EMBEDDED_DEX_BASE64 = "{core_b64}"
EMBEDDED_DEX_SHA256 = "{sha256(CORE_DEX)}"

# MediaPipe 0.10.29 runtime assets are fetched privately and SHA-verified.
MP_MODEL_URL = "https://makey.dev/blur-faces/face_landmarker.task?sha256={sha256(MP_MODEL)}"
MP_MODEL_SHA256 = "{sha256(MP_MODEL)}"
MP_NATIVE_URL = "https://makey.dev/blur-faces/libmediapipe_tasks_vision_jni.so?sha256={sha256(MP_NATIVE)}"
MP_NATIVE_SHA256 = "{sha256(MP_NATIVE)}"
MP_RUNTIME_DEX_URL = "https://makey.dev/blur-faces/mediapipe-face-landmarker.dex?sha256={sha256(MP_DEX)}"
MP_RUNTIME_DEX_SHA256 = "{sha256(MP_DEX)}"

{load_file(os.path.join(PLUGIN_DIR, "metadata.py"))}

{load_file(os.path.join(PLUGIN_DIR, "imports.py"))}

{load_file(os.path.join(PLUGIN_DIR, "constants.py"))}

{load_file(os.path.join(PLUGIN_DIR, "dex.py"))}

{load_file(os.path.join(PLUGIN_DIR, "utils.py"))}

{load_file(os.path.join(PLUGIN_DIR, "plugin.py"))}
'''

output = os.path.join(ROOT_DIR, "build", "plugin", "blur-faces.plugin")
os.makedirs(os.path.dirname(output), exist_ok=True)
with open(output, "w", encoding="utf-8") as target:
    target.write(plugin_code)
print(f"Built {output} ({os.path.getsize(output)} bytes)")

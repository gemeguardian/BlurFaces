# Blur Faces 1.0.0 — Current Handoff

Updated: 2026-09-21

## Current implementation

The release version comes from `BlurFaces/metainfo.yml`: **1.0.0**. The old v3 label referred to development architecture, not the current release artifact.

- `head_detector.cpp`: YOLOv8n, 320×320 RGB, NCNN CPU with **two** threads. Luma CLAHE/EMA in low light, normalized input, a validated 5×2100 float output, geometry guards and NMS. The old MobileNet skin/hemoglobin/anchor filters below are no longer implemented.
- `bytetrack.cpp`: native confirmation and low-score association. `detection_result.h` preserves inference errors and rejects output exceeding the four-mask renderer capacity rather than truncating heads.
- `Main.java`: hooks round-camera preview and encoder; validates native results before publication. Failures invalidate Java geometry and freshness, forcing full-frame protection until recovery. Java smoothing accepts all valid native-confirmed heads without reapplying confidence thresholds. Optical flow and bounded prediction remain.
- Camera switches reset PBOs, queues and native/Java tracks; results captured before the switch are checked again after inference. The current barrier constant is **three frames**, not the historical one-frame claim. Its duration is device-dependent.
- `runtime.py`: stages four bundled assets with SHA-256 checks into app-private runtime directories. No model downloads and no asset-tree writes. Changed native/model bundles require an app restart if a core is already loaded.
- Debug capture is opt-in via Python settings, never restored on reload/restart, expires after five minutes (checked on sampling/status). A bounded single writer saves up to two samples/s into app-private no-backup storage: raw top-left RGBA, candidate/native-track JPEG crops and JSON including read-only Java track state. Limit: 300 files / 32 MiB; old samples are pruned after 24h on configuration/startup; a UI action stops capture and deletes owned files. No uploads. These are not final shader/encoder masks. At startup, legacy captures matching the old filename format are also removed; unrelated files/symlinks are preserved.
- Settings: default blur, mask mode and sensitivity **35/25/18%**. Mask scale is **100%**, resolution is host-selected. The internal synthetic CPU self-test makes no claim to run NCNN, GPU shaders or the encoder.

## Verification and remaining work

See [README](README.md) for build/test commands and their scope. Packaging verifies DEX, ELF, both NCNN files and current Python source. JVM tests execute current Java code; host-native tests execute ByteTrack/result policy with UBSan. These do **not** verify Android hook compatibility, actual NCNN recall/latency or encoded video privacy.

[Device validation](DEVICE_VALIDATION.md) is required before release. Treat startup/asset/hook failures as unavailable protection, not as a protected camera. No detector can guarantee zero missed heads.

## Historical snapshot — not current behavior or verified measurements

The remainder records the 2026-09-20 design. Its latency, one-frame switching, UI, filter and v3 artifact claims are obsolete/unverified; use the current summary above instead.

## Overview

Blur Faces v3.0.0 is a 100% offline, zero-network privacy plugin for exteraGram / Telegram Android. It intercepts camera frames during video-message recording (round videos) and live camera previews, detecting and blurring human faces and heads from any angle (360° detection) before frames reach the preview renderer or video encoder.

Key invariants:
- **Zero-network**: No network requests at any point. All models and native binaries are bundled inside the `.elyx` archive.
- **Fail-closed**: If the native detector or tracking state is not ready, frames remain fully covered by Gaussian blur to prevent any visual privacy leaks.
- **Ultra-low latency**: Hardware NEON SIMD inference via Tencent NCNN and asynchronous OpenGL PBO readbacks ensure <15ms processing latency per frame without UI stalls.
- **Instant camera flipping**: Camera switch barrier reduced to 1 frame (~12ms) with instant PBO queue flushing.

---

## Architecture Breakdown

### 1. Native C++ Core (`src/`)

The native engine is compiled as `libblur_faces.so` for `arm64-v8a`:

- **NCNN 360° Head Detector (`src/head_detector.cpp`, `src/head_detector.h`)**:
  - Lightweight convolutional neural network optimized for mobile ARMv8 CPU with NEON.
  - Detects heads and faces at all angles (front, profile, 90° yaw, tilted, turned back, upside-down).
  - Statically links `libncnn.a` with OpenMP limited to 1 worker thread to avoid CPU core contention.
  - Single input model (`head_det.param`, `head_det.bin`).

- **ByteTrack Multi-Object Tracking (`src/bytetrack.cpp`, `src/bytetrack.h`)**:
  - Two-stage association using Kalman filtering and IoU matching.
  - Discards unconfirmed single-frame false detections while maintaining persistent tracks through brief occlusions.
  - Anti-overshoot braking and yaw-aware profile hold.

- **Multi-Cue Biological False-Positive Suppression (`src/head_detector.cpp`)**:
  - **BT.601 Skin Locus**: Checks $C_r \in [129, 178]$ and $C_b \in [77, 130]$ across dark and light skin tones (Fitzpatrick I–VI).
  - **Euclidean Chrominance Gating**: Rejects pants, knees, denim, and office chairs by enforcing $\Delta_{\text{chroma}} = \sqrt{(C_r - 128)^2 + (C_b - 128)^2} \ge 2.85$.
  - **Non-Biological Yellow Filter**: Drops saturated yellow ceramics and mugs ($C_b < 75$, $R > 90$, $G > 70$).
  - **Spatial Texture Energy Filter**: Rejects uniform flat architectural planes (doors, blank walls, cardboard) via Laplacian gradient energy ($\ge 3.20$).
  - **Micro-Scale Anchor Pruning**: Deactivates Feature 2 anchors (`feature_idx < 2`), cutting 76% of anchor evaluations and eliminating false triggers on small hoops, knobs, and rings.
  - **Selfie Close-Up Extension**: Allows bounding boxes up to 100% of frame ($1.0 \times 1.0$), with skin luminance contrast check to separate large faces from plain cardboard boxes.
  - **Canine / Pet Rejection**: Detects and rejects dogs (e.g. Yorkshire terriers) using biological hemoglobin balance $(R-G)/(R-B) \ge 0.58$, canine snout contrast ratios, and torso aspect-ratio limits.

### 2. Java / OpenGL EGL Pipeline (`src/main/java/com/makey/blurfaces/g2/`)

- **Hooking & Lifecycle (`Main.java`)**:
  - Hooks `SurfaceTexture.updateTexImage()` directly at the camera frame latch boundary.
  - Camera switch handler (`noteCameraSwitch`):
    * Updates `LAST_CAMERA_SWITCH_NANOS` to immediately drop pre-switch frames from the inference queue.
    * Calls `CleanFrameTap.resetPbo()` to flush pending asynchronous PBO readback buffers.
    * Sets `CAMERA_SWITCH_BARRIER_MIN_FRAMES = 1`, providing instant transition (~12ms) without lingering blur artifacts.
  - Shader hooking: intercepts draw calls in `InstantCameraView.CameraGLThread`, binds custom multi-pass blur shaders, and restores original GL state after drawing.

- **Asynchronous PBO Frame Tap (`CleanFrameTap.java`)**:
  - Downsamples the OES camera texture into an FBO.
  - Uses double-buffered OpenGL Pixel Buffer Objects (PBOs) for non-blocking asynchronous DMA readbacks (`glReadPixels` into PBO memory).
  - Multi-pass separable Gaussian blur produces high-quality, artifact-free blur textures.

- **Native JNI Bridge (`NativeBridge.java`)**:
  - Loads `libblur_faces.so` dynamically into the host process.
  - Exposes JNI methods for frame detection, tracking, reconfiguring confidence, and lifecycle reset.

### 3. Elyx Runtime & Python Bridge (`BlurFaces/`)

- **`BlurFaces/runtime.py`**:
  - 100% offline runtime loader.
  - Staging: Copies `core.dex`, `libblur_faces.so`, `head_det.param`, and `head_det.bin` from plugin assets into app-private cache with SHA-256 validation.
  - Loads `core.dex` via `DexClassLoader` and invokes entry point `Main.initAndStart()`.
  - Implements `DexRuntime` with keyword arguments, deferred shutdown, and soft confidence reconfiguration.

- **`BlurFaces/main.py`**:
  - Standard Elyx UI settings:
    * Master Switch: Enable / Disable.
    * Mask Mode: Gaussian Blur, Pixelate, Solid Color.
    * Face Mask Scale: Compact (65%), Standard (82%), Wide (100%).
    * Detection Range: Normal (45%), Extended (35%), Far (25%).
    * Round Video Resolution: Auto, 320x320, 384x384, 448x448, 512x512.
    * Privacy Self-Test button.

---

## Build System

The build script `build.sh` is portable and environment-aware:

```bash
# Variables (with automatic fallback detection):
export JAVA_HOME="/usr/lib/jvm/java-17-openjdk-amd64"
export ELYX_BUILDER="/home/PluginDev/.venv/bin/elyb"
export ANDROID_NDK_ROOT="/home/PluginDev/.android-sdk/ndk/29.0.14206865"

# Build command:
./build.sh
```

Steps executed:
1. Compiles native C++ code with Android NDK (`Android.mk`, `Application.mk`) -> `libs/arm64-v8a/libblur_faces.so`.
2. Builds Java code via Gradle -> `build/dex/core.dex`.
3. Validates and hashes assets via `build.py` -> `BlurFaces/asset_hashes.py`.
4. Packages final installable artifact via `elyb` -> `builds/blur_faces-3.0.0.elyx`.

---

## Testing & Verification

Run the test suite:

```bash
python3 -m pytest tests/tests_*.py
python3 tests/tests_tracking_behavior.py
python3 tests/tests_elyx_contract.py
```

# Blur Faces v3.0.0 — Agent Instructions & Repository Guidelines

This document provides instructions for agents and developers working on the `blur-faces` plugin repository.

## Project Purpose

`blur-faces` is a privacy plugin for exteraGram / Telegram Android that automatically blurs or pixelates human faces and heads in real-time camera previews and recorded round-video messages.

## Key Invariants

1. **100% Offline & Zero-Network**:
   - The plugin must NEVER perform any network calls, socket connections, or HTTP downloads.
   - All models (`head_det.param`, `head_det.bin`), native libraries (`libblur_faces.so`), and Dalvik bytecode (`core.dex`) must remain bundled directly inside the `.elyx` archive.

2. **Fail-Closed Security**:
   - If the detector or tracking state is uninitialized or in transition (e.g. during a camera switch or startup), the frame must remain covered by blur.
   - Never show an unblurred frame if there is any chance a human face is present.

3. **Sub-15ms Real-Time Performance**:
   - Inference uses Tencent NCNN with ARM64 NEON SIMD optimizations.
   - Frame readback uses asynchronous double-buffered OpenGL Pixel Buffer Objects (PBOs) in `CleanFrameTap.java` to prevent stalling the camera GL thread.

4. **Zero Lingering Blur on Camera Flip**:
   - Camera switch barrier is 1 frame (~12ms).
   - Any camera flip event must immediately invoke `noteCameraSwitch()` and `CleanFrameTap.resetPbo()` to flush pending async buffers.

5. **Strict False-Positive Suppression**:
   - The C++ engine in `src/head_detector.cpp` enforces:
     * BT.601 skin chrominance locus check ($C_r \in [129, 178]$, $C_b \in [77, 130]$).
     * Euclidean chrominance radius $\Delta_{\text{chroma}} \ge 2.85$ (rejects pants, denim, knees, office chairs).
     * Non-biological yellow filter ($C_b < 75$, $R > 90$, $G > 70$) (rejects ceramic mugs and yellow objects).
     * Spatial texture energy threshold $\ge 3.20$ (rejects blank walls, flat doors, cardboard).
     * Deactivation of Feature 2 anchors (`feature_idx < 2`) (eliminates micro-clutter false positives).
     * Biological hemoglobin balance ratio $(R - G) / (R - B) \ge 0.58$ and canine snout contrast gating (rejects dogs and pets).

## Repository Structure

```text
/home/PluginDev/plugins/blur-faces/
├── Android.mk / Application.mk  # NDK C++ build definition
├── build.gradle / gradlew        # Java / DEX build configuration
├── build.sh                      # Unified build script (NDK + Gradle + elyb)
├── build.py                      # Asset validation, staging & hash generator
├── HANDOFF.md                    # Architecture handoff documentation
├── README.md                     # Project overview and quickstart
├── src/                          # Native C++ & Java source code
│   ├── head_detector.cpp/.h      # NCNN inference & multi-cue filters
│   ├── bytetrack.cpp/.h          # Multi-object ByteTrack tracker
│   ├── main.cpp                  # JNI entry points
│   └── main/java/                # Android Java hooks & GL shaders
│       └── com/makey/blurfaces/g2/
│           ├── Main.java         # Camera lifecycle, hooks & shader swap
│           ├── CleanFrameTap.java# PBO async readback & Gaussian blur
│           └── NativeBridge.java # JNI loader bridge
├── BlurFaces/                    # Elyx plugin bundle contents
│   ├── metainfo.yml              # Plugin metadata
│   ├── main.py                   # Elyx UI & settings
│   ├── runtime.py                # Asset extraction & DexClassLoader runner
│   ├── strings/                  # Localization (ru, en)
│   └── assets/                   # Bundled binaries & NCNN models
├── legacy/                       # Retired components (SCRFD, old landmarker tests)
└── tests/                        # Unit and regression test contracts
```

## Build Environment

To build the plugin:
```bash
./build.sh
```

The script automatically detects or accepts:
- `JAVA_HOME` (JDK 17+)
- `ELYX_BUILDER` or `ELYB_PATH` (path to `elyb`)
- `ANDROID_NDK_ROOT` or `NDK_PATH` (path to Android NDK)

The output installable artifact is generated at:
`builds/blur_faces-3.0.0.elyx`

## Verification & Testing

Before completing any changes:
```bash
# 1. Run Python test contracts
python3 -m pytest tests/tests_*.py
python3 tests/tests_tracking_behavior.py
python3 tests/tests_elyx_contract.py

# 2. Build the plugin and verify hashes
./build.sh
```

# Blur Faces v1.0.0 — Agent Instructions & Repository Guidelines

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

3. **Real-Time Performance**:
   - Inference uses Tencent NCNN (YOLOv8n head detector, 320x320 input, CPU, 2 threads) with ARM64 NEON SIMD optimizations.
   - Frame readback uses asynchronous double-buffered OpenGL Pixel Buffer Objects (PBOs) in `CleanFrameTap.java` (192x192) to prevent stalling the camera GL thread.
   - Measured on device (SM8750, 2026-09-22): 50–120 ms per inference, ~7–8 detector fps, 40–60% of captured frames dropped. Any tracker constant expressed in "frames" must be read against this cadence, not 30 fps.

4. **Zero Lingering Blur on Camera Flip / New Session**:
   - Camera switch barrier is `CAMERA_SWITCH_BARRIER_MIN_FRAMES` frames.
   - `activateSource()` must call `noteCameraSwitch()` on **every** source activation (not only flips): it resets the native ByteTracker, whose lost tracks age by detector frame count rather than wall time, and calls `CleanFrameTap.resetPbo()` to flush pending async buffers.
   - `NativeBridge.reset()` is lock-free (it runs on the GL/UI thread): it only bumps a reset generation that the inference worker applies before its next tracker update. A frame whose inference overlapped a reset returns `STALE_AFTER_RESET` (-6) and is dropped. Lost tracks stay re-identifiable for `ByteTracker::kMaxTimeLostFrames` (8 detector frames ≈ 1 s).

5. **False-Positive Suppression (current implementation)**:
   - The detector is a learned model; there are no hand-written skin/texture/colour filters anymore (they were removed together with the anchor-based model in `f2f0187`). Do not re-document them without re-adding them.
   - `src/head_detector.cpp` applies only geometric sanity gates on YOLO boxes: min size 5%, aspect 0.35–2.5, small-clutter (`w<0.11 || h<0.13` needs `prob>=0.38`), large-clutter (`area>0.35` needs `prob>=0.60`).
   - Low-light CLAHE gain is capped at 2.5x; JNI relaxes only `high`/`low` thresholds in the dark, never `instant`.
   - `src/bytetrack.cpp` provides the temporal suppression via a Wald SPRT: each detector frame adds `clamp(logit(score) - logit(neutral), ±kEvidenceCap) × size_reliability`, where `neutral = (high+low)/2` and `size_reliability = clamp(area / kReliableArea, kMinSizeReliability, 1)`. A candidate is published once accumulated log-odds reach `kLogOddsConfirm` (1.5) — 2 frames for a confident selfie head, 3 for a confident 10%-wide head, never for the recorded backpack trace (0.71/0.45/0.42 in a 10%×9% box). IoU is **not** evidence (a static object has perfect IoU). Single-frame instant activation requires `score >= instant_thresh` (floor 0.70) **and** `area >= kInstantMinArea`. Confirmed tracks whose log-odds fall below 0 stop being published. Lost tracks are coasted only if they had `>= kMinHitsForCoast` (6) observations and for at most `kMaxCoastPublishFrames` (4) frames; the Java `SourceTracks` hold adds its own time-based coast on top.
   - `tests/native_tracking_test.cpp` replays the false-positive traces recorded on device (15:46 empty room, 16:28 backpack) and the real-face trace (16:27 front camera); keep it green when tuning thresholds.

## Repository Structure

```text
/home/PluginDev/plugins/blur-faces/
├── Android.mk / Application.mk  # NDK C++ build definition
├── build.gradle / gradlew        # Java / DEX build configuration
├── build.sh                      # Unified build script (NDK + Gradle + elyb)
├── build.py                      # Asset validation, staging & hash generator
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
`builds/blur_faces-1.0.0.elyx`

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

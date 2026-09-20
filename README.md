# BlurFaces

<div align="center">

**Real-Time, Zero-Network Privacy Shield for exteraGram / Telegram Android**

[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Platform](https://img.shields.io/badge/platform-Android%20%7C%20ARM64--v8a-blue.svg)]()
[![Inference](https://img.shields.io/badge/engine-Tencent%20NCNN-orange.svg)]()
[![Code Size](https://img.shields.io/badge/code%20size-413%20KB%20%7C%209.1k%20LoC-informational.svg)]()
[![Bundle Size](https://img.shields.io/badge/bundle-2.0%20MB-purple.svg)]()
[![Privacy](https://img.shields.io/badge/policy-100%25%20Offline%20%7C%20Fail--Closed-red.svg)]()

</div>

---

## Highlights

- **100% Offline & Private**: Zero external dependencies, zero network requests. All weights and native libraries are bundled within the `.elyx` package.
- **Fail-Closed Guarantee**: Before detection stabilizes or across instant camera switches, full-frame privacy blur guards the sensor stream.
- **Sub-12ms Inference**: High-performance Tencent NCNN C++ pipeline with ARM64 NEON FP16 SIMD optimizations.
- **Extreme Low-Light Resilience**: Adaptive logit prior ($\beta(\mu)$), dynamic gray-world reference, Weber-normalized 3D relief verification, and preprocessor gain scaling.
- **Robust Multi-Object Tracking**: ByteTrack 2-stage association with decoupled Kalman filtering and coasting protection.
- **Native UI Integration**: Integrated round-video pill control with live haptic feedback, theme synchronization, and dynamic SVG vector icon packs.

---

## Architecture

```
[ Camera OES Texture ] 
         │
         ▼
[ CleanFrameTap (GLSL) ] ──▶ Double-Buffered Async PBO
         │
         ▼
[ Native C++ Engine (libblur_faces.so) ]
   ├── NCNN FPN Head Detector (320x320 FP16)
   ├── Biological Evidence Fusion & Veto Filters
   └── ByteTrack Multi-Object Kalman Tracking
         │
         ▼
[ Camera / Encoder Shaders ] ──▶ Smooth Mask Rendering (Blur / Pixelate / Solid)
```

---

## Building

### Requirements
- JDK 17+
- Android NDK (r25c+)
- `elyb` (ElyxBuilder)

```bash
./build.sh
```

The output package will be generated at:
```
builds/blur_faces-3.0.0.elyx
```

---

## Verification

```bash
# Run pytest contracts
pytest tests/tests_*.py

# Run tracking and elyx packaging contracts
python3 tests/tests_tracking_behavior.py
python3 tests/tests_elyx_contract.py
```

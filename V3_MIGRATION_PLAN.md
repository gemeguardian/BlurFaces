# BlurFaces v3.0 Architecture Migration Specification & Status

## 1. Goal & Context
Migrated `/home/PluginDev/plugins/blur-faces` from MediaPipe Tasks Vision (Google AAR + downloaded 6.5MB Google model) to a 100% offline, native C++ NCNN pipeline with Head Detection + ByteTrack + Fail-Closed Privacy Policy.
- Backup of v2.3.2 is at `/home/PluginDev/plugins/blur-faces-backup-v2`.
- Target artifact: `/home/PluginDev/plugins/blur-faces/builds/blur_faces-3.0.0.elyx` (4.7 MB).
- Status: **RELEASED & DELIVERED**.

---

## 2. Completed Architecture Components

1. **Pretrained Offline NCNN Head Detector:**
   - Bundled permanently into repo assets:
     - `BlurFaces/assets/model/head_det.param` (8.7 KB)
     - `BlurFaces/assets/model/head_det.bin` (2.7 MB)
   - Anchor-based 3-scale FPN detecting heads at all angles (frontal, 90° profile, and back of head 180°).

2. **Native C++ Engine (`libblur_faces.so` for `arm64-v8a`):**
   - Built with Android NDK r29 and statically linked NCNN + OpenMP runtime.
   - `src/head_detector.cpp`: FP16 NEON inference, letterbox scaling, and anchor decoding.
   - `src/bytetrack.cpp`: 2-stage IoU association with Kalman filter and fail-closed oval expansion (+8% per lost frame).
   - `src/main.cpp`: JNI interface for `NativeBridge` with in-place buffer flipping.

3. **Java / DEX Layer (`core.dex`):**
   - Cleaned of all Google/MediaPipe dependencies.
   - `NativeBridge.java` JNI declarations.
   - OpenGL shaders, camera control hooks, and frame capture loops preserved.

4. **Elyx Plugin Package & Python Runtime:**
   - `BlurFaces/runtime.py` stages native `.so`, `.dex`, and model files directly from plugin assets into application cache. Zero network downloads.
   - Bumper version to `3.0.0` in `BlurFaces/metainfo.yml`.
   - All 10 contract and regression test suites passing.

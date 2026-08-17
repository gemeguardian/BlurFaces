# Blur Faces Plugin Handoff

Updated: 2026-08-15

## Current Architecture

Version `1.9.11` uses MediaPipe Face Landmarker `0.10.29` as its only face
geometry source:

- `Delegate.GPU`; there is no CPU fallback.
- `RunningMode.LIVE_STREAM` with one in-flight input and one latest-frame slot.
- Up to four faces, represented by PCA ellipses derived from the 36-point face oval.
- One source-bound result map shared by preview and encoder.
- Positive geometry is held for at most 350 ms; the plugin is not fail-closed.

If GPU initialization fails, hooks are not retained and the host camera remains
untouched. Do not report privacy protection in that state.

## Capture Point

`Main.java` hooks `SurfaceTexture.updateTexImage()` after the host latches a new
camera frame. Capture runs only when the receiver is the exact object in a live
`InstantCameraView.CameraGLThread.cameraSurface` array and its slot equals the
active `InstantCameraView.surfaceIndex`.

At this boundary the camera EGL context is current and the host has not yet
created its encoder snapshot or rendered preview. `CleanFrameTap` renders the
active OES texture into a 320x320 FBO using the host MVP, SurfaceTexture matrix,
and texture coordinates. The `glReadPixels` rows are reversed before creating
the MediaPipe `MPImage`.

Source keys contain surface slot, host generation, and SurfaceTexture identity.
Results from retired sources are dropped, including results completing after a
camera switch.

## Rendering

Preview and encoder use separate context-owned shader programs. Host program and
location fields are swapped in each draw before-hook and restored in that same
draw's after-hook. The plugin does not leave its program installed between draws.

The shaders blur and pixelate only inside the face-aligned ellipse. The same
geometry drives the visible preview and encoded round video.

## Runtime Assets

The installable plugin embeds only the compact core DEX. It downloads and
SHA-verifies:

- `face_landmarker.task`
- `mediapipe-face-landmarker.dex`
- `libmediapipe_tasks_vision_jni.so` for `arm64-v8a`

Each new `DexClassLoader` receives the JNI library through a unique load
directory while retaining the canonical library basename. This avoids native
ownership conflicts during plugin hot reload.

Public asset URLs are under `https://makey.dev/blur-faces/`. The hashes embedded
in the current plugin were verified against those public files on 2026-08-15.

## Build

Run:

```bash
./build-native.sh
```

Despite its historical name, this is now a clean Gradle/package build. It does
not invoke the NDK or compile the legacy SCRFD sources. Output:

```text
build/plugin/blur-faces.plugin
```

The legacy C++ and model files remain only as reference material and are not in
the active build or runtime.

## Device Validation

Use `DEVICE_VALIDATION_1.9.11.md`. Required checks include GPU startup, front/back
camera switching, rapid yaw/roll/translation, entry/exit, disable/re-enable,
hot reload, preview/encoder parity, and public asset hash availability.

Do not claim TikTok-level latency or stability until those cases pass on the
target arm64 device. The OES FBO still performs a 320x320 `glReadPixels`, so only
device measurements can establish actual end-to-end latency.

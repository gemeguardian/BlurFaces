# Blur Faces Plugin Handoff

Updated: 2026-08-18

## Current Architecture

Version `1.0.0` is a source-based Elyx plugin using MediaPipe Face Landmarker `0.10.29` as its only face
geometry source:

- `Delegate.GPU`; there is no CPU fallback.
- Synchronous `RunningMode.VIDEO` on one GPU worker with one latest-frame slot.
- Up to four faces, represented by PCA ellipses derived from the 36-point face oval.
- Source-local, order-independent tracks shared by preview and encoder.
- Each positive track is held independently for at most 350 ms and predicted for at most 120 ms.
- Detection/presence thresholds use `0.60` to reject object pareidolia; tracking uses `0.50`.

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

The installable `.elyx` bundles and SHA-verifies exactly three runtime binary
payloads, so core code, the MediaPipe Java runtime, and JNI remain available
offline:

- generated `core.dex`
- generated `mediapipe-runtime.dex`
- `libmediapipe_tasks_vision_jni.so` for `arm64-v8a`

The Face Landmarker model is intentionally not bundled. On first load, the
plugin downloads only version `float16/1` from Google's official pinned URL:

`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

Its fixed expected SHA-256 is
`64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The download uses Java `URLConnection` for Chaquopy compatibility, explicit
connect/read timeouts, streaming output to a temporary file, `fsync`, hash
verification, atomic replacement, and read-only permissions. A verified model
is reused from the stable app-private `blur_faces_models` directory without a
network request. Missing, corrupt, or failed downloads are fail-closed: no Java
classes are loaded and no camera hooks are installed; logs identify the failure
and recommend checking the network and reloading the plugin.

Python accesses payloads through documented `from elyx import assets`, copies
them into one unique canonical app-private load directory, verifies fixed
build-generated SHA-256 values after both source access and staging, and chmods
DEX/JNI files read-only before class loading. The host runtime loader parents the
MediaPipe runtime loader, which parents the child core loader. The Python logger
proxy is strongly retained for the active runtime and released after Java
`onUnload`; Java also clears its static logger reference.

## Build

Run:

```bash
./build.sh
```

This clean command generates core/runtime DEX, extracts the arm64 MediaPipe JNI,
stages and validates the three bundled assets, updates fixed hashes, invokes
ElyxBuilder with AST validation, and runs the Elyx archive contract. It does not
invoke the NDK or compile the legacy SCRFD sources. Output:

```text
builds/blur_faces-1.0.0.elyx
```

The legacy C++ and model files remain only as reference material and are not in
the active build or runtime.

## Device Validation

Use `DEVICE_VALIDATION_1.0.0.md` as the current device checklist. Required checks
include first-load model download,
cached-model airplane-mode startup, corrupt-cache fail-closed behavior, GPU startup, front/back
camera switching, rapid yaw/roll/translation, entry/exit, disable/re-enable,
hot reload, preview/encoder parity, bundled asset hash verification, and a clean
install with no prior Blur Faces cache.

Do not claim TikTok-level latency or stability until those cases pass on the
target arm64 device. The OES FBO still performs a 320x320 `glReadPixels`, so only
device measurements can establish actual end-to-end latency.

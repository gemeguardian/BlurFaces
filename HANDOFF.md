# Blur Faces Plugin Handoff

Updated: 2026-08-08

## Goal

Build an exteraGram plugin named `blur-faces` that automatically pixelates or blurs faces in Telegram round-video recording:

- The live round-video preview must blur detected faces.
- The encoded and sent round video must retain the face blur.
- The background must remain unmodified.
- It must remain stable across plugin reloads and must not crash exteraGram.
- The target visual behavior is TikTok-like: the blur stays attached to a moving face with minimal visible latency.

The target device observed in logs is a OnePlus `CPH2653`, Android 16, arm64. The application process is `com.exteragram.messenger`.

## Project Locations

- Project root: `/home/PluginDev/plugins/blur-faces`
- Java renderer hooks: `src/main/java/com/makey/blurfaces/Main.java`
- JNI declarations: `src/main/java/com/makey/blurfaces/NativeBridge.java`
- Native JNI and state: `src/main.cpp`
- SCRFD implementation: `src/scrfd.cpp`, `src/scrfd.h`
- Plugin loader: `loader/plugin.py`
- Plugin metadata/version: `loader/metadata.py`
- Plugin packing script: `loader/build.py`
- Native build script: `build-native.sh`
- Native build config: `Android.mk`, `Application.mk`
- Output plugin: `build/plugin/blur-faces.plugin`
- Native output: `build/arm64-v8a/libblur_faces.so`
- SCRFD model: `model/scrfd_10g-opt2.param`, `model/scrfd_10g-opt2.bin`

## Distribution

The plugin is small and downloads model/native assets from HTTPS.

- Public native URL: `https://makey.dev/blur-faces/libblur_faces.so`
- Server-native path: `/var/www/makey-downloads/blur-faces/libblur_faces.so`
- Model URLs are embedded by `loader/build.py`.
- The plugin verifies SHA-256 for every remote asset.
- Latest published native SHA as of this handoff: `355bfc35cb917de02719a9abb0f8c44c9f52301d2e1cb193893a97624905228b`

The loader uses a versioned cache name and a UUID copy before `System.load`, to avoid Android linker/ClassLoader `already opened` failures on hot reload:

```text
libblur_faces_<plugin_version>.so
libblur_faces_load_<uuid>.so
```

Device cache directory:

```text
/data/user/0/com.exteragram.messenger/files/blur_faces_cache/
```

## Building

Java DEX build:

```bash
./gradlew buildDex
```

Native build plus plugin pack:

```bash
./build-native.sh
```

After a native change, publish the exact file before sending the plugin, otherwise the device rejects it with a SHA mismatch:

```bash
cp build/arm64-v8a/libblur_faces.so /var/www/makey-downloads/blur-faces/libblur_faces.so
python3 loader/build.py
curl -fsSL https://makey.dev/blur-faces/libblur_faces.so -o /tmp/blur_faces_remote.so
sha256sum /tmp/blur_faces_remote.so
```

The SHA printed by the final command must match the `NATIVE_SHA256` embedded in `build/plugin/blur-faces.plugin`.

## Sending Artifacts

Artifacts have been sent to Telegram Saved Messages using Telethon:

- Session: `/tmp/telethon_session.session`
- Credentials: `/opt/Goroku/config.json`
- Python environment: `/tmp/telethon-venv/bin/python`
- Latest plugin sent: `blur-faces 1.3.7`, message id `933992`
- Latest log received: message id `933991`, local copy `/tmp/latest_blur_log (1).txt`

The temporary send helper is `/tmp/send_blur_faces.py`.

## Current Architecture

### Rendering

`Main.java` hooks `GLES20.glShaderSource`.

For fragment shaders containing `samplerExternalOES`, it replaces the app shader with `FACE_BLUR_FS`:

- The replacement shader samples the original external camera texture normally.
- It pixelates only inside `uFaces[10]` UV rectangles.
- `uFaceCount` controls active rectangles.
- This path previously proved that the shader replacement works: an earlier full-frame shader mosaic affected the whole round preview.

`Main.java` also hooks `GLES20.glDrawArrays`.

Before full-screen `GL_TRIANGLE_STRIP, 0, 4` draws, it:

1. Verifies the current program exposes `uFaceCount`, so only the replacement camera shader is handled.
2. Reads the current viewport with `glReadPixels`.
3. Schedules SCRFD inference on a single background Java executor.
4. Uses the latest face rectangles for `uFaces` uniforms.

### Face Detection

- Detector: SCRFD-10G using NCNN.
- Native calls: `NativeBridge.init`, `NativeBridge.process`, `NativeBridge.getLastFaceCount`, `NativeBridge.getFaceRects`, `NativeBridge.cleanup`.
- Current Java code requests detection every matching draw (`DETECT_INTERVAL = 1`).
- Only one native detection is allowed in flight (`detectionPending`). This avoids queueing stale frames.
- Small viewports below `256x256` are ignored because the hook also sees unrelated surfaces such as `48x48`.
- The current implementation Y-flips face rectangles after `glReadPixels` before sending them to shader UV coordinates.
- Version `1.3.5` added simple two-point velocity extrapolation in Java between detector results. This code is still present in `Main.java`, but has not run successfully because current native builds crash during initialization.

### Native Threading

`SCRFD::load()` sets `net.opt.num_threads = 1`.

NCNN is supplied as a static library that was built with OpenMP. Therefore `Android.mk` must link:

```make
-fopenmp -static-openmp
```

Attempting to remove those flags fails at link time with unresolved `__kmpc_*`, `omp_get_thread_num`, and `kmp_*` symbols.

## What Was Confirmed Working

1. Plugin packaging, DEX loading, runtime download, SHA verification, model caching, and unique `.so` loading all work.
2. Native `System.load` works with the unique UUID path.
3. `NativeBridge.init` previously returned `rc=0` in versions before the current regression.
4. `glShaderSource` and `glDrawArrays` hooks install successfully.
5. OES shader replacement is called for both preview/encoder shaders:

```text
shader replaced: OES -> face-blur
```

6. The old full-frame shader test visibly pixelated the full round preview. Therefore the rendering hook can affect the video pipeline.
7. SCRFD sometimes detected `1 face` in round-video viewports (`448x448` and `1328x1328`).
8. A native SHA mismatch was fixed by publishing the matching `.so` before sending the plugin.

## Known Failures and Root Causes

### 1. Current blocker: native crash during SCRFD initialization

The latest received log is for `1.3.6`. It shows:

```text
init: System.load ok
Initializing SCRFD model...
Loading param: .../scrfd_10g-opt2.param
Fatal signal 11 (SIGSEGV)
```

The crash occurs inside `net.load_param()`, before `load_model()`, before `NativeBridge.init rc=0`, and before GL hooks are installed.

Previous variant `1.3.5` crashed slightly earlier, just after `Initializing SCRFD model...`, with `SIGABRT`.

The device backtraces refer to the plugin `.so` offsets but lack symbols because Android cannot read native symbols from the app data path. It is a native process crash; Java `try/catch` cannot recover from it.

Current source changes intended to avoid the regression:

- `scrfd.cpp`: no preliminary `net.clear()` in `SCRFD::load()`.
- `main.cpp`: if `g_scrfd` already exists, retain it rather than deleting/recreating it.
- `scrfd.cpp`: logs `Loading param` and `Loading model` to identify the exact phase.

Version `1.3.7` includes these changes but has not yet been tested on the device.

### 2. Face blur is unstable and lags behind moving face

When native initialization did work, logs showed repeated `0 faces` and occasional `1 faces` on various surfaces:

```text
detect frame ...: 0 faces (1328x1328)
detect frame ...: 0 faces (448x448)
detect frame ...: 0 faces (48x48)
detect frame ...: 1 faces (...)
```

The user observed that blur appears/disappears and moves with delay when the face moves quickly.

Why the current approach cannot reach TikTok-quality tracking by itself:

- `glReadPixels` reads a rendered framebuffer and forces a GPU/CPU synchronization.
- Inference starts after that readback, so detector coordinates are necessarily from an older frame.
- SCRFD is a detector, not a per-frame tracker.
- Preview and encoder use different surfaces and viewport sizes.
- The shader UVs are transformed by the app `uSTMatrix`; simple framebuffer UV conversion can be wrong for crop/rotation/mirror cases.

The current code has partial mitigations only:

- single in-flight detector job;
- no unbounded queue;
- ignore tiny surfaces;
- short stale-face hold;
- Y conversion;
- simple velocity prediction.

They are not a complete low-latency tracking solution.

### 3. Older EGL encoder hook crashed

An earlier approach hooked encoder EGL drawing (`onDrawEncoderFrame`). It caused surface state failures and `NULL_DEREF` in `libtmessages.49.so`. Do not restore that approach without an isolated EGL/FBO implementation and device testing.

### 4. Wrong readback region was fixed

An older build used `Math.min(viewportWidth, 96)` and read only the lower-left `96x96` region. It did not downscale the full image, so most faces were absent from input. This was corrected to full-viewport readback. Do not reintroduce crop-by-size as a downscale mechanism.

### 5. Cached native hash mismatches happened

The loader rejects mismatched downloads. Always publish `libblur_faces.so` first, verify it over HTTPS, then rerun `loader/build.py`, then send the generated plugin.

## Immediate Next Steps

1. Install and test the already sent `1.3.7` plugin, message `933992`.
2. Request only the log range beginning at `Initializing SCRFD model...` through either `Model initialized successfully` or `Fatal signal`.
3. If `1.3.7` still crashes in `load_param`, stop iterating on Java tracking. The detector backend must be made stable first.
4. Symbolize the plugin crash offsets from a locally retained unstripped `.so` or make a debug build with symbols. The NDK utilities were not found via basic PATH/glob lookup, so locate the appropriate `llvm-addr2line` binary or use another Android toolchain.
5. Compare the known working native `.so` with the crashing native `.so`. The known working plugin line was version `1.3.4` or earlier, which logged `NativeBridge.init rc=0` and reached detection. Obtain its exact artifact if still cached locally or on the device, and diff build flags/source.

## Correct Long-Term Design for TikTok-Like Behavior

Do not rely on CPU SCRFD plus window-surface `glReadPixels` as the primary tracker.

Required architecture:

1. Get frames from the actual camera source before preview/encoder rendering.
   - Prefer `CameraX`/Camera2 `ImageReader` YUV stream, or the app's camera frame callback if available.
   - Alternative: consume the camera OES texture through a dedicated GPU FBO, not from the display framebuffer.
2. Use a fast face tracker on every frame, not a full detector every frame.
   - Run SCRFD only to initialize/recover face boxes.
   - Track existing faces every frame with a lightweight landmark/model tracker, optical flow, or a GPU tracker.
3. Preserve and use the exact `SurfaceTexture` transform (`uSTMatrix`) for mapping tracker coordinates to the shader texture.
   - Hook or intercept `GLES20.glUniformMatrix4fv` when the camera renderer sets `uSTMatrix`.
   - Apply its crop/mirror/rotation transform to each tracked face box before uploading uniforms.
4. Run preview and encoder from the same tracked camera-space box data.
   - Do not detect separately from unrelated `448`, `1328`, and `48` surfaces.
5. Measure end-to-end latency on device.
   - Record timestamps for capture, tracking completion, uniform upload, and output frame.
   - Reject stale results rather than displaying old boxes.

This is the minimum path toward the requested no-visible-lag behavior. Exact zero-millisecond latency is physically impossible because a frame must first exist, but a direct camera tracker can maintain within roughly one frame instead of detector-sized delays.

## Useful Log Markers

Successful initialization should contain, in order:

```text
init: System.load ok
Initializing SCRFD model...
Loading param: ...
Loading model: ...
Model loaded successfully
Model initialized successfully
init: NativeBridge.init rc=0
Hook registered: glShaderSource
Hook registered: glDrawArrays
Face blur active ...
```

Current failure signature:

```text
Loading param: ...
Fatal signal 11 (SIGSEGV)
```

## Constraints

- Preserve unrelated user changes; the project may be dirty and the workspace root is not a Git repository.
- Do not use destructive Git commands.
- Use `apply_patch` for source edits.
- Do not claim TikTok-level tracking before it is demonstrated on the target device.
- Never ship a plugin after a native source change without verifying the public `.so` hash matches the plugin's embedded hash.

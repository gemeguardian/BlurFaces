# Blur Faces 2.3.0 device validation

## New 2.1-2.3 checks

- Verify `Protection style` applies the same way to preview and encoded output: Gaussian blur, pixelation, and solid mask.
- Verify logs periodically contain `Metrics state=...` with captured, processed, dropped, and inference latency counters.
- Verify the runtime reports `ACTIVE`, `DEGRADED`, `FAILED`, `STOPPING`, and `DISABLED` transitions honestly.
- Verify a delayed detector result never replaces the active camera source and that a stale source is rendered as full-frame protected until a fresh result arrives.
- On a slow device, verify dropped frames increase without unbounded latency; the latest frame must replace queued input.

This release uses MediaPipe Face Landmarker 0.10.29 GPU VIDEO mode as the sole face geometry source for both preview and encoder. Creation, inference, and shutdown run on one worker. There is deliberately no CPU or SCRFD fallback. If GPU setup fails, logs must state that the host camera is untouched and no blur may be claimed active.

## Install and startup

1. Install `builds/blur_faces-1.0.0.elyx`, enable the plugin, and open the round-video camera.
2. Confirm `MediaPipe Face Landmarker 0.10.29 armed with GPU + VIDEO on one worker` appears before capture logs.
3. Confirm `Encoder hooks registered`, `Encoder blur shader ready`, and then `Encoder face protection active` while recording a visible face.
4. Disable the setting while the camera is open. Preview and encoded output must immediately return to host rendering; re-enable and verify startup occurs once.
5. Start disabled and restart the app. No runtime download, DEX startup, or hook log may appear.
6. If GPU setup fails, confirm `GPU Face Landmarker unavailable; host camera left untouched` and do not treat the device as privacy-protected.

## Motion and transform matrix

With the front camera, record separate 10-second samples for: rapid left/right yaw, rapid nodding, roll, moving close/far, walking, entering/exiting the frame, covering half the face, and camera switching/mirroring.

Pass criteria for every sample:

- The oval/PCA mask follows translation, zoom, and roll without vertical inversion.
- Camera switching never shows geometry from the retired `SurfaceTexture` source.
- A slow GPU replaces queued input with the latest frame rather than accumulating latency.
- Every detected face is held independently for at most 350 ms; motion prediction stops after 120 ms.
- Test one to four faces, distant faces, physical mirror reflections, and faces entering at every frame edge.
- After a camera switch, the whole frame stays smoothly blurred until a fresh result from the new source arrives.
- Face masks composite a strongly blurred downsampled three-cycle separable Gaussian texture with no pixel grid or sparse-sample ghosting.
- The detected face contour remains inside the fully blurred core; feathering starts outside the protected oval.
- Recorded/encoded output must have the same blur coverage as preview.

## Logs to send back if anything goes wrong

Export the plugin log around the failure and include the matching video. Search for `BlurFaces`, especially `GPU`, `VIDEO`, `SurfaceTexture`, `source`, and `encoder`.

Do not accept a release based only on a static front-facing test: fast motion, profile, occlusion, mirror, and encoded output are the acceptance cases.

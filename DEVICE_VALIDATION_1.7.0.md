# Blur Faces 1.7.0 device validation

This release keeps the 5-point affine privacy root authoritative. MediaPipe Face Landmarker runs in an independent, latest-frame, best-effort worker and produces 478 mesh points, a 4×4 facial pose matrix, and blendshape scores for future face-anchored effects. It cannot change blur position, size, count, or fail-closed state.

## Install and startup

1. Install `blur-faces-1.7.0.plugin` in exteraGram and open the round-video camera.
2. Confirm that the normal root blur appears exactly as in 1.6.3.
3. Read the plugin log. Expected successful path includes both:
   - `Live blur armed: stable 5-point privacy root + optional dense mesh effect layer`
   - `Dense mesh initialized: 478 landmarks; stable 5-point root still owns privacy`
4. If the second line is absent or says unavailable, the 5-point blur must still work normally. That fallback is expected behavior, not a privacy failure.

## Motion and transform matrix

With the front camera, record separate 10-second samples for: rapid left/right yaw, rapid nodding, roll, moving close/far, walking, entering/exiting the frame, covering half the face, and camera switching/mirroring.

Pass criteria for every sample:

- The blur remains attached to the current 5-point face basis through translation, zoom, and roll.
- It never becomes smaller because the dense mesh missed, lagged, or failed.
- During detector/tracker uncertainty, the privacy cover remains rather than revealing a guessed face area.
- Rapid motion may cause the optional mesh detail stream to skip frames; it must never stall the root blur worker.
- Recorded/encoded output must have the same blur coverage as preview.

## Logs to send back if anything goes wrong

Export the plugin log around the failure and include the matching video. Search for `BlurFaces`, especially `Dense mesh`, `worker diag`, `emergency`, `source`, and `encoder`.

Do not accept a release based only on a static front-facing test: fast motion, profile, occlusion, mirror, and encoded output are the acceptance cases.

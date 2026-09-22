# Device validation — pending

No phone results are implied by host tests or a successful build. Record device/SoC, Android, exteraGram build, plugin artifact SHA-256, sensitivity and mask mode with each run. Use consenting subjects and local footage. Raw-frame diagnostics must stay off unless explicitly enabled for the test; they contain unblurred images, remain local and must be deleted afterwards.

## Synthetic smoke test — passed 2026-09-22

Device reached over `ssh phone`: **SM8750, Android 16**; installed exteraGram reports **12.10.1 (70389)**. The standalone `app_process` test uses artificial pixels only and does not load the plugin into exteraGram, access app data or open the camera.

```bash
python3 scripts/test_debug_device.py --host phone
```

Observed PASS for:
- Real ES3 PBO readback: pixels paired with original timestamps, warm-up duplicate, reset flushing, and failed PBO/direct-read rejection.
- Real Android JPEG/JSON/ZIP encoding, RGBA channel order, bounded asynchronous writer, pooled-buffer copy isolation and capture deletion.
- Bundled NCNN model initialization and production JNI `process`/`processDebug` parity on flat black/gray/white frames, snapshot ABI and invalid-array/uninitialized-engine errors.

This does **not** verify live ExteraGram settings callbacks, camera/encoder hooks, detector accuracy on faces/clutter, full-frame privacy during transitions or real-time performance. The release gate below remains pending. Temporary synthetic device files were removed; the installed plugin was not changed.

## Release gate

- [ ] Clean install with network disabled: all model resources are available; no download attempts.
- [ ] Update from the old release, restart the app: legacy Blur Faces JPEG captures are removed, unrelated files remain. Verify again after recording with debug **off** that no captures are written.
- [ ] Front and rear cameras, repeated flips while recording: no old-camera masks or uncovered transition frames.
- [ ] Bright/dim scenes, abrupt lighting changes, frontal/profile/back-of-head, close-up and distant heads.
- [ ] Still face, rapid movement, sudden stop, partial occlusion and re-entry: record misses, overshoot and mask persistence.
- [ ] Empty cluttered room (bags, gym equipment, patterns) and then the same room with a person: measure false positives **and** missed heads.
- [ ] Multiple heads including more than four: confirm full-frame fallback rather than silently uncovered heads.
- [ ] All three mask modes and sensitivity settings, enable/disable, app pause/resume and plugin reload.
- [ ] Inject native inference failure in a test build: existing masks/empty-scene state are invalidated immediately; full-frame protection persists until a successful result. Restore the production build afterwards.
- [ ] Missing/corrupt bundled assets or incompatible hooks: initialization failure is visible; do not treat the camera as protected.
- [ ] Decode the **recorded round video**, inspect frame-by-frame. Preview-only inspection is insufficient.

## Measurements

Record detection time and capture-to-result age (p50/p95), preview/encoder FPS, dropped frames, startup/flip duration, memory and temperature over at least a few minutes. Compare the same scene/settings on the previous and candidate builds. A CPU synthetic mask test is not a detector accuracy or GPU performance test.

Release only after reviewing uncovered-head frames and regressions. There is no claim of guaranteed 360-degree recall, zero false positives or irreversible anonymization.

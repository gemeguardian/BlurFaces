# Blur Faces v3.0.0 Device Validation Checklist

Target Hardware: ARM64-v8a (e.g. OnePlus 13, Snapdragon 8 Elite, Android 16)
Plugin Artifact: `builds/blur_faces-3.0.0.elyx`

---

## Pre-Flight Checks

- [ ] **Airplane Mode (Zero-Network)**:
  - Turn ON Airplane Mode (disable Wi-Fi and Mobile Data).
  - Install `blur_faces-3.0.0.elyx` in exteraGram.
  - Verify plugin activates immediately without any network prompts or error toasts.
  - Verify that no network connections are attempted (logcat check: no DNS/HTTP/HTTPS activity from plugin).

- [ ] **Privacy Self-Test**:
  - Open exteraGram Settings -> Plugins -> Blur Faces.
  - Tap "Privacy Self-Test".
  - Verify the toast returns `PASSED` (confirming GL shader and synthetic rasterization pipeline are operational).

---

## 1. Camera Switch & Latency Tests

- [ ] **Instant Camera Flip (Rear to Front / Front to Rear)**:
  - Open round-video recording with the rear camera pointing at a room with NO faces.
  - Verify NO blur is present on the rear camera.
  - Double-tap or switch to the front camera (showing user face).
  - Verify face is immediately detected and blurred (<50ms).
  - Switch back from front camera to rear camera.
  - **Critical**: Verify blur vanishes immediately (1 frame / ~12ms). There must be NO lingering face blur or ghost oval on the rear camera.

- [ ] **Fast Switching Spam**:
  - Rapidly switch between front and rear cameras 5 times.
  - Verify no crashes, no GL errors in logcat, and PBO queue resets cleanly without leak.

---

## 2. False Positive Suppression Tests

Test each of the following non-human objects and verify **NO BLUR** is triggered:

- [ ] **Clothing & Legs**:
  - Aim camera at dark jeans, blue pants, and bare knees under direct light.
  - Verify: No false detections (chroma radius gating $\Delta_{\text{chroma}} \ge 2.85$).

- [ ] **Furniture & Office Items**:
  - Aim camera at mesh office chairs, dark armrests, and door handles.
  - Verify: No false detections.

- [ ] **Yellow & Ceramic Objects**:
  - Aim camera at a yellow coffee mug, yellow bowls, or bright yellow packaging.
  - Verify: No false detections (non-biological yellow filter).

- [ ] **Flat Architectural Surfaces**:
  - Aim camera at blank painted doors, walls, and empty cardboard boxes.
  - Verify: No false detections (spatial texture energy filter $\ge 3.20$).

- [ ] **Small Clutter & Rings**:
  - Aim camera at circular items (gym hoops, round wall clocks, doorknobs).
  - Verify: No false detections (Feature 2 anchor pruning).

- [ ] **Pets (Dogs / Cats)**:
  - Aim camera at a pet dog (e.g. Yorkshire terrier, maltipoo, golden retriever) facing the camera.
  - Verify: Dog face and body are NOT blurred (hemoglobin balance $(R-G)/(R-B) \ge 0.58$ and snout ratio checks).

---

## 3. Human Face & Head Detection Robustness

Test the following scenarios with an actual human user:

- [ ] **Extreme Angles (360° Head Detection)**:
  - Frontal face ($0^\circ$).
  - Profile view ($90^\circ$ yaw).
  - Head tilted backwards or downwards ($45^\circ-90^\circ$ pitch).
  - Back of head (hair/neck visible, face turned away).
  - Verify: Continuous blur coverage across all orientations.

- [ ] **Close-up Selfie**:
  - Bring phone camera extremely close so the face fills 80%–100% of the circular frame.
  - Verify: Face remains fully covered without dropping blur box.

- [ ] **Occlusion & Hand-over-Face**:
  - Cover mouth with hand, wear sunglasses or mask.
  - Verify: Head detector and ByteTrack maintain blur lock.

- [ ] **Multi-Person**:
  - 2 to 4 people in the frame.
  - Verify: All faces are tracked and blurred independently.

---

## 4. UI Modes & Quality Checks

- [ ] **Mask Modes**:
  - Switch to **Gaussian Blur**: Verify smooth, high-quality blur composite without pixelation.
  - Switch to **Pixelate**: Verify mosaic/pixel block effect inside head bounding ellipse.
  - Switch to **Solid Color**: Verify solid mask covering face.

- [ ] **Face Mask Scale**:
  - Test Compact (65%), Standard (82%), and Wide (100%). Verify oval coverage adjusts accordingly.

- [ ] **Recorded Video Output**:
  - Record a 10-second video message and send it to "Saved Messages".
  - Play back the sent round video.
  - Verify: Recorded video has the exact same blur coverage as the live preview, with no frame leaks at start or end.

---

## 5. Performance & Resource Consumption

- [ ] **Inference Latency**:
  - Check logcat: `adb logcat -s BlurFaces`.
  - Verify detection inference latency is $\le 15\text{ms}$ on modern ARM64 chips.
  - Verify preview frame rate stays at smooth 30–60 FPS without stutter.

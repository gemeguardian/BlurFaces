# Architectural Roadmap: Upgrading Head Detection Engine & Verification Pipeline

Date: 2026-09-20  
Project: `blur-faces` v3.0.0 (exteraGram / Telegram Android Privacy Plugin)

---

## 1. Executive Summary & Root Cause Analysis

### The Problem
During real-time camera preview and round video recording, the plugin occasionally locks into full-frame blur (`uFaceCount = -1`) or generates false regional blur ellipses on inanimate objects in empty rooms (dumbbells, backpack prints, geometric floor patterns, doorframes, clothing folds).

### Why the Current Model Fails
The current model (`head_det.param` / `head_det.bin`) is an ancient **MobileNet0.25-FPN (2019)** trained on surveillance datasets (**SCUT-HEAD**):
1. **Severe Representational Bottleneck (Width 0.25x)**: Early layers contain only 8–16 channels. The model lacks the capacity to represent biological micro-textures (hair strands, ear helices, neck/shoulder junctions). It degrades into a simple radial gradient/circular blob detector. Any round object (dumbbell plate, soccer ball print on a backpack) triggers high activation.
2. **Surveillance Perspective Bias**: SCUT-HEAD consists of top-down classroom surveillance cameras. It never encountered domestic household clutter, close-up selfie perspectives, or sports equipment.
3. **Hardcoded C++ Workarounds Overfitting**: Handcrafted checks (hemoglobin ratios, BT.601 skin locus, Laplacian texture energy) were layered on top to suppress false positives. While they blocked specific items (e.g. dogs or yellow mugs), they fragilely fail when lighting changes or when inanimate items match skin locus hues (wood veneer, leather bags, dark gym gear).

---

## 2. Model Replacement: YOLOv8n-Head Detection

### Candidate Selection
- **Model**: [`abhiWanKenobi/yolov8n_head_detection`](https://huggingface.co/abhiWanKenobi/yolov8n_head_detection)
- **Architecture**: Ultralytics **YOLOv8n (Nano)** — single-class anchor-free detection (`{0: 'head'}`) with Decoupled Head and Distribution Focal Loss (DFL).
- **Training Dataset**: **HollywoodHeads** (224,740 frames from 69 diverse movies across indoor rooms, cars, offices, challenging lighting conditions).
- **360° Benchmark**: Evaluated on **DAD-3DHeads** (CVPR 2022 dataset with extreme head yaw, pitch, roll, and back-of-head occiput views):
  - **mAP@50**: `0.937`
  - **Precision**: `0.892`
  - **Recall**: `0.892`

### Performance & Resource Footprint
| Metric | Current (MobileNet0.25-FPN) | Target (YOLOv8n-Head) | Impact |
| :--- | :--- | :--- | :--- |
| **Parameters** | ~0.5M | **3.0M** | 6x feature capacity for anatomical details |
| **Model Size (FP16)**| 2.61 MB | **~3.0 MB** | Bundle size grows by only ~400 KB |
| **Input Resolution**| 320x320 | **320x320** | Identical OpenGL PBO tap & scaler pipeline |
| **Latency (ARM64)** | ~10 ms | **8–11 ms** | Zero drop in camera preview framerate |
| **Anchors** | Static IoU grid (F0/F1/F2) | **Anchor-free decoupled** | Eliminates anchor decoding artifacts |

### Output Tensor Format (YOLOv8n at 320x320)
- Single output blob: `[1, 5, 2100]`
  - `5` channels: `[cx, cy, w, h, confidence]`
  - `2100` candidate cells: `(40x40) + (20x20) + (10x10) = 1600 + 400 + 100` across strides 8, 16, 32.
  - Coordinate decoding in C++ requires simple center/size denormalization without anchor lookups.

---

## 3. Revising C++ Heuristic Filters (What to Keep vs. What to Remove)

Our current C++ codebase (`src/head_detector.cpp`) accumulated complex handcrafted filters that were designed specifically to compensate for the old MobileNet0.25's blindness. With a semantically competent YOLOv8n detector, several of these heuristics become redundant or harmful, while others remain valuable.

### Filters to REMOVE or DEPRECATE (Redundant / Counterproductive)
1. **`logit_bias` (Artificial Low-Light Boost)**:
   - *Why remove*: Artificial logit amplification (`box_score + bias`) in low light forces background sensor noise into 98% confidence detections. YOLOv8n trained on Hollywood films already natively handles shadows and low key lighting.
2. **`yellow_hits` / Non-Biological Amber Filter**:
   - *Why remove*: Hardcoded thresholds on `Cb < 75 && R > 90` were designed to reject coffee mugs. YOLOv8n understands object geometry and will not confuse a cylindrical mug with a human head.
3. **Rigid Aspect Ratio Range (`0.45 <= w/h <= 1.60`)**:
   - *Why relax*: YOLOv8n uses decoupled bounding box regression with DFL, producing clean, tightly bounded boxes. Strict aspect ratio gating risks dropping tilted heads or close-up profile shots.
4. **Feature 2 Anchor Skipping**:
   - *Why remove*: YOLOv8n is anchor-free; F2/F1/F0 anchor indexing logic is obsolete.

### Filters to KEEP as Lightweight Post-NMS Guards
1. **Biological Hemoglobin Balance Ratio ($(R - G) / (R - B) \ge 0.58$)**:
   - *Role*: Kept as an ultra-fast secondary sanity check only on low-confidence border proposals (`conf < 0.55`). Natural human skin has blood perfusion under illuminated conditions; inanimate objects (denim, wood, black gym gear) do not.
2. **Circular Viewport Boundary Clipping**:
   - *Role*: Telegram round video notes are strictly circular. Bounding boxes touching the extreme outer corners or sensor margins outside the circular preview radius ($r > 0.50$) should continue to be clipped to save GPU fill rate.
3. **ByteTrack Kinematic Association**:
   - *Role*: Retained completely. ByteTrack with decoupled Kalman filtering (cx, cy, aspect, height) provides temporal consistency, prevents jitter, and bridges brief motion-blur occlusions.

---

## 4. Reference for Future Exploration: Face Presence Classifiers

*(Note: Kept for architectural reference and future optimization, not for immediate integration).*

In computer vision privacy pipelines, binary classification models can serve two architectural patterns:

### Pattern A: "Gatekeeper" (Pre-Detector Presence Filter)
- **Model Candidate**: **MLCommons Tiny Visual Wake Words (VWW)** — `MobileNetV1-0.25`
  - *Weights / Format*: TFLite INT8 (333 KB) / ONNX FP32 (866 KB).
  - *Input*: 96x96 RGB (or Grayscale Y-channel from Camera2 NV21).
  - *Inference Time*: **0.8 ms** on mobile ARM64.
  - *Concept*: If `P(Person) < 0.20`, skip the 320x320 YOLO detector entirely.
  - *Pros*: Saves 100% of detection compute in empty rooms (~90% battery saving) and guarantees zero false positives on empty scenes.
  - *Cons*: Does not prevent false positives on background objects when a person *is* in the room.

### Pattern B: "Box Verifier" (Post-Detector Crop Confirmation)
- **Model Candidate**: **MiniFASNetV2** ([`minivision-ai/Silent-Face-Anti-Spoofing`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing))
  - *Weights / Format*: FP16 NCNN (~900 KB).
  - *Input*: 80x80 RGB crop expanded 2.7x around candidate box.
  - *Inference Time*: **1.4 ms** per candidate box.
  - *Concept*: Specifically trained to distinguish living biological tissue and facial structure from photographic prints, textile prints (soccer ball patterns on backpacks), and synthetic surfaces.
  - *Pros*: Completely resolves pareidolia on backpacks and gym equipment even when a person is present.
  - *Cons*: Adds ~1.4 ms latency per candidate box.

---

## 5. Phased Implementation Checklist for YOLOv8n

- [ ] **Phase 1: Conversion & Quantization**
  - Download `yolov8n_head_detector.pt` from Hugging Face (`abhiWanKenobi/yolov8n_head_detection`).
  - Export to static ONNX at 320x320: `model.export(format='onnx', imgsz=320, opset=12, dynamic=False)`.
  - Convert via `onnx2ncnn` and optimize with `ncnnoptimize ... 65536` to generate FP16 `head_det_v8.param` and `head_det_v8.bin` (~3.0 MB).
- [ ] **Phase 2: C++ Output Decoder Refactor**
  - Replace anchor grid loop (`kHeadAnchors`) in `src/head_detector.cpp` with standard YOLOv8 anchor-free decoder loop (`[1, 5, 2100]` tensor layout).
  - Standardize coordinate denormalization: $x_1 = (cx - w/2), y_1 = (cy - h/2)$.
  - Apply Fast NMS with IoU threshold 0.45.
- [ ] **Phase 3: Heuristic Cleanup**
  - Remove `logit_bias`, `s_ctx.darknessFactor` distortion, and anchor-specific filters.
  - Clean up `verify_head_candidate` to act as a lightweight guard without breaking real face recall.
- [ ] **Phase 4: Device Validation & Benchmarking**
  - Verify latency on OnePlus 13 (target: $\le 11$ ms).
  - Verify zero false positives in empty domestic rooms (dumbbells, backpacks, wall patterns).
  - Verify seamless 360° head blur on selfie turns and profile movements.

#!/usr/bin/env python3
"""Behavioral model for order-independent, per-face hold and prediction rules."""
import itertools
import math


def match(tracks, detections):
    unmatched_t = set(range(len(tracks)))
    unmatched_d = set(range(len(detections)))
    pairs = []
    while unmatched_t and unmatched_d:
        distance, ti, di = min(
            ((tracks[t][0] - detections[d][0]) ** 2 +
             (tracks[t][1] - detections[d][1]) ** 2, t, d)
            for t in unmatched_t for d in unmatched_d
        )
        radius = max(tracks[ti][2], 0.045)
        if distance > (radius * 1.6) ** 2:
            break
        pairs.append((ti, detections[di]))
        unmatched_t.remove(ti)
        unmatched_d.remove(di)
    return sorted(pairs)


tracks = [(0.20, 0.30, 0.08), (0.75, 0.32, 0.09), (0.48, 0.70, 0.07)]
detections = [(0.205, 0.302), (0.742, 0.318), (0.485, 0.695)]
expected = match(tracks, detections)
for ordering in itertools.permutations(detections):
    actual = match(tracks, ordering)
    assert [index for index, _ in actual] == [0, 1, 2]
    assert all(math.dist(actual[i][1], expected[i][1]) < 1e-9 for i in range(3))

# Slow inference adapts the hold from result cadence. Publication time prevents
# inference latency itself from consuming the mask lifetime.
hold_ns = max(350_000_000, min(1_200_000_000, 280_000_000 * 3 + 100_000_000))
assert hold_ns == 940_000_000
last_seen = [1_000_000_000, 1_200_000_000]
assert [1_300_000_000 - seen <= hold_ns for seen in last_seen] == [True, True]
assert [2_140_000_001 - seen <= hold_ns for seen in last_seen] == [False, False]

# Prediction is capped even when hold lasts longer.
prediction_ns = min(120_000_000, 300_000_000)
assert prediction_ns == 120_000_000

# Face hold/decay expansion and velocity dilation (15-25% radius expansion)
def decay_expansion(elapsed_published_ns, hold_limit_ns, elapsed_seen_ns, max_pred_ns, speed=0.0):
    decay_p = max(0.0, min(1.0, elapsed_published_ns / max(1, hold_limit_ns)))
    pred_p = max(0.0, min(1.0, elapsed_seen_ns / max(1, max_pred_ns)))
    vel_d = max(0.0, min(0.10, speed * 0.15))
    expansion = 1.0 + 0.15 * pred_p + 0.05 * decay_p + vel_d
    return max(1.0, min(1.25, expansion))

assert decay_expansion(0, hold_ns, 0, prediction_ns) == 1.0
assert 1.15 <= decay_expansion(hold_ns, hold_ns, prediction_ns, prediction_ns) <= 1.25
assert 1.15 <= decay_expansion(hold_ns, hold_ns, prediction_ns, prediction_ns, speed=1.0) <= 1.25

# Two-level confidence hysteresis:
# Detections near an active track gate are accepted down to 0.20 confidence.
# New tracks require >= 0.30 confidence (or configured_confidence).
TRACK_GATED_MIN_CONFIDENCE = 0.20
TRACK_NEW_MIN_CONFIDENCE = 0.30


def update_tracks(tracks, detections, configured_confidence=0.35):
    unmatched_t = set(range(len(tracks)))
    unmatched_d = set(range(len(detections)))
    matched = []

    while unmatched_t and unmatched_d:
        candidates = [
            (((tracks[t]["x"] - detections[d]["x"]) ** 2 +
              (tracks[t]["y"] - detections[d]["y"]) ** 2), t, d)
            for t in unmatched_t for d in unmatched_d
            if detections[d].get("score", 1.0) >= TRACK_GATED_MIN_CONFIDENCE
        ]
        if not candidates:
            break
        distance, ti, di = min(candidates)
        radius = max(tracks[ti]["radius"], 0.045)
        gate = max(0.045, radius * 1.6)
        if distance > gate * gate:
            break
        matched.append((ti, di))
        unmatched_t.remove(ti)
        unmatched_d.remove(di)

    for ti, di in matched:
        tracks[ti]["x"] = detections[di]["x"]
        tracks[ti]["y"] = detections[di]["y"]
        tracks[ti]["score"] = detections[di].get("score", 1.0)
        tracks[ti]["observed"] = True

    new_track_threshold = max(TRACK_NEW_MIN_CONFIDENCE, configured_confidence)
    new_tracks = []
    for di in unmatched_d:
        score = detections[di].get("score", 1.0)
        if score >= new_track_threshold:
            nt = {
                "x": detections[di]["x"],
                "y": detections[di]["y"],
                "radius": 0.08,
                "score": score,
                "observed": True,
            }
            tracks.append(nt)
            new_tracks.append(nt)

    return matched, new_tracks


# 1. Low-confidence detection (< 0.30) does NOT create a new track on empty tracker
test_tracks = []
matched, created = update_tracks(test_tracks, [{"x": 0.5, "y": 0.5, "score": 0.25}])
assert len(created) == 0
assert len(test_tracks) == 0

# 2. Detection >= 0.35 on empty tracker DOES create a new track
matched, created = update_tracks(test_tracks, [{"x": 0.5, "y": 0.5, "score": 0.35}])
assert len(created) == 1
assert len(test_tracks) == 1
assert test_tracks[0]["x"] == 0.5 and test_tracks[0]["y"] == 0.5

# 3. Low-confidence detection (>= 0.20) DOES update an existing track within its gating radius
test_tracks[0]["observed"] = False
matched, created = update_tracks(test_tracks, [{"x": 0.52, "y": 0.51, "score": 0.22}])
assert len(matched) == 1
assert len(created) == 0
assert test_tracks[0]["observed"] is True
assert test_tracks[0]["x"] == 0.52 and test_tracks[0]["y"] == 0.51

# 4. Sub-0.20 detection (< 0.20) within gating radius does NOT match or update
test_tracks[0]["observed"] = False
matched, created = update_tracks(test_tracks, [{"x": 0.53, "y": 0.51, "score": 0.15}])
assert len(matched) == 0
assert len(created) == 0
assert test_tracks[0]["observed"] is False

# 5. Low-confidence detection (0.25) outside gating radius does NOT create a new track or update existing
test_tracks[0]["observed"] = False
matched, created = update_tracks(test_tracks, [{"x": 0.85, "y": 0.85, "score": 0.25}])
assert len(matched) == 0
assert len(created) == 0
assert len(test_tracks) == 1
assert test_tracks[0]["observed"] is False
assert len(created) == 0
assert len(test_tracks) == 1
assert test_tracks[0]["observed"] is False

# 6. Yaw-aware track hold limit and profile coasting extension
def track_hold_limit(base_hold_ns, yaw):
    if abs(yaw) > 0.25:
        return int(base_hold_ns * (1.0 + 0.40 * abs(yaw)))
    return base_hold_ns

base_hold = 1_050_000_000 # 350ms adaptive + 700ms coast
assert track_hold_limit(base_hold, 0.0) == base_hold
assert track_hold_limit(base_hold, 0.20) == base_hold
assert track_hold_limit(base_hold, 0.75) == int(base_hold * (1.0 + 0.40 * 0.75))
assert track_hold_limit(base_hold, 0.75) > base_hold

# Profile face survives at 1150ms occlusion while frontal expires
assert 1_150_000_000 <= track_hold_limit(base_hold, 0.75)
assert 1_150_000_000 > track_hold_limit(base_hold, 0.0)

# 7. ByteTrack temporal confirmation: 1-frame transient false positive (< instant_thresh) is dropped
class ByteTrackerModel:
    def __init__(self, high_thresh=0.45, low_thresh=0.20, instant_thresh=0.65):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.instant_thresh = instant_thresh
        self.tracked = []
        self.unconfirmed = []
        self.lost = []

    def update(self, detections):
        det_first = [d for d in detections if d["score"] >= self.high_thresh]
        det_second = [d for d in detections if self.low_thresh <= d["score"] < self.high_thresh]

        # Step 1: match det_first with tracked
        matched_trk, unconf_trk, rem_first = match_iou(self.tracked, det_first)
        # Step 2: match det_second with unconf_trk (remaining tracked)
        matched_sec, still_lost_trk, _ = match_iou(unconf_trk, det_second)
        # Step 3: match rem_first with lost
        matched_lost, rem_lost, rem_first_2 = match_iou(self.lost, rem_first)
        # Step 4: match rem_first_2 with unconfirmed (2nd frame confirmation)
        newly_confirmed, rem_unconfirmed, rem_first_final = match_iou(self.unconfirmed, rem_first_2)

        # Unmatched unconfirmed tracks are DROPPED (false positive suppression)
        self.unconfirmed = []

        # Step 5: Process brand-new detections
        new_instant = []
        for d in rem_first_final:
            if d["score"] >= self.instant_thresh:
                new_instant.append(d)
            else:
                self.unconfirmed.append(d)

        self.tracked = matched_trk + matched_sec + matched_lost + newly_confirmed + new_instant
        self.lost = rem_lost + still_lost_trk
        return self.tracked

def match_iou(tracks, detections):
    # Simple spatial matching for unit testing
    matched = []
    unmatched_t = list(tracks)
    unmatched_d = list(detections)
    i = 0
    while i < len(unmatched_t):
        t = unmatched_t[i]
        found = -1
        for j, d in enumerate(unmatched_d):
            dist_sq = (t["x"] - d["x"])**2 + (t["y"] - d["y"])**2
            if dist_sq < 0.04:
                found = j
                break
        if found >= 0:
            matched.append(unmatched_d[found])
            unmatched_t.pop(i)
            unmatched_d.pop(found)
        else:
            i += 1
    return matched, unmatched_t, unmatched_d

tracker = ByteTrackerModel(high_thresh=0.45, low_thresh=0.20, instant_thresh=0.65)
# Transient false positive with moderate score 0.48 on frame 1
output_f1 = tracker.update([{"x": 0.3, "y": 0.3, "score": 0.48}])
assert len(output_f1) == 0, "1-frame moderate-confidence detection must NOT be output"
assert len(tracker.unconfirmed) == 1

# Frame 2: noise disappears -> unconfirmed track is dropped, output remains empty
output_f2 = tracker.update([])
assert len(output_f2) == 0, "Disappeared unconfirmed detection must not produce output or lost tracks"
assert len(tracker.unconfirmed) == 0
assert len(tracker.lost) == 0

# High-confidence face (>= 0.65) is activated instantly
output_instant = tracker.update([{"x": 0.4, "y": 0.4, "score": 0.85}])
assert len(output_instant) == 1, "High-confidence face must activate instantly"

# Real face at 0.50 confirmed on 2nd frame
tracker.tracked = []
f1 = tracker.update([{"x": 0.6, "y": 0.6, "score": 0.50}])
assert len(f1) == 0
f2 = tracker.update([{"x": 0.61, "y": 0.60, "score": 0.52}])
assert len(f2) == 1, "Face seen across 2 frames must be confirmed and output"

# Physical bounding box sanity check:
def valid_head_box(w, h):
    return 0.04 <= w <= 0.95 and 0.05 <= h <= 0.95 and 0.45 <= (w / h) <= 1.65

assert valid_head_box(0.20, 0.25) is True  # normal head: AR = 0.80
assert valid_head_box(0.02, 0.02) is False # tiny speck rejected
assert valid_head_box(0.03, 0.30) is False # vertical line rejected (AR = 0.1)
assert valid_head_box(0.40, 0.04) is False # horizontal bar rejected (AR = 10)

# 8. Anti-overshoot prediction and active braking behavioral verification:
def compute_next_velocity(raw_vel, current_vel, dt, alpha=0.5, speed_deadband=0.06):
    if abs(raw_vel) < speed_deadband:
        return 0.0
    if raw_vel * current_vel <= 0.0:
        return raw_vel * 0.5
    if abs(raw_vel) < abs(current_vel):
        return current_vel + 0.85 * (raw_vel - current_vel)
    return current_vel + alpha * (raw_vel - current_vel)

def predict_center(values_x, vel_x, dt_stale, radius_x, speed_deadband=0.06):
    speed = abs(vel_x)
    pos_horizon = 0.0 if speed < speed_deadband else min(0.025, dt_stale)
    est_x = values_x + vel_x * pos_horizon
    max_shift_x = max(0.015, radius_x * 0.30)
    return max(values_x - max_shift_x, min(values_x + max_shift_x, est_x))

# Test: rapid movement followed by abrupt stop
# Phase 1: fast movement right at 3.0 screen units/sec
v = 0.0
for _ in range(5):
    v = compute_next_velocity(3.0, v, 0.02, alpha=0.5)
assert v > 2.5, "Velocity ramps up during rapid movement"

# Extrapolation during flight is strictly bounded to 30% of radius (e.g. radius 0.10 -> shift <= 0.03)
radius = 0.10
pred = predict_center(0.50, v, 0.033, radius)
assert pred <= 0.50 + 0.03, f"Prediction during movement must not exceed face radius bound: {pred}"

# Phase 2: abrupt stop (raw velocity = 0.0)
v_stopped = compute_next_velocity(0.0, v, 0.02)
assert v_stopped == 0.0, "Active braking must zero velocity immediately upon stopping"

pred_stopped = predict_center(0.70, v_stopped, 0.033, radius)
assert pred_stopped == 0.70, "Zero overshoot: predicted center matches face position exactly when stopped"

# 9. Advanced Mathematical False Positive Suppression Tests:

# a) Sequential Probability Ratio Test (SPRT) Log-Odds Evidence Accumulation
def logit(p):
    p = max(0.001, min(0.999, p))
    return math.log(p / (1.0 - p))

def compute_evidence_update(score, iou, deformation, s_neutral=0.30):
    delta_score = logit(score) - logit(s_neutral)
    delta_iou = 1.2 * (iou - 0.40)
    delta_deform = (deformation - 0.25) * 4.0 if deformation > 0.25 else 0.0
    return delta_score + delta_iou - delta_deform

# Test: High confidence face (0.75) accumulates strong evidence
delta_high = compute_evidence_update(0.75, 0.70, 0.05)
assert delta_high > 1.5, f"High confidence and tight IoU must accumulate strong positive evidence: {delta_high}"

# Test: Low-confidence detection (0.22) with poor IoU and large deformation loses evidence rapidly
delta_noise = compute_evidence_update(0.22, 0.20, 0.40)
assert delta_noise < -1.0, f"Clutter must yield negative log-odds evidence: {delta_noise}"

# Test: Wald SPRT confirmation barrier (L_confirm = 1.0)
L_confirm = 1.0
L_init_unambiguous = logit(0.75) - logit(0.35)
assert L_init_unambiguous >= L_confirm, "Unambiguous face (>= 0.75) passes confirmation immediately"

L_init_borderline = logit(0.45) - logit(0.35)
assert L_init_borderline < L_confirm, "Borderline face requires sequential confirmation across frames"

# b) Anthropometric Head Aspect Ratio and Radial Viewport Gating
def is_valid_head_geometry(cx, cy, w, h, prob):
    # Radial gating in round video: radius <= 0.52 from center (0.5, 0.5)
    rad_sq = (cx - 0.5) ** 2 + (cy - 0.5) ** 2
    if rad_sq > 0.2704: # 0.52^2
        return False
    # Dimensional bounds
    if not (0.08 <= w <= 0.92 and 0.10 <= h <= 0.92):
        return False
    aspect = w / h
    # Anthropometric skull aspect ratio in [0.52, 1.30]
    if not (0.52 <= aspect <= 1.30):
        return False
    if (aspect < 0.60 or aspect > 1.18) and prob < 0.35:
        return False
    return True

assert is_valid_head_geometry(0.5, 0.5, 0.20, 0.25, 0.50) is True   # normal head at center
assert is_valid_head_geometry(0.05, 0.05, 0.20, 0.25, 0.50) is False # outside circular viewport
assert is_valid_head_geometry(0.5, 0.5, 0.05, 0.30, 0.50) is False  # aspect ratio 0.16 (pole/line)
assert is_valid_head_geometry(0.5, 0.5, 0.35, 0.15, 0.50) is False  # aspect ratio 2.33 (horizontal bar)
assert is_valid_head_geometry(0.5, 0.5, 0.20, 0.35, 0.25) is False  # aspect ratio 0.57 below 0.35 prob threshold

# c) Kalman Chi-Square (chi^2) Mahalanobis Distance Gating
def mahalanobis_dist_sq(z, x, var_p, var_r):
    s = var_p + var_r
    return (z - x) ** 2 / s

# Test kinematic gating with chi^2 threshold = 20.0 (df = 4, alpha = 0.001)
CHI_SQUARE_GATE = 20.0
# Realistic Kalman position variance for head of height h = 0.20:
# r_pos = h / 20.0 = 0.01 -> r = 0.0001
# p00 ~ 0.0004 -> s = 0.0005
var_s = 0.0005
# Normal small innovation (0.005 shift)
d_maha_normal = sum(mahalanobis_dist_sq(0.505, 0.50, var_s, 0.0) for _ in range(4))
assert d_maha_normal < CHI_SQUARE_GATE, "Normal physical motion must be within Chi-Square gate"

# Jump anomaly (e.g. background false positive 15% away from prediction: 0.15^2 / 0.0005 = 45.0)
d_maha_jump = mahalanobis_dist_sq(0.65, 0.50, var_s, 0.0)
assert d_maha_jump > CHI_SQUARE_GATE, f"Unphysical track jumping must be rejected by Chi-Square gate: {d_maha_jump}"

# d) Spatial Texture Gradient Energy
def texture_energy(patch):
    # patch: 2D array of luminance
    h, w = len(patch), len(patch[0])
    total_grad = 0.0
    count = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = abs(patch[y][x + 1] - patch[y][x - 1])
            gy = abs(patch[y + 1][x] - patch[y - 1][x])
            total_grad += (gx + gy)
            count += 1
    return total_grad / max(1, count)

# Flat wall patch (uniform with minor sensor noise)
flat_patch = [[128 + (x % 2) for x in range(10)] for y in range(10)]
assert texture_energy(flat_patch) < 2.5, "Flat textureless background must have low gradient energy"

# Face patch with rich features (eyes, nose, mouth gradients)
face_patch = [[128 + ((x * 15 + y * 20) % 100) for x in range(10)] for y in range(10)]
assert texture_energy(face_patch) > 10.0, "Real face features must have high gradient energy"

print("PASS: matching ignores detection order; face hold/expiry and prediction are independent; confidence hysteresis active; yaw-aware profile hold active; ByteTrack 2-stage false-positive rejection verified; anti-overshoot active braking verified; Bayesian SPRT + Chi-Square gating + anthropometric + texture energy verified")

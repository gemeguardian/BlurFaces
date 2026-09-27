#include "bytetrack.h"
#include <algorithm>
#include <cmath>
#include <limits>

void KalmanState1D::init(float init_x, float init_v, float std_p0, float std_p1) {
    x = init_x;
    v = init_v;
    p00 = std_p0 * std_p0;
    p01 = 0.0f;
    p10 = 0.0f;
    p11 = std_p1 * std_p1;
}

void KalmanState1D::predict(float q00, float q01, float q10, float q11, float s) {
    // Constant-velocity step of s reference frames: F = [1 s; 0 1], Q scaled by s.
    x = x + v * s;
    float new_p00 = p00 + s * (p10 + p01) + s * s * p11 + s * q00;
    float new_p01 = p01 + s * p11 + s * q01;
    float new_p10 = p10 + s * p11 + s * q10;
    float new_p11 = p11 + s * q11;
    p00 = new_p00;
    p01 = new_p01;
    p10 = new_p10;
    p11 = new_p11;
}

void KalmanState1D::update(float measurement, float r) {
    float s = p00 + r;
    if (s <= 1e-6f) return;
    float inv_s = 1.0f / s;
    float k0 = p00 * inv_s;
    float k1 = p10 * inv_s;

    float y = measurement - x;
    x += k0 * y;
    v += k1 * y;

    float new_p00 = (1.0f - k0) * p00;
    float new_p01 = (1.0f - k0) * p01;
    float new_p10 = p10 - k1 * p00;
    float new_p11 = p11 - k1 * p01;
    p00 = new_p00;
    p01 = new_p01;
    p10 = new_p10;
    p11 = new_p11;
}

// STrack implementation
STrack::STrack() : track_id_(0), state_(TrackState::New) {}

namespace {
constexpr float kDefaultNeutralScore = 0.35f;
// Ages are sums of fractional frame scales; absorb float rounding at the limits.
constexpr float kAgeEpsilon = 1e-3f;
// Kinematic (IoU-free) match cost, see linear_assignment: 0.55 at the predicted
// centre, 0.70 (= match_thresh, i.e. rejected) at the chi-square gate of 20.
constexpr float kKinematicBaseCost = 0.55f;
constexpr float kKinematicMahaCost = 0.0075f;
// Stage-2 (below high) observations support a track at the band centre only once
// it has this much strong (>= high) observation time, in reference frames.
constexpr float kEstablishedStrongAge = 6.0f;
// Longest detection gap an unconfirmed candidate survives, in reference frames.
// A recorded miss at the tuned cadence (~130 ms) must break the chain even when
// replayed as 3 x 42 ms frames (0.97); 1-2 missed frames at 24 fps must not.
constexpr float kMaxCandidateGap = 0.75f;

float logit(float p) {
    p = std::clamp(p, 0.01f, 0.99f);
    return std::log(p / (1.0f - p));
}
} // namespace

float STrack::evidence(const HeadBox& box, float neutral_score) {
    float raw = std::clamp(logit(box.score) - logit(neutral_score), -kEvidenceCap, kEvidenceCap);
    float area = std::max(0.0f, box.w) * std::max(0.0f, box.h);
    float reliability = std::clamp(area / kReliableArea, kMinSizeReliability, 1.0f);
    return raw * reliability;
}

STrack::STrack(const HeadBox& box, int track_id)
    : STrack(box, track_id, kDefaultNeutralScore) {}

STrack::STrack(const HeadBox& box, int track_id, float neutral_score, float frame_scale)
    : track_id_(track_id), state_(TrackState::New), frames_lost_(0),
      frames_tracked_(1), score_(box.score), tracked_age_(frame_scale),
      strong_age_(frame_scale) {
    init_kalman(box);
    // Wald SPRT log-odds initialisation from the first observation.
    log_odds_ = std::clamp(evidence(box, neutral_score) * frame_scale, kLogOddsMin, kLogOddsMax);
}

void STrack::init_kalman(const HeadBox& box) {
    float h = std::max(0.01f, box.h);
    float w = std::max(0.01f, box.w);
    float a = w / h;

    float std_pos = 2.0f * (1.0f / 20.0f) * h;
    float std_vel = 10.0f * (1.0f / 160.0f) * h;

    kf_cx_.init(box.cx, 0.0f, std_pos, std_vel);
    kf_cy_.init(box.cy, 0.0f, std_pos, std_vel);
    kf_a_.init(a, 0.0f, 0.1f, 0.05f);
    kf_h_.init(h, 0.0f, std_pos, std_vel);
}

void STrack::predict(float motion_scale, float camera_dx, float camera_dy) {
    const float s = std::isfinite(motion_scale) ? std::max(0.0f, motion_scale) : 1.0f;
    float h = std::max(0.01f, kf_h_.x);
    float q_pos = (1.0f / 20.0f) * h;
    float q_vel = (1.0f / 160.0f) * h;
    float q00 = q_pos * q_pos;
    float q11 = q_vel * q_vel;

    // Decay velocity if coasting in lost state (0.85 per reference frame)
    if (state_ == TrackState::Lost) {
        float decay = std::pow(0.85f, s);
        kf_cx_.v *= decay;
        kf_cy_.v *= decay;
        kf_h_.v *= decay;
        kf_a_.v *= decay;
    }

    // Camera motion compensation: the whole image content moved by
    // (camera_dx, camera_dy) since the previous detector frame, so every track
    // did too. Velocity keeps modelling the head's own motion.
    if (std::isfinite(camera_dx) && std::isfinite(camera_dy)) {
        kf_cx_.x += camera_dx;
        kf_cy_.x += camera_dy;
    }

    kf_cx_.predict(q00, 0.0f, 0.0f, q11, s);
    kf_cy_.predict(q00, 0.0f, 0.0f, q11, s);
    kf_a_.predict(0.01f * 0.01f, 0.0f, 0.0f, 0.005f * 0.005f, s);
    kf_h_.predict(q00, 0.0f, 0.0f, q11, s);

    // Keep aspect ratio and height within physical bounds
    kf_a_.x = std::max(0.4f, std::min(1.8f, kf_a_.x));
    kf_h_.x = std::max(0.02f, std::min(1.0f, kf_h_.x));
}

void STrack::update(const HeadBox& box, int frame_id) {
    update(box, frame_id, 0.5f);
}

void STrack::update(const HeadBox& box, int frame_id, float iou) {
    update(box, frame_id, iou, kDefaultNeutralScore);
}

void STrack::update(const HeadBox& box, int frame_id, float iou, float neutral_score,
                    float frame_scale, bool strong) {
    (void) iou; // spatial alignment is a tracking-quality signal, not head evidence
    float h = std::max(0.01f, box.h);
    float w = std::max(0.01f, box.w);
    float a = w / h;

    float r_pos = (1.0f / 20.0f) * h;
    float r = r_pos * r_pos;

    // Relative biomechanical deformation before state update:
    float pred_h = std::max(0.01f, kf_h_.x);
    float pred_w = std::max(0.01f, kf_a_.x * pred_h);
    float def_w = std::abs(w - pred_w) / pred_w;
    float def_h = std::abs(h - pred_h) / pred_h;
    float deformation = std::max(def_w, def_h);

    kf_cx_.update(box.cx, r);
    kf_cy_.update(box.cy, r);
    kf_a_.update(a, 0.05f * 0.05f);
    kf_h_.update(h, r);

    // An observation alone never confirms a track; only activate() does, once the
    // temporal confirmation policy in ByteTracker::update is satisfied.
    state_ = TrackState::Tracked;
    frames_lost_ = 0;
    frames_tracked_++;
    lost_age_ = 0.0f;
    tracked_age_ += frame_scale;
    if (strong) strong_age_ += frame_scale;
    score_ = box.score;

    // Wald SPRT evidence accumulation (see bytetrack.h). A static object with a
    // perfect IoU used to be rewarded here, which is exactly what a backpack is;
    // only detector confidence (size-weighted) and shape rigidity count now.
    // Weighted by frame_scale: near-duplicate frames at high detector fps are not
    // independent evidence. The deformation penalty is not scaled: a >25% shape
    // jump is at least as implausible over a shorter interval.
    float delta_score = evidence(box, neutral_score) * frame_scale;
    // Anthropometric shape deformation penalty: rigid human skulls don't deform >25% between frames
    float delta_deform = (deformation > 0.25f) ? (deformation - 0.25f) * 4.0f : 0.0f;

    log_odds_ = std::clamp(log_odds_ + delta_score - delta_deform, kLogOddsMin, kLogOddsMax);
}

void STrack::activate(int frame_id) {
    state_ = TrackState::Tracked;
    is_activated_ = true;
    // Keep the accumulated hit count: it feeds the coasting trust policy.
    frames_tracked_ = std::max(1, frames_tracked_);
    frames_lost_ = 0;
    lost_age_ = 0.0f;
    // Boost log_odds to ensure confirmed state upon activation
    log_odds_ = std::max(log_odds_, kLogOddsConfirm + 0.5f);
}

void STrack::mark_lost(float frame_scale) {
    state_ = TrackState::Lost;
    frames_lost_++;
    lost_age_ += frame_scale;
    // Negative evidence decay when observation is missed
    log_odds_ = std::clamp(log_odds_ - 0.75f * frame_scale, kLogOddsMin, kLogOddsMax);
}

void STrack::note_gap(float frame_scale) {
    frames_lost_++;
    lost_age_ += frame_scale;
}

void STrack::mark_removed() {
    state_ = TrackState::Removed;
}

bool STrack::publishable() const {
    if (!is_activated_) return false;
    // A confirmed track that has since been sustained only by below-neutral
    // observations (or misses) has argued itself out of being a head.
    if (log_odds_ < 0.0f) return false;
    if (state_ != TrackState::Lost) return true;
    return tracked_age_ + kAgeEpsilon >= kMinHitsForCoast
            && lost_age_ <= kMaxCoastPublishFrames + kAgeEpsilon;
}

float STrack::mahalanobis_distance_sq(const HeadBox& box) const {
    float h = std::max(0.01f, box.h);
    float w = std::max(0.01f, box.w);
    float a = w / h;

    float r_pos = (1.0f / 20.0f) * h;
    float r = r_pos * r_pos;
    float r_a = 0.05f * 0.05f;

    return kf_cx_.mahalanobis_sq(box.cx, r) +
           kf_cy_.mahalanobis_sq(box.cy, r) +
           kf_a_.mahalanobis_sq(a, r_a) +
           kf_h_.mahalanobis_sq(h, r);
}

float STrack::shape_deformation(const HeadBox& box) const {
    float pred_h = std::max(0.01f, kf_h_.x);
    float pred_w = std::max(0.01f, kf_a_.x * pred_h);
    float def_w = std::abs(box.w - pred_w) / pred_w;
    float def_h = std::abs(box.h - pred_h) / pred_h;
    return std::max(def_w, def_h);
}

HeadBox STrack::current_box() const {
    float h = std::max(0.01f, kf_h_.x);
    float a = std::max(0.4f, std::min(1.8f, kf_a_.x));
    float w = a * h;
    float cx = kf_cx_.x;
    float cy = kf_cy_.x;

    HeadBox box;
    box.cx = cx;
    box.cy = cy;
    box.w = w;
    box.h = h;
    box.x1 = cx - w * 0.5f;
    box.y1 = cy - h * 0.5f;
    box.x2 = cx + w * 0.5f;
    box.y2 = cy + h * 0.5f;
    box.score = score_;
    return box;
}

void STrack::get_geometry(float* out6) const {
    HeadBox box = current_box();
    float halfWidth = box.w * 0.58f;
    float halfHeight = box.h * 0.65f;

    out6[0] = box.cx;
    out6[1] = box.cy;
    out6[2] = halfWidth;
    out6[3] = 0.0f;
    out6[4] = 0.0f;
    out6[5] = halfHeight;
}

// ByteTracker implementation
ByteTracker::ByteTracker(float high_threshold, float low_threshold, float match_threshold, int max_time_lost)
    : high_thresh_(high_threshold), low_thresh_(low_threshold),
      match_thresh_(match_threshold), max_time_lost_(max_time_lost) {}

float ByteTracker::frame_scale(float frame_dt_ms) {
    if (!std::isfinite(frame_dt_ms)) return 1.0f;
    return std::clamp(frame_dt_ms / kReferenceFrameMs, kMinFrameScale, 1.0f);
}

float ByteTracker::motion_scale(float frame_dt_ms) {
    if (!std::isfinite(frame_dt_ms)) return 1.0f;
    return std::clamp(frame_dt_ms / kReferenceFrameMs, kMinFrameScale, kMaxMotionScale);
}

void ByteTracker::reset() {
    tracked_stracks_.clear();
    unconfirmed_stracks_.clear();
    lost_stracks_.clear();
    frame_id_ = 0;
    next_track_id_ = 1;
}

float ByteTracker::iou_distance(const HeadBox& a, const HeadBox& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = (x2 > x1 && y2 > y1) ? (x2 - x1) * (y2 - y1) : 0.0f;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float denom = area_a + area_b - inter;
    float iou = denom > 0.0f ? inter / denom : 0.0f;
    return 1.0f - iou;
}

void ByteTracker::linear_assignment(const std::vector<HeadBox>& detections,
                                    std::vector<STrack>& tracks,
                                    float threshold,
                                    std::vector<std::pair<int, int>>& matches,
                                    std::vector<int>& unmatched_tracks,
                                    std::vector<int>& unmatched_detections,
                                    bool kinematic_fallback) {
    matches.clear();
    unmatched_tracks.clear();
    unmatched_detections.clear();

    const size_t num_det = detections.size();
    const size_t num_trk = tracks.size();

    if (num_det == 0) {
        for (size_t i = 0; i < num_trk; ++i) unmatched_tracks.push_back(static_cast<int>(i));
        return;
    }
    if (num_trk == 0) {
        for (size_t i = 0; i < num_det; ++i) unmatched_detections.push_back(static_cast<int>(i));
        return;
    }

    constexpr float kChiSquareGate = 20.0f; // df=4, p=0.001
    // Max relative box size change between two detector frames (~130 ms apart at
    // the measured 7-8 fps, not 33 ms): covers fast approach/retreat, rejects jumps
    // onto an unrelated box.
    constexpr float kMaxDeformation = 0.50f;

    std::vector<std::vector<float>> cost(num_trk, std::vector<float>(num_det));
    for (size_t t = 0; t < num_trk; ++t) {
        HeadBox track_box = tracks[t].current_box();
        for (size_t d = 0; d < num_det; ++d) {
            float d_iou = iou_distance(track_box, detections[d]);
            float d_maha = tracks[t].mahalanobis_distance_sq(detections[d]);
            float deform = tracks[t].shape_deformation(detections[d]);

            // Kinematic & Anthropometric Validation Gate:
            if (d_maha > kChiSquareGate || deform > kMaxDeformation) {
                cost[t][d] = 1e6f;
            } else {
                cost[t][d] = d_iou + 0.01f * d_maha;
                // At 3-5 detector fps a head can move past its own width between
                // frames, so IoU is 0 although the Kalman gate (whose covariance
                // grows with dt and time lost) still accepts it. Without this the
                // head became a fresh candidate and was unblurred until SPRT
                // re-confirmed it. Costlier than any good overlap match (IoU > .45),
                // and never below the stage threshold at the gate edge.
                if (kinematic_fallback) {
                    cost[t][d] = std::min(cost[t][d],
                            kKinematicBaseCost + kKinematicMahaCost * d_maha);
                }
            }
        }
    }

    std::vector<bool> track_used(num_trk, false);
    std::vector<bool> det_used(num_det, false);

    // Greedy matching for minimal cost
    while (true) {
        float min_cost = threshold;
        int best_t = -1;
        int best_d = -1;

        for (size_t t = 0; t < num_trk; ++t) {
            if (track_used[t]) continue;
            for (size_t d = 0; d < num_det; ++d) {
                if (det_used[d]) continue;
                if (cost[t][d] < min_cost) {
                    min_cost = cost[t][d];
                    best_t = static_cast<int>(t);
                    best_d = static_cast<int>(d);
                }
            }
        }

        if (best_t < 0 || best_d < 0) break;

        matches.emplace_back(best_t, best_d);
        track_used[best_t] = true;
        det_used[best_d] = true;
    }

    for (size_t t = 0; t < num_trk; ++t) {
        if (!track_used[t]) unmatched_tracks.push_back(static_cast<int>(t));
    }
    for (size_t d = 0; d < num_det; ++d) {
        if (!det_used[d]) unmatched_detections.push_back(static_cast<int>(d));
    }
}

std::vector<STrack> ByteTracker::update(const std::vector<HeadBox>& detections) {
    return update(detections, high_thresh_, low_thresh_, high_thresh_ + 0.18f);
}

std::vector<STrack> ByteTracker::update(const std::vector<HeadBox>& detections,
                                        float high_threshold, float low_threshold, float instant_threshold,
                                        float frame_dt_ms, float camera_dx, float camera_dy) {
    frame_id_++;
    const float scale = frame_scale(frame_dt_ms);
    const float motion = motion_scale(frame_dt_ms);
    // SPRT neutral point: the centre of the tracker's confidence band. Scores
    // above it are evidence for a head, below it against (see STrack::evidence).
    const float neutral = 0.5f * (high_threshold + low_threshold);

    std::vector<HeadBox> det_first;
    std::vector<HeadBox> det_second;

    for (const auto& det : detections) {
        if (det.score >= high_threshold) {
            det_first.push_back(det);
        } else if (det.score >= low_threshold) {
            det_second.push_back(det);
        }
    }

    // Predict all tracks (tracked, lost, unconfirmed)
    // Evidence is capped at one reference frame per detector frame (scale), but
    // motion is not (motion): on a slow device the head really moved for the
    // whole interval.
    for (auto& track : tracked_stracks_) {
        track.predict(motion, camera_dx, camera_dy);
    }
    for (auto& track : lost_stracks_) {
        track.predict(motion, camera_dx, camera_dy);
    }
    for (auto& track : unconfirmed_stracks_) {
        track.predict(motion, camera_dx, camera_dy);
    }

    // Step 1: Match high-score detections with confirmed tracked tracks
    std::vector<std::pair<int, int>> matches_first;
    std::vector<int> unmatched_tracks_first;
    std::vector<int> unmatched_detections_first;
    linear_assignment(det_first, tracked_stracks_, match_thresh_,
                      matches_first, unmatched_tracks_first, unmatched_detections_first,
                      /*kinematic_fallback=*/true);

    for (const auto& match : matches_first) {
        float iou = 1.0f - iou_distance(tracked_stracks_[match.first].current_box(), det_first[match.second]);
        tracked_stracks_[match.first].update(det_first[match.second], frame_id_, iou, neutral, scale);
    }

    // Step 2: Match low-score detections with remaining confirmed tracked tracks
    std::vector<STrack> remaining_tracked;
    for (int idx : unmatched_tracks_first) {
        remaining_tracked.push_back(tracked_stracks_[idx]);
    }

    std::vector<std::pair<int, int>> matches_second;
    std::vector<int> unmatched_tracks_second;
    std::vector<int> unmatched_detections_second;
    linear_assignment(det_second, remaining_tracked, 0.5f,
                      matches_second, unmatched_tracks_second, unmatched_detections_second);

    // Stage-2 detections scored below high_threshold: too weak to start a track.
    // For a young track (little strong observation time) they must not feed it
    // either: measured against high their evidence is never positive, so they
    // keep it localised while its log-odds drain. Device trace 2026-09-26: a
    // dumbbell track was sustained at 0.23-0.30 that way. An established head
    // (>= kEstablishedStrongAge reference frames at >= high) is judged against
    // the band centre as before: a real head turned to profile sits at 0.3-0.4
    // for seconds, and draining it against high unblurred it after ~4 s.
    for (const auto& match : matches_second) {
        STrack& trk = remaining_tracked[match.first];
        float iou = 1.0f - iou_distance(trk.current_box(), det_second[match.second]);
        float stage2_neutral = trk.strong_age() + kAgeEpsilon >= kEstablishedStrongAge
                ? neutral : high_threshold;
        trk.update(det_second[match.second], frame_id_, iou, stage2_neutral, scale,
                   /*strong=*/false);
    }

    // Those still unmatched in step 2 become lost (coasting)
    std::vector<STrack> newly_lost;
    for (int idx : unmatched_tracks_second) {
        remaining_tracked[idx].mark_lost(scale);
        newly_lost.push_back(remaining_tracked[idx]);
    }

    // Step 3: Match remaining high-score detections with lost tracks (re-identification)
    std::vector<HeadBox> rem_det_first;
    for (int idx : unmatched_detections_first) {
        rem_det_first.push_back(det_first[idx]);
    }

    std::vector<std::pair<int, int>> matches_lost;
    std::vector<int> unmatched_lost;
    std::vector<int> unmatched_re_detections;
    linear_assignment(rem_det_first, lost_stracks_, match_thresh_,
                      matches_lost, unmatched_lost, unmatched_re_detections,
                      /*kinematic_fallback=*/true);

    std::vector<STrack> reactivated_from_lost;
    for (const auto& match : matches_lost) {
        float iou = 1.0f - iou_distance(lost_stracks_[match.first].current_box(), rem_det_first[match.second]);
        lost_stracks_[match.first].update(rem_det_first[match.second], frame_id_, iou, neutral, scale);
        reactivated_from_lost.push_back(lost_stracks_[match.first]);
    }

    // Update lost tracks list: increment frames_lost for unmatched lost tracks.
    // Suppress persistent ghost blur for transient false positives: tracks that
    // were observed for fewer than kMinHitsForCoast reference frames are kept for
    // re-identification for only 2 (and are never published while lost, see
    // publishable()); established tracks coast up to max_time_lost_. All in
    // reference frames (tracked_age / lost_age), i.e. wall time.
    auto keep_lost = [this](const STrack& trk) {
        float allowed_lost = (trk.tracked_age() + kAgeEpsilon >= STrack::kMinHitsForCoast)
                ? static_cast<float>(max_time_lost_) : 2.0f;
        return trk.lost_age() <= allowed_lost + kAgeEpsilon;
    };
    std::vector<STrack> next_lost;
    for (int idx : unmatched_lost) {
        lost_stracks_[idx].mark_lost(scale);
        if (keep_lost(lost_stracks_[idx])) {
            next_lost.push_back(lost_stracks_[idx]);
        }
    }
    for (const auto& trk : newly_lost) {
        if (keep_lost(trk)) {
            next_lost.push_back(trk);
        }
    }
    lost_stracks_ = next_lost;

    // Step 4: Match remaining high-score detections with unconfirmed tracks (2nd-frame confirmation)
    std::vector<HeadBox> rem_det_for_unconfirmed;
    for (int idx : unmatched_re_detections) {
        rem_det_for_unconfirmed.push_back(rem_det_first[idx]);
    }

    std::vector<std::pair<int, int>> matches_unconfirmed;
    std::vector<int> unmatched_unconfirmed;
    std::vector<int> unmatched_detections_final;
    // Kinematic fallback here too: at 3-5 fps a walking head never overlaps its
    // previous box by IoU .3 and would otherwise never confirm. Evidence still
    // has to accumulate, and the recorded clutter traces jump far beyond the gate.
    linear_assignment(rem_det_for_unconfirmed, unconfirmed_stracks_, match_thresh_,
                      matches_unconfirmed, unmatched_unconfirmed, unmatched_detections_final,
                      /*kinematic_fallback=*/true);

    // A candidate is confirmed once its accumulated SPRT evidence reaches
    // kLogOddsConfirm (see bytetrack.h): confident, large heads confirm in two
    // reference frames, borderline or tiny candidates need several. It must be
    // re-detected at >= high_threshold meanwhile; a gap of kMaxCandidateGap
    // (~100 ms; one miss at the tuned cadence) drops it, since real heads are
    // re-detected consistently and clutter is not. Shorter gaps at high detector
    // fps are tolerated without evidence cost, otherwise the longer run of frames
    // SPRT needs there would be broken (or, with decay, delayed ~2x) by single
    // missed frames of a real, distant head.
    std::vector<STrack> newly_confirmed;
    std::vector<STrack> still_unconfirmed;
    for (const auto& match : matches_unconfirmed) {
        STrack& trk = unconfirmed_stracks_[match.first];
        float iou = 1.0f - iou_distance(trk.current_box(), rem_det_for_unconfirmed[match.second]);
        trk.update(rem_det_for_unconfirmed[match.second], frame_id_, iou, neutral, scale);
        if (trk.frames_tracked() >= STrack::kMinHits && trk.is_confirmed()) {
            trk.activate(frame_id_);
            newly_confirmed.push_back(trk);
        } else {
            still_unconfirmed.push_back(trk);
        }
    }
    for (int idx : unmatched_unconfirmed) {
        STrack& trk = unconfirmed_stracks_[idx];
        trk.note_gap(scale);
        if (trk.lost_age() < kMaxCandidateGap) {
            still_unconfirmed.push_back(trk);
        }
    }

    unconfirmed_stracks_ = still_unconfirmed;

    // Step 5: Process brand-new detections
    std::vector<STrack> newly_spawned_instant;
    for (int idx : unmatched_detections_final) {
        const auto& det = rem_det_for_unconfirmed[idx];
        bool reliable_size = det.w * det.h >= STrack::kInstantMinArea;
        if (det.score >= instant_threshold && reliable_size) {
            // Unambiguous, large head: instant activation on first frame
            STrack trk(det, next_track_id_++, neutral, scale);
            trk.activate(frame_id_);
            newly_spawned_instant.push_back(trk);
        } else {
            // Needs temporal confirmation over the next frames
            STrack trk(det, next_track_id_++, neutral, scale);
            unconfirmed_stracks_.push_back(trk);
        }
    }

    // Step 6: Reassemble active tracked tracks
    std::vector<STrack> next_tracked;
    next_tracked.reserve(matches_first.size() + matches_second.size() +
                         reactivated_from_lost.size() + newly_confirmed.size() +
                         newly_spawned_instant.size());

    for (const auto& match : matches_first) {
        next_tracked.push_back(tracked_stracks_[match.first]);
    }
    for (const auto& match : matches_second) {
        next_tracked.push_back(remaining_tracked[match.first]);
    }
    for (const auto& trk : reactivated_from_lost) {
        next_tracked.push_back(trk);
    }
    for (const auto& trk : newly_confirmed) {
        next_tracked.push_back(trk);
    }
    for (const auto& trk : newly_spawned_instant) {
        next_tracked.push_back(trk);
    }
    tracked_stracks_ = next_tracked;

    // Step 7: Output confirmed active tracks, plus briefly coasted lost tracks that
    // were observed long enough to be trusted (see STrack::publishable). The Java
    // layer adds its own time-based hold, so native coasting stays short.
    std::vector<STrack> output_tracks;
    output_tracks.reserve(tracked_stracks_.size() + lost_stracks_.size());
    for (const auto& trk : tracked_stracks_) {
        if (trk.publishable()) {
            output_tracks.push_back(trk);
        }
    }
    for (const auto& trk : lost_stracks_) {
        if (trk.publishable()) {
            output_tracks.push_back(trk);
        }
    }
    return output_tracks;
}

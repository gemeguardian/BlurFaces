#ifndef BYTETRACK_H
#define BYTETRACK_H

#include "head_detector.h"
#include <vector>
#include <cstdint>

enum class TrackState {
    New = 0,
    Tracked = 1,
    Lost = 2,
    Removed = 3
};

struct KalmanState1D {
    float x = 0.0f;
    float v = 0.0f;
    float p00 = 1.0f;
    float p01 = 0.0f;
    float p10 = 0.0f;
    float p11 = 1.0f;

    void init(float init_x, float init_v, float std_p0, float std_p1);
    // s: step length in reference frames (1 = the tuned ~130 ms cadence).
    void predict(float q00, float q01, float q10, float q11, float s = 1.0f);
    void update(float measurement, float r);

    float innovation_var(float r) const { return p00 + r; }
    float mahalanobis_sq(float measurement, float r) const {
        float s = p00 + r;
        return (s > 1e-6f) ? ((measurement - x) * (measurement - x) / s) : 0.0f;
    }
};

class STrack {
public:
    STrack();
    STrack(const HeadBox& box, int track_id);
    // frame_scale: this detector frame's weight in reference frames (see
    // ByteTracker::kReferenceFrameMs); 1 at the cadence the constants were tuned at.
    STrack(const HeadBox& box, int track_id, float neutral_score, float frame_scale = 1.0f);
    ~STrack() = default;

    // motion_scale: time since the previous prediction in reference frames (not
    // capped at 1, unlike evidence). camera_dx/dy: global image shift since the
    // previous detector frame in normalised coordinates (camera motion compensation).
    void predict(float motion_scale = 1.0f, float camera_dx = 0.0f, float camera_dy = 0.0f);
    void update(const HeadBox& box, int frame_id);
    void update(const HeadBox& box, int frame_id, float iou);
    // strong: the detection scored >= high_threshold (stage 1).
    void update(const HeadBox& box, int frame_id, float iou, float neutral_score,
                float frame_scale = 1.0f, bool strong = true);
    void mark_lost(float frame_scale = 1.0f);
    // An unconfirmed candidate missed this frame: ages it without evidence decay
    // (the gap length itself is bounded by ByteTracker).
    void note_gap(float frame_scale);
    void mark_removed();
    void activate(int frame_id);
    // Whether the renderer may draw this track right now (confirmed, and if lost,
    // trusted enough to coast).
    bool publishable() const;

    int track_id() const { return track_id_; }
    TrackState state() const { return state_; }
    int frames_lost() const { return frames_lost_; }
    int frames_tracked() const { return frames_tracked_; }
    // Observed / missed time in reference frames; equal to the frame counts at
    // the reference cadence. Coasting and lost-track policies use these.
    float tracked_age() const { return tracked_age_; }
    float lost_age() const { return lost_age_; }
    // Observed time with detections >= high_threshold, in reference frames.
    float strong_age() const { return strong_age_; }
    float score() const { return score_; }
    bool is_activated() const { return is_activated_; }
    float log_odds() const { return log_odds_; }
    bool is_confirmed() const { return log_odds_ >= kLogOddsConfirm; }

    float mahalanobis_distance_sq(const HeadBox& box) const;
    float shape_deformation(const HeadBox& box) const;

    HeadBox current_box() const;
    void get_geometry(float* out6) const;

    // Sequential Probability Ratio Test (SPRT) confirmation.
    //
    // Every observation adds evidence = logit(score) - logit(neutral), where
    // neutral is the midpoint of the tracker's low/high thresholds, so scores at
    // the band centre are uninformative, high scores count for, low scores count
    // against. Exception: stage-2 (below high) matches of a young track, with
    // less than 6 reference frames of strong observations, use neutral = high,
    // so they can only drain it (see ByteTracker::update). Evidence is weighted
    // by the frame's duration in reference frames (ByteTracker::frame_scale). The per-frame contribution is capped, so no single frame confirms
    // a track on its own, and scaled by size reliability: at 192 px capture a
    // 10%-wide head is ~19 px, where YOLOv8n is least trustworthy (device trace
    // 2026-09-22: a backpack scored 0.71/0.45/0.42 in a 10%x9% box), so tiny
    // boxes must persist several times longer than large ones.
    //
    // Effect at neutral 0.27 (Normal preset, daylight): a selfie head at 0.76
    // confirms in 2 frames; a 10%x20% head at 0.9 needs 3; a 0.45 candidate of
    // reliable size needs 4; the backpack trace above (0.71, 0.45, then a run
    // of ~0.42) never confirms.
    static constexpr float kLogOddsConfirm = 1.5f;
    static constexpr float kLogOddsMin = -3.0f;
    static constexpr float kLogOddsMax = 6.0f;
    static constexpr float kEvidenceCap = 0.75f;
    static constexpr float kReliableArea = 0.03f;   // ~17% x 17% of the frame
    static constexpr float kMinSizeReliability = 0.35f;
    static float evidence(const HeadBox& box, float neutral_score);

    // Single-frame (instant) activation is reserved for boxes large enough for
    // the detector to be reliable; small boxes always go through SPRT.
    static constexpr float kInstantMinArea = kReliableArea;

    // Hit-count floor on top of SPRT; the evidence cap already guarantees >= 2.
    static constexpr int kMinHits = 2;
    // Lost tracks are only coasted to the renderer when they were observed long
    // enough to be trusted, and only briefly: each reference frame of coasting
    // costs ~130 ms of ghost blur. Both are in reference frames (tracked_age /
    // lost_age), not raw detector frames. The Java layer adds its own hold on top.
    static constexpr int kMinHitsForCoast = 6;
    static constexpr int kMaxCoastPublishFrames = 4;

private:
    int track_id_ = 0;
    TrackState state_ = TrackState::New;
    int frames_lost_ = 0;
    int frames_tracked_ = 0;
    float score_ = 0.0f;
    bool is_activated_ = false;
    float log_odds_ = 0.0f; // Sequential Probability Ratio Test (SPRT) Log-Odds
    float tracked_age_ = 0.0f;
    float lost_age_ = 0.0f;
    float strong_age_ = 0.0f;

    // 4 decoupled Kalman filters: cx, cy, aspect_ratio (w/h), h
    KalmanState1D kf_cx_;
    KalmanState1D kf_cy_;
    KalmanState1D kf_a_;
    KalmanState1D kf_h_;

    void init_kalman(const HeadBox& box);
};

class ByteTracker {
public:
    // How long an established lost track stays re-identifiable (it is published
    // for at most STrack::kMaxCoastPublishFrames of these). Counted in reference
    // frames (lost_age; one per detector frame at or below ~7-8 fps), not camera
    // frames: upstream ByteTrack's 30 assumes 30 fps, which kept ghosts
    // re-identifiable for ~4 s. 8 is the intended ~1 s.
    static constexpr int kMaxTimeLostFrames = 8;

    // All frame-denominated constants (SPRT evidence per frame, coast and
    // lost-track windows) were tuned at ~7-8 detector fps. On a faster device
    // (measured 2026-09-26: ~24 fps, 7-50 ms inference) consecutive frames are
    // near-duplicates, and counting each as an independent observation let
    // dumbbells and clutter confirm in ~100 ms. Each frame is therefore weighted
    // by dt / kReferenceFrameMs, capped at 1 so slow devices behave as tuned.
    static constexpr float kReferenceFrameMs = 130.0f;
    static constexpr float kMinFrameScale = 0.1f;
    static float frame_scale(float frame_dt_ms);
    // Kalman prediction is scaled by the real interval (a slow device's head
    // really moves for the whole ~250-400 ms), up to this many reference frames.
    static constexpr float kMaxMotionScale = 4.0f;
    static float motion_scale(float frame_dt_ms);

    ByteTracker(float high_threshold = 0.45f,
                float low_threshold = 0.20f,
                float match_threshold = 0.70f,
                int max_time_lost = kMaxTimeLostFrames);
    ~ByteTracker() = default;

    // Updates tracks with new detections and returns active confirmed tracks for rendering
    std::vector<STrack> update(const std::vector<HeadBox>& detections);
    std::vector<STrack> update(const std::vector<HeadBox>& detections,
                               float high_threshold, float low_threshold, float instant_threshold,
                               float frame_dt_ms = kReferenceFrameMs,
                               float camera_dx = 0.0f, float camera_dy = 0.0f);
    void reset();

private:
    float high_thresh_;
    float low_thresh_;
    float match_thresh_;
    int max_time_lost_;
    int frame_id_ = 0;
    int next_track_id_ = 1;

    std::vector<STrack> tracked_stracks_;
    std::vector<STrack> unconfirmed_stracks_;
    std::vector<STrack> lost_stracks_;

    static float iou_distance(const HeadBox& a, const HeadBox& b);
    void linear_assignment(const std::vector<HeadBox>& detections,
                           std::vector<STrack>& tracks,
                           float threshold,
                           std::vector<std::pair<int, int>>& matches,
                           std::vector<int>& unmatched_tracks,
                           std::vector<int>& unmatched_detections,
                           bool kinematic_fallback = false);
};

#endif // BYTETRACK_H

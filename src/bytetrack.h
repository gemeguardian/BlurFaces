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
    void predict(float q00, float q01, float q10, float q11);
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
    ~STrack() = default;

    void predict();
    void update(const HeadBox& box, int frame_id);
    void update(const HeadBox& box, int frame_id, float iou);
    void mark_lost();
    void mark_removed();
    void activate(int frame_id);

    int track_id() const { return track_id_; }
    TrackState state() const { return state_; }
    int frames_lost() const { return frames_lost_; }
    int frames_tracked() const { return frames_tracked_; }
    float score() const { return score_; }
    bool is_activated() const { return is_activated_; }
    float log_odds() const { return log_odds_; }
    bool is_confirmed() const { return log_odds_ >= kLogOddsConfirm; }

    float mahalanobis_distance_sq(const HeadBox& box) const;
    float shape_deformation(const HeadBox& box) const;

    HeadBox current_box() const;
    void get_geometry(float* out6) const;

    static constexpr float kLogOddsConfirm = 1.0f; // ~73% posterior confidence
    static constexpr float kLogOddsMin = -3.0f;
    static constexpr float kLogOddsMax = 8.0f;

private:
    int track_id_ = 0;
    TrackState state_ = TrackState::New;
    int frames_lost_ = 0;
    int frames_tracked_ = 0;
    float score_ = 0.0f;
    bool is_activated_ = false;
    float log_odds_ = 0.0f; // Sequential Probability Ratio Test (SPRT) Log-Odds

    // 4 decoupled Kalman filters: cx, cy, aspect_ratio (w/h), h
    KalmanState1D kf_cx_;
    KalmanState1D kf_cy_;
    KalmanState1D kf_a_;
    KalmanState1D kf_h_;

    void init_kalman(const HeadBox& box);
};

class ByteTracker {
public:
    ByteTracker(float high_threshold = 0.45f,
                float low_threshold = 0.20f,
                float match_threshold = 0.70f,
                int max_time_lost = 30);
    ~ByteTracker() = default;

    // Updates tracks with new detections and returns active confirmed tracks for rendering
    std::vector<STrack> update(const std::vector<HeadBox>& detections);
    std::vector<STrack> update(const std::vector<HeadBox>& detections,
                               float high_threshold, float low_threshold, float instant_threshold);
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
                           std::vector<int>& unmatched_detections);
};

#endif // BYTETRACK_H

#ifndef FACE_TRACKER_H
#define FACE_TRACKER_H

#include <vector>
#include <cstdint>

struct TrackedPoint {
    float x, y;           // current position
    float vx, vy;         // velocity
    bool active;
    int missed;           // frames since last seen
};

struct TrackedFace {
    float cx, cy;         // center position (pixels)
    float w, h;           // size
    float vx, vy;         // velocity
    bool active;
    int missed;
    std::vector<TrackedPoint> points;  // feature points for optical flow
};

class FaceTracker {
public:
    FaceTracker();
    ~FaceTracker();

    // Initialize tracker with detected face
    void init(const unsigned char* prev_rgba, int width, int height,
              float face_x, float face_y, float face_w, float face_h);

    // Track face to new frame using optical flow
    // Returns true if tracking successful
    bool track(const unsigned char* curr_rgba, int width, int height);

    // Get current face rect
    void getRect(float* x, float* y, float* w, float* h) const;

    // Predict position after dt seconds
    void predict(float dt, float* x, float* y) const;

    bool isActive() const { return active_; }
    void markMissed() { missed_++; if (missed_ > 10) active_ = false; }
    void reset();

private:
    // Lucas-Kanade optical flow for a single point
    bool trackPointLK(const unsigned char* prev_gray, const unsigned char* curr_gray,
                       int width, int height, float& px, float& py);
    bool searchPointPatch(const unsigned char* prev_gray, const unsigned char* curr_gray,
                          int width, int height, float old_x, float old_y,
                          float guess_x, float guess_y,
                          float& out_x, float& out_y);
    bool estimateGlobalMotion(const unsigned char* prev_gray, const unsigned char* curr_gray,
                              int width, int height, float& dx, float& dy);

    // Extract good features to track inside face region
    void extractFeatures(const unsigned char* rgba, int width, int height,
                         float fx, float fy, float fw, float fh);

    // Convert RGBA to grayscale
    void rgbaToGray(const unsigned char* rgba, unsigned char* gray, int width, int height);

    bool active_ = false;
    int missed_ = 0;
    int width_ = 0, height_ = 0;

    // Face state
    float cx_ = 0, cy_ = 0;   // center
    float w_ = 0, h_ = 0;     // size
    float vx_ = 0, vy_ = 0;   // velocity
    float last_global_dx_ = 0, last_global_dy_ = 0;
    float last_face_dx_ = 0, last_face_dy_ = 0;
    float last_a_ = 1.0f, last_b_ = 0.0f;
    bool have_motion_model_ = false;

    // Previous frame grayscale for optical flow
    std::vector<unsigned char> prev_gray_;
    std::vector<TrackedPoint> points_;

    // LK parameters
    static constexpr int WIN_SIZE = 21;      // window size for LK
    static constexpr int MAX_LEVELS = 3;     // pyramid levels
    static constexpr int MAX_POINTS = 30;    // max feature points per face
    static constexpr float MIN_EIG_THRESHOLD = 1e-4f;
};

#endif // FACE_TRACKER_H

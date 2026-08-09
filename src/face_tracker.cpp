#include "face_tracker.h"
#include <android/log.h>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

#define LOG_TAG "FaceTracker"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

FaceTracker::FaceTracker() {}
FaceTracker::~FaceTracker() {}

void FaceTracker::reset() {
    active_ = false;
    missed_ = 0;
    prev_gray_.clear();
    points_.clear();
    vx_ = vy_ = 0;
}

void FaceTracker::rgbaToGray(const unsigned char* rgba, unsigned char* gray, int width, int height) {
    const int pixels = width * height;
    for (int i = 0; i < pixels; i++) {
        const unsigned char* p = rgba + i * 4;
        // ITU-R BT.601 luma
        gray[i] = static_cast<unsigned char>(
            (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8
        );
    }
}

void FaceTracker::init(const unsigned char* prev_rgba, int width, int height,
                       float face_x, float face_y, float face_w, float face_h) {
    width_ = width;
    height_ = height;
    cx_ = face_x + face_w / 2;
    cy_ = face_y + face_h / 2;
    w_ = face_w;
    h_ = face_h;
    vx_ = vy_ = 0;
    missed_ = 0;
    active_ = true;

    // Store previous frame grayscale
    prev_gray_.resize(width * height);
    rgbaToGray(prev_rgba, prev_gray_.data(), width, height);

    // Extract feature points inside face region
    extractFeatures(prev_rgba, width, height, face_x, face_y, face_w, face_h);

    LOGI("Tracker init: face at (%.1f, %.1f) size %.1fx%.1f, %zu points",
         cx_, cy_, w_, h_, points_.size());
}

void FaceTracker::extractFeatures(const unsigned char* rgba, int width, int height,
                                  float fx, float fy, float fw, float fh) {
    points_.clear();
    const int x0 = static_cast<int>(std::max(0.0f, fx));
    const int y0 = static_cast<int>(std::max(0.0f, fy));
    const int x1 = static_cast<int>(std::min(static_cast<float>(width), fx + fw));
    const int y1 = static_cast<int>(std::min(static_cast<float>(height), fy + fh));
    const int region_w = x1 - x0;
    const int region_h = y1 - y0;
    if (region_w <= 0 || region_h <= 0) return;

    // Grid-based feature selection: pick points with high gradient
    const int grid_cols = 5;
    const int grid_rows = 6;
    const int cell_w = region_w / grid_cols;
    const int cell_h = region_h / grid_rows;

    std::vector<unsigned char> gray(width * height);
    rgbaToGray(rgba, gray.data(), width, height);

    for (int gy = 0; gy < grid_rows && points_.size() < MAX_POINTS; gy++) {
        for (int gx = 0; gx < grid_cols && points_.size() < MAX_POINTS; gx++) {
            int cx = x0 + gx * cell_w + cell_w / 2;
            int cy = y0 + gy * cell_h + cell_h / 2;
            if (cx < 1 || cx >= width - 1 || cy < 1 || cy >= height - 1) continue;

            // Compute gradient magnitude
            int idx = cy * width + cx;
            int gx_val = gray[idx + 1] - gray[idx - 1];
            int gy_val = gray[idx + width] - gray[idx - width];
            int grad = gx_val * gx_val + gy_val * gy_val;

            // Accept point if gradient is strong enough (good for tracking)
            if (grad > 100) {
                TrackedPoint pt;
                pt.x = static_cast<float>(cx);
                pt.y = static_cast<float>(cy);
                pt.vx = pt.vy = 0;
                pt.active = true;
                pt.missed = 0;
                points_.push_back(pt);
            }
        }
    }

    // If not enough gradient points, add grid points uniformly
    if (points_.size() < 10) {
        for (int gy = 0; gy < grid_rows && points_.size() < MAX_POINTS; gy++) {
            for (int gx = 0; gx < grid_cols && points_.size() < MAX_POINTS; gx++) {
                int cx = x0 + gx * cell_w + cell_w / 2;
                int cy = y0 + gy * cell_h + cell_h / 2;
                if (cx < 1 || cx >= width - 1 || cy < 1 || cy >= height - 1) continue;
                TrackedPoint pt;
                pt.x = static_cast<float>(cx);
                pt.y = static_cast<float>(cy);
                pt.vx = pt.vy = 0;
                pt.active = true;
                pt.missed = 0;
                points_.push_back(pt);
            }
        }
    }
}

bool FaceTracker::trackPointLK(const unsigned char* prev_gray, const unsigned char* curr_gray,
                               int width, int height, float& px, float& py) {
    const int x = static_cast<int>(px);
    const int y = static_cast<int>(py);
    const int half = WIN_SIZE / 2;

    if (x < half || x >= width - half || y < half || y >= height - half) {
        return false;
    }

    // Compute spatial gradients on previous frame
    float Ix[WIN_SIZE * WIN_SIZE];
    float Iy[WIN_SIZE * WIN_SIZE];
    float It[WIN_SIZE * WIN_SIZE];
    int idx = 0;

    for (int dy = -half; dy <= half; dy++) {
        for (int dx = -half; dx <= half; dx++) {
            int px_ = x + dx;
            int py_ = y + dy;
            int i = py_ * width + px_;

            // Central differences
            Ix[idx] = (prev_gray[i + 1] - prev_gray[i - 1]) * 0.5f;
            Iy[idx] = (prev_gray[i + width] - prev_gray[i - width]) * 0.5f;
            It[idx] = curr_gray[i] - prev_gray[i];
            idx++;
        }
    }

    // Solve LK equation: (A^T A) v = A^T b
    // where A = [Ix, Iy], b = -It
    float sumIx2 = 0, sumIy2 = 0, sumIxIy = 0;
    float sumIxIt = 0, sumIyIt = 0;

    for (int i = 0; i < idx; i++) {
        sumIx2 += Ix[i] * Ix[i];
        sumIy2 += Iy[i] * Iy[i];
        sumIxIy += Ix[i] * Iy[i];
        sumIxIt += Ix[i] * It[i];
        sumIyIt += Iy[i] * It[i];
    }

    // 2x2 matrix inverse
    float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
    if (std::abs(det) < MIN_EIG_THRESHOLD) {
        return false; // not enough texture
    }

    float invDet = 1.0f / det;
    float u = invDet * (sumIy2 * (-sumIxIt) - sumIxIy * (-sumIyIt));
    float v = invDet * (-sumIxIy * (-sumIxIt) + sumIx2 * (-sumIyIt));

    // Clamp large motions (likely tracking failure)
    const float max_motion = 15.0f;
    if (std::abs(u) > max_motion || std::abs(v) > max_motion) {
        return false;
    }

    px += u;
    py += v;

    // Keep in bounds
    px = std::max(static_cast<float>(half), std::min(static_cast<float>(width - half - 1), px));
    py = std::max(static_cast<float>(half), std::min(static_cast<float>(height - half - 1), py));

    return true;
}

bool FaceTracker::searchPointPatch(const unsigned char* prev_gray, const unsigned char* curr_gray,
                                    int width, int height, float old_x, float old_y,
                                    float guess_x, float guess_y,
                                    float& out_x, float& out_y) {
    // LK's 15 px clamp is too small when a 320 px capture jumps between camera
    // frames. Search around the predicted location, not only around the old
    // location: otherwise a successful coarse match can still be rejected as
    // a zero-motion patch when the face has moved more than one LK window.
    const int radius = 72;
    const int half = 3;
    const int ox = static_cast<int>(old_x);
    const int oy = static_cast<int>(old_y);
    const int gx = static_cast<int>(guess_x);
    const int gy = static_cast<int>(guess_y);
    if (ox < half || oy < half || ox >= width - half || oy >= height - half) return false;

    auto patchScore = [&](int cx, int cy) {
        int score = 0;
        for (int py = -half; py <= half; ++py) for (int px = -half; px <= half; ++px) {
            const int a = prev_gray[(oy + py) * width + (ox + px)];
            const int b = curr_gray[(cy + py) * width + (cx + px)];
            score += std::abs(a - b);
        }
        return score;
    };

    int bestScore = std::numeric_limits<int>::max();
    int bestX = gx, bestY = gy;
    const int minX = std::max(half, gx - radius);
    const int maxX = std::min(width - half - 1, gx + radius);
    const int minY = std::max(half, gy - radius);
    const int maxY = std::min(height - half - 1, gy + radius);
    // Search every pixel in the bounded window. At 320x320 and 30 points this
    // remains worker-side and avoids parity gaps that can miss a true match by
    // one pixel during a diagonal shake.
    for (int cy = minY; cy <= maxY; ++cy) {
        for (int cx = minX; cx <= maxX; ++cx) {
            const int score = patchScore(cx, cy);
            if (score < bestScore) { bestScore = score; bestX = cx; bestY = cy; }
        }
    }

    // Refine around the coarse winner so the box does not quantise to a
    // visible 4 px staircase during fast movement.
    const int refine = 6;
    for (int dy = -refine; dy <= refine; ++dy) {
        for (int dx = -refine; dx <= refine; ++dx) {
            const int cx = bestX + dx, cy = bestY + dy;
            if (cx < half || cy < half || cx >= width - half || cy >= height - half) continue;
            const int score = patchScore(cx, cy);
            if (score < bestScore) { bestScore = score; bestX = cx; bestY = cy; }
        }
    }

    // Keep a looser gate for a large motion/rotation: the robust consensus
    // below rejects isolated false matches, while a strict per-patch gate
    // otherwise turns a partial face overlap into a frozen rectangle.
    if (bestScore > 28 * 49) return false;
    out_x = static_cast<float>(bestX); out_y = static_cast<float>(bestY);
    return true;
}

bool FaceTracker::estimateGlobalMotion(const unsigned char* prev_gray, const unsigned char* curr_gray,
                                       int width, int height, float& dx, float& dy) {
    // Estimate camera shake from the background outside the face ROI. This is
    // a tiny coarse translation search, not a second detector; it gives LK a
    // motion prior before the face points are matched.
    const int step = 8;
    const int radius = 48;
    const int half = 2;
    const int skip_left = std::max(0, static_cast<int>(cx_ - w_ * 1.15f));
    const int skip_top = std::max(0, static_cast<int>(cy_ - h_ * 1.15f));
    const int skip_right = std::min(width, static_cast<int>(cx_ + w_ * 1.15f));
    const int skip_bottom = std::min(height, static_cast<int>(cy_ + h_ * 1.15f));
    int best = std::numeric_limits<int>::max();
    int bestDx = 0, bestDy = 0;
    for (int ddy = -radius; ddy <= radius; ddy += 4) {
        for (int ddx = -radius; ddx <= radius; ddx += 4) {
            int score = 0, samples = 0;
            for (int y = half + step; y < height - half - step; y += step) {
                for (int x = half + step; x < width - half - step; x += step) {
                    if (x >= skip_left && x < skip_right && y >= skip_top && y < skip_bottom) continue;
                    const int nx = x + ddx, ny = y + ddy;
                    if (nx < half || nx >= width - half || ny < half || ny >= height - half) continue;
                    for (int py = -half; py <= half; ++py) for (int px = -half; px <= half; ++px) {
                        score += std::abs(static_cast<int>(prev_gray[y * width + x]) -
                                          static_cast<int>(curr_gray[ny * width + nx]));
                    }
                    if (++samples >= 160) break;
                }
                if (samples >= 160) break;
            }
            if (samples > 0 && score < best) { best = score; bestDx = ddx; bestDy = ddy; }
        }
    }
    if (best == std::numeric_limits<int>::max()) return false;
    dx = static_cast<float>(bestDx); dy = static_cast<float>(bestDy);
    return true;
}

bool FaceTracker::track(const unsigned char* curr_rgba, int width, int height) {
    if (!active_ || prev_gray_.empty() || points_.empty() || width != width_ || height != height_) {
        return false;
    }

    // Convert current frame to grayscale
    std::vector<unsigned char> curr_gray(width * height);
    rgbaToGray(curr_rgba, curr_gray.data(), width, height);

    struct Motion {
        float old_x, old_y, new_x, new_y, dx, dy;
        int point_index;
    };
    float global_dx = 0.0f, global_dy = 0.0f;
    const bool global_motion_ok = estimateGlobalMotion(prev_gray_.data(), curr_gray.data(),
                                                       width, height, global_dx, global_dy);
    std::vector<Motion> motions;
    motions.reserve(points_.size());

    for (int point_index = 0; point_index < static_cast<int>(points_.size()); ++point_index) {
        auto& pt = points_[point_index];
        if (!pt.active) continue;
        float old_x = pt.x;
        float old_y = pt.y;
        float lk_x = old_x;
        float lk_y = old_y;
        const bool lk_ok = trackPointLK(prev_gray_.data(), curr_gray.data(), width, height, lk_x, lk_y);
        // Validate LK against an appearance match on every worker frame. A
        // one-iteration LK solver can return a plausible near-zero vector on a
        // hard shake, which would otherwise publish a frozen mask forever.
        // Global motion is only a fallback prior. The face can move
        // independently of the background, so using it as the unconditional
        // seed freezes the mask during camera shake.
            const float lk_dx = lk_x - old_x;
        const float lk_dy = lk_y - old_y;
        const bool lk_is_background_prior = global_motion_ok &&
                std::abs(lk_dx - global_dx) <= 3.0f &&
                std::abs(lk_dy - global_dy) <= 3.0f;
        const float prior_dx = lk_ok && !lk_is_background_prior
                ? lk_dx : (global_motion_ok ? global_dx : vx_);
        const float prior_dy = lk_ok && !lk_is_background_prior
                ? lk_dy : (global_motion_ok ? global_dy : vy_);
        const float guess_x = old_x + std::max(-64.0f, std::min(64.0f, prior_dx));
        const float guess_y = old_y + std::max(-64.0f, std::min(64.0f, prior_dy));
        float matched_x = guess_x;
        float matched_y = guess_y;
        const bool patch_ok = searchPointPatch(prev_gray_.data(), curr_gray.data(), width, height,
                                               old_x, old_y, guess_x, guess_y,
                                               matched_x, matched_y);
        // On camera shake LK often lands on the previous/background position
        // while the patch search finds the face. Never let that weaker result
        // win after a valid appearance match.
        bool ok = patch_ok || (lk_ok && !lk_is_background_prior);
        if (patch_ok) {
            pt.x = matched_x;
            pt.y = matched_y;
        } else if (lk_ok && !lk_is_background_prior) {
            pt.x = lk_x;
            pt.y = lk_y;
        }

        if (ok) {
            float dx = pt.x - old_x;
            float dy = pt.y - old_y;
            // A patch match can be valid even when the coarse global prior is
            // wrong, so accept the appearance result across the full bounded
            // search window instead of requiring LK's smaller motion gate.
            if (std::isfinite(dx) && std::isfinite(dy) &&
                    std::abs(dx) <= 64.0f && std::abs(dy) <= 64.0f) {
                motions.push_back({old_x, old_y, pt.x, pt.y, dx, dy, point_index});
            } else {
                pt.missed++;
            }
        }

        if (ok && !motions.empty() && motions.back().point_index == point_index) {
            pt.missed = 0;
        } else {
            pt.missed++;
            if (pt.missed > 2) pt.active = false;
        }
    }

    if (motions.size() < 5) {
        // Too few points tracked — fail closed and let the next worker frame
        // force SCRFD reacquisition instead of publishing a frozen rectangle.
        markMissed();
        return false;
    }

    auto median = [](std::vector<float> values) {
        std::sort(values.begin(), values.end());
        const size_t middle = values.size() / 2;
        if ((values.size() & 1u) != 0u) return values[middle];
        return (values[middle - 1] + values[middle]) * 0.5f;
    };

    std::vector<float> dxs, dys;
    dxs.reserve(motions.size()); dys.reserve(motions.size());
    for (const Motion& motion : motions) {
        dxs.push_back(motion.dx); dys.push_back(motion.dy);
    }
    const float median_dx = median(dxs);
    const float median_dy = median(dys);

    std::vector<float> abs_dx, abs_dy;
    abs_dx.reserve(motions.size()); abs_dy.reserve(motions.size());
    for (const Motion& motion : motions) {
        abs_dx.push_back(std::abs(motion.dx - median_dx));
        abs_dy.push_back(std::abs(motion.dy - median_dy));
    }
    const float threshold_x = std::max(5.0f, 3.0f * median(abs_dx));
    const float threshold_y = std::max(5.0f, 3.0f * median(abs_dy));

    std::vector<const Motion*> inliers;
    inliers.reserve(motions.size());
    for (const Motion& motion : motions) {
        if (std::abs(motion.dx - median_dx) <= threshold_x &&
                std::abs(motion.dy - median_dy) <= threshold_y) {
            inliers.push_back(&motion);
        } else {
            points_[motion.point_index].active = false;
        }
    }
    if (inliers.size() < 5) {
        markMissed();
        return false;
    }

    // The median is the robust face displacement. Averaging all accepted
    // points re-introduces a small one-sided patch bias every frame, so the
    // ellipse appears to lag behind a rapidly moving face.
    const float avg_dx = median_dx;
    const float avg_dy = median_dy;
    // Update face center and velocity
    float new_cx = cx_ + avg_dx;
    float new_cy = cy_ + avg_dy;

    // Keep the current box on the consensus motion. Velocity is only a
    // prediction aid for the Java encoder boundary and must not pull the
    // actual mask back toward a stale trajectory.
    vx_ = avg_dx;
    vy_ = avg_dy;

    cx_ = new_cx;
    cy_ = new_cy;

    // Estimate scale from the median radial change of feature points. A fixed
    // box works for translation but slips off the forehead/chin when the face
    // approaches the camera, so follow size conservatively.
    std::vector<float> scales;
    scales.reserve(inliers.size());
    for (const Motion* motion : inliers) {
        const float old_rx = motion->old_x - (cx_ - avg_dx);
        const float old_ry = motion->old_y - (cy_ - avg_dy);
        const float new_rx = motion->new_x - cx_;
        const float new_ry = motion->new_y - cy_;
        const float old_radius = std::sqrt(old_rx * old_rx + old_ry * old_ry);
        const float new_radius = std::sqrt(new_rx * new_rx + new_ry * new_ry);
        if (old_radius > 5.0f && new_radius > 1.0f) {
            const float ratio = new_radius / old_radius;
            if (std::isfinite(ratio) && ratio >= 0.6f && ratio <= 1.6f) scales.push_back(ratio);
        }
    }
    if (scales.size() >= 4) {
        const float scale = std::max(0.75f, std::min(1.33f, median(scales)));
        const float size_alpha = 0.35f;
        w_ *= 1.0f + size_alpha * (scale - 1.0f);
        h_ *= 1.0f + size_alpha * (scale - 1.0f);
        w_ = std::max(12.0f, std::min(static_cast<float>(width_) * 0.9f, w_));
        h_ = std::max(12.0f, std::min(static_cast<float>(height_) * 0.9f, h_));
    }

    // Update previous frame for next iteration
    prev_gray_.swap(curr_gray);

    // Re-extract features if too many lost
    if (inliers.size() < MAX_POINTS / 2) {
        extractFeatures(curr_rgba, width, height,
                       cx_ - w_ / 2, cy_ - h_ / 2, w_, h_);
    }

    return true;
}

void FaceTracker::getRect(float* x, float* y, float* w, float* h) const {
    *x = cx_ - w_ / 2;
    *y = cy_ - h_ / 2;
    *w = w_;
    *h = h_;
}

void FaceTracker::predict(float dt, float* x, float* y) const {
    *x = cx_ + vx_ * dt;
    *y = cy_ + vy_ * dt;
}

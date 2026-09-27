#include "camera_motion.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace {

// Box-filtered luma downscale of an RGBA frame to n x n.
void downscale_luma(const uint8_t* rgba, int width, int height, int n, std::vector<uint8_t>& out) {
    out.assign(static_cast<size_t>(n) * n, 0);
    for (int y = 0; y < n; ++y) {
        int y0 = y * height / n;
        int y1 = std::max(y0 + 1, (y + 1) * height / n);
        for (int x = 0; x < n; ++x) {
            int x0 = x * width / n;
            int x1 = std::max(x0 + 1, (x + 1) * width / n);
            uint32_t sum = 0;
            for (int yy = y0; yy < y1; ++yy) {
                const uint8_t* p = rgba + (static_cast<size_t>(yy) * width + x0) * 4;
                for (int xx = x0; xx < x1; ++xx, p += 4) {
                    sum += (p[0] * 77u + p[1] * 150u + p[2] * 29u) >> 8;
                }
            }
            out[static_cast<size_t>(y) * n + x] =
                    static_cast<uint8_t>(sum / static_cast<uint32_t>((y1 - y0) * (x1 - x0)));
        }
    }
}

void halve(const std::vector<uint8_t>& in, int n, std::vector<uint8_t>& out) {
    const int m = n / 2;
    out.assign(static_cast<size_t>(m) * m, 0);
    for (int y = 0; y < m; ++y) {
        for (int x = 0; x < m; ++x) {
            const uint8_t* a = &in[static_cast<size_t>(2 * y) * n + 2 * x];
            out[static_cast<size_t>(y) * m + x] =
                    static_cast<uint8_t>((a[0] + a[1] + a[n] + a[n + 1] + 2) / 4);
        }
    }
}

// Mean absolute difference between prev(x, y) and cur(x + sx, y + sy) over the
// interior [margin, n - margin)^2, i.e. the cost of "content moved by (sx, sy)".
float sad(const std::vector<uint8_t>& prev, const std::vector<uint8_t>& cur, int n,
          int margin, int sx, int sy) {
    uint32_t sum = 0;
    const int side = n - 2 * margin;
    for (int y = margin; y < n - margin; ++y) {
        // Row bases start at column `margin`: with |sx| <= margin the offset is
        // never negative (a row base at column sx < 0 of row 0 would point before
        // the buffer, which is UB even though no such element is read).
        const uint8_t* p = prev.data() + static_cast<size_t>(y * n + margin);
        const uint8_t* c = cur.data() + static_cast<size_t>((y + sy) * n + margin + sx);
        for (int x = 0; x < side; ++x) {
            sum += static_cast<uint32_t>(std::abs(static_cast<int>(p[x]) - static_cast<int>(c[x])));
        }
    }
    return static_cast<float>(sum) / static_cast<float>(side * side);
}

float mean_gradient(const std::vector<uint8_t>& img, int n, int margin) {
    uint32_t sum = 0;
    for (int y = margin; y < n - margin; ++y) {
        for (int x = margin; x < n - margin; ++x) {
            size_t i = static_cast<size_t>(y) * n + x;
            sum += static_cast<uint32_t>(std::abs(img[i + 1] - img[i]) + std::abs(img[i + n] - img[i]));
        }
    }
    const int side = n - 2 * margin;
    return static_cast<float>(sum) / static_cast<float>(side * side);
}

// Sub-pixel offset of a parabola through (-1, a), (0, b), (1, c), in [-0.5, 0.5].
float parabola(float a, float b, float c) {
    float denom = a - 2.0f * b + c;
    if (denom <= 1e-6f) return 0.0f;
    return std::clamp(0.5f * (a - c) / denom, -0.5f, 0.5f);
}

// Scene detail below this (mean |gradient| in grey levels at 32 px) cannot
// localise a shift: a white wall or a covered lens.
constexpr float kMinTexture = 1.5f;
// The best shift must explain the frame clearly better than the average one.
constexpr float kMaxBestToMean = 0.6f;
// Prefer "no motion" when it is nearly as good: avoids sub-pixel jitter on
// static scenes being fed to every track.
constexpr float kZeroPreference = 1.05f;

} // namespace

void CameraMotionEstimator::reset() {
    has_prev_ = false;
}

bool CameraMotionEstimator::estimate(const uint8_t* rgba, int width, int height,
                                     float& dx, float& dy) {
    dx = 0.0f;
    dy = 0.0f;
    if (!rgba || width < kFine || height < kFine) {
        has_prev_ = false;
        return false;
    }
    downscale_luma(rgba, width, height, kFine, cur_fine_);
    halve(cur_fine_, kFine, cur_coarse_);

    bool ok = false;
    if (has_prev_) {
        do {
            if (mean_gradient(prev_coarse_, kCoarse, kCoarseRadius) < kMinTexture) break;

            // Coarse exhaustive search.
            float best = std::numeric_limits<float>::max(), total = 0.0f;
            int bx = 0, by = 0, count = 0;
            for (int sy = -kCoarseRadius; sy <= kCoarseRadius; ++sy) {
                for (int sx = -kCoarseRadius; sx <= kCoarseRadius; ++sx) {
                    float cost = sad(prev_coarse_, cur_coarse_, kCoarse, kCoarseRadius, sx, sy);
                    total += cost;
                    ++count;
                    if (cost < best) { best = cost; bx = sx; by = sy; }
                }
            }
            float zero = sad(prev_coarse_, cur_coarse_, kCoarse, kCoarseRadius, 0, 0);
            if (zero <= best * kZeroPreference) { best = zero; bx = 0; by = 0; }
            float mean = total / static_cast<float>(count);
            if (best > kMaxBestToMean * mean) break;                       // not distinct / scene cut
            if (std::abs(bx) == kCoarseRadius || std::abs(by) == kCoarseRadius) break; // out of range

            // Fine refinement around the coarse estimate.
            const int margin = 2 * kCoarseRadius + kFineRadius + 1;
            const int cx = 2 * bx, cy = 2 * by;
            float fine[2 * kFineRadius + 3][2 * kFineRadius + 3];
            float fbest = std::numeric_limits<float>::max();
            int fx = cx, fy = cy;
            for (int sy = -kFineRadius - 1; sy <= kFineRadius + 1; ++sy) {
                for (int sx = -kFineRadius - 1; sx <= kFineRadius + 1; ++sx) {
                    float cost = sad(prev_fine_, cur_fine_, kFine, margin, cx + sx, cy + sy);
                    fine[sy + kFineRadius + 1][sx + kFineRadius + 1] = cost;
                    if (std::abs(sx) <= kFineRadius && std::abs(sy) <= kFineRadius && cost < fbest) {
                        fbest = cost; fx = cx + sx; fy = cy + sy;
                    }
                }
            }
            int ix = fx - cx + kFineRadius + 1, iy = fy - cy + kFineRadius + 1;
            if (fx == 0 && fy == 0) {
                // Static: report exactly zero, no sub-pixel drift.
                ok = true;
                break;
            }
            float sub_x = parabola(fine[iy][ix - 1], fine[iy][ix], fine[iy][ix + 1]);
            float sub_y = parabola(fine[iy - 1][ix], fine[iy][ix], fine[iy + 1][ix]);
            dx = (static_cast<float>(fx) + sub_x) / static_cast<float>(kFine);
            dy = (static_cast<float>(fy) + sub_y) / static_cast<float>(kFine);
            ok = true;
        } while (false);
    }

    prev_fine_.swap(cur_fine_);
    prev_coarse_.swap(cur_coarse_);
    has_prev_ = true;
    return ok;
}

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

// YOLOv8 head decoding in C++, replacing the static post-processing tail of the
// bundled export (MemoryData anchors / Reshape fixed at 2100 = 320 px input).
// Reading the raw per-level head outputs instead makes the network fully
// convolutional, so it can run at any input size that is a multiple of 32, and
// lets low-confidence cells be skipped before their box is decoded.
//
// Per level (stride 8/16/32) the head produces:
//   box: 64 channels = 4 sides (left, top, right, bottom) x 16 DFL bins
//   cls: 1 channel of class logits
// Channel c of a w x h level is at ptr + c * cstep + y * w + x (ncnn layout).
// Output boxes are in input pixels, identical to the export's out0 rows
// (cx, cy, w, h, sigmoid(cls)); tests/tests_yolo_decode.py checks this.

struct YoloCandidate {
    float cx, cy, w, h, prob;
};

// Returns false if any class logit, or any box value of a decoded cell, is not
// finite (malformed tensor: the caller must treat the frame as failed).
inline bool decode_yolo_level(const float* box, size_t box_cstep,
                              const float* cls, int w, int h, int stride,
                              float prob_threshold, std::vector<YoloCandidate>& out) {
    constexpr int kSides = 4;
    constexpr int kBins = 16;
    // prob >= t  <=>  logit >= log(t / (1 - t)); avoids exp() for rejected cells.
    const bool accept_all = !(prob_threshold > 0.0f);
    const float logit_threshold = accept_all ? 0.0f
            : std::log(std::min(prob_threshold, 0.999999f) / (1.0f - std::min(prob_threshold, 0.999999f)));
    const float s = static_cast<float>(stride);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;
            const float logit = cls[idx];
            if (!std::isfinite(logit)) return false;
            if (!accept_all && logit < logit_threshold) continue;

            float dist[kSides];
            for (int k = 0; k < kSides; ++k) {
                const float* bins = box + static_cast<size_t>(k * kBins) * box_cstep + idx;
                float m = bins[0];
                for (int b = 0; b < kBins; ++b) {
                    const float v = bins[static_cast<size_t>(b) * box_cstep];
                    if (!std::isfinite(v)) return false;
                    m = std::max(m, v);
                }
                float sum = 0.0f, expect = 0.0f;
                for (int b = 0; b < kBins; ++b) {
                    const float e = std::exp(bins[static_cast<size_t>(b) * box_cstep] - m);
                    sum += e;
                    expect += e * static_cast<float>(b);
                }
                dist[k] = expect / sum;
            }
            const float ax = static_cast<float>(x) + 0.5f;
            const float ay = static_cast<float>(y) + 0.5f;
            YoloCandidate c;
            c.cx = (ax + 0.5f * (dist[2] - dist[0])) * s;
            c.cy = (ay + 0.5f * (dist[3] - dist[1])) * s;
            c.w = (dist[0] + dist[2]) * s;
            c.h = (dist[1] + dist[3]) * s;
            c.prob = 1.0f / (1.0f + std::exp(-logit));
            out.push_back(c);
        }
    }
    return true;
}

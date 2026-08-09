#include "scrfd.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#define LOG_TAG "FaceDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {
constexpr int kInput = 320;
constexpr int kLandmarkCount = 5;
constexpr float kVarianceCenter = 0.1f;
constexpr float kVarianceSize = 0.2f;

struct Prior {
    float cx;
    float cy;
    float w;
    float h;
};

static void rgba_to_bgr(const unsigned char* rgba, std::vector<unsigned char>& bgr,
                        int width, int height) {
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    bgr.resize(pixels * 3u);
    for (size_t i = 0; i < pixels; ++i) {
        bgr[i * 3 + 0] = rgba[i * 4 + 2];
        bgr[i * 3 + 1] = rgba[i * 4 + 1];
        bgr[i * 3 + 2] = rgba[i * 4 + 0];
    }
}

static void generate_priors(int width, int height, std::vector<Prior>& priors) {
    // This released face.param is the 4-head slim detector, not a 3-head
    // RetinaFace graph. Its loc heads expose 3/2/2/3 anchors at strides
    // 8/16/32/64, which must match every decoded landmark record exactly.
    static const int strides[] = {8, 16, 32, 64};
    static const int min_sizes[][3] = {
        {10, 16, 24}, {32, 48, 0}, {64, 96, 0}, {128, 192, 256},
    };
    static const int anchors_per_level[] = {3, 2, 2, 3};
    priors.clear();
    for (int level = 0; level < 4; ++level) {
        const int stride = strides[level];
        const int fw = (width + stride - 1) / stride;
        const int fh = (height + stride - 1) / stride;
        for (int y = 0; y < fh; ++y) for (int x = 0; x < fw; ++x) {
            for (int a = 0; a < anchors_per_level[level]; ++a) {
                Prior prior;
                prior.cx = (x + 0.5f) * stride / width;
                prior.cy = (y + 0.5f) * stride / height;
                prior.w = static_cast<float>(min_sizes[level][a]) / width;
                prior.h = static_cast<float>(min_sizes[level][a]) / height;
                priors.push_back(prior);
            }
        }
    }
}

static inline float intersection_area(const FaceDetection& a, const FaceDetection& b) {
    const float x0 = std::max(a.x, b.x);
    const float y0 = std::max(a.y, b.y);
    const float x1 = std::min(a.x + a.w, b.x + b.w);
    const float y1 = std::min(a.y + a.h, b.y + b.h);
    return (x1 > x0 && y1 > y0) ? (x1 - x0) * (y1 - y0) : 0.0f;
}

static void nms_sorted_bboxes(const std::vector<FaceDetection>& candidates,
                              std::vector<int>& picked, float threshold) {
    picked.clear();
    std::vector<float> areas(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) areas[i] = candidates[i].w * candidates[i].h;
    for (size_t i = 0; i < candidates.size(); ++i) {
        bool keep = true;
        for (int j : picked) {
            const float inter = intersection_area(candidates[i], candidates[j]);
            const float uni = areas[i] + areas[j] - inter;
            if (uni > 0.0f && inter / uni > threshold) { keep = false; break; }
        }
        if (keep) picked.push_back(static_cast<int>(i));
    }
}

// Camera pipelines may mirror an OES preview before readback. The model keeps
// semantic eye/mouth slots, so image-space left/right order then reverses even
// though the five points are otherwise valid. Canonicalize those pairs for the
// affine root instead of rejecting every front-camera face as malformed.
static void canonicalize_landmark_pairs(FaceDetection& d) {
    if (d.kps[0].x > d.kps[1].x) std::swap(d.kps[0], d.kps[1]);
    if (d.kps[3].x > d.kps[4].x) std::swap(d.kps[3], d.kps[4]);
}

static bool valid_landmarks(const FaceDetection& d) {
    if (!std::isfinite(d.x) || !std::isfinite(d.y) || !std::isfinite(d.w) ||
        !std::isfinite(d.h) || d.w < 8.0f || d.h < 8.0f) return false;
    const float left = d.x - d.w * 0.25f;
    const float top = d.y - d.h * 0.25f;
    const float right = d.x + d.w * 1.25f;
    const float bottom = d.y + d.h * 1.25f;
    for (const FacePoint& p : d.kps) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < left || p.x > right ||
            p.y < top || p.y > bottom) return false;
    }
    const FacePoint& le = d.kps[0];
    const FacePoint& re = d.kps[1];
    const FacePoint& nose = d.kps[2];
    const FacePoint& lm = d.kps[3];
    const FacePoint& rm = d.kps[4];
    const float eye_dx = re.x - le.x;
    const float eye_dy = re.y - le.y;
    const float eye_dist = std::sqrt(eye_dx * eye_dx + eye_dy * eye_dy);
    const float eye_y = (le.y + re.y) * 0.5f;
    const float mouth_y = (lm.y + rm.y) * 0.5f;
    return eye_dist >= std::max(4.0f, d.w * 0.12f) &&
           mouth_y > eye_y + d.h * 0.08f && nose.y > eye_y - d.h * 0.10f &&
           nose.y < mouth_y + d.h * 0.18f;
}
} // namespace

SCRFD::SCRFD() : initialized(false) {}
SCRFD::~SCRFD() { net.clear(); }

int SCRFD::load(const char* param_path, const char* bin_path) {
    ncnn::Option opt = net.opt;
    opt.use_vulkan_compute = false;
    opt.num_threads = 1;
    net.opt = opt;
    int ret = net.load_param(param_path);
    if (ret != 0) { LOGE("landmark model load_param failed: %d", ret); return ret; }
    ret = net.load_model(bin_path);
    if (ret != 0) { LOGE("landmark model load_model failed: %d", ret); return ret; }
    initialized = true;
    LOGI("RetinaFace MobileNet-0.25 5-KPS model loaded");
    return 0;
}

int SCRFD::detect(const unsigned char* rgba_pixels, int width, int height,
                  std::vector<FaceDetection>& faces, float prob_threshold,
                  float nms_threshold) {
    if (!initialized || !rgba_pixels || width <= 0 || height <= 0) return -1;
    std::vector<unsigned char> bgr;
    rgba_to_bgr(rgba_pixels, bgr, width, height);

    // The model is cheap enough to run at the fixed tap size. Aspect-preserving
    // letterboxing preserves landmark geometry, unlike a stretched square crop.
    const float scale = std::min(static_cast<float>(kInput) / width,
                                 static_cast<float>(kInput) / height);
    const int resized_w = std::max(1, static_cast<int>(std::round(width * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(height * scale)));
    const int pad_left = (kInput - resized_w) / 2;
    const int pad_top = (kInput - resized_h) / 2;
    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(bgr.data(), ncnn::Mat::PIXEL_BGR,
                                                        width, height, resized_w, resized_h);
    ncnn::Mat input;
    ncnn::copy_make_border(resized, input, pad_top, kInput - resized_h - pad_top,
                           pad_left, kInput - resized_w - pad_left,
                           ncnn::BORDER_CONSTANT, 0.0f);
    const float mean[3] = {104.0f, 117.0f, 123.0f};
    input.substract_mean_normalize(mean, nullptr);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(true);
    if (ex.input("input0", input) != 0) return -2;
    ncnn::Mat loc, confidence, landmarks;
    if (ex.extract("output0", loc) != 0 || ex.extract("530", confidence) != 0 ||
        ex.extract("529", landmarks) != 0) {
        LOGE("landmark endpoint extraction failed");
        return -3;
    }
    std::vector<Prior> priors;
    generate_priors(kInput, kInput, priors);
    const int count = static_cast<int>(priors.size());
    // NCNN represents these endpoint tensors as N×{4,2,10}, so `.w` is
    // the per-prior record width rather than total scalar count.
    if (loc.total() < static_cast<size_t>(count) * 4u ||
        confidence.total() < static_cast<size_t>(count) * 2u ||
        landmarks.total() < static_cast<size_t>(count) * 10u) {
        LOGE("landmark tensor size mismatch loc=%zu conf=%zu lm=%zu expected=%d", loc.total(),
             confidence.total(), landmarks.total(), count);
        return -4;
    }
    const float* p_loc = loc.channel(0);
    const float* p_conf = confidence.channel(0);
    const float* p_lm = landmarks.channel(0);
    std::vector<FaceDetection> candidates;
    candidates.reserve(32);
    bool invalid_landmark_candidate = false;
    for (int i = 0; i < count; ++i) {
        const float prob = p_conf[i * 2 + 1];
        if (!std::isfinite(prob) || prob < prob_threshold) continue;
        const Prior& a = priors[i];
        const float cx = a.cx + p_loc[i * 4] * kVarianceCenter * a.w;
        const float cy = a.cy + p_loc[i * 4 + 1] * kVarianceCenter * a.h;
        const float bw = a.w * std::exp(p_loc[i * 4 + 2] * kVarianceSize);
        const float bh = a.h * std::exp(p_loc[i * 4 + 3] * kVarianceSize);
        FaceDetection d{};
        d.x = (cx - bw * 0.5f) * kInput;
        d.y = (cy - bh * 0.5f) * kInput;
        d.w = bw * kInput;
        d.h = bh * kInput;
        d.prob = prob;
        d.has_kps = true;
        for (int k = 0; k < kLandmarkCount; ++k) {
            d.kps[k].x = (a.cx + p_lm[i * 10 + k * 2] * kVarianceCenter * a.w) * kInput;
            d.kps[k].y = (a.cy + p_lm[i * 10 + k * 2 + 1] * kVarianceCenter * a.h) * kInput;
        }
        // Reverse the exact letterbox transform for box and anchors together.
        d.x = (d.x - pad_left) / scale;
        d.y = (d.y - pad_top) / scale;
        d.w /= scale;
        d.h /= scale;
        for (FacePoint& p : d.kps) { p.x = (p.x - pad_left) / scale; p.y = (p.y - pad_top) / scale; }
        canonicalize_landmark_pairs(d);
        if (!valid_landmarks(d)) {
            // A detector did see a face but its anchors cannot safely define a
            // surface-bound mask. Do not turn that into an authoritative zero-face.
            invalid_landmark_candidate = true;
            continue;
        }
        d.has_kps = true;
        candidates.push_back(d);
    }
    std::sort(candidates.begin(), candidates.end(), [](const FaceDetection& a, const FaceDetection& b) {
        return a.prob > b.prob;
    });
    std::vector<int> picked;
    nms_sorted_bboxes(candidates, picked, nms_threshold);
    faces.clear();
    for (int i : picked) faces.push_back(candidates[i]);
    // This distinct status lets the privacy boundary keep emergency cover on
    // instead of treating rejected landmark geometry as a detector-confirmed
    // zero-face scene.
    return (faces.empty() && invalid_landmark_candidate) ? 1 : 0;
}

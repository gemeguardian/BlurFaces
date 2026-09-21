#include "head_detector.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#define LOG_TAG "BlurFacesHead"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

float compute_iou(const HeadBox& a, const HeadBox& b) {
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);
    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);
    float inter = (x2 > x1 && y2 > y1) ? (x2 - x1) * (y2 - y1) : 0.0f;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float denom = area_a + area_b - inter;
    return denom > 0.0f ? inter / denom : 0.0f;
}

void nms(std::vector<HeadBox>& candidates, std::vector<HeadBox>& picked, float threshold) {
    picked.clear();
    std::sort(candidates.begin(), candidates.end(), [](const HeadBox& a, const HeadBox& b) {
        return a.score > b.score;
    });

    for (const auto& box : candidates) {
        bool keep = true;
        for (const auto& accepted : picked) {
            if (compute_iou(box, accepted) > threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(box);
        }
    }
}

} // namespace

HeadDetector::HeadDetector()
    : initialized_(false), ema_init_(false), y_buf_(kInputW * kInputH, 0.0f) {
    memset(ema_lut_, 0, sizeof(ema_lut_));
}

HeadDetector::~HeadDetector() {
    clear();
}

void HeadDetector::clear() {
    net_.clear();
    initialized_ = false;
    ema_init_ = false;
    enhance_on_ = false;
    prev_mean_lum_ = 128.0f;
    memset(ema_lut_, 0, sizeof(ema_lut_));
}

void HeadDetector::enhance_lowlight(ncnn::Mat& in, float t, float alpha) {
    // t in (0,1] — сила эффекта, из mean_lum
    // alpha — коэффициент временного EMA (больше = быстрее адаптация к смене сцены)
    float* R = in.channel(0);
    float* G = in.channel(1);
    float* B = in.channel(2);
    const int W = kInputW, H = kInputH;

    if (y_buf_.size() < static_cast<size_t>(W * H)) {
        y_buf_.resize(W * H);
    }
    float* Y = y_buf_.data();

    int hist[kTilesY][kTilesX][kBins];
    memset(hist, 0, sizeof(hist));

    const int tw = W / kTilesX, th = H / kTilesY;

    for (int y = 0; y < H; ++y) {
        int ty = std::min(y / th, kTilesY - 1);
        for (int x = 0; x < W; ++x) {
            int i = y * W + x;
            float yy = 0.299f * R[i] + 0.587f * G[i] + 0.114f * B[i];
            Y[i] = yy;
            int b = std::clamp(static_cast<int>(yy * (kBins / 256.0f)), 0, kBins - 1);
            hist[ty][std::min(x / tw, kTilesX - 1)][b]++;
        }
    }

    // clipped CDF -> mapping, + временной EMA
    const int pixels_per_tile = tw * th;
    const float clip = 3.0f * pixels_per_tile / kBins;  // clip limit 3.0
    float map_[kTilesY][kTilesX][kBins];

    for (int ty = 0; ty < kTilesY; ++ty) {
        for (int tx = 0; tx < kTilesX; ++tx) {
            float h[kBins];
            float excess = 0.0f;
            for (int b = 0; b < kBins; ++b) {
                h[b] = static_cast<float>(hist[ty][tx][b]);
                if (h[b] > clip) {
                    excess += h[b] - clip;
                    h[b] = clip;
                }
            }
            float add = excess / kBins;
            float cum = 0.0f, total = static_cast<float>(pixels_per_tile);
            for (int b = 0; b < kBins; ++b) {
                cum += h[b] + add;
                float eq  = 255.0f * cum / total;                 // эквализованное
                float idn = (b + 0.5f) * (256.0f / kBins);        // исходное
                float v   = (1.0f - t) * idn + t * eq;            // сила эффекта
                map_[ty][tx][b] = ema_init_
                    ? (1.0f - alpha) * ema_lut_[ty][tx][b] + alpha * v  // антимигание
                    : v;
                ema_lut_[ty][tx][b] = map_[ty][tx][b];
            }
        }
    }
    ema_init_ = true;

    // билинейная интерполяция между тайлами + перенос гейна на RGB
    for (int y = 0; y < H; ++y) {
        float fy = (y + 0.5f) / th - 0.5f;
        int y0 = std::clamp(static_cast<int>(std::floor(fy)), 0, kTilesY - 1);
        int y1 = std::min(y0 + 1, kTilesY - 1);
        float wy = std::clamp(fy - y0, 0.0f, 1.0f);
        for (int x = 0; x < W; ++x) {
            float fx = (x + 0.5f) / tw - 0.5f;
            int x0 = std::clamp(static_cast<int>(std::floor(fx)), 0, kTilesX - 1);
            int x1 = std::min(x0 + 1, kTilesX - 1);
            float wx = std::clamp(fx - x0, 0.0f, 1.0f);

            int i = y * W + x;

            // Интерполяция ещё и по бинам яркости: без неё все пиксели одного
            // бина получают одинаковый target -> контуринг на плавных градиентах.
            float fb = Y[i] * (kBins / 256.0f) - 0.5f;
            int b0 = std::clamp(static_cast<int>(std::floor(fb)), 0, kBins - 1);
            int b1 = std::min(b0 + 1, kBins - 1);
            float wb = std::clamp(fb - b0, 0.0f, 1.0f);

            float v0 = (1.0f - wy) * ((1.0f - wx) * map_[y0][x0][b0] + wx * map_[y0][x1][b0])
                     +         wy  * ((1.0f - wx) * map_[y1][x0][b0] + wx * map_[y1][x1][b0]);
            float v1 = (1.0f - wy) * ((1.0f - wx) * map_[y0][x0][b1] + wx * map_[y0][x1][b1])
                     +         wy  * ((1.0f - wx) * map_[y1][x0][b1] + wx * map_[y1][x1][b1]);
            float v = (1.0f - wb) * v0 + wb * v1;

            float gain = std::clamp((v + 1.0f) / (Y[i] + 1.0f), 1.0f, 4.0f);
            R[i] = std::min(R[i] * gain, 255.0f);
            G[i] = std::min(G[i] * gain, 255.0f);
            B[i] = std::min(B[i] * gain, 255.0f);
        }
    }
}

int HeadDetector::load(const char* param_path, const char* bin_path) {
    clear();
    ncnn::Option opt = net_.opt;
    opt.use_vulkan_compute = false;
    opt.num_threads = 2;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_packing_layout = true;
    net_.opt = opt;

    int ret = net_.load_param(param_path);
    if (ret != 0) {
        LOGE("head detector load_param failed: %d", ret);
        return ret;
    }
    ret = net_.load_model(bin_path);
    if (ret != 0) {
        LOGE("head detector load_model failed: %d", ret);
        return ret;
    }

    initialized_ = true;
    LOGI("NCNN HeadDetector initialized successfully (input %dx%d)", kInputW, kInputH);
    return 0;
}

int HeadDetector::detect(const unsigned char* rgba_pixels, int width, int height,
                         std::vector<HeadBox>& heads,
                         float prob_threshold, float nms_threshold) {
    heads.clear();
    if (!initialized_ || !rgba_pixels || width <= 0 || height <= 0) {
        return -1;
    }

    // RGBA to RGB and bilinear resize to 320x320
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(rgba_pixels, ncnn::Mat::PIXEL_RGBA2RGB,
                                                    width, height, kInputW, kInputH);

    // Fast adaptive low-light enhancement:
    // Sample frame luminance across a 32x32 grid (1024 samples ~ 0.05 us)
    int step_x = std::max(1, width / 32);
    int step_y = std::max(1, height / 32);
    int sum_lum = 0;
    int sample_count = 0;

    for (int y = 0; y < height; y += step_y) {
        const unsigned char* row = rgba_pixels + y * width * 4;
        for (int x = 0; x < width; x += step_x) {
            const unsigned char* p = row + x * 4;
            int luma = (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
            sum_lum += luma;
            sample_count++;
        }
    }
    float mean_lum = (sample_count > 0) ? (static_cast<float>(sum_lum) / sample_count) : 128.0f;

    // Apply CLAHE on luma + transfer gain uniformly to RGB + temporal EMA.
    // Nominal low-light working point is mean_lum < 65.0f; hysteresis around it
    // (on below 62, off above 70) prevents on/off flicker when the scene sits
    // right on the boundary. t ramps continuously from 0 so there is no step.
    if (!enhance_on_ && mean_lum < 62.0f) enhance_on_ = true;
    if (enhance_on_ && mean_lum > 70.0f) enhance_on_ = false;

    if (enhance_on_) {
        float t = std::clamp((70.0f - mean_lum) / 60.0f, 0.0f, 0.85f);
        // Scene cut / lights toggled -> adapt fast, otherwise smooth hard.
        float delta = std::fabs(mean_lum - prev_mean_lum_);
        float alpha = (delta > 18.0f) ? 0.8f : 0.3f;
        enhance_lowlight(input, t, alpha);
    }
    prev_mean_lum_ = mean_lum;

    // YOLOv8 input normalization: 0..255 -> 0.0..1.0
    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    input.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    if (ex.input("in0", input) != 0) {
        LOGE("head detector input failed");
        return -2;
    }

    ncnn::Mat out;
    if (ex.extract("out0", out) != 0) {
        LOGE("failed to extract out0");
        return -3;
    }

    std::vector<HeadBox> candidates;
    candidates.reserve(32);

    const float* ptr_cx   = out.row(0);
    const float* ptr_cy   = out.row(1);
    const float* ptr_w    = out.row(2);
    const float* ptr_h    = out.row(3);
    const float* ptr_conf = out.row(4);

    const float inv_w = 1.0f / static_cast<float>(kInputW);
    const float inv_h = 1.0f / static_cast<float>(kInputH);

    for (int i = 0; i < out.w; ++i) {
        float prob = ptr_conf[i];
        if (prob < prob_threshold) continue;

        float bbox_cx = ptr_cx[i] * inv_w;
        float bbox_cy = ptr_cy[i] * inv_h;
        float bbox_w  = ptr_w[i]  * inv_w;
        float bbox_h  = ptr_h[i]  * inv_h;

        float xmin = std::max(0.0f, std::min(1.0f, bbox_cx - bbox_w * 0.5f));
        float ymin = std::max(0.0f, std::min(1.0f, bbox_cy - bbox_h * 0.5f));
        float xmax = std::max(0.0f, std::min(1.0f, bbox_cx + bbox_w * 0.5f));
        float ymax = std::max(0.0f, std::min(1.0f, bbox_cy + bbox_h * 0.5f));

        float bw = xmax - xmin;
        float bh = ymax - ymin;

        // Physical bounding box sanity checks:
        // - Min: 5% width, 5% height (rejects tiny noise artifacts)
        // - Max: 100% width, 100% height (supports full-frame close-up selfies)
        if (bw < 0.05f || bh < 0.05f || bw > 1.0f || bh > 1.0f) continue;

        // Small clutter discrimination
        if ((bw < 0.11f || bh < 0.13f) && prob < 0.38f) continue;

        // Aspect ratio [0.35, 2.50] supports all head tilts, profile turns, and full 360° poses
        float aspect = bw / bh;
        if (aspect < 0.35f || aspect > 2.50f) continue;

        HeadBox box;
        box.x1 = xmin;
        box.y1 = ymin;
        box.x2 = xmax;
        box.y2 = ymax;
        box.cx = (xmin + xmax) * 0.5f;
        box.cy = (ymin + ymax) * 0.5f;
        box.w = bw;
        box.h = bh;
        box.score = prob;

        candidates.push_back(box);
    }

    std::vector<HeadBox> picked;
    nms(candidates, picked, nms_threshold);

    heads = std::move(picked);
    return static_cast<int>(heads.size());
}

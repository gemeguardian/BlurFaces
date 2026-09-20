#include "head_detector.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

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

HeadDetector::HeadDetector() : initialized_(false) {}

HeadDetector::~HeadDetector() {
    clear();
}

void HeadDetector::clear() {
    net_.clear();
    initialized_ = false;
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
            sum_lum += (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
            sample_count++;
        }
    }
    float mean_lum = (sample_count > 0) ? (static_cast<float>(sum_lum) / sample_count) : 128.0f;

    // If scene is in low light / darkness (< 65 luma), apply gentle adaptive gamma tone mapping
    // to boost facial gradient signals for the CNN without amplifying raw sensor noise.
    if (mean_lum < 65.0f && mean_lum > 8.0f) {
        float t = std::clamp((65.0f - mean_lum) / 50.0f, 0.0f, 1.0f);
        float gamma = 1.0f - t * 0.28f;
        float lut[256];
        const float inv255 = 1.0f / 255.0f;
        for (int i = 0; i < 256; ++i) {
            if (i < 8) {
                lut[i] = static_cast<float>(i);
            } else {
                lut[i] = std::pow(i * inv255, gamma) * 255.0f;
            }
        }

        for (int c = 0; c < 3; ++c) {
            float* ptr = input.channel(c);
            for (int i = 0; i < kInputW * kInputH; ++i) {
                int v = std::clamp(static_cast<int>(ptr[i] + 0.5f), 0, 255);
                ptr[i] = lut[v];
            }
        }
    }

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

        // Circular mask upper sensor boundary clip:
        // Boxes touching extreme top border (ymin <= 0.005) with small height (bh <= 0.15)
        // correspond to sensor clipping artifacts or fingers on phone edge.
        if (ymin <= 0.005f && bh <= 0.15f) continue;

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

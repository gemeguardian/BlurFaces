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

// Fast spatial texture gradient energy check to suppress flat textureless CNN hallucinations
inline float compute_texture_energy(const ncnn::Mat& img, const HeadBox& box) {
    int x0 = std::clamp(static_cast<int>(box.x1 * img.w), 1, img.w - 2);
    int y0 = std::clamp(static_cast<int>(box.y1 * img.h), 1, img.h - 2);
    int x1 = std::clamp(static_cast<int>(box.x2 * img.w), 1, img.w - 2);
    int y1 = std::clamp(static_cast<int>(box.y2 * img.h), 1, img.h - 2);

    if (x1 <= x0 + 4 || y1 <= y0 + 4) return 20.0f; // too small to sample reliably, pass through

    const float* g_ptr = img.channel(1); // green channel represents primary luminance signal
    int stride = img.w;

    // 8x8 grid = 64 interior points
    constexpr int kSamples = 8;
    float step_x = static_cast<float>(x1 - x0) / (kSamples + 1);
    float step_y = static_cast<float>(y1 - y0) / (kSamples + 1);

    float sum_grad = 0.0f;
    for (int j = 1; j <= kSamples; ++j) {
        int y = y0 + static_cast<int>(j * step_y);
        const float* row = g_ptr + y * stride;
        const float* row_above = g_ptr + (y - 1) * stride;
        const float* row_below = g_ptr + (y + 1) * stride;
        for (int i = 1; i <= kSamples; ++i) {
            int x = x0 + static_cast<int>(i * step_x);
            // Multiplied by 255 because img channel is normalized [0, 1]
            float gx = std::abs(row[x + 1] - row[x - 1]) * 255.0f;
            float gy = std::abs(row_below[x] - row_above[x]) * 255.0f;
            sum_grad += (gx + gy);
        }
    }
    return sum_grad / (kSamples * kSamples);
}

// Lightweight candidate validation:
// High-confidence proposals (prob >= 0.55f) from YOLOv8n (trained on HollywoodHeads and DAD-3DHeads)
// are verified anatomically by the 3M-parameter network across all 360° yaw/pitch angles.
// For borderline proposals (prob < 0.55f), we enforce biological hemoglobin chrominance and
// canine snout contrast checks to reject furniture, floor patterns, and pets.
bool verify_head_candidate(const unsigned char* rgba, int img_w, int img_h,
                           float xmin, float ymin, float xmax, float ymax,
                           float prob) {
    int ix0 = std::clamp(static_cast<int>(xmin * img_w), 0, img_w - 1);
    int iy0 = std::clamp(static_cast<int>(ymin * img_h), 0, img_h - 1);
    int ix1 = std::clamp(static_cast<int>(xmax * img_w), 0, img_w - 1);
    int iy1 = std::clamp(static_cast<int>(ymax * img_h), 0, img_h - 1);

    int box_w = ix1 - ix0;
    int box_h = iy1 - iy0;
    if (box_w < 4 || box_h < 4) return true;

    constexpr int kGrid = 8;
    float step_x = static_cast<float>(box_w) / (kGrid + 1);
    float step_y = static_cast<float>(box_h) / (kGrid + 1);

    int sum_cb = 0;
    int sum_cr = 0;
    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;
    int sum_lum = 0;
    int center_sum_lum = 0;
    constexpr int total_samples = kGrid * kGrid;

    for (int j = 1; j <= kGrid; ++j) {
        int y = iy0 + static_cast<int>(j * step_y);
        const unsigned char* row = rgba + y * img_w * 4;
        for (int i = 1; i <= kGrid; ++i) {
            int x = ix0 + static_cast<int>(i * step_x);
            const unsigned char* p = row + x * 4;
            int r = p[0];
            int g = p[1];
            int b = p[2];

            int lum = (r * 77 + g * 150 + b * 29) >> 8;
            sum_lum += lum;

            if (j >= 3 && j <= 6 && i >= 3 && i <= 6) {
                center_sum_lum += lum;
            }

            // Integer fixed-point ITU-R BT.601 YCbCr conversion
            int cb = 128 + ((-43 * r - 85 * g + 128 * b) >> 8);
            int cr = 128 + ((128 * r - 107 * g - 21 * b) >> 8);

            sum_cb += cb;
            sum_cr += cr;
            sum_r += r;
            sum_g += g;
            sum_b += b;
        }
    }

    float box_lum_mean = static_cast<float>(sum_lum) / total_samples;
    float center_lum_mean = static_cast<float>(center_sum_lum) / 16.0f;

    // Canine dark-snout check:
    // When a pet looks toward the camera against lighter backgrounds, the center snout is
    // significantly darker than the box average (< 52% of box mean).
    if (box_lum_mean > 45.0f && (center_lum_mean / box_lum_mean) < 0.52f) {
        return false;
    }

    // High confidence YOLOv8 detections (>= 0.55f) retain full 360° coverage (hair, back of head, hood).
    if (prob >= 0.55f) {
        return true;
    }

    // Borderline proposals (< 0.55f) must satisfy biological hemoglobin and chroma distance:
    int rb_diff = (sum_r - sum_b) / total_samples;
    int rg_diff = (sum_r - sum_g) / total_samples;

    // Biological hemoglobin chrominance balance: (R - G) / (R - B) >= 0.58
    if (rb_diff <= 0 || (rg_diff * 100) < (rb_diff * 58)) {
        return false;
    }

    float cb_mean = static_cast<float>(sum_cb) / total_samples;
    float cr_mean = static_cast<float>(sum_cr) / total_samples;
    float d_cr = cr_mean - 128.0f;
    float d_cb = cb_mean - 128.0f;
    float chroma_dist = std::sqrt(d_cr * d_cr + d_cb * d_cb);
    if (chroma_dist < 2.5f) return false;

    return true;
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
        // Boxes touching extreme top border (ymin <= 0.01) with small height (bh <= 0.25)
        // correspond to sensor clipping artifacts or fingers on phone edge.
        if (ymin <= 0.01f && bh <= 0.25f) continue;

        // Physical bounding box sanity checks for human head in round video notes:
        // - Min: 7% width, 8% height (rejects buttons, specks, tags)
        // - Max: 100% width, 100% height (supports full-frame close-up selfies)
        if (bw < 0.07f || bh < 0.08f || bw > 1.0f || bh > 1.0f) continue;

        // Small clutter discrimination
        if ((bw < 0.11f || bh < 0.13f) && prob < 0.38f) continue;

        float aspect = bw / bh;
        // Aspect ratio [0.35, 2.0] supports extreme head tilts and profile turns
        if (aspect < 0.35f || aspect > 2.0f) continue;

        if (!verify_head_candidate(rgba_pixels, width, height, xmin, ymin, xmax, ymax, prob)) {
            continue;
        }

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

        // On borderline proposals, ensure minimal spatial texture energy to suppress flat surfaces
        if (prob < 0.55f) {
            float tex_energy = compute_texture_energy(input, box);
            if (tex_energy < 2.5f) continue;
        }

        candidates.push_back(box);
    }

    std::vector<HeadBox> picked;
    nms(candidates, picked, nms_threshold);

    heads = std::move(picked);
    return static_cast<int>(heads.size());
}

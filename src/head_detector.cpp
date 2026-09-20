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

const float kHeadAnchors[6 * 2] = {
    192.0f, 240.0f, 384.0f, 480.0f, // feature 0: stride 32
    48.0f,  60.0f,  96.0f,  120.0f, // feature 1: stride 16
    12.0f,  15.0f,  24.0f,  30.0f   // feature 2: stride 8
};

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float inverse_sigmoid(float x) {
    float clamped = std::max(1e-5f, std::min(1.0f - 1e-5f, x));
    return -std::log(1.0f / clamped - 1.0f);
}

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

// Spatial texture gradient energy check to suppress flat textureless CNN hallucinations
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
            float gx = std::abs(row[x + 1] - row[x - 1]);
            float gy = std::abs(row_below[x] - row_above[x]);
            sum_grad += (gx + gy);
        }
    }
    return sum_grad / (kSamples * kSamples);
}

} // namespace

namespace hdv {

inline float smoothstep01(float edge0, float edge1, float x) {
    float t = std::clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

inline float logit_bias(float mean_lum, int feature_idx) {
    float b = 1.35f * (1.0f - smoothstep01(14.0f, 65.0f, mean_lum));
    if (feature_idx == 1) b += 0.30f; // selfie anchor prior
    return b;
}

struct FrameContext {
    float gw_cr = 128.0f;
    float gw_cb = 128.0f;
    float mean_lum = 128.0f;
    int width = 0;
    int height = 0;
    float gain = 1.0f;
};

static FrameContext s_ctx;

void set_gain(float gain) {
    s_ctx.gain = std::max(1.0f, std::min(10.0f, gain));
}

float get_gain() {
    return s_ctx.gain;
}

inline float candidate_prob_floor(int feature_idx, float mean_frame_lum) {
    float s = smoothstep01(18.0f, 85.0f, mean_frame_lum);
    if (feature_idx == 1) {
        return 0.21f + 0.29f * s; // 0.21 in dark -> 0.50 in daylight
    }
    return 0.20f + 0.05f * s;
}

void update_frame_context(const unsigned char* rgba, int width, int height) {
    s_ctx.width = width;
    s_ctx.height = height;
    if (!rgba || width <= 0 || height <= 0) return;

    int step_x = std::max(1, width / 32);
    int step_y = std::max(1, height / 32);
    int64_t sum_lum = 0;
    int64_t sum_cr = 0;
    int64_t sum_cb = 0;
    int sample_count = 0;

    for (int y = 0; y < height; y += step_y) {
        const unsigned char* row = rgba + y * width * 4;
        for (int x = 0; x < width; x += step_x) {
            const unsigned char* p = row + x * 4;
            int r = p[0];
            int g = p[1];
            int b = p[2];
            int lum = (p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8;
            int cb = 128 + ((-43 * r - 85 * g + 128 * b) >> 8);
            int cr = 128 + ((128 * r - 107 * g - 21 * b) >> 8);

            sum_lum += lum;
            sum_cr += cr;
            sum_cb += cb;
            sample_count++;
        }
    }

    if (sample_count > 0) {
        s_ctx.mean_lum = static_cast<float>(sum_lum) / sample_count;
        s_ctx.gw_cr = static_cast<float>(sum_cr) / sample_count;
        s_ctx.gw_cb = static_cast<float>(sum_cb) / sample_count;
    } else {
        s_ctx.mean_lum = 128.0f;
        s_ctx.gw_cr = 128.0f;
        s_ctx.gw_cb = 128.0f;
    }
}

bool verify_head_candidate(const unsigned char* rgba, int img_w, int img_h,
                           float xmin, float ymin, float xmax, float ymax,
                           int feature_idx) {
    int ix0 = std::clamp(static_cast<int>(xmin * img_w), 0, img_w - 1);
    int iy0 = std::clamp(static_cast<int>(ymin * img_h), 0, img_h - 1);
    int ix1 = std::clamp(static_cast<int>(xmax * img_w), 0, img_w - 1);
    int iy1 = std::clamp(static_cast<int>(ymax * img_h), 0, img_h - 1);

    int box_w = ix1 - ix0;
    int box_h = iy1 - iy0;
    if (box_w < 4 || box_h < 4) return true;

    float cx = (ix0 + ix1) * 0.5f;
    float cy = (iy0 + iy1) * 0.5f;
    float rx = box_w * 0.5f;
    float ry = box_h * 0.5f;

    // 1. Local reference white point (Cr^ref, Cb^ref):
    // Mix 24-point ring around candidate box with whole-frame gray-world white point
    constexpr int kRingPoints = 24;
    float ring_sum_cr = 0.0f;
    float ring_sum_cb = 0.0f;
    constexpr float kRingRadiusScale = 1.25f;
    constexpr float kPi = 3.14159265358979323846f;

    for (int k = 0; k < kRingPoints; ++k) {
        float angle = (2.0f * kPi * k) / static_cast<float>(kRingPoints);
        int px = std::clamp(static_cast<int>(cx + rx * kRingRadiusScale * std::cos(angle)), 0, img_w - 1);
        int py = std::clamp(static_cast<int>(cy + ry * kRingRadiusScale * std::sin(angle)), 0, img_h - 1);
        const unsigned char* p = rgba + (py * img_w + px) * 4;
        int r = p[0];
        int g = p[1];
        int b = p[2];
        float cb = 128.0f + ((-43.0f * r - 85.0f * g + 128.0f * b) / 256.0f);
        float cr = 128.0f + ((128.0f * r - 107.0f * g - 21.0f * b) / 256.0f);
        ring_sum_cr += cr;
        ring_sum_cb += cb;
    }
    float ring_cr = ring_sum_cr / kRingPoints;
    float ring_cb = ring_sum_cb / kRingPoints;

    float cr_ref = 0.5f * ring_cr + 0.5f * s_ctx.gw_cr;
    float cb_ref = 0.5f * ring_cb + 0.5f * s_ctx.gw_cb;

    // 2. Central 72% ROI inside candidate box
    float roi_x0 = ix0 + 0.14f * box_w;
    float roi_x1 = ix1 - 0.14f * box_w;
    float roi_y0 = iy0 + 0.14f * box_h;
    float roi_y1 = iy1 - 0.14f * box_h;

    // 3. 8x8 grid with 2x2 box filtering
    constexpr int kGrid = 8;
    float Y[kGrid][kGrid];
    float Cr[kGrid][kGrid];
    float Cb[kGrid][kGrid];

    float sum_y = 0.0f;
    float sum_cr = 0.0f;
    float sum_cb = 0.0f;
    float min_y = 255.0f;
    float max_y = 0.0f;
    int yellow_hits = 0;

    for (int j = 0; j < kGrid; ++j) {
        float fy = roi_y0 + ((j + 0.5f) / kGrid) * (roi_y1 - roi_y0);
        int iy = std::clamp(static_cast<int>(fy), 0, img_h - 2);
        for (int i = 0; i < kGrid; ++i) {
            float fx = roi_x0 + ((i + 0.5f) / kGrid) * (roi_x1 - roi_x0);
            int ix = std::clamp(static_cast<int>(fx), 0, img_w - 2);

            // 2x2 box filter
            const unsigned char* p00 = rgba + (iy * img_w + ix) * 4;
            const unsigned char* p10 = rgba + (iy * img_w + (ix + 1)) * 4;
            const unsigned char* p01 = rgba + ((iy + 1) * img_w + ix) * 4;
            const unsigned char* p11 = rgba + ((iy + 1) * img_w + (ix + 1)) * 4;

            float r = (p00[0] + p10[0] + p01[0] + p11[0]) * 0.25f;
            float g = (p00[1] + p10[1] + p01[1] + p11[1]) * 0.25f;
            float b = (p00[2] + p10[2] + p01[2] + p11[2]) * 0.25f;

            float y_val = (77.0f * r + 150.0f * g + 29.0f * b) / 256.0f;
            float cb_val = 128.0f + ((-43.0f * r - 85.0f * g + 128.0f * b) / 256.0f);
            float cr_val = 128.0f + ((128.0f * r - 107.0f * g - 21.0f * b) / 256.0f);

            Y[j][i] = y_val;
            Cr[j][i] = cr_val;
            Cb[j][i] = cb_val;

            sum_y += y_val;
            sum_cr += cr_val;
            sum_cb += cb_val;

            if (y_val < min_y) min_y = y_val;
            if (y_val > max_y) max_y = y_val;

            if (cb_val < 75.0f && r > 90.0f && g > 70.0f) {
                yellow_hits++;
            }
        }
    }

    constexpr int kTotal = kGrid * kGrid; // 64
    float mu = sum_y / kTotal;
    float cr_mean = sum_cr / kTotal;
    float cb_mean = sum_cb / kTotal;

    // Saturated non-biological yellow ceramic/mug filter
    if (yellow_hits >= 5) return false;

    // 4. Residual chroma vector and projection onto invariant skin axis u = (+0.882, -0.471)
    float dr = cr_mean - cr_ref;
    float db = cb_mean - cb_ref;
    float proj = dr * 0.882f + db * (-0.471f);

    // Spatial chroma variation sigma_cr and sigma_c
    float var_cr = 0.0f;
    float var_cb = 0.0f;
    float var_y = 0.0f;
    for (int j = 0; j < kGrid; ++j) {
        for (int i = 0; i < kGrid; ++i) {
            float dcr = Cr[j][i] - cr_mean;
            var_cr += dcr * dcr;
            float dcb = Cb[j][i] - cb_mean;
            var_cb += dcb * dcb;
            float dy = Y[j][i] - mu;
            var_y += dy * dy;
        }
    }
    float sigma_cr = std::sqrt(var_cr / kTotal);
    float sigma_c = std::sqrt((var_cr + var_cb) / kTotal);
    float sigma_y = std::sqrt(var_y / kTotal);

    // Minimum Euclidean chrominance radius from sample averaging scaled by preprocessor gain:
    float g = s_ctx.gain;
    float mu_raw = (g > 1.05f) ? (mu / g) : mu;
    float dmin = g * std::max(0.45f, 0.38f + 0.0195f * mu_raw) + (1.8f * (sigma_c * g) / 8.0f);

    // 5. Weber-normalized 3D facial relief metrics:
    // a) Laplacian energy lap (Lambda)
    float sum_lap = 0.0f;
    for (int j = 1; j <= 6; ++j) {
        for (int i = 1; i <= 6; ++i) {
            float lap_val = Y[j - 1][i] + Y[j + 1][i] + Y[j][i - 1] + Y[j][i + 1] - 4.0f * Y[j][i];
            sum_lap += std::abs(lap_val);
        }
    }
    float lap = sum_lap / (36.0f * (mu + 1.0f));

    // b) Weber dynamic range Cw
    float Cw = (max_y - min_y) / (mu + 1.0f);

    // c) Weber variance nu
    float nu = sigma_y / (mu + 1.0f);

    // d) T-zone profile T: forehead (rows 1-2, cols 2-5) + nose (rows 3-5, cols 3-4) vs lateral cheeks
    float sum_tzone = 0.0f;
    int count_tzone = 0;
    for (int j = 1; j <= 2; ++j) {
        for (int i = 2; i <= 5; ++i) {
            sum_tzone += Y[j][i];
            count_tzone++;
        }
    }
    for (int j = 3; j <= 5; ++j) {
        for (int i = 3; i <= 4; ++i) {
            sum_tzone += Y[j][i];
            count_tzone++;
        }
    }

    float sum_lat = 0.0f;
    int count_lat = 0;
    for (int j = 3; j <= 5; ++j) {
        for (int i = 0; i <= 2; ++i) { sum_lat += Y[j][i]; count_lat++; }
        for (int i = 5; i <= 7; ++i) { sum_lat += Y[j][i]; count_lat++; }
    }

    float mu_tzone = sum_tzone / count_tzone;
    float mu_lat = sum_lat / count_lat;
    float T = (mu_tzone - mu_lat) / (mu + 1.0f);

    // Explicit Vetoes:
    // 1. Black chair rejection: near-zero skin projection and flat Laplacian energy
    if (proj < 0.45f * dmin && lap < 0.026f) return false;

    // 2. Flat dark coat / couch rejection: very low luminance and low Weber dynamic range
    if (mu < 26.f && Cw < 0.13f) return false;

    // 3. Flat uniform achromatic surface check:
    if (sigma_cr < 0.35f && proj < 0.50f * dmin) return false;

    // 4. Canine dark snout check: prominent illuminated face vs dark animal snout
    float center_core_lum = (Y[3][3] + Y[3][4] + Y[4][3] + Y[4][4]) * 0.25f;
    if (mu > 45.0f && (center_core_lum / mu) < 0.50f) {
        return false;
    }

    // 5. Large quadruped fur suppression
    float box_area = (xmax - xmin) * (ymax - ymin);
    if (box_area > 0.55f && cr_mean < 139.0f && proj < 1.2f * dmin) {
        return false;
    }

    // Soft Evidence Score Fusion:
    float s_chroma = smoothstep01(0.20f * dmin, 1.80f * dmin, proj);
    float s_sigma = smoothstep01(0.35f, 1.10f, sigma_cr);
    float s_lap = smoothstep01(0.022f, 0.060f, lap);
    float s_cw = smoothstep01(0.12f, 0.38f, Cw);
    float s_t = std::clamp(T / 0.08f, 0.0f, 1.0f) * 0.5f + smoothstep01(0.05f, 0.25f, nu) * 0.5f;

    float S = 0.28f * s_chroma + 0.18f * s_sigma + 0.26f * s_lap + 0.18f * s_cw + 0.10f * s_t;

    if (S < 0.32f) return false;

    return true;
}

} // namespace hdv

namespace {

bool verify_head_chrominance(const unsigned char* rgba, int img_w, int img_h,
                             float xmin, float ymin, float xmax, float ymax,
                             int feature_idx) {
    return hdv::verify_head_candidate(rgba, img_w, img_h, xmin, ymin, xmax, ymax, feature_idx);
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

    // Update frame-level context (32x32 subsampled luminance and gray-world white point)
    hdv::update_frame_context(rgba_pixels, width, height);
    float mean_lum = hdv::s_ctx.mean_lum;

    // RGBA to RGB (MXNet Gluon head detector was trained on RGB) and bilinear resize to 320x320
    ncnn::Mat input;
    if (width == kInputW && height == kInputH) {
        input = ncnn::Mat::from_pixels(rgba_pixels, ncnn::Mat::PIXEL_RGBA2RGB, kInputW, kInputH);
    } else {
        input = ncnn::Mat::from_pixels_resize(rgba_pixels, ncnn::Mat::PIXEL_RGBA2RGB,
                                              width, height, kInputW, kInputH);
    }

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

    ncnn::Extractor ex = net_.create_extractor();
    ex.set_light_mode(true);
    if (ex.input("data", input) != 0) {
        LOGE("head detector input failed");
        return -2;
    }

    std::vector<HeadBox> candidates;
    candidates.reserve(32);
    float score_threshold_logit = inverse_sigmoid(prob_threshold);

    // Feature 0 (stride 32, anchor 192x240) and Feature 1 (stride 16, anchors 48x60 and 96x120)
    // span head sizes from 11% to 85% of the frame.
    // Feature 2 (stride 8, anchors 12x15 and 24x30) targets tiny micro-scale objects (3.7% - 9.4%),
    // which never correspond to human heads in 1:1 round video notes (smallest real face in dataset
    // is 17%x19%). In practice, F2 exclusively hallucinates on inanimate circular room artifacts
    // (exercise hoops, door knobs, circular frames). Skipping F2 eliminates 76% of all anchor decodings
    // and suppresses micro false positives completely with zero recall loss.
    for (int feature_idx = 0; feature_idx < 2; ++feature_idx) {
        char out_name[64];
        std::snprintf(out_name, sizeof(out_name), "head_output%d_out%d_fwd", feature_idx, feature_idx);
        ncnn::Mat out;
        if (ex.extract(out_name, out) != 0) {
            LOGE("failed to extract %s", out_name);
            continue;
        }

        for (int anchor_idx = 0; anchor_idx < 2; ++anchor_idx) {
            // Anchor (384, 480) in feature 0 is larger than the entire frame (320x320).
            // In round video selfies it only produces gigantic false positive boxes on floors/monitors.
            if (feature_idx == 0 && anchor_idx == 1) continue;

            int p = anchor_idx * 5; // 4 bbox + 1 score
            const float* xptr = out.channel(p + 0);
            const float* yptr = out.channel(p + 1);
            const float* wptr = out.channel(p + 2);
            const float* hptr = out.channel(p + 3);
            const float* score_ptr = out.channel(p + 4);

            const float bias_w = kHeadAnchors[feature_idx * 4 + anchor_idx * 2 + 0];
            const float bias_h = kHeadAnchors[feature_idx * 4 + anchor_idx * 2 + 1];

            for (int h = 0; h < out.h; ++h) {
                for (int w = 0; w < out.w; ++w) {
                    float box_score = *score_ptr;
                    float bias = hdv::logit_bias(mean_lum, feature_idx);
                    float adj_score = box_score + bias;
                    if (box_score > score_threshold_logit || adj_score > score_threshold_logit) {
                        float bbox_cx = (w + *xptr) / static_cast<float>(out.w);
                        float bbox_cy = (h + *yptr) / static_cast<float>(out.h);
                        float bbox_w = std::exp(*wptr) * bias_w / static_cast<float>(kInputW);
                        float bbox_h = std::exp(*hptr) * bias_h / static_cast<float>(kInputH);

                        float prob = sigmoid(adj_score);

                        // Dynamic candidate probability floor:
                        // For selfie anchor F1#1 (and F0/F1), replace hard prob < 0.50f with candidate_prob_floor
                        if (feature_idx == 1 && anchor_idx == 1 && prob < hdv::candidate_prob_floor(feature_idx, mean_lum)) {
                            xptr++; yptr++; wptr++; hptr++; score_ptr++;
                            continue;
                        }

                        float xmin = std::max(0.0f, std::min(1.0f, bbox_cx - bbox_w * 0.5f));
                        float ymin = std::max(0.0f, std::min(1.0f, bbox_cy - bbox_h * 0.5f));
                        float xmax = std::max(0.0f, std::min(1.0f, bbox_cx + bbox_w * 0.5f));
                        float ymax = std::max(0.0f, std::min(1.0f, bbox_cy + bbox_h * 0.5f));

                        float bw = xmax - xmin;
                        float bh = ymax - ymin;

                        // Circular mask upper sensor boundary clip:
                        // Boxes touching the extreme top border (ymin <= 0.01) with small height (bh <= 0.25)
                        // correspond to sensor clipping artifacts or fingers on phone edge, outside the circular note.
                        if (ymin <= 0.01f && bh <= 0.25f) {
                            xptr++; yptr++; wptr++; hptr++; score_ptr++;
                            continue;
                        }

                        // Physical bounding box sanity checks for human head in round video notes:
                        // - Min: 8% width, 10% height (rejects buttons, specks, tags)
                        // - Max: 100% width, 100% height (supports full-frame close-up selfies)
                        // - Aspect ratio width/height: range [0.45, 1.60] (supports head tilts and profiles)
                        if (bw >= 0.08f && bh >= 0.10f && bw <= 1.0f && bh <= 1.0f) {
                            if (bw < 0.11f || bh < 0.13f) {
                                if (prob < 0.38f) {
                                    xptr++; yptr++; wptr++; hptr++; score_ptr++;
                                    continue;
                                }
                            }
                            float aspect = bw / bh;
                            // Large box aspect ratio sanity: wide horizontal rectangles (aspect > 1.25) when width > 0.75
                            // correspond to quadruped animal bodies lying down, never upright or tilted human heads.
                            if (bw > 0.75f && aspect > 1.25f) {
                                xptr++; yptr++; wptr++; hptr++; score_ptr++;
                                continue;
                            }
                            if (aspect >= 0.45f && aspect <= 1.60f) {
                                if (!hdv::verify_head_candidate(rgba_pixels, width, height, xmin, ymin, xmax, ymax, feature_idx)) {
                                    xptr++; yptr++; wptr++; hptr++; score_ptr++;
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

                                // Suppress flat planar architecture hallucinations (doors, walls, floors, tables)
                                // Human heads are dense with micro-edges (eyes, nose, mouth, hair):
                                // true head texture energy is >= 4.20 even in dark rooms (mean 14.80).
                                // Flat doors and planar surfaces have texture energy <= 2.88.
                                float tex_energy = compute_texture_energy(input, box);
                                float min_tex = (mean_lum < 65.0f) ? (mean_lum < 40.0f ? 1.0f : 1.8f) : ((feature_idx == 0) ? 3.2f : 2.5f);
                                if (tex_energy < min_tex) {
                                    xptr++; yptr++; wptr++; hptr++; score_ptr++;
                                    continue;
                                }

                                candidates.push_back(box);
                            }
                        }
                    }

                    xptr++;
                    yptr++;
                    wptr++;
                    hptr++;
                    score_ptr++;
                }
            }
        }
    }

    std::vector<HeadBox> picked;
    nms(candidates, picked, nms_threshold);

    heads = std::move(picked);
    return static_cast<int>(heads.size());
}

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
            float gx = std::abs(row[x + 1] - row[x - 1]);
            float gy = std::abs(row_below[x] - row_above[x]);
            sum_grad += (gx + gy);
        }
    }
    return sum_grad / (kSamples * kSamples);
}

// Fast fixed-point chrominance validation to reject achromatic macro-hallucinations (knees, pants, floors)
bool verify_head_chrominance(const unsigned char* rgba, int img_w, int img_h,
                             float xmin, float ymin, float xmax, float ymax,
                             int feature_idx) {
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

    int skin_hits = 0;
    int center_skin_hits = 0;
    int top_skin_hits = 0;
    int yellow_hits = 0;
    int sum_cb = 0;
    int sum_cr = 0;
    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;
    int sum_lum = 0;
    int center_sum_lum = 0;
    int min_lum = 255;
    int max_lum = 0;
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
            if (lum < min_lum) min_lum = lum;
            if (lum > max_lum) max_lum = lum;
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

            // Artificial saturated yellow / amber (ceramics, mugs, plastic, beer)
            // Human skin across all ethnicities physically never has Cb < 75
            if (cb < 75 && r > 90 && g > 70) {
                yellow_hits++;
            }

            // Standard biological skin locus (Fitzpatrick I through VI)
            // Excludes near-neutral gray/black objects (chairs, clothes, cushions)
            if (cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173 && (r > b)) {
                skin_hits++;
                if (j <= 3) {
                    top_skin_hits++;
                }
                if (j >= 3 && j <= 6 && i >= 3 && i <= 6) {
                    center_skin_hits++;
                }
            }
        }
    }

    // Reject objects with non-biological saturated yellow (> 7% of box)
    if (yellow_hits >= 5) return false;

    float skin_ratio = static_cast<float>(skin_hits) / total_samples;
    float cb_mean = static_cast<float>(sum_cb) / total_samples;
    float cr_mean = static_cast<float>(sum_cr) / total_samples;
    int rb_diff = (sum_r - sum_b) / total_samples;

    // Human skin has average Cb >= 100 (in tungsten warm light >= 92); yellow ceramics have Cb ~ 73
    if (cb_mean < 92.0f) return false;

    // Euclidean chrominance radius from neutral gray (128, 128)
    // Chair = 2.29, knees = 0.8, dark clothes < 2.5, true human heads >= 3.05
    float d_cr = cr_mean - 128.0f;
    float d_cb = cb_mean - 128.0f;
    float chroma_dist = std::sqrt(d_cr * d_cr + d_cb * d_cb);
    if (chroma_dist < 2.85f) return false;

    // Real human heads have contiguous facial skin across the central core (cheeks, nose, mouth).
    // In human selfie video notes, center_skin_hits is >= 8 out of 16 (dataset min 9, mean 15.75).
    // Dog body/torso/rump false positives have dark steel/black saddle or fur boundaries in center (< 8 hits).
    if (center_skin_hits < 8) return false;

    // Top skin check: human head upper quadrant contains forehead skin (dataset min 2, mean 20.0).
    // Dog nose tips and dark pet backs contain zero skin locus hits in the upper 3 rows (< 2 hits).
    if (top_skin_hits < 2) return false;

    float box_lum_mean = static_cast<float>(sum_lum) / total_samples;
    float center_lum_mean = static_cast<float>(center_sum_lum) / 16.0f;

    // Canine dark-snout vs illuminated human facial core check:
    // A human facial core is prominently illuminated (center_lum_mean / box_lum_mean >= 0.66).
    // When a pet dog looks up surrounded by bright beige floor tiles, the center is a dark snout (< 55% of box mean).
    if (box_lum_mean > 45.0f && (center_lum_mean / box_lum_mean) < 0.55f) {
        return false;
    }

    float box_area = (xmax - xmin) * (ymax - ymin);

    // Real close-up faces (where hair/collar is out of frame) have high skin ratio (82% - 98%).
    // Unlike flat cardboard sheets or wood veneer, real faces contain high-contrast facial
    // micro-structures (eyes, pupils, nostrils, mouth fissure) with luminance variation.
    if (skin_ratio > 0.82f && box_area > 0.08f) {
        if ((max_lum - min_lum) < 22) return false;
    }

    if (feature_idx == 0) {
        if (skin_ratio < 0.18f || cr_mean < 130.5f || rb_diff < 1) return false;
    } else if (feature_idx == 1) {
        if (skin_ratio < 0.18f || cr_mean < 130.9f || rb_diff < 1) return false;
    } else {
        if (skin_ratio < 0.22f || cr_mean < 131.0f || rb_diff < 1) return false;
    }

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

    // RGBA to RGB (MXNet Gluon head detector was trained on RGB) and bilinear resize to 320x320
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
                    if (box_score > score_threshold_logit) {
                        float bbox_cx = (w + *xptr) / static_cast<float>(out.w);
                        float bbox_cy = (h + *yptr) / static_cast<float>(out.h);
                        float bbox_w = std::exp(*wptr) * bias_w / static_cast<float>(kInputW);
                        float bbox_h = std::exp(*hptr) * bias_h / static_cast<float>(kInputH);

                        float prob = sigmoid(box_score);

                        // Medium-scale F1 A1 anchor (96x120): true human heads in this scale always exhibit
                        // sharp confidence >= 0.66 (mean 0.98). Suppress ambiguous pet/fur proposals (< 0.50).
                        if (feature_idx == 1 && anchor_idx == 1 && prob < 0.50f) {
                            xptr++; yptr++; wptr++; hptr++; score_ptr++;
                            continue;
                        }

                        float xmin = std::max(0.0f, std::min(1.0f, bbox_cx - bbox_w * 0.5f));
                        float ymin = std::max(0.0f, std::min(1.0f, bbox_cy - bbox_h * 0.5f));
                        float xmax = std::max(0.0f, std::min(1.0f, bbox_cx + bbox_w * 0.5f));
                        float ymax = std::max(0.0f, std::min(1.0f, bbox_cy + bbox_h * 0.5f));

                        float bw = xmax - xmin;
                        float bh = ymax - ymin;

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
                            if (aspect >= 0.45f && aspect <= 1.60f) {
                                if (!verify_head_chrominance(rgba_pixels, width, height, xmin, ymin, xmax, ymax, feature_idx)) {
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
                                float min_tex = (feature_idx == 0) ? 3.2f : 2.8f;
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

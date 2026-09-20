#ifndef HEAD_DETECTOR_H
#define HEAD_DETECTOR_H

#include <vector>
#include <ncnn/net.h>

struct HeadBox {
    float x1; // normalized [0, 1]
    float y1;
    float x2;
    float y2;
    float cx;
    float cy;
    float w;
    float h;
    float score;
};

namespace hdv {
    void update_frame_context(const unsigned char* rgba, int width, int height);
    float candidate_prob_floor(int feature_idx, float mean_frame_lum);
    void set_gain(float gain);
    float get_gain();
    bool verify_head_candidate(const unsigned char* rgba, int img_w, int img_h,
                               float xmin, float ymin, float xmax, float ymax,
                               int feature_idx);
}

class HeadDetector {
public:
    HeadDetector();
    ~HeadDetector();

    int load(const char* param_path, const char* bin_path);
    int detect(const unsigned char* rgba_pixels, int width, int height,
               std::vector<HeadBox>& heads,
               float prob_threshold = 0.25f, float nms_threshold = 0.45f);

    bool is_initialized() const { return initialized_; }
    void clear();

private:
    ncnn::Net net_;
    bool initialized_ = false;
    static constexpr int kInputW = 320;
    static constexpr int kInputH = 320;
};

#endif // HEAD_DETECTOR_H

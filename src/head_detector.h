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

#ifndef SCRFD_H
#define SCRFD_H

#include <vector>
#include <ncnn/net.h>

struct FacePoint {
    float x;
    float y;
};

// A detector result carries semantic facial anchors, not only an axis-aligned
// box. The five points are left eye, right eye, nose, left mouth, right mouth,
// in upright top-left pixel coordinates of the captured frame.
struct FaceDetection {
    float x;
    float y;
    float w;
    float h;
    float prob;
    FacePoint kps[5];
    bool has_kps;
};

// Kept as SCRFD to avoid broad loader/JNI churn. The implementation deliberately
// validates the explicit RetinaFace MobileNet-0.25 5-KPS model interface.
class SCRFD {
public:
    SCRFD();
    ~SCRFD();

    int load(const char* param_path, const char* bin_path);
    int detect(const unsigned char* rgba_pixels, int width, int height,
               std::vector<FaceDetection>& faces,
               float prob_threshold = 0.60f, float nms_threshold = 0.40f);

private:
    ncnn::Net net;
    bool initialized;
};

#endif // SCRFD_H

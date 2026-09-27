// Host harness for tests_yolo_decode.py: decodes raw YOLOv8 head blobs with the
// production decoder and prints every cell (threshold 0) in export order.
// Input file: for each level (stride 8, 16, 32): int32 w, h, then 64*h*w box
// floats (channel-major, dense) and h*w class logits.
#include "yolo_decode.h"
#include <cstdint>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 2;
    std::vector<YoloCandidate> out;
    for (int stride : {8, 16, 32}) {
        int32_t wh[2];
        if (std::fread(wh, sizeof(int32_t), 2, f) != 2) return 3;
        const size_t cells = static_cast<size_t>(wh[0]) * wh[1];
        std::vector<float> box(64 * cells), cls(cells);
        if (std::fread(box.data(), sizeof(float), box.size(), f) != box.size()) return 3;
        if (std::fread(cls.data(), sizeof(float), cls.size(), f) != cls.size()) return 3;
        if (!decode_yolo_level(box.data(), cells, cls.data(), wh[0], wh[1], stride, 0.0f, out)) return 4;
    }
    std::fclose(f);
    for (const auto& c : out) std::printf("%.6f %.6f %.6f %.6f %.7f\n", c.cx, c.cy, c.w, c.h, c.prob);
    return 0;
}

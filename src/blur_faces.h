#ifndef BLUR_FACES_H
#define BLUR_FACES_H

#ifdef __cplusplus
extern "C" {
#endif

// Initializes the process-global SCRFD detector from NCNN model files.
int blur_faces_init(const char* param_path, const char* bin_path);

// Detects faces in a tightly packed RGBA8888 frame. out_faces contains at most
// max_faces records of five floats: x, y, width, height, confidence.
// Returns the number of records written, -1 when uninitialized/detection fails,
// or -2 for invalid arguments.
int blur_faces_detect(unsigned char* rgba_pixels, int width, int height,
                      float* out_faces, int max_faces);

// Pixelates face rectangles directly in a tightly packed RGBA8888 frame.
// faces uses the same five-float records emitted by blur_faces_detect.
// Returns 0 on success, -2 for invalid frame/array arguments, or -3 for a
// non-finite or invalid face rectangle. Off-frame valid rectangles are clipped.
int blur_faces_blur_rgba(unsigned char* rgba_pixels, int width, int height,
                         const float* faces, int face_count);

void blur_faces_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // BLUR_FACES_H
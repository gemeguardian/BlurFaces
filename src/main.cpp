#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "blur_faces.h"
#include "scrfd.h"
#include "face_tracker.h"

#define TAG "BlurFaces"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static SCRFD* g_scrfd = nullptr;
static std::mutex g_scrfd_mutex;
// 14 floats per face: box x/y/w/h then five semantic anchors x/y.
// Typed detector results stay in native storage so an accidental stride mismatch
// cannot silently turn landmark coordinates into a rectangle.
static FaceDetection g_cached_faces[10] = {};
static int g_cached_count = 0;
static bool g_last_was_detection = false;
static bool g_last_detection_geometry_valid = false;
static int g_detect_width = 0;
static int g_detect_height = 0;

// Face trackers — one per face, optical flow between detections
static FaceTracker g_trackers[10];
static int g_active_tracker_count = 0;
static unsigned char* g_prev_frame = nullptr;
static int g_prev_width = 0;
static int g_prev_height = 0;

static bool valid_frame(const unsigned char* rgba_pixels, int width, int height) {
    if (!rgba_pixels || width <= 0 || height <= 0 ||
        width > std::numeric_limits<int>::max() / 4) return false;
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    return pixels <= static_cast<size_t>(std::numeric_limits<int>::max()) / 3;
}

// glReadPixels returns rows bottom-up, so the buffer handed to us is the frame
// upside down. SCRFD barely detects inverted faces, which is why detection only
// landed on some frames. Flip in place once, before detection or tracking, and
// every coordinate downstream is then plain top-left origin screen space.
static std::vector<unsigned char> g_flip_row;
static void flip_vertical(unsigned char* rgba, int width, int height) {
    const size_t stride = static_cast<size_t>(width) * 4u;
    if (g_flip_row.size() < stride) g_flip_row.resize(stride);
    unsigned char* scratch = g_flip_row.data();
    for (int y = 0; y < height / 2; ++y) {
        unsigned char* top = rgba + static_cast<size_t>(y) * stride;
        unsigned char* bottom = rgba + static_cast<size_t>(height - 1 - y) * stride;
        memcpy(scratch, top, stride);
        memcpy(top, bottom, stride);
        memcpy(bottom, scratch, stride);
    }
}

// Initialize trackers only as short-lived recovery support. Landmark detections
// are authoritative for actual geometry, while tracker output still obtains
// landmark motion from its updated box when the detector is between passes.
static void init_trackers(const unsigned char* rgba, int width, int height) {
    g_active_tracker_count = 0;
    for (int i = 0; i < g_cached_count && i < 10; i++) {
        const FaceDetection& face = g_cached_faces[i];
        g_trackers[i].init(rgba, width, height, face.x, face.y, face.w, face.h);
        g_active_tracker_count++;
    }
}

// Track faces in new frame using optical flow
// Returns number of successfully tracked faces
static int track_faces(const unsigned char* rgba, int width, int height) {
    int tracked = 0;
    for (int i = 0; i < g_active_tracker_count; i++) {
        if (!g_trackers[i].isActive()) continue;
        if (!g_trackers[i].track(rgba, width, height)) {
            // Never publish a stale rectangle for one lost face while another
            // tracker survives. Reacquire the complete set on this frame so a
            // hard shake cannot leave a frozen mask behind.
            return 0;
        }
        tracked++;
    }
    return tracked;
}

// Carry anchors through a tracker frame by the box similarity transform. This
// gives translation and scale only during the 33ms gap to the next authoritative
// landmark detector result; an invalid tracker forces
// immediate detector recovery rather than publishing an invented face.
static void get_tracked_faces(int max_faces) {
    int count = 0;
    for (int i = 0; i < g_active_tracker_count && count < max_faces; i++) {
        if (!g_trackers[i].isActive()) continue;
        float x, y, w, h;
        g_trackers[i].getRect(&x, &y, &w, &h);
        FaceDetection& face = g_cached_faces[count];
        if (!(face.w > 0.0f && face.h > 0.0f && face.has_kps)) continue;
        const float sx = w / face.w;
        const float sy = h / face.h;
        for (FacePoint& point : face.kps) {
            point.x = x + (point.x - face.x) * sx;
            point.y = y + (point.y - face.y) * sy;
        }
        face.x = x; face.y = y; face.w = w; face.h = h;
        ++count;
    }
    g_cached_count = count;
}

// Pixelate each face in small blocks. The block average is computed before
// writing, so this is safe for a tightly packed RGBA buffer in-place.
static void blur_face(unsigned char* rgba, int width, int height,
                      int left, int top, int right, int bottom) {
    const int block_size = 12;
    for (int y = top; y < bottom; y += block_size) {
        const int y_end = std::min(y + block_size, bottom);
        for (int x = left; x < right; x += block_size) {
            const int x_end = std::min(x + block_size, right);
            uint32_t sums[4] = {0, 0, 0, 0};
            const int count = (y_end - y) * (x_end - x);
            for (int sample_y = y; sample_y < y_end; ++sample_y) {
                const unsigned char* row = rgba + static_cast<size_t>(sample_y) * width * 4;
                for (int sample_x = x; sample_x < x_end; ++sample_x) {
                    const unsigned char* pixel = row + sample_x * 4;
                    for (int channel = 0; channel < 4; ++channel) sums[channel] += pixel[channel];
                }
            }
            unsigned char average[4];
            for (int channel = 0; channel < 4; ++channel)
                average[channel] = static_cast<unsigned char>(sums[channel] / count);
            for (int write_y = y; write_y < y_end; ++write_y) {
                unsigned char* row = rgba + static_cast<size_t>(write_y) * width * 4;
                for (int write_x = x; write_x < x_end; ++write_x)
                    for (int channel = 0; channel < 4; ++channel) row[write_x * 4 + channel] = average[channel];
            }
        }
    }
}

// C API for ctypes
extern "C" {

__attribute__((visibility("default")))
int blur_faces_init(const char* param_path, const char* bin_path) {
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    LOGI("Initializing SCRFD model...");
    
    if (g_scrfd) {
        LOGI("SCRFD already initialized; keeping existing model");
        return 0;
    }
    g_scrfd = new SCRFD();
    int ret = g_scrfd->load(param_path, bin_path);
    
    if (ret != 0) {
        LOGE("Model load failed: %d", ret);
        delete g_scrfd;
        g_scrfd = nullptr;
        return ret;
    }
    
    LOGI("Model initialized successfully");
    return 0;
}

// Detection body without locking. The JNI entry point already holds
// g_scrfd_mutex for the whole frame, and std::mutex is not recursive, so
// re-locking here deadlocked the detector thread on its very first frame —
// which is exactly why the blur appeared once and then never came back.
static int detect_locked(unsigned char* rgba_pixels, int width, int height,
                         FaceDetection* out_faces, int max_faces) {
    if (!valid_frame(rgba_pixels, width, height) || max_faces < 0 || max_faces > 10 ||
        (max_faces > 0 && !out_faces)) {
        LOGE("Invalid detect arguments");
        return -2;
    }
    if (!g_scrfd) { LOGE("Model not initialized"); return -1; }
    std::vector<FaceDetection> faces;
    const int ret = g_scrfd->detect(rgba_pixels, width, height, faces);
    if (ret < 0) { LOGE("Detection failed: %d", ret); return -1; }
    g_last_detection_geometry_valid = (ret == 0);
    const int count = std::min(static_cast<int>(faces.size()), max_faces);
    for (int i = 0; i < count; ++i) out_faces[i] = faces[i];
    LOGI("Detected %d landmark-anchored faces", count);
    return count;
}

// Compatibility C API remains rectangle-based for offline callers. Runtime JNI
// uses the typed cache below and retains all five anchors.
__attribute__((visibility("default")))
int blur_faces_detect(unsigned char* rgba_pixels, int width, int height,
                      float* out_faces, int max_faces) {
    if (!out_faces || max_faces < 0 || max_faces > 10) return -2;
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    FaceDetection detected[10] = {};
    const int count = detect_locked(rgba_pixels, width, height, detected, max_faces);
    if (count < 0) return count;
    for (int i = 0; i < count; ++i) {
        out_faces[i * 5 + 0] = detected[i].x;
        out_faces[i * 5 + 1] = detected[i].y;
        out_faces[i * 5 + 2] = detected[i].w;
        out_faces[i * 5 + 3] = detected[i].h;
        out_faces[i * 5 + 4] = detected[i].prob;
    }
    return count;
}

// Blur detected faces in-place. Each face is five floats: x, y, w, h, prob.
// Coordinates are in the original frame's pixel space, as returned by detect.
// Body without locking, for callers that already hold g_scrfd_mutex.
static int blur_rgba_locked(unsigned char* rgba_pixels, int width, int height,
                            const float* faces, int face_count) {
    if (!valid_frame(rgba_pixels, width, height) || face_count < 0 ||
        face_count > std::numeric_limits<int>::max() / 5 ||
        (face_count > 0 && !faces)) {
        LOGE("Invalid blur arguments");
        return -2;
    }
    for (int i = 0; i < face_count; ++i) {
        const float x = faces[i * 5 + 0];
        const float y = faces[i * 5 + 1];
        const float w = faces[i * 5 + 2];
        const float h = faces[i * 5 + 3];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) ||
            !std::isfinite(h) || w <= 0.0f || h <= 0.0f) {
            LOGE("Invalid face rectangle at index %d", i);
            return -3;
        }
        // SCRFD bounds the visible facial features, especially when yaw/profile
        // hides one cheek. Blur a head-sized safety area instead: it keeps ears,
        // jaw and side contours private while clipping precisely at frame edges.
        constexpr float kSidePadding = 0.38f;
        constexpr float kTopPadding = 0.24f;
        constexpr float kBottomPadding = 0.42f;
        const float expanded_x = x - w * kSidePadding;
        const float expanded_y = y - h * kTopPadding;
        const float x_end = x + w * (1.0f + kSidePadding);
        const float y_end = y + h * (1.0f + kBottomPadding);
        if (!std::isfinite(expanded_x) || !std::isfinite(expanded_y) ||
            !std::isfinite(x_end) || !std::isfinite(y_end)) {
            LOGE("Face rectangle overflow at index %d", i);
            return -3;
        }
        const float clipped_left = std::max(0.0f, std::min(expanded_x, static_cast<float>(width)));
        const float clipped_top = std::max(0.0f, std::min(expanded_y, static_cast<float>(height)));
        const float clipped_right = std::max(0.0f, std::min(x_end, static_cast<float>(width)));
        const float clipped_bottom = std::max(0.0f, std::min(y_end, static_cast<float>(height)));
        const int left = static_cast<int>(std::floor(clipped_left));
        const int top = static_cast<int>(std::floor(clipped_top));
        const int right = static_cast<int>(std::ceil(clipped_right));
        const int bottom = static_cast<int>(std::ceil(clipped_bottom));
        if (left < right && top < bottom) blur_face(rgba_pixels, width, height,
                                                     left, top, right, bottom);
    }
    return 0;
}

// Public entry point: takes the lock, then runs the same body.
__attribute__((visibility("default")))
int blur_faces_blur_rgba(unsigned char* rgba_pixels, int width, int height,
                         const float* faces, int face_count) {
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    return blur_rgba_locked(rgba_pixels, width, height, faces, face_count);
}

__attribute__((visibility("default")))
void blur_faces_cleanup() {
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    LOGI("Cleaning up...");
    if (g_scrfd) {
        delete g_scrfd;
        g_scrfd = nullptr;
    }
    for (int i = 0; i < 10; i++) {
        g_trackers[i].reset();
    }
    g_active_tracker_count = 0;
    g_cached_count = 0;
    g_last_was_detection = false;
    g_last_detection_geometry_valid = false;
    g_detect_width = 0;
    g_detect_height = 0;
    if (g_prev_frame) {
        delete[] g_prev_frame;
        g_prev_frame = nullptr;
    }
    g_prev_width = 0;
    g_prev_height = 0;
}

} // extern "C"

// Allow a second OpenMP runtime in this process.
//
// ncnn is built with OpenMP and we link it statically, so every copy of this
// library carries its own libomp. Plugin hot reload loads a NEW file each time
// (unique UUID path) and the previous one is never unloaded, so the process ends
// up holding two registered libomp copies. The second one hits
// __kmp_register_library_startup, finds the first, and calls abort() — OMP error
// #15. That is the SIGABRT seen on reload: param loads fine, then load_model
// touches a parallel region and the process dies. On a cold app start there is
// only one copy, which is why it looked like it worked sometimes.
//
// KMP_DUPLICATE_LIB_OK downgrades that fatal error to a warning. It must be set
// before the first OpenMP call, and JNI_OnLoad runs at System.load time, well
// before ncnn touches anything. Oversubscription is not a concern here because
// SCRFD::load pins ncnn to a single worker.
static void allow_duplicate_openmp() {
    setenv("KMP_DUPLICATE_LIB_OK", "TRUE", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
}

// JNI entry points (required for Android to load .so)
extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
    allow_duplicate_openmp();
    LOGI("native plugin loaded (KMP_DUPLICATE_LIB_OK=TRUE)");
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    LOGI("native plugin unloaded");
    blur_faces_cleanup();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_makey_blurfaces_NativeBridge_init(JNIEnv* env, jclass,
                                            jstring param_path, jstring bin_path) {
    if (!param_path || !bin_path) return -2;
    const char* param = env->GetStringUTFChars(param_path, nullptr);
    const char* bin = env->GetStringUTFChars(bin_path, nullptr);
    int result = (param && bin) ? blur_faces_init(param, bin) : -2;
    if (param) env->ReleaseStringUTFChars(param_path, param);
    if (bin) env->ReleaseStringUTFChars(bin_path, bin);
    return result;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_makey_blurfaces_NativeBridge_process(JNIEnv* env, jclass, jobject rgba,
                                               jint width, jint height, jboolean detect) {
    if (!rgba || width <= 0 || height <= 0) return -2;
    auto* pixels = static_cast<unsigned char*>(env->GetDirectBufferAddress(rgba));
    jlong capacity = env->GetDirectBufferCapacity(rgba);
    const size_t required = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    if (!pixels || capacity < 0 || static_cast<size_t>(capacity) < required) return -2;
    
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);

    // The buffer arrives bottom-up from glReadPixels. Flip once here so both
    // SCRFD and the optical-flow tracker see an upright frame, and every
    // rectangle produced below is plain top-left origin.
    flip_vertical(pixels, width, height);

    g_detect_width = width;
    g_detect_height = height;
    g_last_was_detection = detect || g_active_tracker_count <= 0;

    if (g_last_was_detection) {
        // Full SCRFD detection: either the scheduled detect pass, or a recovery
        // because tracking lost every face.
        int count = detect_locked(pixels, width, height, g_cached_faces, 10);
        if (count < 0) return count;
        g_cached_count = count;
        if (count > 0) {
            init_trackers(pixels, width, height);
            const FaceDetection& first = g_cached_faces[0];
            LOGI("detect frame: %d landmark faces (%dx%d) first=(%.1f,%.1f %.1fx%.1f p=%.3f)",
                 count, width, height, first.x, first.y, first.w, first.h, first.prob);
        } else {
            // A zero-face detection is authoritative. Do not let trackers from
            // the previous scene keep publishing their last rectangle.
            for (int i = 0; i < 10; ++i) g_trackers[i].reset();
            g_active_tracker_count = 0;
            std::memset(g_cached_faces, 0, sizeof(g_cached_faces));
            LOGI("detect frame: 0 faces (%dx%d); trackers reset", width, height);
        }
        return 0;
    }

    // Tracking pass: optical flow moves the existing boxes with the face.
    int tracked = track_faces(pixels, width, height);
    if (tracked > 0) {
        get_tracked_faces(10);
        return 0;
    }

    // Do not wait for the next scheduled detector tick after a complete
    // tracking miss. Snapchat-style reacquisition is detect-on-recovery: the
    // current frame is the best chance to find the face before the mask drifts
    // or the fail-closed transition cover becomes visible.
    LOGI("tracking miss: immediate detector recovery");
    g_last_was_detection = true;
    int recovered = detect_locked(pixels, width, height, g_cached_faces, 10);
    if (recovered < 0) return recovered;
    g_cached_count = recovered;
    if (recovered > 0) {
        init_trackers(pixels, width, height);
        LOGI("recovery frame: %d faces (%dx%d)", recovered, width, height);
    } else {
        for (int i = 0; i < 10; ++i) g_trackers[i].reset();
        g_active_tracker_count = 0;
        std::memset(g_cached_faces, 0, sizeof(g_cached_faces));
        LOGI("recovery frame: 0 faces; trackers reset");
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_makey_blurfaces_NativeBridge_getLastFaceCount(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    return g_cached_count;
}

extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_NativeBridge_resetTracking(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    for (int i = 0; i < 10; ++i) g_trackers[i].reset();
    g_active_tracker_count = 0;
    g_cached_count = 0;
    g_last_was_detection = false;
    g_last_detection_geometry_valid = false;
    std::memset(g_cached_faces, 0, sizeof(g_cached_faces));
    LOGI("camera source changed; native trackers and cached faces reset");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_makey_blurfaces_NativeBridge_wasLastProcessDetection(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    // Only a detector run with valid landmark geometry may clear the emergency
    // cover on a zero-face result. Tracker output and rejected KPS remain unknown.
    return (g_last_was_detection && g_last_detection_geometry_valid) ? JNI_TRUE : JNI_FALSE;
}

// Return 14 UV floats per face in the captured upright top-left coordinate
// system: box x/y/w/h followed by left-eye/right-eye/nose/left-mouth/right-mouth.
// The caller derives an affine ellipse from semantic anchors, so roll and scale
// are encoded in its basis instead of being guessed from an axis-aligned box.
extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_NativeBridge_getFaceAnchors(JNIEnv* env, jclass,
                                                     jfloatArray out) {
    std::lock_guard<std::mutex> lock(g_scrfd_mutex);
    if (!out || g_cached_count <= 0 || g_detect_width <= 0 || g_detect_height <= 0) return;
    const int stride = 14;
    int n = std::min(g_cached_count, static_cast<int>(env->GetArrayLength(out)) / stride);
    float uv[10 * stride] = {};
    const float invW = 1.0f / g_detect_width;
    const float invH = 1.0f / g_detect_height;
    for (int i = 0; i < n; ++i) {
        const FaceDetection& face = g_cached_faces[i];
        float* dst = uv + i * stride;
        dst[0] = face.x * invW; dst[1] = face.y * invH;
        dst[2] = face.w * invW; dst[3] = face.h * invH;
        for (int k = 0; k < 5; ++k) {
            dst[4 + k * 2] = face.kps[k].x * invW;
            dst[5 + k * 2] = face.kps[k].y * invH;
        }
    }
    env->SetFloatArrayRegion(out, 0, n * stride, uv);
}

extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_NativeBridge_cleanup(JNIEnv*, jclass) {
    blur_faces_cleanup();
}

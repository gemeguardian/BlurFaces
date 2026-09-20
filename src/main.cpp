#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "blur_faces.h"
#include "head_detector.h"
#include "bytetrack.h"

#define TAG "BlurFaces"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static HeadDetector* g_detector = nullptr;
static ByteTracker* g_tracker = nullptr;
static std::mutex g_engine_mutex;

static void allow_duplicate_openmp() {
    setenv("KMP_DUPLICATE_LIB_OK", "TRUE", 1);
    setenv("OMP_NUM_THREADS", "2", 1);
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    allow_duplicate_openmp();
    LOGI("native plugin v1.0 loaded (NCNN HeadDetector + ByteTrack)");
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM*, void*) {
    LOGI("native plugin v1.0 unloaded");
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (g_detector) { delete g_detector; g_detector = nullptr; }
    if (g_tracker) { delete g_tracker; g_tracker = nullptr; }
}

extern "C" {

int blur_faces_init(const char* param_path, const char* bin_path) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    LOGI("Initializing HeadDetector and ByteTrack...");
    if (!g_detector) g_detector = new HeadDetector();
    if (!g_tracker) g_tracker = new ByteTracker(0.45f, 0.20f, 0.70f, 30);

    int ret = g_detector->load(param_path, bin_path);
    if (ret != 0) {
        LOGE("HeadDetector load failed: %d", ret);
        delete g_detector; g_detector = nullptr;
        delete g_tracker; g_tracker = nullptr;
        return ret;
    }
    g_tracker->reset();
    LOGI("HeadDetector + ByteTrack initialized successfully");
    return 0;
}

void blur_faces_cleanup() {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (g_detector) { delete g_detector; g_detector = nullptr; }
    if (g_tracker) { delete g_tracker; g_tracker = nullptr; }
    LOGI("HeadDetector + ByteTrack cleaned up");
}

} // extern "C"

// JNI bindings for com.makey.blurfaces.g2.NativeBridge
extern "C" JNIEXPORT jint JNICALL
Java_com_makey_blurfaces_g2_NativeBridge_init(JNIEnv* env, jclass,
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
Java_com_makey_blurfaces_g2_NativeBridge_process(JNIEnv* env, jclass,
                                                 jobject rgba_buf, jint width, jint height,
                                                 jfloatArray out_geom, jfloatArray out_scores,
                                                 jfloatArray out_yaws, jint max_faces,
                                                 jfloat min_confidence) {
    if (!rgba_buf || width <= 0 || height <= 0 || max_faces <= 0) return -2;
    auto* pixels = static_cast<unsigned char*>(env->GetDirectBufferAddress(rgba_buf));
    jlong capacity = env->GetDirectBufferCapacity(rgba_buf);
    const size_t required = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    if (!pixels || capacity < 0 || static_cast<size_t>(capacity) < required) return -2;

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (!g_detector || !g_detector->is_initialized() || !g_tracker) {
        LOGE("Engine not initialized");
        return -1;
    }

    // Map min_confidence to ByteTrack thresholds for YOLOv8
    float min_conf = static_cast<float>(min_confidence);
    if (min_conf <= 0.0f) min_conf = 0.35f;
    if (min_conf > 1.0f) min_conf /= 100.0f;

    // high_thresh: threshold for high-confidence detections
    // low_thresh: floor for ByteTrack stage 2 matching
    // instant_thresh: threshold for single-frame instant activation
    float high_thresh = std::max(0.20f, std::min(0.60f, min_conf));
    float low_thresh = std::max(0.15f, high_thresh * 0.60f);
    float instant_thresh = std::max(0.40f, std::min(0.70f, high_thresh + 0.15f));

    // Run NCNN Head Detection with low_thresh as detection floor
    std::vector<HeadBox> detected_heads;
    g_detector->detect(pixels, width, height, detected_heads, low_thresh, 0.45f);

    // Update ByteTrack multi-object tracker (returns only confirmed active tracks)
    std::vector<STrack> active_tracks = g_tracker->update(detected_heads, high_thresh, low_thresh, instant_thresh);

    int count = std::min(static_cast<int>(active_tracks.size()), max_faces);
    if (count > 0 && out_geom) {
        std::vector<float> geom_buffer(count * 6);
        std::vector<float> score_buffer(count);
        std::vector<float> yaw_buffer(count, 0.0f);

        for (int i = 0; i < count; ++i) {
            active_tracks[i].get_geometry(geom_buffer.data() + i * 6);
            float score = active_tracks[i].score();
            if (active_tracks[i].state() == TrackState::Lost) {
                score *= 0.5f;
            }
            score_buffer[i] = score;
            LOGI("[Face #%d] score=%.2f center=(%.2f, %.2f) radius=(%.2f, %.2f)",
                 active_tracks[i].track_id(), score,
                 geom_buffer[i * 6 + 0], geom_buffer[i * 6 + 1],
                 geom_buffer[i * 6 + 2], geom_buffer[i * 6 + 5]);
        }

        env->SetFloatArrayRegion(out_geom, 0, count * 6, geom_buffer.data());
        if (out_scores) {
            env->SetFloatArrayRegion(out_scores, 0, count, score_buffer.data());
        }
        if (out_yaws) {
            env->SetFloatArrayRegion(out_yaws, 0, count, yaw_buffer.data());
        }
    }

    return count;
}

extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_g2_NativeBridge_reset(JNIEnv*, jclass) {
    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (g_tracker) {
        g_tracker->reset();
        LOGI("ByteTracker reset on camera switch");
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_g2_NativeBridge_cleanup(JNIEnv*, jclass) {
    blur_faces_cleanup();
}

// Backward-compatibility JNI bindings for com.makey.blurfaces.NativeBridge
extern "C" JNIEXPORT jint JNICALL
Java_com_makey_blurfaces_NativeBridge_init(JNIEnv* env, jclass clazz,
                                           jstring param_path, jstring bin_path) {
    return Java_com_makey_blurfaces_g2_NativeBridge_init(env, clazz, param_path, bin_path);
}

extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_NativeBridge_resetTracking(JNIEnv* env, jclass clazz) {
    Java_com_makey_blurfaces_g2_NativeBridge_reset(env, clazz);
}

extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_NativeBridge_cleanup(JNIEnv* env, jclass clazz) {
    Java_com_makey_blurfaces_g2_NativeBridge_cleanup(env, clazz);
}

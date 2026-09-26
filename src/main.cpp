#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "blur_faces.h"
#include "head_detector.h"
#include "bytetrack.h"
#include "detection_result.h"
#include "debug_snapshot.h"

#define TAG "BlurFaces"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static HeadDetector* g_detector = nullptr;
static ByteTracker* g_tracker = nullptr;
static std::mutex g_engine_mutex;

// Tracker resets are requested from the camera GL thread and the UI thread on
// every camera flip / source activation, while the inference worker holds
// g_engine_mutex for a whole NCNN forward pass (50-120 ms on device). Taking the
// mutex there stalled rendering, so a reset only bumps this generation; the
// worker applies it before touching the tracker. g_tracker_generation is the
// generation the tracker state belongs to and is guarded by g_engine_mutex.
static std::atomic<uint32_t> g_reset_generation{0};
static uint32_t g_tracker_generation = 0;

// Detections from a frame whose inference overlapped a reset are dropped
// without advancing the tracker. Negative, so Java never reads it as "no heads".
static constexpr jint kStaleAfterReset = -6;

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
    if (!g_tracker) g_tracker = new ByteTracker(0.45f, 0.20f, 0.70f, ByteTracker::kMaxTimeLostFrames);

    int ret = g_detector->load(param_path, bin_path);
    if (ret != 0) {
        LOGE("HeadDetector load failed: %d", ret);
        delete g_detector; g_detector = nullptr;
        delete g_tracker; g_tracker = nullptr;
        return ret;
    }
    g_tracker->reset();
    g_tracker_generation = g_reset_generation.load(std::memory_order_acquire);
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

static jint process_frame(JNIEnv* env, jobject rgba_buf, jint width, jint height,
                          jfloatArray out_geom, jfloatArray out_scores,
                          jfloatArray out_yaws, jint max_faces,
                          jfloat min_confidence, jfloatArray debug) {
    if (debug && env->GetArrayLength(debug) < kDebugFloats) return -2;
    if (!rgba_buf || width <= 0 || height <= 0 || max_faces <= 0 || max_faces > 4
            || !out_geom || env->GetArrayLength(out_geom) < max_faces * 6
            || (out_scores && env->GetArrayLength(out_scores) < max_faces)
            || (out_yaws && env->GetArrayLength(out_yaws) < max_faces)
            || !std::isfinite(min_confidence)) return -2;
    auto* pixels = static_cast<unsigned char*>(env->GetDirectBufferAddress(rgba_buf));
    jlong capacity = env->GetDirectBufferCapacity(rgba_buf);
    const size_t required = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    if (!pixels || capacity < 0 || static_cast<size_t>(capacity) < required) return -2;

    std::lock_guard<std::mutex> lock(g_engine_mutex);
    if (!g_detector || !g_detector->is_initialized() || !g_tracker) {
        LOGE("Engine not initialized");
        return -1;
    }
    const uint32_t generation = g_reset_generation.load(std::memory_order_acquire);
    if (generation != g_tracker_generation) {
        g_tracker->reset();
        g_tracker_generation = generation;
    }

    // Map min_confidence to ByteTrack thresholds for YOLOv8
    float min_conf = static_cast<float>(min_confidence);
    if (min_conf <= 0.0f) min_conf = 0.35f;
    if (min_conf > 1.0f) min_conf /= 100.0f;

    // high_thresh: threshold for high-confidence detections (needs kMinHits frames)
    // low_thresh: floor for ByteTrack stage 2 matching of already-confirmed tracks
    // instant_thresh: single-frame activation. Device logs (rear camera, empty
    // room) show clutter scoring 0.41-0.61 for one or two frames, so a single
    // frame is only trusted when the model is genuinely certain.
    float high_thresh = std::max(0.18f, std::min(0.50f, min_conf));
    float low_thresh = std::max(0.12f, high_thresh * 0.55f);
    float instant_thresh = std::max(0.70f, std::min(0.85f, high_thresh + 0.35f));

    // Low-light relaxation: model confidence physically drops in darkness, so
    // keeping daylight thresholds costs recall. Uses the previous detector
    // frame's luminance: ~130 ms old at the measured 7-8 detector fps, well
    // inside the CLAHE hysteresis band and EMA smoothing. The CLAHE enhancement
    // already amplifies noise in the same regime, so only the multi-frame
    // thresholds are relaxed; single-frame instant activation never is.
    float scene_lum = g_detector->last_mean_lum();
    if (scene_lum < 45.0f) {
        float k = std::clamp((45.0f - scene_lum) / 35.0f, 0.0f, 1.0f);
        high_thresh = std::max(0.14f, high_thresh * (1.0f - 0.35f * k));
        low_thresh  = std::max(0.10f, low_thresh  * (1.0f - 0.35f * k));
    }

    // Run NCNN Head Detection with low_thresh as detection floor
    std::vector<HeadBox> detected_heads;
    int status = g_detector->detect(pixels, width, height, detected_heads, low_thresh, 0.45f);

    // A reset arrived during inference: these detections belong to the previous
    // camera/session and must not seed tracks in the new one. The next frame
    // applies the reset; Java drops this result.
    if (g_reset_generation.load(std::memory_order_acquire) != generation) {
        if (debug) {
            auto snapshot = debug_snapshot(kStaleAfterReset, high_thresh, low_thresh, instant_thresh,
                    scene_lum, g_detector->last_mean_lum(), {}, {});
            env->SetFloatArrayRegion(debug, 0, kDebugFloats, snapshot.data());
        }
        return kStaleAfterReset;
    }

    // Preserve native failures all the way to Java's full-frame fallback.
    std::vector<STrack> active_tracks;
    int count = update_detection_result(status, detected_heads, *g_tracker,
            high_thresh, low_thresh, instant_thresh, max_faces, active_tracks);
    if (debug) {
        auto snapshot = debug_snapshot(status, high_thresh, low_thresh, instant_thresh,
                scene_lum, g_detector->last_mean_lum(), detected_heads, active_tracks);
        env->SetFloatArrayRegion(debug, 0, kDebugFloats, snapshot.data());
    }
    if (count < 0) return count;
    if (count > 0 && out_geom) {
        std::vector<float> geom_buffer(count * 6);
        std::vector<float> score_buffer(count);
        std::vector<float> yaw_buffer(count, 0.0f);

        for (int i = 0; i < count; ++i) {
            active_tracks[i].get_geometry(geom_buffer.data() + i * 6);
            float score = active_tracks[i].score();
            if (active_tracks[i].state() == TrackState::Lost) {
                score = std::max(0.25f, score * 0.85f);
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

extern "C" JNIEXPORT jint JNICALL
Java_com_makey_blurfaces_g2_NativeBridge_process(JNIEnv* env, jclass,
        jobject rgba, jint width, jint height, jfloatArray geom, jfloatArray scores,
        jfloatArray yaws, jint max_faces, jfloat confidence) {
    return process_frame(env, rgba, width, height, geom, scores, yaws, max_faces, confidence, nullptr);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_makey_blurfaces_g2_NativeBridge_processDebug(JNIEnv* env, jclass,
        jobject rgba, jint width, jint height, jfloatArray geom, jfloatArray scores,
        jfloatArray yaws, jint max_faces, jfloat confidence, jfloatArray debug) {
    if (!debug) return -2;
    return process_frame(env, rgba, width, height, geom, scores, yaws, max_faces, confidence, debug);
}

extern "C" JNIEXPORT void JNICALL
Java_com_makey_blurfaces_g2_NativeBridge_reset(JNIEnv*, jclass) {
    // Lock-free on purpose, see g_reset_generation.
    g_reset_generation.fetch_add(1, std::memory_order_acq_rel);
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

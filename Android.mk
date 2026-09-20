LOCAL_PATH := $(call my-dir)

# NCNN prebuilt
include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := ncnn-prebuilt/arm64-v8a/libncnn.a
LOCAL_EXPORT_C_INCLUDES := $(LOCAL_PATH)/ncnn-prebuilt/arm64-v8a/include
include $(PREBUILT_STATIC_LIBRARY)

# Main shared library
include $(CLEAR_VARS)
LOCAL_MODULE := blur_faces
LOCAL_SRC_FILES := src/head_detector.cpp src/bytetrack.cpp src/main.cpp
LOCAL_C_INCLUDES := $(LOCAL_PATH)/ncnn-prebuilt/arm64-v8a/include
LOCAL_STATIC_LIBRARIES := ncnn
# ncnn was built with OpenMP; keep its runtime statically contained in the
# hot-loaded library and limit execution to one worker in HeadDetector::load().
LOCAL_LDLIBS := -landroid -llog -lz -ljnigraphics -fopenmp -static-openmp
LOCAL_CPPFLAGS := -std=c++17 -fvisibility=hidden -fopenmp -ffunction-sections -fdata-sections -Os
LOCAL_LDFLAGS := -Wl,-Bsymbolic -Wl,--exclude-libs,ALL -Wl,--gc-sections
include $(BUILD_SHARED_LIBRARY)

LOCAL_PATH := $(call my-dir)

# NCNN prebuilt
include $(CLEAR_VARS)
LOCAL_MODULE := ncnn
LOCAL_SRC_FILES := libs/arm64-v8a/libncnn.a
LOCAL_EXPORT_C_INCLUDES := libs/arm64-v8a/include/ncnn
include $(PREBUILT_STATIC_LIBRARY)

# Main shared library
include $(CLEAR_VARS)
LOCAL_MODULE := blur_faces
LOCAL_SRC_FILES := src/scrfd.cpp src/face_tracker.cpp src/main.cpp
LOCAL_C_INCLUDES := libs/arm64-v8a/include
LOCAL_STATIC_LIBRARIES := ncnn
# ncnn was built with OpenMP; keep its runtime statically contained in the
# hot-loaded library and limit execution to one worker in SCRFD::load().
LOCAL_LDLIBS := -landroid -llog -lz -ljnigraphics -fopenmp -static-openmp
LOCAL_CPPFLAGS := -std=c++17 -fvisibility=hidden -fopenmp
LOCAL_LDFLAGS := -Wl,-Bsymbolic -Wl,--exclude-libs,ALL -Wl,-soname,libblur_faces_@@SONAME_VERSION@@.so
include $(BUILD_SHARED_LIBRARY)

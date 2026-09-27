#pragma once

#include "bytetrack.h"

// Shared by JNI and host-native behavioral tests. A failed inference must not
// advance the tracker with an empty observation or publish a successful count.
inline int update_detection_result(int status, const std::vector<HeadBox>& heads,
                                   ByteTracker& tracker, float high, float low,
                                   float instant, int capacity,
                                   std::vector<STrack>& output,
                                   float frame_dt_ms = ByteTracker::kReferenceFrameMs,
                                   float camera_dx = 0.0f, float camera_dy = 0.0f) {
    output.clear();
    if (status < 0) {
        tracker.reset();
        return status;
    }
    if (status != static_cast<int>(heads.size())) {
        tracker.reset();
        return -4;
    }
    output = tracker.update(heads, high, low, instant, frame_dt_ms, camera_dx, camera_dy);
    if (static_cast<int>(output.size()) > capacity) {
        output.clear();
        return -5; // Full-frame protection rather than silently dropping heads.
    }
    return static_cast<int>(output.size());
}

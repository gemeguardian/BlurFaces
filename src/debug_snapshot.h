#pragma once
#include "bytetrack.h"
#include <array>
#include <algorithm>

// Versioned diagnostic-only ABI. No effect on detection/tracking decisions.
constexpr int kDebugCandidates = 64;
constexpr int kDebugTracks = 4;
constexpr int kDebugHeader = 10;
constexpr int kDebugFloats = kDebugHeader + kDebugCandidates * 5 + kDebugTracks * 8;

inline std::array<float, kDebugFloats> debug_snapshot(
        int status, float high, float low, float instant, float previous_lum,
        float current_lum, const std::vector<HeadBox>& candidates,
        const std::vector<STrack>& tracks) {
    std::array<float, kDebugFloats> out{};
    int nc = status < 0 ? 0 : std::min<int>(candidates.size(), kDebugCandidates);
    int nt = std::min<int>(tracks.size(), kDebugTracks);
    out[0] = 1; out[1] = status; out[2] = high; out[3] = low; out[4] = instant;
    out[5] = previous_lum; out[6] = current_lum;
    out[7] = status < 0 ? 0 : candidates.size(); out[8] = nc; out[9] = nt;
    for (int i = 0; i < nc; ++i) {
        int p = kDebugHeader + i * 5;
        const auto& b = candidates[i];
        out[p] = b.x1; out[p+1] = b.y1; out[p+2] = b.x2;
        out[p+3] = b.y2; out[p+4] = b.score;
    }
    for (int i = 0; i < nt; ++i) {
        int p = kDebugHeader + kDebugCandidates * 5 + i * 8;
        const auto& t = tracks[i];
        HeadBox b = t.current_box();
        out[p] = t.track_id(); out[p+1] = static_cast<int>(t.state());
        out[p+2] = t.frames_lost(); out[p+3] = t.score();
        out[p+4] = b.x1; out[p+5] = b.y1; out[p+6] = b.x2; out[p+7] = b.y2;
    }
    return out;
}

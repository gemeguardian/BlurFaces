#include "detection_result.h"
#include "debug_snapshot.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

static HeadBox head(float x, float score) {
    return {x - .05f, .4f, x + .05f, .6f, x, .5f, .1f, .2f, score};
}

int main() {
    ByteTracker tracker;
    std::vector<STrack> output;
    auto process = [&](int status, const std::vector<HeadBox>& heads, int capacity = 4) {
        return update_detection_result(status, heads, tracker, .25f, .12f, .35f, capacity, output);
    };
    // Native confirmation, including the low-light regime used by JNI.
    assert(process(1, {head(.5f, .30f)}) == 0);
    assert(process(1, {head(.505f, .30f)}) == 1);
    int id = output.front().track_id();
    assert(process(1, {head(.51f, .15f)}) == 1);
    assert(output.front().track_id() == id);
    assert(output.front().state() == TrackState::Tracked);
    assert(std::fabs(output.front().score() - .15f) < .00001f);
    float geometry[6];
    output.front().get_geometry(geometry);
    for (float value : geometry) assert(std::isfinite(value));
    assert(geometry[2] > 0.f && geometry[5] > 0.f);

    // Diagnostic ABI preserves observations vs native coasting without updating tracks.
    constexpr int track_offset = kDebugHeader + kDebugCandidates * 5;
    auto snapshot = debug_snapshot(1, .25f, .12f, .35f, 20.f, 25.f,
            {head(.51f, .15f)}, output);
    assert(snapshot.size() == 362 && snapshot[0] == 1);
    assert(snapshot[7] == 1 && snapshot[8] == 1 && snapshot[9] == 1);
    assert(snapshot[track_offset] == id);
    assert(snapshot[track_offset + 1] == static_cast<int>(TrackState::Tracked));
    assert(snapshot[track_offset + 2] == 0);
    assert(process(0, {}) == 1);
    snapshot = debug_snapshot(0, .25f, .12f, .35f, 25.f, 25.f, {}, output);
    assert(snapshot[8] == 0 && snapshot[9] == 1);
    assert(snapshot[track_offset + 1] == static_cast<int>(TrackState::Lost));
    assert(snapshot[track_offset + 2] > 0);
    std::vector<HeadBox> many(70, head(.5f, .9f));
    snapshot = debug_snapshot(70, .25f, .12f, .35f, 25.f, 25.f, many, {});
    assert(snapshot[7] == 70 && snapshot[8] == 64 && snapshot[9] == 0);
    snapshot = debug_snapshot(-4, .25f, .12f, .35f, 25.f, 25.f, many, {});
    assert(snapshot[1] == -4 && snapshot[7] == 0 && snapshot[8] == 0);

    // Every inference failure preserves its status and clears old tracker state.
    for (int error : {-1, -2, -3, -4}) {
        assert(process(1, {head(.5f, .9f)}) == 1);
        assert(process(error, {}) == error);
        assert(output.empty());
        assert(process(0, {}) == 0);
        assert(output.empty());
    }
    assert(process(1, {}) == -4); // inconsistent native count
    assert(process(0, {}) == 0);

    // Transient borderline clutter is not confirmed on its first frame.
    assert(process(1, {head(.5f, .30f)}) == 0);
    assert(process(0, {}) == 0);
    assert(process(0, {}) == 0);

    // Independent heads survive detection permutation and reset on camera flip.
    std::vector<HeadBox> heads = {head(.2f, .9f), head(.5f, .9f), head(.8f, .9f)};
    assert(process(3, heads) == 3);
    std::reverse(heads.begin(), heads.end());
    assert(process(3, heads) == 3);
    for (const auto& track : output) assert(track.state() == TrackState::Tracked);
    tracker.reset();
    assert(process(0, {}) == 0);

    // Renderer overflow must cover the whole frame, not omit the fifth head.
    heads = {head(.1f, .9f), head(.3f, .9f), head(.5f, .9f), head(.7f, .9f), head(.9f, .9f)};
    assert(process(5, heads) == -5);
    assert(output.empty());
    tracker.reset();
    assert(process(0, {}) == 0);
    std::cout << "PASS: real ByteTrack + JNI result policy: confirmation, errors, recovery, overflow, reset\n";
}

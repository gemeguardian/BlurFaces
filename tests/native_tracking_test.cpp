#include "detection_result.h"
#include "debug_snapshot.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

// Distant head: 10% x 20% of the frame (area .02 -> size reliability 2/3).
static HeadBox head(float x, float score) {
    return {x - .05f, .4f, x + .05f, .6f, x, .5f, .1f, .2f, score};
}
// Selfie head: 50% x 60% of the frame (area .3 -> fully reliable).
static HeadBox selfie(float x, float score) {
    return {x - .25f, .2f, x + .25f, .8f, x, .5f, .5f, .6f, score};
}
// Device trace 2026-09-22 16:28: backpack box 10% x 9% (area .009 -> reliability floor).
static HeadBox backpack(float score) {
    return {.15f, .345f, .25f, .435f, .20f, .39f, .10f, .09f, score};
}

int main() {
    ByteTracker tracker;
    std::vector<STrack> output;
    // Thresholds mirror the JNI mapping for the "Normal" preset in low light:
    // high .25, low .12, instant .70 (instant is never relaxed by lighting).
    auto process = [&](int status, const std::vector<HeadBox>& heads, int capacity = 4) {
        return update_detection_result(status, heads, tracker, .25f, .12f, .70f, capacity, output);
    };
    // Same preset in daylight: high .35, low .19.
    auto process_day = [&](int status, const std::vector<HeadBox>& heads) {
        return update_detection_result(status, heads, tracker, .35f, .19f, .70f, 4, output);
    };
    static_assert(STrack::kMinHits == 2, "test written for a 2-hit floor");
    static_assert(STrack::kLogOddsConfirm == 1.5f && STrack::kEvidenceCap == 0.75f,
                  "test written for confirm=1.5, cap=.75");
    static_assert(STrack::kMinHitsForCoast == 6, "test written for 6-hit coast trust");
    static_assert(STrack::kMaxCoastPublishFrames == 4, "test written for 4-frame coast");

    // SPRT confirmation: a confident selfie head confirms on the second frame,
    // a confident distant head on the third, a borderline distant head
    // (.30 vs neutral .185) on the fourth.
    assert(process(1, {selfie(.5f, .55f)}) == 0);
    assert(process(1, {selfie(.5f, .55f)}) == 1);
    tracker.reset();
    for (int i = 0; i < 2; ++i) assert(process(1, {head(.5f, .90f)}) == 0);
    assert(process(1, {head(.5f, .90f)}) == 1);
    tracker.reset();
    for (int i = 0; i < 3; ++i) assert(process(1, {head(.5f, .30f)}) == 0);
    assert(process(1, {head(.5f, .30f)}) == 1);
    tracker.reset();

    // After confirmation, low-score (stage 2) matches keep the track alive.
    assert(process(1, {selfie(.5f, .55f)}) == 0);
    assert(process(1, {selfie(.505f, .55f)}) == 1);
    int id = output.front().track_id();
    assert(process(1, {selfie(.51f, .15f)}) == 1);
    assert(output.front().track_id() == id);
    assert(output.front().state() == TrackState::Tracked);
    assert(std::fabs(output.front().score() - .15f) < .00001f);
    float geometry[6];
    output.front().get_geometry(geometry);
    for (float value : geometry) assert(std::isfinite(value));
    assert(geometry[2] > 0.f && geometry[5] > 0.f);

    // Diagnostic ABI preserves observations vs native coasting without updating tracks.
    constexpr int track_offset = kDebugHeader + kDebugCandidates * 5;
    auto snapshot = debug_snapshot(1, .25f, .12f, .70f, 20.f, 25.f,
            {selfie(.51f, .15f)}, output);
    assert(snapshot.size() == 362 && snapshot[0] == 1);
    assert(snapshot[7] == 1 && snapshot[8] == 1 && snapshot[9] == 1);
    assert(snapshot[track_offset] == id);
    assert(snapshot[track_offset + 1] == static_cast<int>(TrackState::Tracked));
    assert(snapshot[track_offset + 2] == 0);

    // A track with fewer than kMinHitsForCoast observations is not coasted when it
    // disappears: this was the 0.26/0.31/0.43 ghost blur seen on device.
    assert(output.front().frames_tracked() == 3);
    assert(process(0, {}) == 0);
    snapshot = debug_snapshot(0, .25f, .12f, .70f, 25.f, 25.f, {}, output);
    assert(snapshot[8] == 0 && snapshot[9] == 0);
    // It is still kept for re-identification for two frames and keeps its identity.
    assert(process(1, {selfie(.515f, .55f)}) == 1);
    assert(output.front().track_id() == id);
    assert(output.front().frames_tracked() == 4);
    assert(process(1, {selfie(.52f, .55f)}) == 1);
    assert(process(1, {selfie(.52f, .55f)}) == 1);
    assert(output.front().frames_tracked() == 6);

    // An established track coasts, but only for kMaxCoastPublishFrames frames.
    for (int i = 0; i < STrack::kMaxCoastPublishFrames; ++i) {
        assert(process(0, {}) == 1);
        assert(output.front().track_id() == id);
        assert(output.front().state() == TrackState::Lost);
    }
    snapshot = debug_snapshot(0, .25f, .12f, .70f, 25.f, 25.f, {}, output);
    assert(snapshot[8] == 0 && snapshot[9] == 1);
    assert(snapshot[track_offset + 1] == static_cast<int>(TrackState::Lost));
    assert(snapshot[track_offset + 2] > 0);
    assert(process(0, {}) == 0);
    // ...while remaining re-identifiable within max_time_lost.
    assert(process(1, {selfie(.53f, .55f)}) == 1);
    assert(output.front().track_id() == id);
    tracker.reset();

    std::vector<HeadBox> many(70, selfie(.5f, .9f));
    snapshot = debug_snapshot(70, .25f, .12f, .70f, 25.f, 25.f, many, {});
    assert(snapshot[7] == 70 && snapshot[8] == 64 && snapshot[9] == 0);
    snapshot = debug_snapshot(-4, .25f, .12f, .70f, 25.f, 25.f, many, {});
    assert(snapshot[1] == -4 && snapshot[7] == 0 && snapshot[8] == 0);

    // Device false positives 2026-09-22 15:46 (rear camera, nobody in frame):
    // one frame at .48, two frames at .50/.61, and .41/.54 followed by a miss.
    // None of them may reach the renderer, at any point.
    assert(process(1, {head(.3f, .48f)}) == 0);
    assert(process(0, {}) == 0);
    assert(process(1, {head(.1f, .50f)}) == 0);
    assert(process(1, {head(.1f, .61f)}) == 0);
    assert(process(0, {}) == 0);
    assert(process(1, {head(.7f, .41f)}) == 0);
    assert(process(1, {head(.7f, .54f)}) == 0);
    assert(process(1, {head(.7f, .15f)}) == 0); // below high: confirmation chain broken
    assert(process(1, {head(.7f, .52f)}) == 0); // restarts from one hit
    assert(process(0, {}) == 0);
    assert(process(0, {}) == 0);
    assert(output.empty());
    tracker.reset();

    // Device false positive 2026-09-22 16:28 (rear camera, backpack, daylight):
    // .71 in a 10%x9% box (instant score, but too small to trust a single frame),
    // .45, a miss, then a run of ~.42. Six consistent frames still stay silent.
    assert(process_day(1, {backpack(.71f)}) == 0);
    assert(process_day(1, {backpack(.45f)}) == 0);
    assert(process_day(0, {}) == 0);
    for (int i = 0; i < 6; ++i) assert(process_day(1, {backpack(.42f)}) == 0);
    assert(output.empty());
    tracker.reset();
    // The same score on a selfie-sized head (device trace 16:27, front camera:
    // .70-.84 sustained) is trusted immediately.
    assert(process_day(1, {selfie(.5f, .76f)}) == 1);
    tracker.reset();

    // A confirmed track that keeps scoring below neutral argues itself out of
    // being a head and stops being published, even though it is still matched.
    assert(process(1, {selfie(.5f, .55f)}) == 0);
    assert(process(1, {selfie(.5f, .55f)}) == 1);
    for (int i = 0; i < 12 && !output.empty(); ++i) process(1, {selfie(.5f, .13f)});
    assert(output.empty());
    tracker.reset();

    // Every inference failure preserves its status and clears old tracker state.
    for (int error : {-1, -2, -3, -4}) {
        assert(process(1, {selfie(.5f, .9f)}) == 1);
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
    assert(process(3, heads) == 0); // distant heads: no instant, SPRT needs 3 frames
    assert(process(3, heads) == 0);
    assert(process(3, heads) == 3);
    std::reverse(heads.begin(), heads.end());
    assert(process(3, heads) == 3);
    for (const auto& track : output) assert(track.state() == TrackState::Tracked);
    tracker.reset();
    assert(process(0, {}) == 0);

    // Renderer overflow must cover the whole frame, not omit the fifth head.
    heads = {head(.1f, .9f), head(.3f, .9f), head(.5f, .9f), head(.7f, .9f), head(.9f, .9f)};
    assert(process(5, heads) == 0);
    assert(process(5, heads) == 0);
    assert(process(5, heads) == -5);
    assert(output.empty());
    tracker.reset();
    assert(process(0, {}) == 0);
    std::cout << "PASS: real ByteTrack + JNI result policy: SPRT confirmation, size gating, "
                 "device false-positive replays, errors, recovery, overflow, reset\n";
}

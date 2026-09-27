#include "detection_result.h"
#include "debug_snapshot.h"
#include "camera_motion.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
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

// Distant head (10% x 20%) at an arbitrary position.
static HeadBox head_at(float x, float y, float score) {
    return {x - .05f, y - .1f, x + .05f, y + .1f, x, y, .1f, .2f, score};
}

// Smooth, non-periodic 192x192 RGBA texture (two octaves of bilinear value
// noise), sampled with the content shifted by (sx, sy) pixels.
static std::vector<uint8_t> texture(float sx, float sy, uint32_t seed = 7) {
    constexpr int kSize = 192;
    auto lattice = [&](int n) {
        std::vector<float> v(static_cast<size_t>(n) * n);
        uint32_t s = seed * 2654435761u + static_cast<uint32_t>(n);
        for (auto& x : v) { s = s * 1664525u + 1013904223u; x = static_cast<float>(s >> 24); }
        return v;
    };
    auto sample = [](const std::vector<float>& v, int n, float u, float w) {
        u = std::clamp(u, 0.f, n - 1.001f); w = std::clamp(w, 0.f, n - 1.001f);
        int x0 = static_cast<int>(u), y0 = static_cast<int>(w);
        float fx = u - x0, fy = w - y0;
        auto at = [&](int x, int y) { return v[static_cast<size_t>(y) * n + x]; };
        return (1 - fy) * ((1 - fx) * at(x0, y0) + fx * at(x0 + 1, y0))
                  + fy  * ((1 - fx) * at(x0, y0 + 1) + fx * at(x0 + 1, y0 + 1));
    };
    auto coarse = lattice(16), fine = lattice(40);
    std::vector<uint8_t> rgba(kSize * kSize * 4);
    for (int y = 0; y < kSize; ++y) {
        for (int x = 0; x < kSize; ++x) {
            // Pad the lattice so shifted content never samples outside it.
            float u = (x - sx + 32.f) / (kSize + 64.f), w = (y - sy + 32.f) / (kSize + 64.f);
            float value = .7f * sample(coarse, 16, u * 15, w * 15) + .3f * sample(fine, 40, u * 39, w * 39);
            auto g = static_cast<uint8_t>(std::clamp(value, 0.f, 255.f));
            uint8_t* p = &rgba[(static_cast<size_t>(y) * kSize + x) * 4];
            p[0] = p[1] = p[2] = g; p[3] = 255;
        }
    }
    return rgba;
}

static void camera_motion_tests() {
    CameraMotionEstimator cmc;
    float dx = 1, dy = 1;
    auto base = texture(0, 0);
    assert(!cmc.estimate(base.data(), 192, 192, dx, dy) && dx == 0 && dy == 0); // no previous frame
    // A static scene reports exactly zero (no sub-pixel jitter fed to tracks).
    assert(cmc.estimate(base.data(), 192, 192, dx, dy) && dx == 0 && dy == 0);
    // Content moved right by 20 px and up by 9 px (10.4% / 4.7% of the frame).
    auto moved = texture(20, -9);
    assert(cmc.estimate(moved.data(), 192, 192, dx, dy));
    assert(std::fabs(dx - 20 / 192.f) < 1.f / 64 && std::fabs(dy + 9 / 192.f) < 1.f / 64);
    // Large pan: 45 px (~23%) is still inside the coarse search range.
    auto panned = texture(65, -9);
    assert(cmc.estimate(panned.data(), 192, 192, dx, dy));
    assert(std::fabs(dx - 45 / 192.f) < 1.f / 64 && std::fabs(dy) < 1.f / 64);
    // Scene cut (unrelated content): no motion is claimed.
    auto other = texture(0, 0, 99);
    assert(!cmc.estimate(other.data(), 192, 192, dx, dy) && dx == 0 && dy == 0);
    // A flat scene (wall, covered lens) cannot localise a shift.
    std::vector<uint8_t> flat(192 * 192 * 4, 120);
    cmc.reset();
    cmc.estimate(flat.data(), 192, 192, dx, dy);
    assert(!cmc.estimate(flat.data(), 192, 192, dx, dy) && dx == 0 && dy == 0);
    // Invalid input fails safe.
    assert(!cmc.estimate(nullptr, 192, 192, dx, dy) && dx == 0 && dy == 0);
}

static void cadence_and_motion_tests() {
    ByteTracker tracker;
    std::vector<STrack> output;
    auto run = [&](const std::vector<HeadBox>& heads, float dt_ms, float cdx = 0, float cdy = 0) {
        return update_detection_result(static_cast<int>(heads.size()), heads, tracker,
                .25f, .12f, .70f, 4, output, dt_ms, cdx, cdy);
    };

    // --- Fail-closed for stage-2 support (review of the frame-scale diff) ---
    // An established head (>= kEstablishedStrongAge reference frames at >= high)
    // turned to profile, scoring .22 (between band centre .185 and high .25) for
    // 40 frames (~5 s), stays blurred.
    for (int i = 0; i < 10; ++i) run({selfie(.5f, .55f)}, 130);
    assert(output.size() == 1);
    int id = output.front().track_id();
    for (int i = 0; i < 40; ++i) {
        assert(run({selfie(.5f, .22f)}, 130) == 1);
        assert(output.front().track_id() == id);
    }
    tracker.reset();
    // A young track sustained only by the same .22 drains out (dumbbell trace).
    run({selfie(.5f, .55f)}, 130);
    assert(run({selfie(.5f, .55f)}, 130) == 1);
    bool drained = false;
    for (int i = 0; i < 20 && !drained; ++i) drained = run({selfie(.5f, .22f)}, 130) == 0;
    assert(drained);
    tracker.reset();

    // --- High detector fps (24 fps, 42 ms): evidence is per wall time ---
    // A confident selfie head no longer confirms on the 2nd near-duplicate frame
    // (~84 ms), but within ~2 reference frames of wall time.
    int confirmed_at = -1;
    for (int i = 0; i < 12 && confirmed_at < 0; ++i) {
        if (run({selfie(.5f, .55f)}, 42) == 1) confirmed_at = i;
    }
    assert(confirmed_at >= 3 && confirmed_at <= 8);
    tracker.reset();
    // A real distant head at 24 fps with a single missed frame every 4th frame
    // still confirms (the gap is tolerated without evidence cost).
    confirmed_at = -1;
    for (int i = 0; i < 40 && confirmed_at < 0; ++i) {
        std::vector<HeadBox> h;
        if (i % 4 != 3) h.push_back(head(.5f, .90f));
        if (run(h, 42) == 1) confirmed_at = i;
    }
    assert(confirmed_at > 0 && confirmed_at * 42 <= 1000);
    tracker.reset();
    // The 16:28 backpack trace (.71, .45, miss, 6 x .42 at ~130 ms) replayed at
    // 24 fps over the same wall time: each recorded frame becomes ~3 near-duplicates.
    // Silent, as at the tuned cadence.
    auto day42 = [&](const std::vector<HeadBox>& h) {
        return update_detection_result(static_cast<int>(h.size()), h, tracker,
                .35f, .19f, .70f, 4, output, 42);
    };
    for (int i = 0; i < 3; ++i) assert(day42({backpack(.71f)}) == 0);
    for (int i = 0; i < 3; ++i) assert(day42({backpack(.45f)}) == 0);
    for (int i = 0; i < 3; ++i) assert(day42({}) == 0);
    for (int i = 0; i < 18; ++i) assert(day42({backpack(.42f)}) == 0);
    tracker.reset();
    // 15:46 empty-room trace, same treatment.
    auto low42 = [&](const std::vector<HeadBox>& h) {
        return update_detection_result(static_cast<int>(h.size()), h, tracker,
                .25f, .12f, .70f, 4, output, 42);
    };
    const std::vector<std::vector<HeadBox>> empty_room = {
            {head(.3f, .48f)}, {}, {head(.1f, .50f)}, {head(.1f, .61f)}, {},
            {head(.7f, .41f)}, {head(.7f, .54f)}, {head(.7f, .15f)}, {head(.7f, .52f)}, {}, {}};
    for (const auto& frame : empty_room) {
        for (int i = 0; i < 3; ++i) assert(low42(frame) == 0);
    }
    tracker.reset();

    // --- Slow device (3 fps, 330 ms): a walking distant head ---
    // It moves 6% of the frame per detector frame, so consecutive 10%-wide boxes
    // overlap by IoU .25 only. It must still confirm and keep one identity.
    int first_published = -1;
    for (int i = 0; i < 12; ++i) {
        int n = run({head_at(.15f + .06f * i, .5f, .80f)}, 330);
        if (n == 1 && first_published < 0) { first_published = i; id = output.front().track_id(); }
        if (first_published >= 0) {
            assert(n == 1);
            assert(output.front().track_id() == id);
        }
    }
    assert(first_published >= 0 && first_published <= 3);
    // Two missed frames while walking: re-identified at the extrapolated position.
    run({}, 330);
    run({}, 330);
    assert(run({head_at(.15f + .06f * 14, .5f, .80f)}, 330) == 1);
    assert(output.front().track_id() == id);
    tracker.reset();

    // --- Irregular cadence (throttling / GC): 130 ms and 520 ms intervals ---
    // A head crossing the frame at 0.3 frame-widths per second moves 3.9% or
    // 15.6% (more than its own width) between detector frames. The prediction
    // must follow wall time, keeping the head Tracked (not lost, not re-seeded)
    // on every frame.
    {
        float x = .05f;
        for (int i = 0; i < 6; ++i) { // established while walking at the tuned cadence
            x += .3f * .13f;
            run({head_at(x, .5f, .80f)}, 130);
        }
        assert(output.size() == 1);
        id = output.front().track_id();
        for (int i = 0; i < 6; ++i) {
            float dt = (i % 2 == 0) ? 520.f : 130.f;
            x += .3f * dt / 1000.f;
            assert(run({head_at(x, .5f, .80f)}, dt) == 1);
            assert(output.front().track_id() == id);
            assert(output.front().state() == TrackState::Tracked);
        }
        tracker.reset();
    }

    // --- Camera pan (7-8 fps): a static head sweeps 15% per frame in the image ---
    // With the measured camera shift the prediction lands on it and the track
    // stays Tracked with one identity.
    for (int i = 0; i < 8; ++i) run({head_at(.2f, .5f, .80f)}, 130);
    assert(output.size() == 1);
    id = output.front().track_id();
    for (int i = 1; i <= 4; ++i) {
        assert(run({head_at(.2f + .15f * i, .5f, .80f)}, 130, .15f, 0.f) == 1);
        assert(output.front().track_id() == id);
        assert(output.front().state() == TrackState::Tracked);
    }
    tracker.reset();

    // Kinematic gating never merges two far-apart heads into one track.
    for (int i = 0; i < 6; ++i) run({head_at(.2f, .5f, .8f), head_at(.8f, .5f, .8f)}, 330);
    assert(output.size() == 2 && output[0].track_id() != output[1].track_id());
    tracker.reset();
}

int main() {
    camera_motion_tests();
    cadence_and_motion_tests();
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
    // One more confident frame: since stage-2 scores count against high (not the
    // band centre), the earlier .15 observation cost this track more log-odds.
    assert(process(1, {selfie(.52f, .55f)}) == 1);
    assert(output.front().frames_tracked() == 7);

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

    // max_time_lost is counted in reference frames (one per ~130 ms detector frame, not 30 fps):
    // an established head is re-identified up to ~1 s of misses, and after that a
    // re-detection is a fresh candidate that must earn confirmation again. With
    // the upstream 30 it was re-identified and published straight away.
    static_assert(ByteTracker::kMaxTimeLostFrames == 8, "test written for ~1 s at 7-8 fps");
    for (int misses : {ByteTracker::kMaxTimeLostFrames, ByteTracker::kMaxTimeLostFrames + 1}) {
        for (int i = 0; i < 8; ++i) process(1, {selfie(.5f, .55f)});
        assert(process(1, {selfie(.5f, .55f)}) == 1);
        int established = output.front().track_id();
        for (int i = 0; i < misses; ++i) process(0, {});
        if (misses <= ByteTracker::kMaxTimeLostFrames) {
            assert(process(1, {selfie(.5f, .55f)}) == 1);
            assert(output.front().track_id() == established);
        } else {
            assert(process(1, {selfie(.5f, .55f)}) == 0);
            assert(process(1, {selfie(.5f, .55f)}) == 1);
            assert(output.front().track_id() != established);
        }
        tracker.reset();
    }

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

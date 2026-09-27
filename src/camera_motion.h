#ifndef CAMERA_MOTION_H
#define CAMERA_MOTION_H

#include <cstdint>
#include <vector>

// Global image translation between consecutive detector frames (camera motion
// compensation for the tracker, as in BoT-SORT). Runs on the same small RGBA
// frame the detector gets, so no extra readback or JNI data is needed.
//
// Coarse-to-fine SAD block matching on a luma pyramid: 32x32 with +-8 px
// (+-25% of the frame per detector frame) then 64x64 refined by +-2 px with a
// sub-pixel parabola fit. ~0.3M abs-diffs per frame.
//
// Fail-safe: when the scene is too flat, the minimum is not distinct, or the
// motion is out of range, estimate() reports no motion (shift 0), which is
// exactly the behaviour without compensation.
class CameraMotionEstimator {
public:
    static constexpr int kFine = 64;
    static constexpr int kCoarse = 32;
    static constexpr int kCoarseRadius = 8;
    static constexpr int kFineRadius = 2;

    // Forget the previous frame (tracker reset, camera switch, error).
    void reset();

    // Shift of the image content since the previous call, in normalised [0,1]
    // frame coordinates (content that was at x is now at x + dx). Returns false
    // (dx = dy = 0) for the first frame after reset or an unreliable estimate.
    bool estimate(const uint8_t* rgba, int width, int height, float& dx, float& dy);

private:
    std::vector<uint8_t> prev_fine_, prev_coarse_;
    std::vector<uint8_t> cur_fine_, cur_coarse_;
    bool has_prev_ = false;
};

#endif // CAMERA_MOTION_H

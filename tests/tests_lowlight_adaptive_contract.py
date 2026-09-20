#!/usr/bin/env python3
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEAD_CPP = (ROOT / "src/head_detector.cpp").read_text(encoding="utf-8")
BYTE_CPP = (ROOT / "src/bytetrack.cpp").read_text(encoding="utf-8")
MAIN_CPP = (ROOT / "src/main.cpp").read_text(encoding="utf-8")


class LowLightAdaptiveContract(unittest.TestCase):
    def test_head_detector_has_fast_subsampled_luminance(self):
        self.assertIn("step_x = std::max(1, width / 32)", HEAD_CPP)
        self.assertIn("step_y = std::max(1, height / 32)", HEAD_CPP)
        self.assertIn("p[0] * 77 + p[1] * 150 + p[2] * 29", HEAD_CPP)

    def test_head_detector_has_adaptive_gamma_lut(self):
        self.assertIn("mean_lum < 65.0f", HEAD_CPP)
        self.assertIn("1.0f - t * 0.28f", HEAD_CPP)
        self.assertIn("lut[256]", HEAD_CPP)
        self.assertIn("std::pow(i * inv255, gamma) * 255.0f", HEAD_CPP)

    def test_head_detector_has_small_clutter_discrimination(self):
        self.assertIn("bw < 0.11f || bh < 0.13f", HEAD_CPP)
        self.assertIn("prob < 0.38f", HEAD_CPP)

    def test_head_detector_uses_anchor_free_yolov8_decoder(self):
        self.assertIn('ex.extract("out0", out)', HEAD_CPP)
        self.assertIn("out.row(4)", HEAD_CPP)

    def test_bytetrack_suppresses_transient_ghost_coasting(self):
        self.assertIn("frames_tracked() >= 4", BYTE_CPP)
        self.assertIn("max_time_lost_ : 4", BYTE_CPP)

    def test_main_calibrates_instant_and_low_thresholds(self):
        self.assertIn("low_thresh = std::max(0.12f, high_thresh * 0.55f)", MAIN_CPP)
        self.assertIn("instant_thresh = std::max(0.35f, std::min(0.60f, high_thresh + 0.05f))", MAIN_CPP)


if __name__ == "__main__":
    unittest.main()

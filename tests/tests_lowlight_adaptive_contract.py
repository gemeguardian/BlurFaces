#!/usr/bin/env python3
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
HEAD_CPP = (ROOT / "src/head_detector.cpp").read_text(encoding="utf-8")
BYTE_CPP = (ROOT / "src/bytetrack.cpp").read_text(encoding="utf-8")
BYTE_H = (ROOT / "src/bytetrack.h").read_text(encoding="utf-8")
MAIN_CPP = (ROOT / "src/main.cpp").read_text(encoding="utf-8")
MAIN_JAVA = (ROOT / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text(encoding="utf-8")


class LowLightAdaptiveContract(unittest.TestCase):
    def test_head_detector_has_fast_subsampled_luminance(self):
        self.assertIn("step_x = std::max(1, width / 32)", HEAD_CPP)
        self.assertIn("step_y = std::max(1, height / 32)", HEAD_CPP)
        self.assertIn("p[0] * 77 + p[1] * 150 + p[2] * 29", HEAD_CPP)

    def test_head_detector_has_adaptive_gamma_lut(self):
        self.assertIn("mean_lum < 65.0f", HEAD_CPP)
        self.assertIn("enhance_lowlight", HEAD_CPP)
        self.assertIn("kTilesX", HEAD_CPP)
        self.assertIn("kTilesY", HEAD_CPP)
        self.assertIn("clip", HEAD_CPP)
        self.assertIn("ema_lut_", HEAD_CPP)

    def test_head_detector_has_small_clutter_discrimination(self):
        self.assertIn("bw < 0.11f || bh < 0.13f", HEAD_CPP)
        self.assertIn("prob < 0.38f", HEAD_CPP)

    def test_head_detector_uses_anchor_free_yolov8_decoder(self):
        self.assertIn('ex.extract("out0", out)', HEAD_CPP)
        self.assertIn("out.row(4)", HEAD_CPP)

    def test_head_detector_has_large_clutter_discrimination(self):
        self.assertIn("bw * bh > 0.35f && prob < 0.60f", HEAD_CPP)

    def test_head_detector_caps_lowlight_gain(self):
        self.assertIn("(Y[i] + 1.0f), 1.0f, 2.5f)", HEAD_CPP)

    def test_bytetrack_suppresses_transient_ghost_coasting(self):
        self.assertIn("static constexpr int kMinHits = 3;", BYTE_H)
        self.assertIn("static constexpr int kMinHitsForCoast = 6;", BYTE_H)
        self.assertIn("static constexpr int kMaxCoastPublishFrames = 4;", BYTE_H)
        self.assertIn("frames_tracked() >= STrack::kMinHits", BYTE_CPP)
        self.assertIn("if (trk.publishable())", BYTE_CPP)
        self.assertNotIn("frames_lost() <= 12", BYTE_CPP)

    def test_main_calibrates_instant_and_low_thresholds(self):
        self.assertIn("low_thresh = std::max(0.12f, high_thresh * 0.55f)", MAIN_CPP)
        self.assertIn("instant_thresh = std::max(0.70f, std::min(0.85f, high_thresh + 0.35f))", MAIN_CPP)

    def test_main_never_relaxes_instant_threshold_in_low_light(self):
        low_light = MAIN_CPP[MAIN_CPP.index("if (scene_lum < 45.0f)"):]
        low_light = low_light[:low_light.index("}")]
        self.assertNotIn("instant_thresh", low_light)

    def test_java_resets_native_tracker_on_every_source_activation(self):
        activate = MAIN_JAVA[MAIN_JAVA.index("static void activateSource("):]
        activate = activate[:activate.index("emit(\"Camera source changed")]
        self.assertIn("noteCameraSwitch(now);", activate)
        self.assertNotIn("if (prevSource != null) {\n            noteCameraSwitch(now);", activate)


if __name__ == "__main__":
    unittest.main()

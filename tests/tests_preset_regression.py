#!/usr/bin/env python3
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "BlurFaces/main.py").read_text(encoding="utf-8")


class PresetRegressionTest(unittest.TestCase):
    def test_settings_structure_is_preset_free(self):
        self.assertNotIn("preset_rows", SOURCE)
        self.assertNotIn("model_rows", SOURCE)
        self.assertNotIn("PRESET_PROFILES", SOURCE)
        self.assertIn("ROUND_VIDEO_WIDTHS", SOURCE)
        self.assertIn("FACE_MASK_SCALE = 100", SOURCE)
        self.assertNotIn("FACE_MASK_SCALES", SOURCE)
        self.assertNotIn("self.face_mask_index", SOURCE)
        self.assertIn("DETECTION_CONFIDENCES", SOURCE)
        self.assertIn("MASK_MODES", SOURCE)


if __name__ == "__main__":
    unittest.main()

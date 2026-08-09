import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent
MAIN = (ROOT / "src/main/java/com/makey/blurfaces/Main.java").read_text()


def method_body(name: str) -> str:
    match = re.search(rf"private static void {name}\([^)]*\) \{{", MAIN)
    if not match:
        raise AssertionError(f"method {name} not found")
    start = match.end()
    depth = 1
    i = start
    while depth and i < len(MAIN):
        depth += (MAIN[i] == "{") - (MAIN[i] == "}")
        i += 1
    return MAIN[start : i - 1]


class PreviewRegressionContract(unittest.TestCase):
    def test_pending_detection_does_not_hide_published_face(self):
        """1.8.2 introduced a pending-source gate that kept preview at faceCount=0.

        Worker publication is source-bound already, so a positive result from the
        active source must be renderable immediately even if the detector cadence
        gate has not yet been cleared by its owning worker iteration.
        """
        body = method_body("prepareNextPreviewDraw")
        self.assertNotRegex(
            body,
            r"if\s*\(awaitingSwitchScan\)\s*\{[^}]*s\.faceCount\s*=\s*0",
        )

    def test_pending_detection_still_forces_full_detector_pass(self):
        after = method_body("afterDraw")
        self.assertIn("source.equals(PENDING_CAMERA_SOURCE.get())", after)
        self.assertRegex(after, r"switchScan\s*\|\|")

    def test_encoder_path_remains_independent(self):
        encoder = method_body("beforeEncoderDraw")
        self.assertNotRegex(
            encoder,
            r"if\s*\(awaitingSwitchScan\)\s*\{[^}]*e\.faceCount\s*=\s*0",
        )


if __name__ == "__main__":
    unittest.main()

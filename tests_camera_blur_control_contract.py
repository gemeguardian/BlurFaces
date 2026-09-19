import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
MAIN = (ROOT / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text(encoding="utf-8")


class CameraBlurControlContract(unittest.TestCase):
    def test_control_uses_separate_pill_below_zoom_and_icons_only(self):
        self.assertIn('getMethod("getZoomSlider")', MAIN)
        self.assertIn("R.drawable.msg_photo_blur", MAIN)
        self.assertIn("R.drawable.msg_blur_off", MAIN)
        control = MAIN[MAIN.index("private static final class BlurControl") :]
        self.assertNotIn("TextView", control)
        self.assertIn("AndroidUtilities.dp(96), AndroidUtilities.dp(48)", control)
        self.assertGreaterEqual(control.count("AndroidUtilities.dp(48)"), 4)
        self.assertIn("AndroidUtilities.dp(44), AndroidUtilities.dp(44)", control)
        self.assertIn("zoomSlider.getY() + zoomSlider.getHeight() - AndroidUtilities.dp(8)", control)
        self.assertIn("Gravity.TOP | Gravity.CENTER_HORIZONTAL", control)
        self.assertIn("ViewTreeObserver.OnPreDrawListener", control)
        self.assertIn("pill.setAlpha(zoomSlider.getAlpha())", control)
        self.assertNotIn("pill.setVisibility(zoomSlider.getVisibility())", control)

    def test_control_follows_theme_and_has_tactile_selected_state(self):
        self.assertIn("Theme.key_chat_messagePanelBackground", MAIN)
        self.assertIn("Theme.key_featuredStickers_addButton", MAIN)
        self.assertIn("Theme.key_chats_actionIcon", MAIN)
        self.assertIn("Theme.key_chat_messagePanelText", MAIN)
        self.assertIn("HapticFeedbackConstants.KEYBOARD_TAP", MAIN)
        self.assertIn("view.setSelected(selected)", MAIN)
        self.assertIn("selector.animate().translationX(target).setDuration(180L)", MAIN)

    def test_toggle_gates_capture_preview_and_encoder(self):
        self.assertIn("private static volatile boolean blurEnabled = true", MAIN)
        self.assertIn("private static volatile int maskMode", MAIN)
        self.assertIn("getDiagnostics()", MAIN)
        self.assertIn("uMaskMode", MAIN)
        self.assertIn("if (!acceptingFrames || !blurEnabled) return;", MAIN)
        self.assertIn("if (!acceptingFrames || !blurEnabled || !state.ready", MAIN)
        self.assertIn("if (!acceptingFrames || !blurEnabled", MAIN)
        self.assertIn("SOURCE_TRACKS.clear()", MAIN)
        self.assertIn("SOURCE_ACTIVE_SINCE.clear()", MAIN)

    def test_reenable_reactivates_current_camera_source(self):
        toggle = MAIN[MAIN.index("private static void setBlurEnabled") :
                      MAIN.index("private static void refreshBlurControls")]
        self.assertIn("for (CameraState state : CAMERA_STATES.values())", toggle)
        self.assertIn("state.activeSource = null", toggle)
        self.assertIn("SOURCE_ACTIVE_SINCE.clear()", toggle)

    def test_new_camera_defaults_to_privacy_on(self):
        install = MAIN[MAIN.index("private static void installBlurControl") :
                       MAIN.index("private static void setBlurEnabled")]
        self.assertIn("setBlurEnabled(true)", install)

    def test_existing_camera_gets_control_when_opened(self):
        hook = MAIN[MAIN.index("private static void hookCameraControls") :
                    MAIN.index("private static void installBlurControl")]
        self.assertIn('getDeclaredMethod("startAnimation", boolean.class, boolean.class)', hook)
        self.assertIn("Boolean.TRUE.equals(param.args[0])", hook)
        install = MAIN[MAIN.index("private static void installBlurControl") :
                       MAIN.index("private static void setBlurEnabled")]
        self.assertIn("BLUR_CONTROLS.get(cameraView)", install)
        self.assertIn("existing.attach()", install)

    def test_degraded_and_failed_protection_state_indication(self):
        self.assertIn("private static void setProtectionState(String newState)", MAIN)
        control = MAIN[MAIN.index("private static final class BlurControl") :]
        self.assertIn('"DEGRADED".equals(protectionState)', control)
        self.assertIn('"FAILED".equals(protectionState)', control)
        self.assertIn("0xFFFFA000", control)
        self.assertIn("0xFFE53935", control)
        self.assertIn('"Face blur active (degraded)"', control)
        self.assertIn('"Blur faces"', control)


if __name__ == "__main__":
    unittest.main()

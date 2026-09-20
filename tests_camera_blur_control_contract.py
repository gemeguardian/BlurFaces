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
        self.assertIn("pill.setAlpha(alpha)", control)
        self.assertIn("cameraBottom + AndroidUtilities.dp(16)", control)
        self.assertIn("setBlurBackground", MAIN)
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

    def test_dynamic_icon_pack_detection_and_svgs(self):
        self.assertIn("SVG_TG_BLUR", MAIN)
        self.assertIn("SVG_TG_BLUR_OFF", MAIN)
        self.assertIn("SVG_SOLAR_BLUR", MAIN)
        self.assertIn("SVG_SOLAR_BLUR_OFF", MAIN)
        self.assertIn("SVG_REMIX_BLUR", MAIN)
        self.assertIn("SVG_REMIX_BLUR_OFF", MAIN)
        self.assertIn("detectActiveIconPack", MAIN)
        self.assertIn("ExteraConfig.getIconPack()", MAIN)
        self.assertIn('"SOLAR"', MAIN)
        self.assertIn('"REMIX"', MAIN)
        self.assertIn('"DEFAULT"', MAIN)
        self.assertIn("renderSvgToBitmap", MAIN)
        self.assertIn("getBlurIconDrawable", MAIN)
        control = MAIN[MAIN.index("private static final class BlurControl") :]
        self.assertIn("updateIcons()", control)

    def test_embedded_svgs_match_user_icons(self):
        icons_dir = pathlib.Path("/tmp/user_icons")
        if icons_dir.exists():
            import re
            def extract_constant(name):
                m = re.search(r'String\s+' + name + r'\s*=\s*(.*?);', MAIN, re.DOTALL)
                if not m:
                    return ''
                expr = m.group(1)
                parts = re.findall(r'"((?:[^"\\]|\\.)*)"', expr)
                return ''.join(parts).encode('utf-8').decode('unicode_escape')

            const_map = {
                'tg_blur.svg': 'SVG_TG_BLUR',
                'tg_blur_off.svg': 'SVG_TG_BLUR_OFF',
                'solar_blur.svg': 'SVG_SOLAR_BLUR',
                'solar_blur_off.svg': 'SVG_SOLAR_BLUR_OFF',
                'remix_blur.svg': 'SVG_REMIX_BLUR',
                'remix_blur_off.svg': 'SVG_REMIX_BLUR_OFF'
            }
            for file_name, const_name in const_map.items():
                raw = (icons_dir / file_name).read_text().strip()
                extracted = extract_constant(const_name).strip()
                self.assertEqual(raw, extracted, f"{file_name} does not match {const_name}")


if __name__ == "__main__":
    unittest.main()

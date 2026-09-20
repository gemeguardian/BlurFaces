import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
PLUGIN = (ROOT / "BlurFaces/main.py").read_text()
RUNTIME = (ROOT / "BlurFaces/runtime.py").read_text()
JAVA = (ROOT / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text()


class ReconfigureReloadContract(unittest.TestCase):
    def test_soft_reconfigure_does_not_restart_or_rehook(self):
        self.assertNotIn("def _restart_runtime", PLUGIN)
        self.assertIn("def _request_reconfigure", PLUGIN)
        self.assertNotIn("_stop_runtime", PLUGIN[PLUGIN.index("def _request_reconfigure"):])
        self.assertNotIn("_prepare_runtime", PLUGIN[PLUGIN.index("def _request_reconfigure"):])
        self.assertIn("runtime.switch_confidence", PLUGIN)
        self.assertIn("runtime.reserve_reconfigure_request()", PLUGIN)
        self.assertIn("threading.Thread(", PLUGIN)
        self.assertEqual(1, JAVA.count("hookSurfaceUpdates();"))
        self.assertIn("public static boolean reconfigure", JAVA)

    def test_switch_is_generation_guarded_and_recovers(self):
        self.assertIn("public static boolean reconfigure", JAVA)
        self.assertIn("NativeBridge.reset();", JAVA)
        self.assertIn("if success:", PLUGIN)
        self.assertIn('self.set_setting("detection_range", range_index)', PLUGIN)

    def test_hot_reload_registry_reuses_compatible_runtime(self):
        self.assertNotIn("_LEGACY_REGISTRY_KEYS", RUNTIME)
        self.assertNotIn("_blur_faces_mediapipe_runtime", RUNTIME)
        self.assertIn("_cancel_deferred_shutdown(registry)", RUNTIME)
        self.assertIn("The process-global Java runtime and hooks survive", RUNTIME)
        self.assertIn('registry.get("core_bundle_id") == CORE_BUNDLE_ID', RUNTIME)
        self.assertIn("Blur Faces core DEX changed while loaded", RUNTIME)


if __name__ == "__main__":
    unittest.main()

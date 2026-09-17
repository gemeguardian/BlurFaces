import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
PLUGIN = (ROOT / "BlurFaces/main.py").read_text()
RUNTIME = (ROOT / "BlurFaces/runtime.py").read_text()
JAVA = (ROOT / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text()
BRIDGE = (ROOT / "src/main/java/com/makey/blurfaces/g2/ModelSettingsBridge.java").read_text()


class ReconfigureReloadContract(unittest.TestCase):
    def test_soft_reconfigure_does_not_restart_or_rehook(self):
        self.assertNotIn("def _restart_runtime", PLUGIN)
        self.assertIn("def _request_reconfigure", PLUGIN)
        block = PLUGIN[PLUGIN.index("def _request_reconfigure"):PLUGIN.index("def _delete_model_file")]
        self.assertNotIn("_stop_runtime", block)
        self.assertNotIn("_prepare_runtime", block)
        self.assertIn("runtime.switch_model", block)
        self.assertIn("runtime.reserve_reconfigure_request()", block)
        self.assertIn("threading.Thread(", block)
        self.assertEqual(1, JAVA.count("hookSurfaceUpdates();"))
        self.assertIn("public static boolean reconfigure", JAVA)

    def test_switch_is_generation_guarded_and_recovers(self):
        self.assertIn("RECONFIGURE_GENERATION", JAVA)
        self.assertIn("createEngineCandidate", JAVA)
        self.assertIn("requestGeneration != RECONFIGURE_GENERATION.get()", JAVA)
        self.assertIn("if (candidate == null) return false;", JAVA)
        self.assertIn("candidate.close();", JAVA)
        self.assertIn("if (initialized) acceptingFrames = true;", JAVA)
        self.assertIn("executor.execute(() -> closeEngines(oldLandmarker, oldDetector))", JAVA)
        self.assertIn("if success:", PLUGIN)
        self.assertIn('self.set_setting("model", model_index)', PLUGIN)

    def test_hot_reload_registry_reuses_compatible_runtime(self):
        self.assertIn("_LEGACY_REGISTRY_KEYS", RUNTIME)
        self.assertIn("delattr(sys, key)", RUNTIME)
        self.assertIn("_cancel_deferred_shutdown(registry)", RUNTIME)
        self.assertIn("The process-global Java runtime and hooks survive", RUNTIME)
        self.assertIn('registry.get("core_bundle_id") == CORE_BUNDLE_ID', RUNTIME)
        self.assertIn("changed plugin Java", RUNTIME)
        self.assertIn("MediaPipe DEX/JNI bundle changed while loaded", RUNTIME)

    def test_settings_cross_chaquopy_as_host_uitems(self):
        self.assertIn("Custom(item=model_settings_item", PLUGIN)
        self.assertIn("Custom(item=preset_settings_item", PLUGIN)
        self.assertNotIn("Custom(factory=model_settings_factory", PLUGIN)
        self.assertNotIn("Custom(factory=preset_settings_factory", PLUGIN)
        self.assertIn('getMethod("getItem", Integer.TYPE)', RUNTIME)
        self.assertIn('getMethod("getPresetItem", Integer.TYPE)', RUNTIME)
        self.assertIn("public static UItem getItem", BRIDGE)
        self.assertIn("public static UItem getPresetItem", BRIDGE)

    def test_live_settings_refresh_and_download_cancellation(self):
        self.assertIn("def _reload_settings_structure", PLUGIN)
        self.assertNotIn("if not self._is_model_ready(self.model_index)", PLUGIN)
        self.assertIn("show_sensitivity = self._settings_model_index == 0", PLUGIN)
        self.assertIn("self._reload_settings_structure(model_index)", PLUGIN)
        self.assertIn("self._reload_settings_structure(previous_value)", PLUGIN)
        self.assertIn("Integer(jint(self._settings_model_index))", PLUGIN)
        failure = PLUGIN[PLUGIN.index("else:\n                if setting_key == \"model\""):]
        self.assertIn("self._sync_model_settings_bridge()", failure)
        self.assertIn('"model_settings_layout"', PLUGIN)
        self.assertIn("if self._settings_show_sensitivity:", PLUGIN)
        self.assertIn("if not self._settings_has_models:", PLUGIN)
        self.assertIn("has_models = self._get_downloaded_mask() != 0", PLUGIN)
        self.assertNotIn('key="enabled"', PLUGIN)
        self.assertIn("self._sync_model_settings_bridge()", PLUGIN)
        self.assertIn('self.set_setting("model", index)', PLUGIN)
        self.assertIn("def _schedule_bridge_refresh", PLUGIN)
        self.assertIn("REFRESH_POSTED.compareAndSet(false, true)", BRIDGE)
        self.assertIn("applySnapshot(animated)", BRIDGE)
        self.assertIn("radioButton.setChecked(downloaded && state.selectedModel == boundIndex", BRIDGE)
        self.assertIn("synchronized (DOWNLOAD_LOCK)", BRIDGE)
        self.assertIn("download_generation != self._download_generation", PLUGIN)
        self.assertIn("model download cancelled", RUNTIME)


if __name__ == "__main__":
    unittest.main()

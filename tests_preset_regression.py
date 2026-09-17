import ast
import inspect
import pathlib
import threading
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = (ROOT / "BlurFaces/main.py").read_text(encoding="utf-8")


class BasePlugin:
    def __init__(self):
        pass


def load_plugin_class():
    tree = ast.parse(SOURCE)
    body = [
        node for node in tree.body
        if isinstance(node, (ast.Assign, ast.ClassDef))
    ]
    namespace = {
        "BasePlugin": BasePlugin,
        "threading": threading,
        "DETECTION_CONFIDENCES": (60, 50, 40),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "BlurFaces/main.py", "exec"), namespace)
    return namespace["BlurFacesPlugin"]


BlurFacesPlugin = load_plugin_class()


class ImmediateThread:
    def __init__(self, target, **_kwargs):
        self.target = target

    def start(self):
        self.target()


class Runtime:
    def __init__(self, success=False):
        self.success = success

    def reserve_reconfigure_request(self):
        return 7

    def switch_model(self, *_args):
        return self.success


class PresetRegressionTest(unittest.TestCase):
    def test_localized_preset_labels_expose_actual_profiles(self):
        catalogs = []
        for locale in ("en", "ru"):
            path = ROOT / f"BlurFaces/strings/strings_{locale}.yml"
            catalogs.append({
                key: ast.literal_eval(value.strip())
                for key, value in (
                    line.split(":", 1)
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
            })
        self.assertEqual(set(catalogs[0]), set(catalogs[1]))
        for catalog in catalogs:
            self.assertIn("GPU", catalog["preset_flagship"])
            self.assertIn("GPU", catalog["preset_balanced"])
            self.assertIn("384", catalog["preset_balanced"])
            self.assertIn("CPU", catalog["preset_eco"])
            self.assertIn("320", catalog["preset_eco"])

    def test_profiles_match_the_parameters_shown_in_preset_labels(self):
        self.assertEqual({
            0: (0, 0, 0, 0),
            1: (1, 0, 0, 2),
            2: (1, 1, 0, 1),
        }, BlurFacesPlugin.PRESET_PROFILES)

    def test_preset_bridge_can_refresh_without_prebuilt_labels(self):
        parameter = inspect.signature(BlurFacesPlugin._sync_preset_bridge).parameters["labels"]
        self.assertIsNone(parameter.default)

    def make_plugin(self):
        plugin = BlurFacesPlugin.__new__(BlurFacesPlugin)
        plugin.enabled = True
        plugin._loaded = True
        plugin._generation = 4
        plugin._config_generation = 0
        plugin.model_index = 0
        plugin.detection_range_index = 0
        plugin.processor_index = 0
        plugin.preset_index = 2
        plugin._applied_preset_index = 1
        plugin.dex_loader = Runtime()
        plugin.log = lambda _message: None
        plugin.settings = []
        plugin.set_setting = lambda *args, **kwargs: plugin.settings.append((args, kwargs))
        plugin._reload_settings_structure = lambda *_args: None
        plugin._schedule_bridge_refresh = lambda: None
        plugin._sync_model_settings_bridge = lambda: None
        plugin.preset_syncs = 0
        plugin._sync_preset_bridge = lambda: setattr(
            plugin, "preset_syncs", plugin.preset_syncs + 1
        )
        return plugin

    def test_failed_preset_reconfigure_rolls_back_persistent_and_bridge_state(self):
        plugin = self.make_plugin()
        original_thread = threading.Thread
        threading.Thread = ImmediateThread
        try:
            plugin._request_reconfigure(1, 0, 1, "preset", 1)
        finally:
            threading.Thread = original_thread

        self.assertEqual(1, plugin.preset_index)
        self.assertEqual(1, plugin._applied_preset_index)
        self.assertEqual(1, plugin.preset_syncs)
        self.assertIn((("preset", 1), {"reload_settings": True}), plugin.settings)

    def test_stale_preset_download_does_not_apply_after_manual_selection(self):
        plugin = self.make_plugin()
        plugin._download_generation = 3
        plugin._downloads_lock = threading.Lock()
        plugin._downloading_models = {1}
        plugin._downloaded_mask = 0
        plugin._preset_downloading = 2
        plugin._preset_downloading_model = 1
        plugin.preset_index = 3
        plugin._requested_model_index = 0
        plugin.log = lambda _message: None
        plugin._sync_model_settings_bridge = lambda: None
        plugin._reload_settings_structure = lambda: None
        plugin._schedule_bridge_refresh = lambda: None
        plugin._activate_model = lambda _index: self.fail("stale model activated")
        plugin._apply_preset = lambda *_args: self.fail("stale preset applied")

        method_globals = plugin._download_and_activate.__globals__
        originals = method_globals.get("download_model"), method_globals.get("is_model_downloaded")
        method_globals["download_model"] = lambda *_args, **_kwargs: object()
        method_globals["is_model_downloaded"] = lambda _index: False
        try:
            plugin._download_and_activate(1, 3)
        finally:
            method_globals["download_model"], method_globals["is_model_downloaded"] = originals

        self.assertIsNone(plugin._preset_downloading)
        self.assertIsNone(plugin._preset_downloading_model)
        self.assertEqual(2, plugin._downloaded_mask)

    def test_old_download_preserves_new_pending_preset(self):
        plugin = self.make_plugin()
        plugin._download_generation = 3
        plugin._downloads_lock = threading.Lock()
        plugin._downloading_models = {0, 1}
        plugin._downloaded_mask = 0
        plugin._preset_downloading = 0
        plugin._preset_downloading_model = 0
        plugin.preset_index = 0
        plugin._requested_model_index = 0
        plugin._sync_model_settings_bridge = lambda: None
        plugin._reload_settings_structure = lambda: None
        plugin._activate_model = lambda _index: None
        plugin._apply_preset = lambda *_args: self.fail("wrong model download applied preset")

        method_globals = plugin._download_and_activate.__globals__
        originals = method_globals.get("download_model"), method_globals.get("is_model_downloaded")
        method_globals["download_model"] = lambda *_args, **_kwargs: object()
        method_globals["is_model_downloaded"] = lambda _index: False
        try:
            plugin._download_and_activate(1, 3)
        finally:
            method_globals["download_model"], method_globals["is_model_downloaded"] = originals

        self.assertEqual(0, plugin._preset_downloading)
        self.assertEqual(0, plugin._preset_downloading_model)

    def test_tapping_selected_preset_retries_its_missing_model(self):
        plugin = self.make_plugin()
        plugin.preset_index = 2
        plugin._preset_downloading = None
        plugin._preset_downloading_model = None
        plugin._requested_model_index = 0
        plugin._is_model_ready = lambda _index: False
        requested = []
        plugin._on_model_click = requested.append

        plugin._dex_on_preset_click(2)

        self.assertEqual(2, plugin._preset_downloading)
        self.assertEqual(1, plugin._preset_downloading_model)
        self.assertEqual(1, plugin._requested_model_index)
        self.assertEqual([1], requested)

    def test_preset_switch_between_non_manual_syncs_bridge(self):
        plugin = self.make_plugin()
        plugin.preset_index = 0
        plugin._applied_preset_index = 0
        plugin._is_model_ready = lambda _index: True
        plugin._apply_preset = lambda *_args: None

        plugin._dex_on_preset_click(4)  # factory index 4 = preset 1 (Balanced)

        self.assertEqual(1, plugin.preset_index)
        self.assertGreaterEqual(plugin.preset_syncs, 1)

    def test_unload_detaches_preset_before_model_bridge(self):
        plugin = self.make_plugin()
        plugin._download_generation = 0
        events = []
        plugin.log = lambda _message: None
        plugin._stop_runtime = lambda: events.append("runtime")
        plugin._detach_preset_bridge = lambda: events.append("preset")
        plugin._detach_model_settings_bridge = lambda: events.append("model")

        plugin.on_plugin_unload()

        self.assertEqual(["runtime", "preset", "model"], events)


if __name__ == "__main__":
    unittest.main()

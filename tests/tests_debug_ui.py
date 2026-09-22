"""Execute debug UI/runtime methods, replacing only ExteraGram scheduling/Java boundaries."""
import ast
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest

ROOT = Path(__file__).resolve().parent.parent


class HostPlugin:
    def __init__(self):
        self.settings = {}
        self.messages = []

    def get_setting(self, key, default=None):
        return self.settings.get(key, default)

    def set_setting(self, key, value, reload_settings=False):
        self.settings[key] = value

    def log(self, message):
        self.messages.append(message)


class DebugUiTest(unittest.TestCase):
    def setUp(self):
        self.queue, self.ui, self.timers = [], [], []
        def schedule(fn, delay=0):
            (self.timers if delay else self.ui).append(fn)
        self.namespace = {
            "BasePlugin": HostPlugin, "run_on_queue": self.queue.append,
            "run_on_ui_thread": schedule,
            "strings": SimpleNamespace(get=lambda key: key),
            **{name: lambda *args, **kw: SimpleNamespace(**kw)
               for name in ("Switch", "Text", "Header", "Divider", "Selector")},
        }
        path = ROOT / "BlurFaces/main.py"
        tree = ast.parse(path.read_text())
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, str(path), "exec"), self.namespace)
        self.plugin = self.namespace["BlurFacesPlugin"]()
        self.plugin._loaded = True
        self.plugin.show_toast = self.plugin.messages.append
        self.active = False
        self.calls = []
        def configure(value):
            self.calls.append(value)
            self.active = value
        self.runtime = SimpleNamespace(
            set_debug_capture=configure,
            is_debug_capture_enabled=lambda: self.active,
            get_debug_capture_status=lambda: "state=" + ("recording" if self.active else "off"),
            clear_debug_captures=lambda: configure(False),
            unload=lambda: configure(False),
        )
        self.plugin.dex_loader = self.runtime

    def drain(self):
        while self.queue or self.ui:
            while self.queue:
                self.queue.pop(0)()
            while self.ui:
                self.ui.pop(0)()

    def test_switch_callbacks_expiry_and_status(self):
        rows = self.plugin.create_settings()
        switch = next(row for row in rows if getattr(row, "key", None) == "debug_capture")
        self.assertFalse(switch.default)
        switch.on_change(True)
        self.assertEqual(self.calls, [])  # no disk/JNI work in UI callback
        self.drain()
        self.assertEqual(self.calls, [True])
        self.assertTrue(self.plugin.settings["debug_capture"])
        self.assertEqual(len(self.timers), 1)
        # Native session expires, even if no camera frames arrive.
        self.active = False
        self.timers.pop(0)()
        self.drain()
        self.assertFalse(self.plugin.settings["debug_capture"])
        self.assertEqual(self.plugin._debug_status, "state=off")
        self.assertEqual(self.timers, [])
        next(row for row in rows if getattr(row, "text", None) == "debug_status").on_click(None)
        self.drain()
        self.assertEqual(self.plugin.messages[-1], "state=off")

    def test_unload_cancels_queued_enable_and_poll(self):
        self.plugin._on_debug_capture_change(True)
        self.plugin.on_plugin_unload()
        self.drain()
        self.assertNotIn(True, self.calls)
        self.assertFalse(self.plugin.settings["debug_capture"])
        self.assertEqual(self.timers, [])

    def test_latest_toggle_wins_and_clear_stops_poll(self):
        self.plugin._on_debug_capture_change(True)
        self.plugin._on_debug_capture_change(False)
        self.drain()
        self.assertEqual(self.calls, [False])
        self.plugin._on_debug_capture_change(True)
        self.drain()
        self.plugin._on_debug_clear_click()
        self.drain()
        self.timers.pop(0)()
        self.drain()
        self.assertFalse(self.plugin.settings["debug_capture"])
        self.assertEqual(self.timers, [])

    def test_missing_runtime_and_io_failure_are_visible(self):
        self.plugin.dex_loader = None
        self.plugin._on_debug_capture_change(True)
        self.assertFalse(self.plugin.settings["debug_capture"])
        self.assertIn("debug_unavailable", self.plugin.messages)
        self.plugin.dex_loader = self.runtime
        def fail(_):
            raise OSError("storage full")
        self.runtime.set_debug_capture = fail
        self.plugin._on_debug_capture_change(True)
        self.drain()
        self.assertFalse(self.plugin.settings["debug_capture"])
        self.assertIn("storage full", self.plugin.messages[-1])

    def test_load_never_restores_persisted_consent(self):
        self.plugin.settings["debug_capture"] = True
        self.plugin.on_plugin_load()
        self.assertFalse(self.plugin.settings["debug_capture"])
        self.assertFalse(self.plugin._debug_enabled)
        # Runtime startup itself is covered separately; don't run Android here.


class DebugRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.registry = {"lock": threading.RLock(), "owner": None}
        def method(registry, cls, name, *types):
            return SimpleNamespace(invoke=lambda _, *args: self.calls.append((name, args)) or
                                   (True if name == "isDebugCaptureEnabled" else "state=off"))
        path = ROOT / "BlurFaces/runtime.py"
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == "DexRuntime")
        namespace = {"_runtime_registry": lambda: self.registry, "_method": method,
                     "String": str, "RUNTIME_BUNDLE_ID": "b" * 64}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        self.runtime = namespace["DexRuntime"](HostPlugin())
        self.runtime.dex_main_class = object()
        self.registry["owner"] = self.runtime.owner_token

    def test_java_api_and_bundle_identity(self):
        self.runtime.set_debug_capture(True)
        self.assertTrue(self.runtime.is_debug_capture_enabled())
        self.assertEqual(self.runtime.get_debug_capture_status(), "state=off")
        self.runtime.clear_debug_captures()
        self.assertEqual(self.calls, [
            ("setDebugCapture", ("true", "b" * 64)),
            ("isDebugCaptureEnabled", ()), ("getDebugCaptureStatus", ()),
            ("clearDebugCaptures", ()),
        ])

    def test_old_owner_cannot_enable_capture(self):
        self.registry["owner"] = object()
        with self.assertRaisesRegex(RuntimeError, "not active"):
            self.runtime.set_debug_capture(True)
        self.assertEqual(self.calls, [])

    def test_unload_stops_capture_before_core(self):
        self.runtime.unload(deferred=False)
        self.assertEqual(self.calls[:2], [
            ("setDebugCapture", ("false", "b" * 64)), ("onUnload", ()),
        ])
        self.assertIsNone(self.runtime.dex_main_class)
        self.assertIsNone(self.registry["owner"])


if __name__ == "__main__":
    unittest.main()

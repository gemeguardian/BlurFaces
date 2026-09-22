import threading

from android_utils import run_on_ui_thread
from base_plugin import BasePlugin
from client_utils import run_on_queue
from elyx import strings
from ui.settings import Divider, Header, Selector, Switch, Text

from .runtime import DexRuntime


ROUND_VIDEO_WIDTHS = (0, 320, 384, 448, 512)
FACE_MASK_SCALE = 100
DETECTION_CONFIDENCES = (35, 25, 18)
MASK_MODES = (0, 1, 2)


class BlurFacesPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.blur_by_default = True
        self.round_video_width_index = 0
        self.detection_range_index = 0
        self.mask_mode_index = 0
        self.dex_loader = None
        self._generation = 0
        self._config_generation = 0
        self._loaded = False
        self._debug_request = 0
        self._debug_enabled = False
        self._debug_status = "state=off"

    def on_plugin_load(self):
        self.log("[BlurFaces] Plugin loading...")
        self._loaded = True
        self.enabled = True
        # Persisted Switch state is display-only, never consent after a reload.
        self._debug_request += 1
        self._debug_enabled = False
        self._debug_status = "state=off"
        self.set_setting("debug_capture", False)
        self.blur_by_default = bool(self.get_setting("blur_by_default", True))
        self.round_video_width_index = 0
        self.detection_range_index = self._valid_range_index(
            self.get_setting("detection_range", 0)
        )
        self.mask_mode_index = self._valid_mask_mode_index(self.get_setting("mask_mode", 0))
        self._generation += 1
        if self.enabled:
            generation = self._generation
            run_on_queue(lambda: self._prepare_runtime(generation))

    def _is_current(self, generation):
        return self.enabled and generation == self._generation

    def _prepare_runtime(self, generation):
        runtime = None
        try:
            runtime = DexRuntime(
                plugin=self,
                round_video_width=ROUND_VIDEO_WIDTHS[self.round_video_width_index],
                face_mask_scale=FACE_MASK_SCALE,
                detection_confidence=DETECTION_CONFIDENCES[self.detection_range_index],
                blur_by_default=self.blur_by_default,
            )
            if not runtime.stage_and_start():
                return
            runtime.set_mask_mode(self.mask_mode_index)
            if not self._is_current(generation):
                runtime.unload()
                return
            previous = self.dex_loader
            self.dex_loader = runtime
            if previous:
                previous.unload()
            self.log("[BlurFaces] Native NCNN 360° HeadDetector + ByteTrack active")
        except Exception as error:
            if runtime:
                runtime.unload()
            self.log(f"[BlurFaces] Runtime preparation failed: {type(error).__name__}: {error}")

    def on_plugin_unload(self):
        self.log("[BlurFaces] Plugin unloading...")
        self._loaded = False
        self.enabled = False
        self._debug_request += 1
        self._debug_enabled = False
        self.set_setting("debug_capture", False)
        self._generation += 1
        self._stop_runtime()

    def _stop_runtime(self):
        runtime = self.dex_loader
        self.dex_loader = None
        if runtime:
            runtime.unload()

    def create_settings(self):
        self._loaded = True
        return [
            Header(strings.get("settings_header")),
            Switch(
                key="blur_by_default",
                text=strings.get("settings_blur_by_default"),
                default=self.blur_by_default,
                on_change=self._on_blur_by_default_change,
            ),
            Divider(text=strings.get("settings_blur_by_default_subtext")),
            Selector(
                key="mask_mode",
                text=strings.get("settings_mask_mode"),
                icon="msg_photo_blur",
                default=self.mask_mode_index,
                items=[
                    strings.get("mask_mode_blur"),
                    strings.get("mask_mode_pixelate"),
                    strings.get("mask_mode_solid"),
                ],
                on_change=self._on_mask_mode_change,
            ),
            Divider(text=strings.get("settings_mask_mode_subtext")),
            Selector(
                key="detection_range",
                text=strings.get("settings_range"),
                icon="msg_search",
                default=self.detection_range_index,
                items=[
                    strings.get("range_normal"),
                    strings.get("range_extended"),
                    strings.get("range_far"),
                ],
                on_change=self._on_range_change,
            ),
            Divider(text=strings.get("settings_range_subtext")),
            Header(text=strings.get("debug_header")),
            Switch(
                key="debug_capture",
                text=strings.get("debug_capture"),
                default=False,
                on_change=self._on_debug_capture_change,
            ),
            Divider(text=strings.get("debug_warning")),
            Text(
                text=strings.get("debug_status"),
                on_click=self._on_debug_status_click,
            ),
            Divider(text=self._debug_status),
            Text(
                text=strings.get("debug_clear"),
                red=True,
                on_click=self._on_debug_clear_click,
            ),
        ]

    def _debug_action(self, action=None, value=False, notify=False):
        runtime = self.dex_loader
        if runtime is None or not self._loaded:
            self.set_setting("debug_capture", False, reload_settings=True)
            self.show_toast(strings.get("debug_unavailable"))
            return
        self._debug_request += 1
        request = self._debug_request
        generation = self._generation

        def current():
            return (self._loaded and self._is_current(generation)
                    and request == self._debug_request and runtime is self.dex_loader)

        def apply():
            if not current():
                return
            error = None
            try:
                if action == "configure":
                    runtime.set_debug_capture(value)
                elif action == "clear":
                    runtime.clear_debug_captures()
                enabled = runtime.is_debug_capture_enabled()
                status = runtime.get_debug_capture_status()
            except Exception as exc:
                error = str(exc)
                try:
                    runtime.set_debug_capture(False)
                except Exception as stop_error:
                    self.log(f"[BlurFaces] Debug stop after error failed: {stop_error}")
                enabled = False
                status = f"state=error: {error}"
                self.log(f"[BlurFaces] Debug capture: {error}")

            def publish():
                if not current():
                    return
                self._debug_enabled = enabled
                self._debug_status = status
                self.set_setting("debug_capture", enabled, reload_settings=True)
                if notify or error:
                    self.show_toast(status)
                if enabled:
                    # Poll off the UI thread, including expiry with the camera closed.
                    run_on_ui_thread(lambda: self._debug_action() if current() else None, 1000)

            run_on_ui_thread(publish)

        run_on_queue(apply)

    def _on_debug_capture_change(self, value):
        self._debug_action("configure", bool(value))

    def _on_debug_status_click(self, _view=None):
        self._debug_action(notify=True)

    def _on_debug_clear_click(self, _view=None):
        self._debug_action("clear", notify=True)

    @staticmethod
    def _valid_width_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if 0 <= index < len(ROUND_VIDEO_WIDTHS) else 0

    @staticmethod
    def _valid_range_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if 0 <= index < len(DETECTION_CONFIDENCES) else 0

    @staticmethod
    def _valid_mask_mode_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if index in MASK_MODES else 0

    def _on_blur_by_default_change(self, value):
        self.blur_by_default = bool(value)
        self.set_setting("blur_by_default", self.blur_by_default)
        runtime = self.dex_loader
        if runtime:
            run_on_queue(lambda: runtime.set_blur_by_default(self.blur_by_default))
        self.log(f"[BlurFaces] Blur by default: {self.blur_by_default}")

    def _on_width_change(self, value):
        index = self._valid_width_index(value)
        self.round_video_width_index = index
        self.set_setting("round_video_width", index)
        width = ROUND_VIDEO_WIDTHS[index]
        runtime = self.dex_loader
        if runtime:
            run_on_queue(lambda: runtime.set_round_video_width(width))
        label = "app default" if width == 0 else f"{width}x{width}"
        self.log(f"[BlurFaces] Round-video resolution: {label}; applies when camera opens next")

    def _on_mask_mode_change(self, value):
        index = self._valid_mask_mode_index(value)
        self.mask_mode_index = index
        self.set_setting("mask_mode", index)
        runtime = self.dex_loader
        if runtime:
            run_on_queue(lambda: runtime.set_mask_mode(index))

    def _on_range_change(self, value):
        index = self._valid_range_index(value)
        if index == self.detection_range_index:
            return
        previous = self.detection_range_index
        if not self.enabled:
            self.detection_range_index = index
            self.set_setting("detection_range", index)
            return
        self._request_reconfigure(index, "detection_range", previous)

    def _request_reconfigure(self, range_index, setting_key, previous_value):
        runtime = self.dex_loader
        if runtime is None:
            return
        self._config_generation += 1
        request_generation = self._config_generation
        lifecycle_generation = self._generation
        runtime_request = runtime.reserve_reconfigure_request()

        def apply():
            try:
                success = runtime.switch_confidence(
                    DETECTION_CONFIDENCES[range_index],
                    lifecycle_generation, runtime_request,
                )
            except Exception as error:
                self.log(f"[BlurFaces] RECONFIG_EXCEPTION: {error}")
                success = False
            if (not self._loaded or request_generation != self._config_generation
                    or lifecycle_generation != self._generation):
                return
            if success:
                self.detection_range_index = range_index
                self.set_setting("detection_range", range_index)
            else:
                self.set_setting(setting_key, previous_value, reload_settings=True)

        threading.Thread(
            target=apply,
            name=f"BlurFacesConfig{request_generation}", daemon=True,
        ).start()

    def _on_self_test_click(self, _view=None):
        mask_scale = FACE_MASK_SCALE / 100.0
        mask_mode = self.mask_mode_index
        runtime = self.dex_loader
        result = None
        if runtime is not None and hasattr(runtime, "run_privacy_self_test"):
            try:
                result = runtime.run_privacy_self_test(mask_scale, mask_mode)
            except Exception as error:
                result = f"FAILED: {error}"
        else:
            try:
                temp_runtime = DexRuntime(
                    plugin=self,
                    round_video_width=ROUND_VIDEO_WIDTHS[self.round_video_width_index],
                    face_mask_scale=FACE_MASK_SCALE,
                    detection_confidence=DETECTION_CONFIDENCES[self.detection_range_index],
                )
                if hasattr(temp_runtime, "run_privacy_self_test"):
                    result = temp_runtime.run_privacy_self_test(mask_scale, mask_mode)
            except Exception as error:
                result = f"FAILED: {error}"
        if not result:
            result = "FAILED: Could not invoke self-test"
        self.log(f"[BlurFaces] Self-test: {result}")
        self.show_toast(result)

    def show_toast(self, message):
        plugin_toast = getattr(super(), "show_toast", None)
        if callable(plugin_toast):
            try:
                plugin_toast(message)
                return
            except Exception:
                pass
        try:
            from ui.bulletin import BulletinHelper
            BulletinHelper.show_info(str(message))
            return
        except Exception:
            pass
        try:
            from org.telegram.messenger import AndroidUtilities, ApplicationLoader
            import android.widget.Toast
            context = getattr(ApplicationLoader, "applicationContext", None)
            if context is not None:
                def _show():
                    try:
                        android.widget.Toast.makeText(
                            context,
                            str(message),
                            android.widget.Toast.LENGTH_LONG
                        ).show()
                    except Exception:
                        pass
                AndroidUtilities.runOnUIThread(_show)
        except Exception:
            pass

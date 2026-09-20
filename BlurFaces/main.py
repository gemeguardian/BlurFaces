import threading

from base_plugin import BasePlugin
from client_utils import run_on_queue
from elyx import strings
from ui.settings import Divider, Header, Selector, Switch, Text

from .runtime import DexRuntime


ROUND_VIDEO_WIDTHS = (0, 320, 384, 448, 512)
FACE_MASK_SCALES = (65, 82, 100)
DETECTION_CONFIDENCES = (45, 35, 25)
MASK_MODES = (0, 1, 2)


class BlurFacesPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.round_video_width_index = 0
        self.face_mask_index = 2
        self.detection_range_index = 0
        self.mask_mode_index = 0
        self.dex_loader = None
        self._generation = 0
        self._config_generation = 0
        self._loaded = False

    def on_plugin_load(self):
        self.log("[BlurFaces] Plugin loading...")
        self._loaded = True
        self.enabled = bool(self.get_setting("enabled", True))
        self.round_video_width_index = 0
        self.face_mask_index = self._valid_mask_index(self.get_setting("face_mask_size", 2))
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
                face_mask_scale=FACE_MASK_SCALES[self.face_mask_index],
                detection_confidence=DETECTION_CONFIDENCES[self.detection_range_index],
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
        self._generation += 1
        self._stop_runtime()

    def _stop_runtime(self):
        runtime = self.dex_loader
        self.dex_loader = None
        if runtime:
            runtime.unload()

    def create_settings(self):
        self._loaded = True
        mask_items = [
            strings.get("mask_compact"),
            strings.get("mask_standard"),
            strings.get("mask_wide"),
        ]
        return [
            Header(strings.get("settings_header")),
            Switch(
                key="enabled",
                text=strings.get("settings_enabled"),
                default=self.enabled,
                on_change=self._on_enabled_change,
            ),
            Divider(text=strings.get("settings_enabled_subtext")),
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
                key="face_mask_size",
                text=strings.get("settings_mask"),
                icon="msg_view_file",
                default=self.face_mask_index,
                items=mask_items,
                on_change=self._on_mask_change,
            ),
            Divider(text=strings.get("settings_mask_subtext")),
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
        ]

    @staticmethod
    def _valid_width_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if 0 <= index < len(ROUND_VIDEO_WIDTHS) else 0

    @staticmethod
    def _valid_mask_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 2
        return index if 0 <= index < len(FACE_MASK_SCALES) else 2

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

    def _on_enabled_change(self, value):
        self.enabled = bool(value)
        self.set_setting("enabled", self.enabled)
        self._generation += 1
        generation = self._generation
        if self.enabled:
            run_on_queue(lambda: self._prepare_runtime(generation))
        else:
            run_on_queue(self._stop_runtime)
        self.log(f"[BlurFaces] Enabled: {self.enabled}")

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

    def _on_mask_change(self, value):
        index = self._valid_mask_index(value)
        self.face_mask_index = index
        self.set_setting("face_mask_size", index)
        scale = FACE_MASK_SCALES[index]
        runtime = self.dex_loader
        if runtime:
            run_on_queue(lambda: runtime.set_face_mask_scale(scale))
        self.log(f"[BlurFaces] Face mask scale: {scale}%")

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
        mask_scale = FACE_MASK_SCALES[self.face_mask_index] / 100.0
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
                    face_mask_scale=FACE_MASK_SCALES[self.face_mask_index],
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

import threading

from base_plugin import BasePlugin
from client_utils import run_on_queue
from elyx import strings
from java import jint, jlong
from java.lang import Integer, Long, String
from java.util.function import IntConsumer
from ui.settings import Custom, Divider, Header, Selector, Text

from .runtime import (
    DexRuntime,
    ModelClickCallback,
    ModelDeleteCallback,
    PresetClickCallback,
    delete_model,
    download_model,
    get_model_settings_bridge,
    is_model_downloaded,
    model_settings_factory,
    model_settings_item,
    preset_settings_factory,
    preset_settings_item,
)


ROUND_VIDEO_WIDTHS = (0, 320, 384, 448, 512)
FACE_MASK_SCALES = (65, 82, 100)
DETECTION_CONFIDENCES = (60, 50, 40)
MASK_MODES = (0, 1, 2)


class BlurFacesPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.enabled = True
        self.round_video_width_index = 0
        self.face_mask_index = 1
        self.detection_range_index = 0
        self.mask_mode_index = 0
        self.processor_index = 0
        self.model_index = 0
        self.dex_loader = None
        self._generation = 0
        self._config_generation = 0
        self._download_generation = 0
        self._downloading_models = set()
        self._downloads_lock = threading.Lock()
        self._downloaded_mask = None
        self._requested_model_index = 0
        self._pending_model_switch = None
        self._model_settings_bridge = None
        self._model_click_callback = None
        self._model_delete_callback = None
        self._preset_click_callback = None
        self.preset_index = 3  # manual: current behaviour, no engine changes
        self._applied_preset_index = 3
        self._preset_downloading = None
        self._preset_downloading_model = None
        self._loaded = False
        self._bridge_error_logged = False
        self._bridge_refresh_pending = False
        self._settings_show_sensitivity = True
        self._settings_has_models = False
        self._settings_model_index = 0

    def on_plugin_load(self):
        self.log("[BlurFaces] Plugin loading...")
        self._loaded = True
        self.enabled = True
        self.set_setting("enabled", True)
        self.round_video_width_index = self._valid_width_index(
            self.get_setting("round_video_width", 0)
        )
        self.face_mask_index = self._valid_mask_index(self.get_setting("face_mask_size", 1))
        self.detection_range_index = self._valid_range_index(
            self.get_setting("detection_range", 0)
        )
        self.mask_mode_index = self._valid_mask_mode_index(self.get_setting("mask_mode", 0))
        self.processor_index = self._valid_processor_index(self.get_setting("processor", 0))
        self.model_index = self._valid_model_index(self.get_setting("model", 0))
        self.preset_index = self._valid_preset_index(self.get_setting("preset", 3))
        self._applied_preset_index = self.preset_index
        self._settings_model_index = self.model_index
        self._requested_model_index = self.model_index
        self._generation += 1
        self._download_generation += 1
        if self.enabled:
            generation = self._generation
            run_on_queue(lambda: self._prepare_runtime(generation))

    def _is_current(self, generation):
        return self.enabled and generation == self._generation

    def _prepare_runtime(self, generation):
        runtime = None
        try:
            runtime = DexRuntime(
                self,
                ROUND_VIDEO_WIDTHS[self.round_video_width_index],
                FACE_MASK_SCALES[self.face_mask_index],
                DETECTION_CONFIDENCES[self.detection_range_index],
                self.processor_index == 0,
                self.model_index,
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
            self.log("[BlurFaces] Verified GPU face geometry enabled for preview and encoder")
            pending = self._pending_model_switch
            self._pending_model_switch = None
            if pending is not None and runtime.model_kind != ("precise", "near", "far")[pending]:
                switched = runtime.switch_model(
                    pending, DETECTION_CONFIDENCES[self.detection_range_index],
                    self.processor_index == 0, generation,
                )
                if switched:
                    self.model_index = pending
                    self.set_setting("model", pending)
        except Exception as error:
            if runtime:
                runtime.unload()
            self.log(f"[BlurFaces] Runtime preparation failed: {type(error).__name__}: {error}")

    def on_plugin_unload(self):
        self.log("[BlurFaces] Plugin unloading...")
        self._loaded = False
        self.enabled = False
        self._generation += 1
        self._download_generation += 1
        self._stop_runtime()
        self._detach_preset_bridge()
        self._detach_model_settings_bridge()

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
        mask_subtext = strings.get("settings_mask_subtext")
        rows = [Header(strings.get("settings_performance"))]
        rows.extend(self._preset_rows())
        rows.append(Divider(text=strings.get("settings_preset_subtext")))
        if self.preset_index != 3:
            # Preset selected: the manual section stays hidden, model downloads
            # run from the preset row itself.
            return rows
        rows.append(Header(strings.get("settings_model")))
        rows.extend(self._model_rows())
        rows.append(Divider(text=strings.get("settings_model_subtext")))
        self._settings_has_models = self._get_downloaded_mask() != 0
        if not self._settings_has_models:
            return rows
        # Keep the settings structure stable while models download. Rebuilding an
        # open UniversalRecyclerView is host-version dependent, while these values
        # are safe to configure before the selected model is available.
        self._settings_show_sensitivity = self._settings_model_index == 0
        rows.extend([
            Selector(
                key="round_video_width",
                text=strings.get("settings_width"),
                icon="msg_media",
                default=self.round_video_width_index,
                items=[
                    strings.get("width_auto"),
                    strings.get("width_fast"),
                    strings.get("width_balanced"),
                    strings.get("width_quality"),
                    strings.get("width_maximum"),
                ],
                on_change=self._on_width_change,
            ),
            Divider(text=strings.get("settings_width_subtext")),
            Selector(
                key="face_mask_size",
                text=strings.get("settings_mask"),
                icon="msg_view_file",
                default=self.face_mask_index,
                items=mask_items,
                on_change=self._on_mask_change,
            ),
            Divider(text=mask_subtext),
            Selector(
                key="mask_mode",
                text=strings.get("settings_mask_mode"),
                icon="msg_photo_blur",
                default=self.mask_mode_index,
                items=[strings.get("mask_mode_blur"), strings.get("mask_mode_pixelate"), strings.get("mask_mode_solid")],
                on_change=self._on_mask_mode_change,
            ),
            Divider(text=strings.get("settings_mask_mode_subtext")),
        ])
        if self._settings_show_sensitivity:
            rows.extend([
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
            ])
        rows.extend([
            Selector(
                key="processor",
                text=strings.get("settings_processor"),
                icon="msg_settings",
                default=self.processor_index,
                items=[
                    strings.get("processor_gpu"),
                    strings.get("processor_cpu"),
                ],
                on_change=self._on_processor_change,
            ),
            Divider(text=strings.get("settings_processor_subtext")),
        ])
        return rows

    def _model_rows(self):
        labels = (
            strings.get("model_precise"),
            strings.get("model_lite_near"),
            strings.get("model_lite_far"),
        )
        try:
            bridge = self._sync_model_settings_bridge(labels)
            rows = []
            for index in range(3):
                row = Custom(item=model_settings_item(bridge, index))
                row.factory = model_settings_factory(bridge, index)
                rows.append(row)
            return rows
        except Exception as error:
            if not self._bridge_error_logged:
                self.log(
                    "[BlurFaces] Model settings bridge failed to load "
                    f"({type(error).__name__}: {error})"
                )
                self._bridge_error_logged = True
            return [Text(text=strings.get("model_settings_unavailable"), red=True)]

    def _preset_rows(self):
        labels = (
            strings.get("preset_flagship"),
            strings.get("preset_balanced"),
            strings.get("preset_eco"),
            strings.get("preset_manual"),
        )
        bridge = get_model_settings_bridge()
        self._sync_preset_bridge(labels)
        rows = []
        for index in range(4):
            row = Custom(item=preset_settings_item(bridge, 3 + index))
            row.factory = preset_settings_factory(bridge, 3 + index)
            rows.append(row)
        return rows

    def _sync_preset_bridge(self, labels=None):
        if not self._loaded:
            return
        try:
            if labels is None:
                labels = (
                    strings.get("preset_flagship"),
                    strings.get("preset_balanced"),
                    strings.get("preset_eco"),
                    strings.get("preset_manual"),
                )
            bridge = get_model_settings_bridge()
            if bridge is None:
                return
            parameter_types = (
                IntConsumer, String, String, String, String, String,
                String, String, String, String, String, Integer.TYPE,
            )
            callback = self._preset_click_callback
            if callback is None:
                callback = PresetClickCallback(self)
                self._preset_click_callback = callback
            status_labels = self._model_status_labels()
            args = (
                callback,
                labels[0], labels[1], labels[2], labels[3],
                strings.get("preset_manual_subtext"),
                status_labels[0], status_labels[1], status_labels[2],
                status_labels[3], status_labels[4],
                Integer(jint(self.preset_index)),
            )
            update_presets = bridge.getMethod("updatePresets", *parameter_types)
            updated = update_presets.invoke(
                None, callback, labels[0], labels[1], labels[2], labels[3],
                strings.get("preset_manual_subtext"),
                status_labels[0], status_labels[1], status_labels[2],
                status_labels[3], status_labels[4],
                Integer(jint(self.preset_index)),
            )
            if not updated:
                attach_presets = bridge.getMethod("attachPresets", *parameter_types)
                attach_presets.invoke(
                    None, callback, labels[0], labels[1], labels[2], labels[3],
                    strings.get("preset_manual_subtext"),
                    status_labels[0], status_labels[1], status_labels[2],
                    status_labels[3], status_labels[4],
                    Integer(jint(self.preset_index)),
                )
        except Exception as error:
            self.log(f"[BlurFaces] Preset settings bridge sync failed: {error}")

    def _dex_on_preset_click(self, index):
        self.log(
            f"[BlurFaces] PRESET_CLICK raw={index} current={self.preset_index} "
            f"applied={self._applied_preset_index} enabled={self.enabled} "
            f"runtime={self.dex_loader is not None}"
        )
        # Java bridge versions before 2.3 sent factory indices (3..6), while
        # one intermediate build sent local indices (0..3). Prefer the stable
        # factory contract and retain the local form for 0..2 during migration.
        preset = self._valid_preset_index(index - 3 if 3 <= int(index) <= 6 else index)
        self.log(f"[BlurFaces] PRESET_CLICK resolved={preset}")
        if preset == self.preset_index:
            profile = self.PRESET_PROFILES.get(preset)
            if profile is not None and not self._is_model_ready(profile[0]):
                # A failed download leaves the preset selected. Let a second tap
                # retry it instead of forcing the user through Manual first.
                self._preset_downloading = preset
                self._preset_downloading_model = profile[0]
                self._requested_model_index = profile[0]
                self._on_model_click(profile[0])
            elif profile is not None and self._applied_preset_index != preset:
                # The Java cell can be tapped again while an earlier async
                # reconfigure is still finishing. Do not treat that tap as a
                # no-op: retry the profile until the engine confirms it.
                self.log(f"[BlurFaces] Retrying unapplied preset {preset}")
                self._apply_preset(preset, self._applied_preset_index)
            else:
                self.log(f"[BlurFaces] PRESET_CLICK no-op profile={profile}")
            return
        previous = self.preset_index
        self._config_generation += 1
        self.preset_index = preset
        self._preset_downloading = None
        self._preset_downloading_model = None
        self._requested_model_index = (
            self.PRESET_PROFILES[preset][0] if preset in self.PRESET_PROFILES
            else self.model_index
        )
        # Preset-to-preset switches never rebuild the page: the radio dot has
        # already moved in Java and the engine reconfigures in background.
        # Only the manual <-> preset boundary changes the row structure.
        structure_changed = (previous == 3) != (preset == 3)
        self.set_setting("preset", preset)
        if preset == 3:
            self._applied_preset_index = preset
        if preset != 3:
            profile = self.PRESET_PROFILES.get(preset)
            self.log(
                f"[BlurFaces] PRESET_PROFILE preset={preset} profile={profile} "
                f"model_ready={profile is not None and self._is_model_ready(profile[0])}"
            )
            if profile is not None and not self._is_model_ready(profile[0]):
                # Model missing: download right from the preset row; the
                # progress shows inside the preset cell itself.
                self._preset_downloading = preset
                self._preset_downloading_model = profile[0]
                self.log(
                    "[BlurFaces] Preset model is not downloaded; downloading it now"
                )
                self._on_model_click(profile[0])
                if structure_changed:
                    self.set_setting("preset", preset, reload_settings=True)
                else:
                    self._sync_preset_bridge()
                return
            self._apply_preset(preset, self._applied_preset_index)
        if structure_changed:
            self.set_setting("preset", preset, reload_settings=True)
        else:
            self._sync_preset_bridge()
        self.log(f"[BlurFaces] Performance preset: {previous} -> {preset}")

    PRESET_PROFILES = {
        # model_index, processor_index, range_index, width_index
        0: (0, 0, 0, 0),  # quality: precise, GPU, normal, auto width
        1: (1, 0, 0, 2),  # balanced: near, GPU, normal, 384
        2: (1, 1, 0, 1),  # compatibility: near, CPU, normal, 320
    }

    def _apply_preset(self, index, previous_preset=None):
        profile = self.PRESET_PROFILES.get(index)
        self.log(
            f"[BlurFaces] PRESET_APPLY index={index} previous={previous_preset} "
            f"profile={profile} runtime={self.dex_loader is not None}"
        )
        if profile is None:
            return
        model_index, processor_index, range_index, width_index = profile
        width_changed = width_index != self.round_video_width_index
        def commit():
            self._applied_preset_index = index
            if width_changed:
                self._on_width_change(width_index)
            self._sync_preset_bridge()

        if not self.enabled:
            self.model_index = model_index
            self._settings_model_index = model_index
            self.detection_range_index = range_index
            self.processor_index = processor_index
            commit()
            return
        runtime = self.dex_loader
        if runtime is None:
            # Commit now; the next runtime preparation picks these up.
            self.model_index = model_index
            self._settings_model_index = model_index
            self.detection_range_index = range_index
            self.processor_index = processor_index
            self.set_setting("model", model_index)
            self.set_setting("detection_range", range_index)
            self.set_setting("processor", processor_index)
            commit()
            if self.enabled:
                self._generation += 1
                generation = self._generation
                run_on_queue(lambda: self._prepare_runtime(generation))
            return
        self._request_reconfigure(model_index, range_index, processor_index,
                                  "preset", previous_preset, commit)

    def _detach_preset_bridge(self):
        bridge = self._model_settings_bridge
        callback = self._preset_click_callback
        self._preset_click_callback = None
        if bridge is not None and callback is not None:
            try:
                bridge.getMethod("detachPresets", IntConsumer).invoke(None, callback)
            except Exception as error:
                self.log(f"[BlurFaces] Preset settings bridge detach failed: {error}")
        if callback is not None:
            callback.release()

    def _model_snapshot(self, labels=None):
        if labels is None:
            labels = (
                strings.get("model_precise"),
                strings.get("model_lite_near"),
                strings.get("model_lite_far"),
            )
        downloaded_mask = self._get_downloaded_mask()
        return labels, strings.get("model_delete"), downloaded_mask

    def _get_downloaded_mask(self):
        with self._downloads_lock:
            if self._downloaded_mask is None:
                self._downloaded_mask = sum(
                    (1 << index) for index in range(3) if is_model_downloaded(index)
                )
            return self._downloaded_mask

    def _is_model_ready(self, index):
        return (self._get_downloaded_mask() & (1 << index)) != 0

    @staticmethod
    def _model_status_labels():
        return (
            strings.get("model_connecting"),
            strings.get("model_downloading"),
            strings.get("model_verifying"),
            strings.get("model_downloaded"),
            strings.get("model_download_failed"),
        )

    def _sync_model_settings_bridge(self, labels=None):
        if not self._loaded:
            return self._model_settings_bridge
        bridge = get_model_settings_bridge()
        labels, delete_label, downloaded_mask = self._model_snapshot(labels)
        parameter_types = (
            IntConsumer, IntConsumer, String, String, String, String,
            String, String, String, String, String,
            Integer.TYPE, Integer.TYPE,
        )
        status_labels = self._model_status_labels()
        click_callback = self._model_click_callback
        delete_callback = self._model_delete_callback
        if click_callback is None or delete_callback is None:
            click_callback = ModelClickCallback(self)
            delete_callback = ModelDeleteCallback(self)
            self._model_click_callback = click_callback
            self._model_delete_callback = delete_callback
        if self._model_settings_bridge is bridge:
            update = bridge.getMethod("update", *parameter_types)
            updated = update.invoke(
                None, click_callback, delete_callback, labels[0], labels[1], labels[2],
                delete_label, status_labels[0], status_labels[1], status_labels[2],
                status_labels[3], status_labels[4],
                Integer(jint(self._settings_model_index)), Integer(jint(downloaded_mask)),
            )
            if updated:
                return bridge
        attach = bridge.getMethod("attach", *parameter_types)
        attach.invoke(
            None, click_callback, delete_callback, labels[0], labels[1], labels[2],
            delete_label, status_labels[0], status_labels[1], status_labels[2],
            status_labels[3], status_labels[4],
            Integer(jint(self._settings_model_index)), Integer(jint(downloaded_mask)),
        )
        self._model_settings_bridge = bridge
        self._bridge_error_logged = False
        return bridge

    def _schedule_bridge_refresh(self):
        with self._downloads_lock:
            if self._bridge_refresh_pending or not self._loaded:
                return
            self._bridge_refresh_pending = True

        def refresh():
            with self._downloads_lock:
                self._bridge_refresh_pending = False
            if self._loaded:
                self._sync_model_settings_bridge()
                self._sync_preset_bridge()

        run_on_queue(refresh)

    def _reload_settings_structure(self, model_index=None):
        if model_index is not None:
            self._settings_model_index = self._valid_model_index(model_index)
        has_models = self._get_downloaded_mask() != 0
        show_sensitivity = has_models and self._settings_model_index == 0
        if (show_sensitivity == self._settings_show_sensitivity
                and has_models == self._settings_has_models):
            return
        self._settings_show_sensitivity = show_sensitivity
        self._settings_has_models = has_models
        self.set_setting(
            "model_settings_layout", (2 if has_models else 0) | (1 if show_sensitivity else 0),
            reload_settings=True,
        )

    def _detach_model_settings_bridge(self):
        bridge = self._model_settings_bridge
        click_callback = self._model_click_callback
        delete_callback = self._model_delete_callback
        self._model_settings_bridge = None
        self._model_click_callback = None
        self._model_delete_callback = None
        if bridge is not None and click_callback is not None and delete_callback is not None:
            try:
                bridge.getMethod("detach", IntConsumer, IntConsumer).invoke(
                    None, click_callback, delete_callback
                )
            except Exception as error:
                self.log(f"[BlurFaces] Model settings bridge detach failed: {error}")
        if click_callback is not None:
            click_callback.release()
        if delete_callback is not None:
            delete_callback.release()

    def _dex_on_model_click(self, index):
        self.log(
            f"[BlurFaces] MODEL_CLICK raw={index} current={self.model_index} "
            f"requested={self._requested_model_index} enabled={self.enabled} "
            f"runtime={self.dex_loader is not None}"
        )
        self._on_model_click(index)

    def _dex_on_model_delete(self, index):
        index = self._valid_model_index(index)
        if not self._is_model_ready(index):
            self._sync_model_settings_bridge()
            return
        run_on_queue(lambda: self._delete_model_file(index))

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
            return 1
        return index if 0 <= index < len(FACE_MASK_SCALES) else 1

    @staticmethod
    def _valid_range_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if 0 <= index < len(DETECTION_CONFIDENCES) else 0

    @staticmethod
    def _valid_processor_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if index in (0, 1) else 0

    @staticmethod
    def _valid_model_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if index in (0, 1, 2) else 0

    @staticmethod
    def _valid_mask_mode_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 0
        return index if index in MASK_MODES else 0

    @staticmethod
    def _valid_preset_index(value):
        try:
            index = int(value)
        except (TypeError, ValueError):
            return 3
        return index if index in (0, 1, 2, 3) else 3

    def _on_enabled_change(self, value):
        self.enabled = value
        self.set_setting("enabled", value)
        self._generation += 1
        generation = self._generation
        if value:
            run_on_queue(lambda: self._prepare_runtime(generation))
        else:
            run_on_queue(self._stop_runtime)
        self.log(f"[BlurFaces] Enabled: {value}")

    def _on_width_change(self, value):
        index = self._valid_width_index(value)
        self.round_video_width_index = index
        self.set_setting("round_video_width", index)
        width = ROUND_VIDEO_WIDTHS[index]
        runtime = self.dex_loader
        if runtime:
            run_on_queue(lambda: runtime.set_round_video_width(width))
        label = "app default" if width == 0 else f"{width}x{width}"
        self.log(f"[BlurFaces] Round-video resolution: {label}; applies when the camera is opened next")

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
        try:
            index = int(value)
        except (TypeError, ValueError):
            index = 0
        if index not in MASK_MODES:
            index = 0
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
        self._request_reconfigure(self.model_index, index, self.processor_index,
                                  "detection_range", previous)

    def _on_processor_change(self, value):
        index = self._valid_processor_index(value)
        if index == self.processor_index:
            return
        previous = self.processor_index
        if not self.enabled:
            self.processor_index = index
            self.set_setting("processor", index)
            return
        self._request_reconfigure(self.model_index, self.detection_range_index, index,
                                  "processor", previous)

    def _on_model_click(self, value):
        index = self._valid_model_index(value)
        self.log(f"[BlurFaces] MODEL_CLICK resolved={index} ready={self._is_model_ready(index)}")
        self._requested_model_index = index
        if self._is_model_ready(index):
            self._activate_model(index)
            return
        with self._downloads_lock:
            if index in self._downloading_models:
                return
            self._downloading_models.add(index)
        download_generation = self._download_generation
        thread = threading.Thread(
            target=lambda: self._download_and_activate(index, download_generation),
            name=f"BlurFacesModel{index}", daemon=True,
        )
        thread.start()

    def _download_and_activate(self, index, download_generation):
        model_path = download_model(
            self, index,
            lambda phase, downloaded, total: self._update_model_download(
                index, phase, downloaded, total, download_generation
            ),
            lambda: self._loaded and download_generation == self._download_generation,
        )
        with self._downloads_lock:
            self._downloading_models.discard(index)
        if not self._loaded or download_generation != self._download_generation:
            return
        # Resolve the preset-download state before any settings rebuild: the
        # reload must not expand the manual section for preset-triggered
        # downloads.
        pending_preset = None
        if self._preset_downloading_model == index:
            pending_preset = self._preset_downloading
            self._preset_downloading = None
            self._preset_downloading_model = None
        if model_path is None:
            self._schedule_bridge_refresh()
            return
        with self._downloads_lock:
            mask = self._downloaded_mask
            if mask is None:
                mask = sum((1 << candidate) for candidate in range(3)
                           if is_model_downloaded(candidate))
            self._downloaded_mask = mask | (1 << index)
        # Publish the verified bit immediately. Delaying this behind runtime
        # creation leaves the attached row showing the download icon.
        self._sync_model_settings_bridge()
        self._reload_settings_structure()
        if self._requested_model_index == index:
            self._activate_model(index)
        if (pending_preset is not None and pending_preset == self.preset_index
                and pending_preset in self.PRESET_PROFILES):
            self._apply_preset(pending_preset, self._applied_preset_index)
            self._sync_preset_bridge()

    def _update_model_download(self, index, phase, downloaded, total, download_generation):
        bridge = self._model_settings_bridge
        if (bridge is None or not self._loaded
                or download_generation != self._download_generation):
            return
        try:
            if self._preset_downloading is not None and self._requested_model_index == index:
                # Download was started from a preset row: progress shows there.
                # The flag itself is cleared by _download_and_activate when the
                # transfer finishes, so the preset can auto-apply afterwards.
                bridge.getMethod(
                    "updatePresetDownload", Integer.TYPE, Integer.TYPE, Long.TYPE, Long.TYPE
                ).invoke(
                    None, Integer(jint(self._preset_downloading)), Integer(jint(phase)),
                    Long(jlong(downloaded)), Long(jlong(total)),
                )
            bridge.getMethod(
                "updateDownload", Integer.TYPE, Integer.TYPE, Long.TYPE, Long.TYPE
            ).invoke(
                None, Integer(jint(index)), Integer(jint(phase)),
                Long(jlong(downloaded)), Long(jlong(total)),
            )
        except Exception as error:
            self.log(f"[BlurFaces] Download progress update failed: {error}")

    def _activate_model(self, index):
        if not self.enabled:
            self.model_index = index
            self._settings_model_index = index
            self.set_setting("model", index)
            self._schedule_bridge_refresh()
            self._reload_settings_structure()
            return
        runtime = self.dex_loader
        if runtime is None:
            # Startup may previously have stopped because no model existed. Commit
            # the now-verified selection, refresh the open page, and start it again.
            self.model_index = index
            self._pending_model_switch = None
            self.set_setting("model", index)
            self._sync_model_settings_bridge()
            self._reload_settings_structure()
            if self.enabled:
                self._generation += 1
                generation = self._generation
                run_on_queue(lambda: self._prepare_runtime(generation))
            return
        self._request_reconfigure(index, self.detection_range_index, self.processor_index,
                                  "model", self.model_index)

    def _request_reconfigure(self, model_index, range_index, processor_index,
                             setting_key, previous_value, on_success=None):
        runtime = self.dex_loader
        if runtime is None:
            if setting_key == "model":
                self._pending_model_switch = model_index
            return
        self._config_generation += 1
        request_generation = self._config_generation
        lifecycle_generation = self._generation
        runtime_request = runtime.reserve_reconfigure_request()
        self.log(
            f"[BlurFaces] RECONFIG_REQUEST key={setting_key} model={model_index} "
            f"range={range_index} processor={processor_index} request={runtime_request}"
        )
        if setting_key == "model":
            # Update row visibility before MediaPipe creates the new graph. Engine
            # creation can take seconds, but the settings page should react now.
            self._reload_settings_structure(model_index)

        def apply():
            try:
                success = runtime.switch_model(
                    model_index,
                    DETECTION_CONFIDENCES[range_index] if model_index == 0
                    else DETECTION_CONFIDENCES[0],
                    processor_index == 0,
                    lifecycle_generation, runtime_request,
                )
                self.log(f"[BlurFaces] RECONFIG_RESULT request={runtime_request} success={success}")
            except Exception as error:
                self.log(f"[BlurFaces] RECONFIG_EXCEPTION request={runtime_request}: {error}")
                success = False
            if (not self._loaded or request_generation != self._config_generation
                    or lifecycle_generation != self._generation):
                return
            if success:
                self.model_index = model_index
                self.detection_range_index = range_index
                self.processor_index = processor_index
                self.set_setting("model", model_index)
                self.set_setting("detection_range", range_index)
                self.set_setting("processor", processor_index)
                if on_success is not None:
                    on_success()
                self._schedule_bridge_refresh()
                self._reload_settings_structure()
            else:
                if setting_key == "model":
                    self._reload_settings_structure(previous_value)
                    self._sync_model_settings_bridge()
                elif setting_key == "preset":
                    self.preset_index = previous_value
                    self._sync_preset_bridge()
                self.set_setting(setting_key, previous_value, reload_settings=True)

        # Let every request supersede queued intermediate models immediately.
        threading.Thread(
            target=apply,
            name=f"BlurFacesConfig{request_generation}", daemon=True,
        ).start()

    def _delete_model_file(self, index):
        deleting_active = index == self.model_index
        if deleting_active:
            self._generation += 1
            self._stop_runtime()
        deleted = delete_model(self, index)
        if not deleted:
            if deleting_active and self.enabled:
                generation = self._generation
                self._prepare_runtime(generation)
            self._schedule_bridge_refresh()
            self._reload_settings_structure()
            return

        self._downloaded_mask = self._get_downloaded_mask() & ~(1 << index)

        replacement = next(
            (candidate for candidate in range(3) if self._is_model_ready(candidate)),
            None,
        )
        if self.model_index == index and replacement is not None:
            self.model_index = replacement
            self._settings_model_index = replacement
            self.set_setting("model", replacement)
        self._schedule_bridge_refresh()
        self._reload_settings_structure()
        if deleting_active and replacement is not None and self.enabled:
            generation = self._generation
            self._prepare_runtime(generation)

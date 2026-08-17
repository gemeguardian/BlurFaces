import os
from android_utils import log
from client_utils import run_on_queue
from base_plugin import BasePlugin
from ui.settings import Header, Text, Switch, Divider


class Plugin(BasePlugin):
    """MediaPipe GPU face blur for round-video preview and encoding."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.dex_loader = None
        self._generation = 0

    def on_plugin_load(self):
        log("[BlurFaces] Plugin loading...")
        self.enabled = self.get_setting("enabled", True)
        self._generation += 1
        if self.enabled:
            generation = self._generation
            run_on_queue(lambda: self._prepare_runtime(generation))

    def _is_current(self, generation):
        return self.enabled and generation == self._generation

    def _prepare_runtime(self, generation):
        try:
            from java.io import File
            from org.telegram.messenger import ApplicationLoader

            files_dir = ApplicationLoader.getFilesDirFixed()
            cache_dir = str(File(files_dir, "blur_faces_cache").getAbsolutePath())
            os.makedirs(cache_dir, exist_ok=True)
            native_path = os.path.join(cache_dir, "libmediapipe_tasks_vision_jni.so")
            model_path = os.path.join(cache_dir, "face_landmarker.task")
            runtime_path = os.path.join(cache_dir, "mediapipe-face-landmarker.dex")
            download_file(MP_NATIVE_URL, MP_NATIVE_SHA256, native_path, "MediaPipe JNI")
            if not self._is_current(generation):
                return
            native_load_path = prepare_native_load_copy(native_path, cache_dir)
            download_file(MP_MODEL_URL, MP_MODEL_SHA256, model_path, "Face Landmarker model")
            if not self._is_current(generation):
                return
            download_file(MP_RUNTIME_DEX_URL, MP_RUNTIME_DEX_SHA256, runtime_path,
                          "MediaPipe runtime DEX")
            if not self._is_current(generation):
                return
            loader = DexLoader(self)
            if not loader.load_and_start(model_path, native_load_path, runtime_path):
                return
            if not self._is_current(generation):
                loader.unload()
                return
            self.dex_loader = loader
            log("[BlurFaces] GPU LIVE_STREAM face geometry enabled for preview and encoder")
        except Exception as error:
            log(f"[BlurFaces] Runtime preparation failed: {type(error).__name__}: {error}")

    def on_plugin_unload(self):
        log("[BlurFaces] Plugin unloading...")
        self.enabled = False
        self._generation += 1
        self._stop_runtime()

    def _stop_runtime(self):
        loader = self.dex_loader
        self.dex_loader = None
        if loader:
            loader.unload()

    def create_settings(self):
        return [
            Header("Blur Faces"),
            Text("GPU MediaPipe dense face mesh shared by preview and encoded round video."),
            Divider(),
            Switch(key="enabled", text="Enable round-video face blur", default=self.enabled, on_change=self._on_enabled_change),
        ]

    def _on_enabled_change(self, value):
        self.enabled = value
        self.set_setting("enabled", value)
        self._generation += 1
        generation = self._generation
        if value:
            run_on_queue(lambda: self._prepare_runtime(generation))
        else:
            run_on_queue(self._stop_runtime)
        log(f"[BlurFaces] Enabled: {value}")

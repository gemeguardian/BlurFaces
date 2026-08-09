import os
from android_utils import log
from client_utils import run_on_queue
from base_plugin import BasePlugin
from ui.settings import Header, Text, Switch, Divider


class Plugin(BasePlugin):
    """Blur Faces - preview-only round-video face blur."""

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.dex_loader = None

    def on_plugin_load(self):
        log("[BlurFaces] Plugin loading...")
        self.enabled = self.get_setting("enabled", True)
        run_on_queue(self._prepare_runtime)

    def _prepare_runtime(self):
        try:
            from java.io import File
            from org.telegram.messenger import ApplicationLoader

            files_dir = ApplicationLoader.getFilesDirFixed()
            cache_dir = str(File(files_dir, "blur_faces_cache").getAbsolutePath())
            os.makedirs(cache_dir, exist_ok=True)
            so_filename = "libblur_faces_" + __version__.replace(".", "_") + ".so"
            so_cached = download_native(NATIVE_URL, NATIVE_SHA256, cache_dir, filename=so_filename)
            so_path = unique_load_path(so_cached, cache_dir)
            # The optional mesh runtime uses the same verified cache discipline,
            # but failure leaves the five-point privacy root fully functional.
            mesh_so_path = None
            try:
                mesh_so_cached = download_native(MESH_NATIVE_URL, MESH_NATIVE_SHA256, cache_dir,
                                                 filename="libmediapipe_tasks_vision_jni.so")
                # MediaPipe itself calls System.loadLibrary("mediapipe_tasks_vision_jni").
                # Keep the canonical basename; renaming it would make that later
                # class initializer look for a second, missing library.
                mesh_so_path = mesh_so_cached
            except Exception as error:
                log(f"[BlurFaces] Dense mesh JNI unavailable; stable root continues: {error}")
            param_path, bin_path, mesh_path, downloaded = extract_model(
                MODEL_PARAM_URL, MODEL_PARAM_SHA256, MODEL_BIN_URL, MODEL_BIN_SHA256,
                MESH_MODEL_URL, MESH_MODEL_SHA256, cache_dir,
            )
            mesh_dex_path = os.path.join(cache_dir, "mediapipe-face-landmarker.dex")
            try:
                download_file(
                    MESH_DEX_URL, MESH_DEX_SHA256, mesh_dex_path,
                    "dense mesh runtime DEX",
                )
            except Exception as error:
                # The core DEX has no direct MediaPipe class reference. A failed
                # optional runtime download leaves the stable root runnable.
                log(f"[BlurFaces] Dense mesh DEX unavailable; stable root continues: {error}")
                mesh_dex_path = None
            log(f"[BlurFaces] Runtime assets ready ({'downloaded' if downloaded else 'cached'})")
            self.dex_loader = DexLoader(self)
            if not self.dex_loader.load_and_start(so_path, param_path, bin_path, mesh_path, mesh_so_path, mesh_dex_path):
                return
            log("[BlurFaces] Stable privacy root enabled; dense mesh is an additive effect layer and cannot move or shrink blur.")
        except Exception as error:
            log(f"[BlurFaces] Runtime preparation failed: {type(error).__name__}: {error}")

    def on_plugin_unload(self):
        log("[BlurFaces] Plugin unloading...")
        if self.dex_loader:
            self.dex_loader.unload()
            self.dex_loader = None

    def create_settings(self):
        return [
            Header("Blur Faces"),
            Text("Stable face-root blur with a dense mesh layer for future face-anchored effects."),
            Divider(),
            Switch(key="enabled", text="Enable preview face blur", default=self.enabled, on_change=self._on_enabled_change),
        ]

    def _on_enabled_change(self, value):
        self.enabled = value
        self.set_setting("enabled", value)
        log(f"[BlurFaces] Enabled: {value}")

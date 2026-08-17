import base64
import hashlib
import os
from dalvik.system import DexClassLoader
from java.lang import String
from org.telegram.messenger import ApplicationLoader


class _PythonLogger(dynamic_proxy(Consumer)):
    """java.util.function.Consumer<String> bound to the plugin log sink."""
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin
    def accept(self, message):
        self.plugin.log(message)


class DexLoader:
    def __init__(self, plugin):
        self.plugin = plugin
        self.dex_main_class = None
        self.dex_loader = None

    def load_and_start(self, model_path, native_path, runtime_dex_path):
        try:
            # Core hooks stay embedded. The SHA-verified MediaPipe runtime is the
            # parent so Main resolves all task classes through one runtime loader.
            dex_bytes = base64.b64decode(EMBEDDED_DEX_BASE64)
            actual_core_sha256 = hashlib.sha256(dex_bytes).hexdigest()
            if actual_core_sha256 != EMBEDDED_DEX_SHA256:
                raise ValueError(
                    f"Embedded core DEX SHA-256 mismatch: {actual_core_sha256}"
                )
            cache = ApplicationLoader.applicationContext.getDir(CACHE_DIR_NAME, 0)
            core_path = os.path.join(cache.getAbsolutePath(), "core.dex")
            try:
                os.chmod(core_path, 0o644)
            except OSError:
                pass
            with open(core_path, "wb") as output:
                output.write(dex_bytes)
            # Android 14+ rejects a DEX which remains writable after extraction.
            # This must happen before DexClassLoader opens the file, including on
            # subsequent plugin loads where the filename is reused.
            os.chmod(core_path, 0o444)
            opt = ApplicationLoader.applicationContext.getDir(DEX_OPT_DIR_NAME, 0)
            media_pipe_library_dir = os.path.dirname(native_path)
            parent = ApplicationLoader.applicationContext.getClassLoader()
            runtime_loader = DexClassLoader(
                runtime_dex_path, opt.getAbsolutePath(), media_pipe_library_dir, parent,
            )
            loader = DexClassLoader(
                core_path, opt.getAbsolutePath(), media_pipe_library_dir, runtime_loader,
            )
            dex_class = loader.loadClass(CLASS_NAME)
            # Install the diagnostic sink before initAndStart so Java/native
            # messages flow into the same stream the user reads.
            consumer_class = ApplicationLoader.applicationContext.getClassLoader()
            consumer_type = consumer_class.loadClass("java.util.function.Consumer")
            logger_proxy = _PythonLogger(self.plugin)
            set_logger = dex_class.getMethod("setLogger", consumer_type)
            set_logger.invoke(None, logger_proxy)
            self.plugin.log("[BlurFaces] logger callback installed")
            dex_class.getMethod("initAndStart", String).invoke(None, model_path)
            self.dex_loader = loader
            self.dex_main_class = dex_class
            self.plugin.log("[BlurFaces] DEX loaded and started")
            return True
        except Exception as error:
            self.plugin.log(f"[BlurFaces] DEX load failed: {error}")
            return False

    def unload(self):
        if self.dex_main_class is not None:
            try:
                self.dex_main_class.getMethod("onUnload").invoke(None)
            except Exception as error:
                self.plugin.log(f"[BlurFaces] DEX unload failed: {error}")
            self.dex_main_class = None
            self.dex_loader = None

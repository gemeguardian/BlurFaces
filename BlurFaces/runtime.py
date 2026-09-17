import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time

from android.view import View
from dalvik.system import DexClassLoader
from elyx import assets
from java import dynamic_proxy, jarray, jbyte, jint
from java.lang import Integer, String
from java.net import URL
from java.util.function import Consumer, IntConsumer
from org.telegram.messenger import ApplicationLoader

from .asset_hashes import ASSET_HASHES


CLASS_NAME = "com.makey.blurfaces.g2.Main"
SETTINGS_BRIDGE_CLASS_NAME = "com.makey.blurfaces.g2.ModelSettingsBridge"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_FILENAME = "face_landmarker-float16-v1.task"
LITE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
LITE_MODEL_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
LITE_MODEL_FILENAME = "blaze_face_short_range-float16-v1.tflite"
FAR_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"
FAR_MODEL_SHA256 = "3698b18f063835bc609069ef052228fbe86d9c9a6dc8dcb7c7c2d69aed2b181b"
FAR_MODEL_FILENAME = "blaze_face_full_range-float16-v1.tflite"
CONNECT_TIMEOUT_MS = 15_000
READ_TIMEOUT_MS = 60_000

MODEL_SPECS = {
    "precise": (MODEL_URL, MODEL_SHA256, MODEL_FILENAME, "Face Landmarker"),
    "near": (LITE_MODEL_URL, LITE_MODEL_SHA256, LITE_MODEL_FILENAME, "short-range Face Detector"),
    "far": (FAR_MODEL_URL, FAR_MODEL_SHA256, FAR_MODEL_FILENAME, "full-range Face Detector"),
}

LOADER_ABI_SALT = "hot-swap-core-v3"
CORE_BUNDLE_ID = hashlib.sha256((LOADER_ABI_SALT + "\0" +
    ASSET_HASHES["dex/core.dex"]).encode("ascii")).hexdigest()
RUNTIME_BUNDLE_ID = hashlib.sha256((LOADER_ABI_SALT + "\0" + "".join(
    ASSET_HASHES[name] for name in (
        "dex/mediapipe-runtime.dex",
        "jni/arm64-v8a/libmediapipe_tasks_vision_jni.so",
    )
)).encode("ascii")).hexdigest()
_REGISTRY_KEY = "_blur_faces_mediapipe_runtime_v3"
_LEGACY_REGISTRY_KEYS = (
    "_blur_faces_mediapipe_runtime_v2",
    "_blur_faces_mediapipe_runtime",
)
_EPOCH_KEY = "_blur_faces_runtime_module_epoch"
_MODULE_EPOCH = getattr(sys, _EPOCH_KEY, 0) + 1
setattr(sys, _EPOCH_KEY, _MODULE_EPOCH)
_CORE_LOAD_TOKEN = object()


def _runtime_registry():
    registry = getattr(sys, _REGISTRY_KEY, None)
    if registry is None:
        legacy = next((getattr(sys, key, None) for key in _LEGACY_REGISTRY_KEYS
                       if isinstance(getattr(sys, key, None), dict)), None)
        registry = {
            "lock": threading.RLock(),
            "module_epoch": _MODULE_EPOCH,
            "runtime_bundle_id": None,
            "core_bundle_id": None,
            "core_load_token": None,
            "runtime_dir": None,
            "runtime_loader": None,
            "core_loader": None,
            "main_class": None,
            "settings_bridge_class": None,
            "owner": None,
            "broken": False,
            "restart_reason": None,
            "shutdown_timer": None,
            "shutdown_token": None,
            "methods": {},
            "reconfigure_sequence": 0,
        }
        if isinstance(legacy, dict):
            legacy_lock = legacy.get("lock")
            if legacy_lock is not None:
                with legacy_lock:
                    if legacy.get("runtime_bundle_id") in (None, RUNTIME_BUNDLE_ID):
                        registry["runtime_bundle_id"] = legacy.get("runtime_bundle_id")
                        registry["runtime_dir"] = legacy.get("runtime_dir")
                        registry["runtime_loader"] = legacy.get("runtime_loader")
                        registry["core_bundle_id"] = legacy.get("core_bundle_id")
                        registry["core_loader"] = legacy.get("core_loader")
                        registry["main_class"] = legacy.get("main_class")
                        registry["settings_bridge_class"] = legacy.get("settings_bridge_class")
                        registry["owner"] = legacy.get("owner")
                        registry["broken"] = bool(legacy.get("broken", False))
            for key in _LEGACY_REGISTRY_KEYS:
                if hasattr(sys, key):
                    delattr(sys, key)
        setattr(sys, _REGISTRY_KEY, registry)
    else:
        registry.setdefault("module_epoch", 0)
        registry.setdefault("settings_bridge_class", None)
        registry.setdefault("runtime_bundle_id", None)
        registry.setdefault("core_bundle_id", None)
        registry.setdefault("core_load_token", None)
        registry.setdefault("shutdown_timer", None)
        registry.setdefault("shutdown_token", None)
        registry.setdefault("methods", {})
        registry.setdefault("restart_reason", None)
        registry.setdefault("reconfigure_sequence", 0)
    return registry


def _method(registry, dex_class, name, *types):
    key = (dex_class, name, types)
    method = registry["methods"].get(key)
    if method is None:
        method = dex_class.getMethod(name, *types)
        registry["methods"][key] = method
    return method


def _cancel_deferred_shutdown(registry):
    timer = registry.get("shutdown_timer")
    registry["shutdown_timer"] = None
    registry["shutdown_token"] = None
    if timer is not None:
        timer.cancel()


class _PythonLogger(dynamic_proxy(Consumer)):
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin

    def accept(self, message):
        plugin = self.plugin
        if plugin is not None:
            plugin.log(message)

    def release(self):
        self.plugin = None


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        while True:
            block = source.read(1024 * 1024)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def _stage_asset(asset_name, target, read_only):
    source = assets.get(asset_name).path_str
    expected = ASSET_HASHES[asset_name]
    if _sha256(source) != expected:
        raise ValueError(f"Bundled asset SHA-256 mismatch: {asset_name}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.isfile(target) and _sha256(target) == expected:
        if read_only:
            os.chmod(target, 0o444)
        return target
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(target) + ".", dir=os.path.dirname(target))
    try:
        with os.fdopen(fd, "wb") as output, open(source, "rb") as payload:
            shutil.copyfileobj(payload, output, 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(temporary) != expected:
            raise ValueError(f"Staged asset SHA-256 mismatch: {asset_name}")
        os.replace(temporary, target)
        if read_only:
            os.chmod(target, 0o444)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _loaded_runtime_matches(registry):
    runtime_dir = registry.get("runtime_dir")
    if not runtime_dir:
        return False
    expected = {
        os.path.join(runtime_dir, "mediapipe-runtime.dex"):
            ASSET_HASHES["dex/mediapipe-runtime.dex"],
        os.path.join(runtime_dir, "jni", "arm64-v8a", "libmediapipe_tasks_vision_jni.so"):
            ASSET_HASHES["jni/arm64-v8a/libmediapipe_tasks_vision_jni.so"],
    }
    return all(os.path.isfile(path) and _sha256(path) == digest
               for path, digest in expected.items())


def _release_loaded_core(registry):
    dex_class = registry.get("main_class")
    if registry.get("owner") is not None and dex_class is not None:
        clean = dex_class.getMethod("onUnload").invoke(None)
        if clean is False:
            registry["broken"] = True
            raise RuntimeError("MediaPipe worker did not stop; restart the application")
        registry["owner"] = None
        try:
            dex_class.getMethod("clearLogger").invoke(None)
        except Exception:
            pass
    registry["main_class"] = None
    registry["settings_bridge_class"] = None
    registry["core_loader"] = None
    registry["core_load_token"] = None
    registry["methods"].clear()


def _ensure_runtime_loaders(context, registry):
    loaded_epoch = registry.get("module_epoch", 0)
    if loaded_epoch > _MODULE_EPOCH:
        raise RuntimeError("Stale Blur Faces plugin instance")
    if loaded_epoch < _MODULE_EPOCH:
        registry["module_epoch"] = _MODULE_EPOCH
    root = context.getDir("blur_faces_runtime_v2", 0).getCanonicalPath()
    runtime_dir = os.path.join(root, "runtime_" + RUNTIME_BUNDLE_ID)
    core_dir = os.path.join(root, "core_" + CORE_BUNDLE_ID)
    native_dir = os.path.join(runtime_dir, "jni", "arm64-v8a")
    os.makedirs(native_dir, exist_ok=True)
    core_path = _stage_asset("dex/core.dex", os.path.join(core_dir, "core.dex"), True)
    runtime_path = _stage_asset(
        "dex/mediapipe-runtime.dex",
        os.path.join(runtime_dir, "mediapipe-runtime.dex"), True,
    )
    _stage_asset(
        "jni/arm64-v8a/libmediapipe_tasks_vision_jni.so",
        os.path.join(native_dir, "libmediapipe_tasks_vision_jni.so"), True,
    )

    opt = context.getDir("blur_faces_dex_opt_v2", 0).getCanonicalPath()
    parent = context.getClassLoader()
    if registry["runtime_loader"] is not None:
        if registry["runtime_bundle_id"] not in (None, RUNTIME_BUNDLE_ID):
            registry["restart_reason"] = "MediaPipe DEX/JNI bundle changed while loaded"
            raise RuntimeError(registry["restart_reason"] + "; restart the application")
        if registry["runtime_bundle_id"] is None and not _loaded_runtime_matches(registry):
            registry["restart_reason"] = "Loaded MediaPipe DEX/JNI bundle cannot be verified"
            raise RuntimeError(registry["restart_reason"] + "; restart the application")
        registry["runtime_bundle_id"] = RUNTIME_BUNDLE_ID
    else:
        registry["runtime_loader"] = DexClassLoader(runtime_path, opt, native_dir, parent)
        registry["runtime_bundle_id"] = RUNTIME_BUNDLE_ID

    if (registry["core_loader"] is not None
            and registry.get("core_bundle_id") == CORE_BUNDLE_ID):
        registry["core_load_token"] = _CORE_LOAD_TOKEN
        registry["runtime_dir"] = runtime_dir
        return
    if registry["core_loader"] is not None:
        # A compatible MediaPipe parent can stay loaded, but changed plugin Java
        # code must be adopted now. Otherwise an update keeps executing stale model
        # rows until the whole client restarts. Host UItem may retain old factory
        # classes, but new settings definitions use factories from this loader.
        _release_loaded_core(registry)
    registry["core_loader"] = DexClassLoader(
        core_path, opt, native_dir, registry["runtime_loader"]
    )
    registry["core_bundle_id"] = CORE_BUNDLE_ID
    registry["core_load_token"] = _CORE_LOAD_TOKEN
    registry["runtime_dir"] = runtime_dir


def _model_location(context, model_kind):
    model_filename = MODEL_SPECS[model_kind][2]
    model_dir = context.getDir("blur_faces_models", 0).getCanonicalPath()
    return model_dir, os.path.join(model_dir, model_filename)


def cached_model_path(context, model_kind="precise"):
    _, model_sha256, _, _ = MODEL_SPECS[model_kind]
    _, model_path = _model_location(context, model_kind)
    if os.path.isfile(model_path):
        actual = _sha256(model_path)
        if actual == model_sha256:
            os.chmod(model_path, 0o444)
            return model_path
        os.chmod(model_path, 0o600)
        os.unlink(model_path)
    return None


def is_model_downloaded(model_index):
    context = ApplicationLoader.applicationContext
    model_kind = ("precise", "near", "far")[model_index]
    return cached_model_path(context, model_kind) is not None


def download_model(plugin, model_index, progress_callback=None, is_current=None):
    context = ApplicationLoader.applicationContext
    model_kind = ("precise", "near", "far")[model_index]
    model_url, model_sha256, model_filename, model_label = MODEL_SPECS[model_kind]
    cached = cached_model_path(context, model_kind)
    if cached is not None:
        return cached
    model_dir, model_path = _model_location(context, model_kind)

    os.makedirs(model_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=model_filename + ".", suffix=".download", dir=model_dir)
    os.close(fd)
    connection = None
    input_stream = None
    try:
        plugin.log(f"[BlurFaces] Downloading {model_label} model from official Google URL: {model_url}")
        connection = URL(model_url).openConnection()
        connection.setConnectTimeout(CONNECT_TIMEOUT_MS)
        connection.setReadTimeout(READ_TIMEOUT_MS)
        connection.setUseCaches(False)
        connection.setRequestProperty("Accept-Encoding", "identity")
        connection.connect()
        total_bytes = int(connection.getContentLengthLong())
        if total_bytes <= 0:
            total_bytes = -1
        if progress_callback is not None:
            progress_callback(2, 0, total_bytes)
        input_stream = connection.getInputStream()
        buffer = jarray(jbyte)(64 * 1024)
        downloaded_bytes = 0
        last_update = 0.0
        with open(temporary, "wb") as output:
            while True:
                if is_current is not None and not is_current():
                    raise InterruptedError("model download cancelled")
                count = input_stream.read(buffer)
                if count == -1:
                    break
                if count:
                    output.write(bytearray(value & 0xFF for value in buffer[:count]))
                    downloaded_bytes += count
                    now = time.monotonic()
                    if progress_callback is not None and now - last_update >= 0.075:
                        progress_callback(2, downloaded_bytes, total_bytes)
                        last_update = now
            output.flush()
            os.fsync(output.fileno())

        if progress_callback is not None:
            progress_callback(2, downloaded_bytes, total_bytes)
            progress_callback(3, downloaded_bytes, total_bytes)
        actual = _sha256(temporary)
        if actual != model_sha256:
            raise ValueError(
                f"downloaded model SHA-256 mismatch: expected {model_sha256}, got {actual}"
            )
        os.replace(temporary, model_path)
        os.chmod(model_path, 0o444)
        if progress_callback is not None:
            progress_callback(4, downloaded_bytes, total_bytes)
        plugin.log(f"[BlurFaces] {model_label} model downloaded and verified: {model_path}")
        return model_path
    except Exception as error:
        cancelled = isinstance(error, InterruptedError)
        if progress_callback is not None and not cancelled:
            progress_callback(5, locals().get("downloaded_bytes", 0),
                              locals().get("total_bytes", -1))
        if not cancelled:
            plugin.log(
                f"[BlurFaces] {model_label} model unavailable; camera hooks will not be installed. "
                f"Check the network and reload the plugin ({type(error).__name__}: {error})"
            )
        return None
    finally:
        if input_stream is not None:
            try:
                input_stream.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.disconnect()
            except Exception:
                pass
        if os.path.exists(temporary):
            os.unlink(temporary)


def delete_model(plugin, model_index):
    context = ApplicationLoader.applicationContext
    model_kind = ("precise", "near", "far")[model_index]
    _, model_path = _model_location(context, model_kind)
    if not os.path.isfile(model_path):
        return False
    os.chmod(model_path, 0o600)
    os.unlink(model_path)
    plugin.log(f"[BlurFaces] Deleted cached {MODEL_SPECS[model_kind][3]} model")
    return True


def get_model_settings_bridge():
    context = ApplicationLoader.applicationContext
    registry = _runtime_registry()
    with registry["lock"]:
        _ensure_runtime_loaders(context, registry)
        if registry["settings_bridge_class"] is None:
            registry["settings_bridge_class"] = registry["core_loader"].loadClass(
                SETTINGS_BRIDGE_CLASS_NAME
            )
        return registry["settings_bridge_class"]


def model_settings_item(bridge_class, index):
    return bridge_class.getMethod("getItem", Integer.TYPE).invoke(
        None, Integer(jint(index))
    )


def preset_settings_item(bridge_class, index):
    return bridge_class.getMethod("getPresetItem", Integer.TYPE).invoke(
        None, Integer(jint(index))
    )


def model_settings_factory(bridge_class, index):
    return bridge_class.getMethod("getFactory", Integer.TYPE).invoke(
        None, Integer(jint(index))
    )


def preset_settings_factory(bridge_class, index):
    return bridge_class.getMethod("getPresetFactory", Integer.TYPE).invoke(
        None, Integer(jint(index))
    )


def model_cell_click(bridge_class, index, view):
    bridge_class.getMethod("onCellClick", Integer.TYPE, View).invoke(
        None, Integer(jint(index)), view
    )


def model_cell_long_click(bridge_class, index, view):
    bridge_class.getMethod("onCellLongClick", Integer.TYPE, View).invoke(
        None, Integer(jint(index)), view
    )


class ModelClickCallback(dynamic_proxy(IntConsumer)):
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin

    def accept(self, index):
        plugin = self.plugin
        if plugin is not None:
            plugin._dex_on_model_click(index)

    def release(self):
        self.plugin = None


class ModelDeleteCallback(dynamic_proxy(IntConsumer)):
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin

    def accept(self, index):
        plugin = self.plugin
        if plugin is not None:
            plugin._dex_on_model_delete(index)

    def release(self):
        self.plugin = None


class PresetClickCallback(dynamic_proxy(IntConsumer)):
    def __init__(self, plugin):
        super().__init__()
        self.plugin = plugin

    def accept(self, index):
        plugin = self.plugin
        if plugin is not None:
            plugin._dex_on_preset_click(index)

    def release(self):
        self.plugin = None


class DexRuntime:
    def __init__(self, plugin, round_video_width=0, face_mask_scale=100,
                 detection_confidence=60, use_gpu=True, model_index=0):
        self.plugin = plugin
        self.round_video_width = round_video_width
        self.face_mask_scale = face_mask_scale
        self.detection_confidence = detection_confidence
        self.use_gpu = use_gpu
        self.model_kind = ("precise", "near", "far")[model_index]
        self.mask_mode = 0
        self.dex_main_class = None
        self.core_loader = None
        self.runtime_loader = None
        self.logger_proxy = None
        self.runtime_dir = None
        self.owner_token = object()

    def stage_and_start(self):
        try:
            context = ApplicationLoader.applicationContext
            get_model_settings_bridge()
            model_path = cached_model_path(context, self.model_kind)
            if model_path is None:
                self.plugin.log(
                    f"[BlurFaces] {MODEL_SPECS[self.model_kind][3]} model is not downloaded; "
                    "open plugin settings to download it"
                )
                return False
            self.plugin.log(
                f"[BlurFaces] Reusing verified cached {MODEL_SPECS[self.model_kind][3]} model: {model_path}"
            )
            registry = _runtime_registry()
            with registry["lock"]:
                _cancel_deferred_shutdown(registry)
                if registry["broken"]:
                    self.plugin.log("[BlurFaces] MediaPipe runtime requires an application restart")
                    return False
                self.runtime_dir = registry["runtime_dir"]
                if registry["main_class"] is None:
                    registry["main_class"] = registry["core_loader"].loadClass(CLASS_NAME)
                dex_class = registry["main_class"]
                parent = context.getClassLoader()
                consumer_type = parent.loadClass("java.util.function.Consumer")
                self.logger_proxy = _PythonLogger(self.plugin)
                _method(registry, dex_class, "setLogger", consumer_type).invoke(None, self.logger_proxy)
                _method(registry, dex_class, "setRoundVideoResolution", String).invoke(
                    None, str(self.round_video_width)
                )
                _method(registry, dex_class, "setFaceMaskScale", String).invoke(
                    None, str(self.face_mask_scale))
                _method(registry, dex_class, "setMaskMode", String).invoke(None, str(self.mask_mode))
                if registry["owner"] is None:
                    _method(registry, dex_class, "initAndStart", String, String, String, String).invoke(
                        None, model_path, str(self.detection_confidence),
                        "gpu" if self.use_gpu else "cpu", self.model_kind
                    )
                else:
                    # The process-global Java runtime and hooks survive the short unload/load
                    # window. Reconfigure the engine and transfer ownership without re-hooking.
                    ok = self._invoke_reconfigure(
                        registry, dex_class, model_path, self.detection_confidence,
                        self.use_gpu, self.model_kind,
                    )
                    if ok is False:
                        raise RuntimeError("hot-reload runtime reconfigure was superseded or failed")
                    # Re-apply visual settings after a model graph replacement. The
                    # Java runtime is process-global and may have been initialized by
                    # an older plugin instance.
                    _method(registry, dex_class, "setMaskMode", String).invoke(None, str(self.mask_mode))
                registry["owner"] = self.owner_token
                self.runtime_loader = registry["runtime_loader"]
                self.core_loader = registry["core_loader"]
                self.dex_main_class = dex_class
                self.plugin.log(f"[BlurFaces] Runtime reused from {self.runtime_dir}")
                return True
        except Exception as error:
            self.plugin.log(f"[BlurFaces] DEX load failed: {error}")
            self.unload()
            return False

    def set_round_video_width(self, width):
        self.round_video_width = width
        dex_class = self.dex_main_class
        if dex_class is None:
            return
        try:
            registry = _runtime_registry()
            _method(registry, dex_class, "setRoundVideoResolution", String).invoke(None, str(width))
        except Exception as error:
            plugin = self.plugin
            if plugin is not None:
                plugin.log(f"[BlurFaces] Resolution update failed: {error}")

    def set_face_mask_scale(self, scale):
        self.face_mask_scale = scale
        dex_class = self.dex_main_class
        if dex_class is None:
            return
        try:
            registry = _runtime_registry()
            _method(registry, dex_class, "setFaceMaskScale", String).invoke(None, str(scale))
        except Exception as error:
            plugin = self.plugin
            if plugin is not None:
                plugin.log(f"[BlurFaces] Face mask update failed: {error}")

    def set_mask_mode(self, mode):
        self.mask_mode = mode
        dex_class = self.dex_main_class
        if dex_class is None:
            return
        try:
            registry = _runtime_registry()
            _method(registry, dex_class, "setMaskMode", String).invoke(None, str(mode))
        except Exception as error:
            plugin = self.plugin
            if plugin is not None:
                plugin.log(f"[BlurFaces] Mask mode update failed: {error}")

    @staticmethod
    def reserve_reconfigure_request():
        registry = _runtime_registry()
        with registry["lock"]:
            registry["reconfigure_sequence"] += 1
            return registry["reconfigure_sequence"]

    def switch_model(self, model_index, detection_confidence, use_gpu, generation, request_generation=None):
        plugin = self.plugin
        dex_class = self.dex_main_class
        if plugin is not None:
            plugin.log(
                f"[BlurFaces] SWITCH_MODEL start model={model_index} confidence={detection_confidence} "
                f"gpu={use_gpu} generation={generation} request={request_generation} "
                f"runtime_class={dex_class is not None}"
            )
        if plugin is None or dex_class is None or not plugin._is_current(generation):
            if plugin is not None:
                plugin.log("[BlurFaces] SWITCH_MODEL rejected: stale generation or missing Java class")
            return
        model_kind = ("precise", "near", "far")[model_index]
        model_path = cached_model_path(ApplicationLoader.applicationContext, model_kind)
        if model_path is None:
            return
        try:
            registry = _runtime_registry()
            request = request_generation
            if request is None:
                request = self.reserve_reconfigure_request()
            switched = self._invoke_reconfigure(
                registry, dex_class, model_path, detection_confidence, use_gpu,
                model_kind, request,
            )
            if switched and plugin._is_current(generation):
                self.model_kind = model_kind
                plugin.log(f"[BlurFaces] Active model switched to {MODEL_SPECS[model_kind][3]}")
                return True
            plugin.log(f"[BlurFaces] SWITCH_MODEL returned false switched={switched}")
            return False
        except Exception as error:
            plugin.log(f"[BlurFaces] Model switch failed: {error}")
            return False

    @staticmethod
    def _invoke_reconfigure(registry, dex_class, model_path, detection_confidence,
                            use_gpu, model_kind, request=None):
        if request is None:
            registry["reconfigure_sequence"] += 1
            request = registry["reconfigure_sequence"]
        try:
            method = _method(registry, dex_class, "reconfigure", String, String,
                             String, String, String)
        except Exception:
            # Migration from the previous compatible core: its stable switchModel
            # remains usable until the process naturally adopts this release's core.
            method = _method(registry, dex_class, "switchModel", String, String,
                             String, String)
            result = method.invoke(
                None, model_path, str(detection_confidence),
                "gpu" if use_gpu else "cpu", model_kind,
            )
            return True if result is None else bool(result)
        return method.invoke(
            None, model_path, str(detection_confidence),
            "gpu" if use_gpu else "cpu", model_kind, str(request),
        )
    def unload(self, deferred=True):
        dex_class = self.dex_main_class
        logger_proxy = self.logger_proxy
        self.dex_main_class = None
        registry = _runtime_registry()
        with registry["lock"]:
            if dex_class is not None and registry["owner"] is self.owner_token:
                token = object()
                registry["shutdown_token"] = token
                def finish_shutdown():
                    with registry["lock"]:
                        if registry.get("shutdown_token") is not token:
                            return
                        registry["shutdown_timer"] = None
                        registry["shutdown_token"] = None
                        try:
                            clean = _method(registry, dex_class, "onUnload").invoke(None)
                            if clean is False:
                                registry["broken"] = True
                            registry["owner"] = None
                            _method(registry, dex_class, "clearLogger").invoke(None)
                        except Exception as error:
                            registry["broken"] = True
                            if self.plugin is not None:
                                self.plugin.log(f"[BlurFaces] DEX unload failed: {error}")
                if deferred:
                    timer = threading.Timer(1.5, finish_shutdown)
                    timer.daemon = True
                    registry["shutdown_timer"] = timer
                    timer.start()
                else:
                    finish_shutdown()
        if logger_proxy is not None:
            logger_proxy.release()
        self.logger_proxy = None
        self.core_loader = None
        self.runtime_loader = None
        self.plugin = None

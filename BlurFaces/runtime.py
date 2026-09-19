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
from java import dynamic_proxy, jarray, jbyte, jfloat, jint
from java.lang import Float, Integer, String
from java.util.function import Consumer, IntConsumer
from org.telegram.messenger import ApplicationLoader

from .asset_hashes import ASSET_HASHES


CLASS_NAME = "com.makey.blurfaces.g2.Main"
SETTINGS_BRIDGE_CLASS_NAME = "com.makey.blurfaces.g2.ModelSettingsBridge"

MODEL_FILENAME = "head_det.param"
BIN_FILENAME = "head_det.bin"

MODEL_SPECS = {
    "precise": ("offline://ncnn/head_det.param", ASSET_HASHES["model/head_det.param"], MODEL_FILENAME, "NCNN Head Detector"),
    "near": ("offline://ncnn/head_det.param", ASSET_HASHES["model/head_det.param"], MODEL_FILENAME, "NCNN Head Detector (Near)"),
    "far": ("offline://ncnn/head_det.param", ASSET_HASHES["model/head_det.param"], MODEL_FILENAME, "NCNN Head Detector (Far)"),
}

LOADER_ABI_SALT = "ncnn-native-v3"
CORE_BUNDLE_ID = hashlib.sha256((LOADER_ABI_SALT + "\0" +
    ASSET_HASHES["dex/core.dex"]).encode("ascii")).hexdigest()
RUNTIME_BUNDLE_ID = hashlib.sha256((LOADER_ABI_SALT + "\0" + "".join(
    ASSET_HASHES[name] for name in (
        "dex/core.dex",
        "jni/arm64-v8a/libblur_faces.so",
        "model/head_det.param",
        "model/head_det.bin",
    )
)).encode("ascii")).hexdigest()
_REGISTRY_KEY = "_blur_faces_ncnn_runtime_v3"
_LEGACY_REGISTRY_KEYS = (
    "_blur_faces_mediapipe_runtime_v3",
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
        os.path.join(runtime_dir, "jni", "arm64-v8a", "libblur_faces.so"):
            ASSET_HASHES["jni/arm64-v8a/libblur_faces.so"],
    }
    return all(os.path.isfile(path) and _sha256(path) == digest
               for path, digest in expected.items())


def _release_loaded_core(registry):
    dex_class = registry.get("main_class")
    if registry.get("owner") is not None and dex_class is not None:
        clean = dex_class.getMethod("onUnload").invoke(None)
        if clean is False:
            registry["broken"] = True
            raise RuntimeError("Native worker did not stop; restart the application")
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
    root = context.getDir("blur_faces_runtime_v3", 0).getCanonicalPath()
    runtime_dir = os.path.join(root, "runtime_" + RUNTIME_BUNDLE_ID)
    core_dir = os.path.join(root, "core_" + CORE_BUNDLE_ID)
    native_dir = os.path.join(runtime_dir, "jni", "arm64-v8a")
    model_dir = os.path.join(root, "models")
    os.makedirs(native_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    core_path = _stage_asset("dex/core.dex", os.path.join(core_dir, "core.dex"), True)
    so_path = _stage_asset(
        "jni/arm64-v8a/libblur_faces.so",
        os.path.join(native_dir, "libblur_faces.so"), True,
    )
    _stage_asset(
        "model/head_det.param",
        os.path.join(model_dir, "head_det.param"), True,
    )
    _stage_asset(
        "model/head_det.bin",
        os.path.join(model_dir, "head_det.bin"), True,
    )

    opt = context.getDir("blur_faces_dex_opt_v3", 0).getCanonicalPath()
    parent = context.getClassLoader()

    if (registry.get("core_loader") is not None
            and registry.get("core_bundle_id") == CORE_BUNDLE_ID):
        registry["core_load_token"] = _CORE_LOAD_TOKEN
        registry["runtime_dir"] = runtime_dir
        return

    if registry.get("core_loader") is not None:
        registry["restart_reason"] = "Blur Faces core DEX changed while loaded"
        raise RuntimeError(registry["restart_reason"] + "; restart the application")

    loader = DexClassLoader(core_path, opt, native_dir, parent)
    registry["core_loader"] = loader
    registry["runtime_loader"] = loader
    registry["core_bundle_id"] = CORE_BUNDLE_ID
    registry["runtime_bundle_id"] = RUNTIME_BUNDLE_ID
    registry["core_load_token"] = _CORE_LOAD_TOKEN
    registry["runtime_dir"] = runtime_dir

    bridge_class = loader.loadClass("com.makey.blurfaces.g2.NativeBridge")
    bridge_class.getMethod("ensureLoaded", String).invoke(None, so_path)


def cached_model_path(context, model_kind="precise"):
    root = context.getDir("blur_faces_runtime_v3", 0).getCanonicalPath()
    param_path = os.path.join(root, "models", "head_det.param")
    if os.path.isfile(param_path) and _sha256(param_path) == ASSET_HASHES["model/head_det.param"]:
        return param_path
    registry = _runtime_registry()
    with registry["lock"]:
        _ensure_runtime_loaders(context, registry)
    return param_path if os.path.isfile(param_path) else None


def is_model_downloaded(model_index):
    return True


def download_model(plugin, model_index, progress_callback=None, is_current=None):
    if is_current is not None and not is_current():
        raise InterruptedError("model download cancelled")
    context = ApplicationLoader.applicationContext
    registry = _runtime_registry()
    with registry["lock"]:
        _ensure_runtime_loaders(context, registry)
    path = cached_model_path(context)
    if progress_callback is not None:
        progress_callback(2, 1, 1)
        progress_callback(3, 1, 1)
        progress_callback(4, 1, 1)
    if plugin is not None:
        plugin.log(f"[BlurFaces] Embedded NCNN model ready: {path}")
    return path


def delete_model(plugin, model_index):
    if plugin is not None:
        plugin.log("[BlurFaces] Embedded NCNN model is built-in and cannot be deleted")
    return False


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


def runtime_restart_reason():
    return _runtime_registry().get("restart_reason")


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
                 detection_confidence=38, use_gpu=True, model_index=0):
        self.plugin = plugin
        self.round_video_width = round_video_width
        self.face_mask_scale = face_mask_scale
        self.detection_confidence = detection_confidence
        self.use_gpu = False
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
                registry = _runtime_registry()
                with registry["lock"]:
                    _ensure_runtime_loaders(context, registry)
                model_path = cached_model_path(context, self.model_kind)
            if model_path is None:
                self.plugin.log("[BlurFaces] Offline model staging failed")
                return False
            self.plugin.log(
                f"[BlurFaces] Active offline NCNN Head Detector model: {model_path}"
            )
            registry = _runtime_registry()
            with registry["lock"]:
                _cancel_deferred_shutdown(registry)
                if registry["broken"]:
                    self.plugin.log("[BlurFaces] Native runtime requires an application restart")
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
                        "cpu", self.model_kind
                    )
                else:
                    # The process-global Java runtime and hooks survive the short unload/load window.
                    ok = self._invoke_reconfigure(
                        registry, dex_class, model_path, self.detection_confidence,
                        False, self.model_kind,
                    )
                    if ok is False:
                        raise RuntimeError("hot-reload runtime reconfigure was superseded or failed")
                    _method(registry, dex_class, "setMaskMode", String).invoke(None, str(self.mask_mode))
                registry["owner"] = self.owner_token
                self.runtime_loader = registry["runtime_loader"]
                self.core_loader = registry["core_loader"]
                self.dex_main_class = dex_class
                self.plugin.log(f"[BlurFaces] Runtime armed from {self.runtime_dir}")
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

    def run_privacy_self_test(self, mask_scale=None, mask_mode=None):
        if mask_scale is None:
            mask_scale = float(self.face_mask_scale) / 100.0
        if mask_mode is None:
            mask_mode = int(self.mask_mode)
        dex_class = self.dex_main_class
        if dex_class is None:
            registry = _runtime_registry()
            dex_class = registry.get("main_class")
            if dex_class is None and registry.get("core_loader") is not None:
                try:
                    dex_class = registry["core_loader"].loadClass(CLASS_NAME)
                    registry["main_class"] = dex_class
                except Exception:
                    pass
        if dex_class is None:
            return "FAILED: Runtime core not loaded"
        try:
            registry = _runtime_registry()
            method = _method(registry, dex_class, "runPrivacySelfTest", Float.TYPE, Integer.TYPE)
            res = method.invoke(None, Float(jfloat(mask_scale)), Integer(jint(mask_mode)))
            return str(res)
        except Exception as error:
            return f"FAILED: {error}"

    @staticmethod
    def reserve_reconfigure_request():
        registry = _runtime_registry()
        with registry["lock"]:
            registry["reconfigure_sequence"] += 1
            return registry["reconfigure_sequence"]

    def switch_confidence(self, detection_confidence, generation, request_generation=None):
        return self.switch_model(0, detection_confidence, False, generation, request_generation)

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
                plugin.log(f"[BlurFaces] Active model reconfigured to {MODEL_SPECS[model_kind][3]}")
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
            return method.invoke(
                None, model_path, str(detection_confidence),
                "cpu", model_kind, str(request),
            )
        except Exception:
            method = _method(registry, dex_class, "switchModel", String, String,
                             String, String)
            result = method.invoke(
                None, model_path, str(detection_confidence),
                "cpu", model_kind,
            )
            return True if result is None else bool(result)

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

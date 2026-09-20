#!/usr/bin/env python3
"""Loader lifecycle and native NCNN + ByteTrack asset contracts."""
from pathlib import Path

root = Path(__file__).resolve().parent
plugin = (root / "BlurFaces/main.py").read_text()
runtime = (root / "BlurFaces/runtime.py").read_text()
main = (root / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text()

assert "if self.enabled:" in plugin
assert "run_on_queue(lambda: self._prepare_runtime(generation))" in plugin
assert "def _is_current(self, generation):" in plugin
assert "generation == self._generation" in plugin
assert "run_on_queue(self._stop_runtime)" in plugin
assert "self._generation += 1" in plugin
assert "ROUND_VIDEO_WIDTHS = (0, 320, 384, 448, 512)" in plugin
assert "FACE_MASK_SCALES = (65, 82, 100)" in plugin
assert "DETECTION_CONFIDENCES = (45, 35, 25)" in plugin
assert "parsed != 65 && parsed != 82 && parsed != 100" in main
assert 'key="round_video_width"' in plugin
assert "def _on_width_change(self, value):" in plugin
assert "Native head detector unavailable; host camera left untouched" in main
assert 'throw new IllegalStateException("Native head detector initialization failed"' in main
assert "MAX_TRACK_HOLD_NS = 1_200_000_000L" in main
assert "tracks.update(detections, System.nanoTime())" in main
assert "interval * 3L + 100_000_000L" in main
assert "lastSeenPublishedNanos" in main
assert "tracks.hasFreshPublication(now)" in main
assert "NativeBridge.ensureLoaded(null);" in main
assert "NativeBridge.init(paramPath, binPath);" in main
assert "NativeBridge.process(frame.rgba" in main
assert "NativeBridge.reset();" in main
assert 'assets.get(asset_name).path_str' in runtime
assert '"model/face_landmarker.task"' not in runtime
assert "if model_path is None:" in runtime
assert "return False" in runtime
assert "model_path = cached_model_path(context, self.model_kind)" in runtime
assert "def get_model_settings_bridge():" in runtime
assert "SETTINGS_BRIDGE_CLASS_NAME" in runtime
assert "os.chmod(target, 0o444)" in runtime
assert "def _ensure_runtime_loaders(context, registry):" in runtime
assert 'registry["core_loader"] = loader' in runtime
assert "os.makedirs(native_dir, exist_ok=True)" in runtime
assert 'LOADER_ABI_SALT = "ncnn-native-v3"' in runtime
assert '_REGISTRY_KEY = "_blur_faces_ncnn_runtime_v3"' in runtime
assert 'if loaded_epoch > _MODULE_EPOCH:' in runtime
assert 'if loaded_epoch < _MODULE_EPOCH:' in runtime
assert '"_blur_faces_mediapipe_runtime_v3",' in runtime
assert 'context.getDir("blur_faces_runtime_v3", 0)' in runtime
assert "dex_class.getMethod(\"onUnload\").invoke(None)" in runtime
assert "logger_proxy.release()" in runtime
assert "public static void clearLogger() { logger = null; }" in main
assert "public static void setRoundVideoResolution(String value)" in main
assert "public static void setFaceMaskScale(String value)" in main
assert 'glGetUniformLocation(program, "uMaskScale")' in main
assert '"com.exteragram.messenger.utils.system.SystemUtils"' in main
assert 'type.getDeclaredMethod("getRoundVideoResolution")' in main
assert "if (override != 0) param.setResult(override);" in main
assert 'dex_class.getMethod("clearLogger").invoke(None)' in runtime
assert '_method(registry, dex_class, "setRoundVideoResolution", String).invoke' in runtime
assert '_method(registry, dex_class, "setFaceMaskScale", String).invoke' in runtime
assert '_method(registry, dex_class, "initAndStart", String, String, String, String).invoke' in runtime
assert "def switch_model(self, model_index" in runtime
assert "public static boolean reconfigure" in main
assert "RadioCell" not in plugin
assert "SimpleSettingFactory" not in plugin
assert "ImageView" not in plugin
assert "LayoutHelper" not in plugin
assert "PyObject.fromJava" not in plugin
assert "PythonPluginsEngine" not in plugin
assert "dynamic_proxy" not in plugin
assert '"load_" + uuid.uuid4().hex' not in runtime
assert '"runtime_" + RUNTIME_BUNDLE_ID' in runtime
assert '"core_" + CORE_BUNDLE_ID' in runtime
assert 'registry["main_class"]' in runtime
assert "public static synchronized boolean onUnload()" in main
assert 'key="detection_range"' in plugin
assert "def _request_reconfigure(" in plugin
assert "def _restart_runtime" not in plugin
print("PASS: Elyx lifecycle, loader hierarchy, and native NCNN resource contracts")

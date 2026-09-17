#!/usr/bin/env python3
"""Loader lifecycle and MediaPipe asset contracts."""
from pathlib import Path

root = Path(__file__).resolve().parent
plugin = (root / "BlurFaces/main.py").read_text()
runtime = (root / "BlurFaces/runtime.py").read_text()
main = (root / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text()
settings_bridge = (root / "src/main/java/com/makey/blurfaces/g2/ModelSettingsBridge.java").read_text()
package = (root / "loader/build.py").read_text()

assert "if self.enabled:" in plugin
assert "run_on_queue(lambda: self._prepare_runtime(generation))" in plugin
assert "def _is_current(self, generation):" in plugin
assert "generation == self._generation" in plugin
assert "run_on_queue(self._stop_runtime)" in plugin
assert "self._generation += 1" in plugin
assert "ROUND_VIDEO_WIDTHS = (0, 320, 384, 448, 512)" in plugin
assert "FACE_MASK_SCALES = (65, 82, 100)" in plugin
assert "DETECTION_CONFIDENCES = (60, 50, 40)" in plugin
assert "float radiusU = (maxU - minU) * .78f + .004f;" in main
assert "float radiusV = (maxV - minV) * .78f + .004f;" in main
assert "parsed != 65 && parsed != 82 && parsed != 100" in main
assert 'key="round_video_width"' in plugin
assert "def _on_width_change(self, value):" in plugin
assert "Face Landmarker unavailable; host camera left untouched" in main
assert "never switch modes silently" in main
assert 'throw new IllegalStateException("Face Landmarker initialization failed"' in main
assert "executor.submit(active::close)" in main
assert "MAX_TRACK_HOLD_NS = 1_200_000_000L" in main
assert "tracks.update(detections, System.nanoTime())" in main
assert "interval * 3L + 100_000_000L" in main
assert "lastSeenPublishedNanos" in main
assert "tracks.hasFreshPublication(now)" in main
assert "if (image != null) image.close()" in main
assert ".setMinFaceDetectionConfidence(confidence)" in main
assert ".setMinFacePresenceConfidence(confidence)" in main
assert ".setMinTrackingConfidence(0.50f)" in main
assert 'assets.get(asset_name).path_str' in runtime
assert '"model/face_landmarker.task"' not in runtime
assert "MODEL_SHA256" in runtime
assert "if model_path is None:" in runtime
assert "return False" in runtime
assert "def download_model(plugin, model_index, progress_callback=None, is_current=None):" in runtime
assert "def delete_model(plugin, model_index):" in runtime
assert "def is_model_downloaded(model_index):" in runtime
assert "model_path = cached_model_path(context, self.model_kind)" in runtime
assert "def get_model_settings_bridge():" in runtime
assert "SETTINGS_BRIDGE_CLASS_NAME" in runtime
assert "os.chmod(target, 0o444)" in runtime
assert "def _ensure_runtime_loaders(context, registry):" in runtime
assert 'registry["runtime_loader"] = DexClassLoader(runtime_path' in runtime
assert 'registry["core_loader"] = DexClassLoader(' in runtime
runtime_loader_index = runtime.index('registry["runtime_loader"] = DexClassLoader')
assert runtime.index('_stage_asset("dex/core.dex"') < runtime_loader_index
assert runtime.index('"dex/mediapipe-runtime.dex",') < runtime_loader_index
assert runtime.index('"jni/arm64-v8a/libmediapipe_tasks_vision_jni.so",') < runtime_loader_index
loader_reuse = runtime.index('if registry["core_loader"] is not None:')
assert runtime.index('_stage_asset("dex/core.dex"') < loader_reuse
assert runtime.index('"dex/mediapipe-runtime.dex",') < loader_reuse
assert runtime.index('"jni/arm64-v8a/libmediapipe_tasks_vision_jni.so",') < loader_reuse
assert "os.makedirs(native_dir, exist_ok=True)" in runtime
assert 'LOADER_ABI_SALT = "hot-swap-core-v3"' in runtime
assert '_REGISTRY_KEY = "_blur_faces_mediapipe_runtime_v3"' in runtime
assert 'if loaded_epoch > _MODULE_EPOCH:' in runtime
assert 'if loaded_epoch < _MODULE_EPOCH:' in runtime
assert '"_blur_faces_mediapipe_runtime_v2",' in runtime
assert 'context.getDir("blur_faces_runtime_v2", 0)' in runtime
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
assert "FaceDetector.createFromOptions" in main
assert "boxesToGeometry" in main
assert "blaze_face_short_range" in runtime
assert "blaze_face_full_range" in runtime
assert "box.width() * 1.50f" in main
assert "box.height() * 1.65f" in main
assert 'self.set_setting("model", index)' in plugin
assert "runtime.switch_model(" in plugin
assert "run_on_queue(lambda: self._prepare_runtime(generation))" in plugin
assert "if runtime is None:" in plugin
assert "def switch_model(self, model_index" in runtime
assert "public static boolean reconfigure" in main
assert "self._downloading_models = set()" in plugin
assert "threading.Thread(" in plugin
assert "DownloadState[] downloadStates" in settings_bridge
assert "RadioCell" not in plugin
assert "SimpleSettingFactory" not in plugin
assert "ImageView" not in plugin
assert "LayoutHelper" not in plugin
assert "_model_factories" not in plugin
assert "public final class ModelSettingsBridge" in settings_bridge
assert "ModelRadioCell extends FrameLayout" in settings_bridge
assert "new RadioButton(context)" in settings_bridge
assert "radioButton.setSize(AndroidUtilities.dp(20))" in settings_bridge
assert "AndroidUtilities.dp(50)" in settings_bridge
assert "AndroidUtilities.dp(60)" in settings_bridge
assert "extends CustomSetting.Factory<ModelRadioCell>" in settings_bridge
assert "class PreciseFactory extends ModelFactory" in settings_bridge
assert "class NearFactory extends ModelFactory" in settings_bridge
assert "class FarFactory extends ModelFactory" in settings_bridge
assert settings_bridge.count("for (ModelFactory factory : FACTORIES)") == 1
assert "public UItem create(Plugin plugin, CustomSetting setting, PyObject args)" in settings_bridge
assert "setClickableValue(true)" in settings_bridge
assert "setShadowValue(false)" in settings_bridge
assert "private static volatile IntConsumer clickCallback;" in settings_bridge
assert "private static volatile IntConsumer deleteCallback;" in settings_bridge
assert "try { target.accept(index); }" in settings_bridge
assert "private static void dispatchClick" in settings_bridge
assert "private static void dispatchDelete" in settings_bridge
assert "PyObject.fromJava" not in plugin
assert "PythonPluginsEngine" not in plugin
assert "PythonPluginsEngine" not in settings_bridge
assert "class ModelClickCallback(dynamic_proxy(IntConsumer))" in runtime
assert "class ModelDeleteCallback(dynamic_proxy(IntConsumer))" in runtime
assert "self._model_click_callback = click_callback" in plugin
assert "self._model_delete_callback = delete_callback" in plugin
assert 'bridge.getMethod("detach", IntConsumer, IntConsumer)' in plugin
assert "clickCallback == expectedClickCallback" in settings_bridge
assert "deleteCallback == expectedDeleteCallback" in settings_bridge
assert "static Context" not in settings_bridge
assert "static View" not in settings_bridge
assert "ItemOptions.makeOptions(fragment, row)" in settings_bridge
assert ".setBlur(true)" in settings_bridge
assert ".setDrawScrim(true)" in settings_bridge
assert ".forceBelowScrim(true)" in settings_bridge
assert ".forceBottom(false)" in settings_bridge
assert ".setGravity(Gravity.LEFT)" in settings_bridge
assert ".allowMoveScrim()" not in settings_bridge
assert ".hideScrimUnder()" not in settings_bridge
assert ".forceBottom(true)" not in settings_bridge
assert "recyclerListView.getClipBackground(row)" in settings_bridge
assert ".setScrimViewBackground(scrimBackground)" in settings_bridge
assert "row.setBackground(null)" in settings_bridge
assert ".setOnDismiss(restoreSelector)" in settings_bridge
assert "View deleteButton = options.getLastView()" in settings_bridge
assert "Theme.key_windowBackgroundWhite, fragment.getResourceProvider()" in settings_bridge
assert "state.deleteLabel.isEmpty()" in settings_bridge
assert 'strings.get("model_delete")' in plugin
assert "String nearLabel, String farLabel, String deleteLabel" in settings_bridge
assert "final String deleteLabel;" in settings_bridge
assert "LaunchActivity.getSafeLastFragment()" in settings_bridge
assert "createActionMode()" not in plugin
assert "showActionMode()" not in plugin
assert "dynamic_proxy" not in plugin
assert "extends RadioCell" not in settings_bridge
assert "import org.telegram.ui.Cells.RadioCell" not in settings_bridge
assert "getChildAt" not in settings_bridge
assert "msg_camera" not in settings_bridge
assert "deleteSelection" not in settings_bridge
assert "_dex_on_model_long_click" not in plugin
assert "_delete_model_index" not in plugin
assert "LayoutHelper.MATCH_PARENT" not in plugin
assert "def _download_and_activate(self, index, download_generation):" in plugin
assert "def _dex_on_model_delete(self, index):" in plugin
assert "if not self._is_model_ready(index):" in plugin
assert plugin.count("download_model(") == 1
assert "if ((state.downloadedMask & (1 << index)) == 0) return;" in settings_bridge
assert "cancelClickRunnables(true)" in settings_bridge
assert "setMinimumHeight(AndroidUtilities.dp(50))" in settings_bridge
assert "radioButton.setClickable(false)" in settings_bridge
assert "downloadIcon.setClickable(false)" in settings_bridge
assert "int controlCenter = width - AndroidUtilities.dp(31);" in settings_bridge
assert "int downloadCenter = controlCenter;" in settings_bridge
assert "resourcesProvider),\n                    2));" in settings_bridge
assert "LocaleController.isRTL\n                    ? width - AndroidUtilities.dp(31)" not in settings_bridge
assert "float start = AndroidUtilities.dp(24);" in settings_bridge
assert "textView.layout(AndroidUtilities.dp(24), 0," in settings_bridge
assert "deleting_active = index == self.model_index" in plugin
assert "if self.model_index == index and replacement is not None:" in plugin
assert 'self.set_setting("model", replacement)' in plugin
assert "if deleting_active and replacement is not None and self.enabled:" in plugin
assert "download_model(" not in plugin[plugin.index("def create_settings"):plugin.index("def _on_model_click")]
assert '"load_" + uuid.uuid4().hex' not in runtime
assert '"runtime_" + RUNTIME_BUNDLE_ID' in runtime
assert '"core_" + CORE_BUNDLE_ID' in runtime
assert 'registry["main_class"]' in runtime
assert 'registry["settings_bridge_class"]' in runtime
assert "public static synchronized boolean onUnload()" in main
assert 'key="detection_range"' in plugin
assert "Keep the settings structure stable while models download" in plugin
assert '"gpu" if self.use_gpu else "cpu"' in runtime
assert "def _request_reconfigure(" in plugin
assert "def _restart_runtime" not in plugin
print("PASS: Elyx lifecycle, loader hierarchy, and explicit GPU/CPU resource contracts")

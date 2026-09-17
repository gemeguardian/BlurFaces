#!/usr/bin/env python3
import ast
import hashlib
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TREE = ROOT / "BlurFaces"
MANDATORY = {
    "dex/core.dex": ROOT / "build/dex/core.dex",
    "dex/mediapipe-runtime.dex": ROOT / "build/dex/mesh-runtime.dex",
    "jni/arm64-v8a/libmediapipe_tasks_vision_jni.so": ROOT / "build/mesh-runtime/libmediapipe_tasks_vision_jni.so",
}
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
LITE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
FAR_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite"


def digest(data):
    return hashlib.sha256(data).hexdigest()


meta = (TREE / "metainfo.yml").read_text(encoding="utf-8")
assert "id: blur_faces" in meta
assert 'version: "2.3.0"' in meta
assert 'author: "@gemeguardian"' in meta
assert 'requirements: ""' in meta
refmap = (ROOT / "refmap.yml").read_text(encoding="utf-8")
assert "assets: BlurFaces/assets" in refmap

python_files = list(TREE.glob("*.py"))
classes = []
all_urls = []
for source_path in python_files:
    source = source_path.read_text(encoding="utf-8")
    ast.parse(source, filename=str(source_path))
    assert "http://" not in source
    all_urls.extend(re.findall(r'https://[^"\s]+', source))
    tree = ast.parse(source)
    classes.extend(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "BasePlugin" for base in node.bases)
    )
assert len(classes) == 1
assert sorted(all_urls) == sorted([MODEL_URL, LITE_MODEL_URL, FAR_MODEL_URL])
main_source = (TREE / "main.py").read_text(encoding="utf-8")
runtime_source = (TREE / "runtime.py").read_text(encoding="utf-8")
bridge_source = (ROOT / "src/main/java/com/makey/blurfaces/g2/ModelSettingsBridge.java").read_text(
    encoding="utf-8"
)
main_tree = ast.parse(main_source)
selector_calls = [
    node for node in ast.walk(main_tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "Selector"
]
assert len(selector_calls) == 5
assert all("subtext" not in {keyword.arg for keyword in call.keywords} for call in selector_calls)
assert 'key="detection_range"' in main_source
assert "Keep the settings structure stable while models download" in main_source
assert "rows.extend([" in main_source
assert 'Divider(text=strings.get("settings_width_subtext"))' in main_source
assert "DexRuntime" in main_source
assert "from .asset_hashes import ASSET_HASHES" in runtime_source
assert re.findall(r'https://[^"\s]+', runtime_source) == [MODEL_URL, LITE_MODEL_URL, FAR_MODEL_URL]
assert "http://" not in runtime_source
assert f'MODEL_SHA256 = "{MODEL_SHA256}"' in runtime_source
assert 'context.getDir("blur_faces_models", 0)' in runtime_source
assert "connection.setConnectTimeout(CONNECT_TIMEOUT_MS)" in runtime_source
assert "connection.setReadTimeout(READ_TIMEOUT_MS)" in runtime_source
assert "os.fsync(output.fileno())" in runtime_source
assert "os.replace(temporary, model_path)" in runtime_source
assert "os.chmod(model_path, 0o444)" in runtime_source
assert "if actual == model_sha256:" in runtime_source
assert "model_path = cached_model_path(context, self.model_kind)" in runtime_source
assert "if model_path is None:" in runtime_source
assert "open plugin settings to download it" in runtime_source
assert "def get_model_settings_bridge():" in runtime_source
assert "get_model_settings_bridge()" in runtime_source
assert 'dex_class.getMethod("clearLogger").invoke(None)' in runtime_source
assert 'getMethod("setLogger", consumer_type).invoke(None, None)' not in runtime_source
assert "def download_model(plugin, model_index, progress_callback=None, is_current=None):" in runtime_source
assert "def delete_model(plugin, model_index):" in runtime_source
assert "ModelSettingsBridge" in bridge_source
assert "ModelRadioCell extends FrameLayout" in bridge_source
assert "CustomSetting.Factory<ModelRadioCell>" in bridge_source
assert "SimpleSettingFactory" not in main_source
assert "RadioCell" not in main_source
assert "ImageView" not in main_source
assert "LayoutHelper" not in main_source
assert "_model_factories" not in main_source
assert "private static volatile IntConsumer clickCallback;" in bridge_source
assert "private static volatile IntConsumer deleteCallback;" in bridge_source
assert "try { target.accept(index); }" in bridge_source
assert "PyObject.fromJava" not in main_source
assert "PythonPluginsEngine" not in main_source
assert "PythonPluginsEngine" not in bridge_source
assert "class ModelClickCallback(dynamic_proxy(IntConsumer))" in runtime_source
assert "class ModelDeleteCallback(dynamic_proxy(IntConsumer))" in runtime_source
assert 'bridge.getMethod("detach", IntConsumer, IntConsumer)' in main_source
assert "clickCallback == expectedClickCallback" in bridge_source
assert "deleteCallback == expectedDeleteCallback" in bridge_source
assert "new PreciseFactory(), new NearFactory(), new FarFactory()," in bridge_source
assert bridge_source.count("for (ModelFactory factory : FACTORIES)") == 1
assert "setClickableValue(true)" in bridge_source
assert "setShadowValue(false)" in bridge_source
assert "new RadioButton(context)" in bridge_source
assert "radioButton.setVisibility(downloaded ? View.VISIBLE : View.GONE)" in bridge_source
assert "downloadIcon.setVisibility(downloaded ? View.GONE : View.VISIBLE)" in bridge_source
assert "AndroidUtilities.dp(50)" in bridge_source
assert "AndroidUtilities.dp(60)" in bridge_source
assert "int controlCenter = width - AndroidUtilities.dp(31);" in bridge_source
assert "int downloadCenter = controlCenter;" in bridge_source
assert "resourcesProvider),\n                    2));" in bridge_source
assert "LocaleController.isRTL\n                    ? width - AndroidUtilities.dp(31)" not in bridge_source
assert "textView.layout(AndroidUtilities.dp(24), 0," in bridge_source
assert "float start = AndroidUtilities.dp(24);" in bridge_source
assert "Theme.dividerPaint" in bridge_source
assert "ItemOptions.makeOptions(fragment, row)" in bridge_source
assert ".setBlur(true)" in bridge_source
assert ".setDrawScrim(true)" in bridge_source
assert ".forceBelowScrim(true)" in bridge_source
assert ".forceBottom(false)" in bridge_source
assert ".setGravity(Gravity.LEFT)" in bridge_source
assert ".allowMoveScrim()" not in bridge_source
assert ".hideScrimUnder()" not in bridge_source
assert ".forceBottom(true)" not in bridge_source
assert "recyclerListView.getClipBackground(row)" in bridge_source
assert ".setScrimViewBackground(scrimBackground)" in bridge_source
assert "row.setBackground(null)" in bridge_source
assert ".setOnDismiss(restoreSelector)" in bridge_source
assert "View deleteButton = options.getLastView()" in bridge_source
assert "Theme.key_windowBackgroundWhite, fragment.getResourceProvider()" in bridge_source
assert 'strings.get("model_delete")' in main_source
assert "String nearLabel, String farLabel, String deleteLabel" in bridge_source
assert "final String deleteLabel;" in bridge_source
assert "extends RadioCell" not in bridge_source
assert "import org.telegram.ui.Cells.RadioCell" not in bridge_source
assert "getChildAt" not in bridge_source
assert "msg_camera" not in bridge_source
assert "createActionMode()" not in main_source
assert "dynamic_proxy" not in main_source
assert "_dex_on_model_long_click" not in main_source
assert "_delete_model_index" not in main_source
assert main_source.count("download_model(") == 1
assert "if ((state.downloadedMask & (1 << index)) == 0) return;" in bridge_source
assert "cancelClickRunnables(true)" in bridge_source
assert "setMinimumHeight(AndroidUtilities.dp(50))" in bridge_source
assert "radioButton.setClickable(false)" in bridge_source
assert "downloadIcon.setClickable(false)" in bridge_source
assert "deleting_active = index == self.model_index" in main_source
assert "if self.model_index == index and replacement is not None:" in main_source
assert 'self.set_setting("model", replacement)' in main_source
assert "if deleting_active and replacement is not None and self.enabled:" in main_source
assert "download_model(" not in main_source[
    main_source.index("def create_settings"):main_source.index("def _on_model_click")
]
assert '"load_" + uuid.uuid4().hex' not in runtime_source
assert '"runtime_" + RUNTIME_BUNDLE_ID' in runtime_source
assert '"core_" + CORE_BUNDLE_ID' in runtime_source
assert 'LOADER_ABI_SALT = "hot-swap-core-v3"' in runtime_source
assert '_REGISTRY_KEY = "_blur_faces_mediapipe_runtime_v3"' in runtime_source
assert 'if loaded_epoch > _MODULE_EPOCH:' in runtime_source
assert '"_blur_faces_mediapipe_runtime_v2",' in runtime_source
loader_index = runtime_source.index('registry["runtime_loader"] = DexClassLoader')
assert runtime_source.index('_stage_asset("dex/core.dex"') < loader_index
assert runtime_source.index('"dex/mediapipe-runtime.dex",') < loader_index
assert runtime_source.index('"jni/arm64-v8a/libmediapipe_tasks_vision_jni.so",') < loader_index
early_return_index = runtime_source.index('if registry["core_loader"] is not None:')
assert runtime_source.index('_stage_asset("dex/core.dex"') < early_return_index
assert runtime_source.index('"dex/mediapipe-runtime.dex",') < early_return_index
assert runtime_source.index('"jni/arm64-v8a/libmediapipe_tasks_vision_jni.so",') < early_return_index
assert "os.makedirs(native_dir, exist_ok=True)" in runtime_source
assert "get_model_settings_bridge" in main_source
assert "def _release_loaded_core(registry):" in runtime_source
assert 'registry["settings_bridge_class"] = None' in runtime_source
assert 'registry["core_loader"] = None' in runtime_source
assert 'registry["core_load_token"] = _CORE_LOAD_TOKEN' in runtime_source
assert 'registry["runtime_loader"] = DexClassLoader(runtime_path' in runtime_source

hash_namespace = {}
exec((TREE / "asset_hashes.py").read_text(encoding="ascii"), hash_namespace)
expected_hashes = hash_namespace["ASSET_HASHES"]
assert set(expected_hashes) == set(MANDATORY)
assert len(MANDATORY) == 3
for name, generated in MANDATORY.items():
    bundled = TREE / "assets" / name
    assert bundled.read_bytes() == generated.read_bytes()
    assert digest(bundled.read_bytes()) == expected_hashes[name]

artifacts = sorted((ROOT / "builds").glob("*.elyx"), key=lambda path: path.stat().st_mtime_ns)
assert artifacts, "ElyxBuilder did not produce builds/*.elyx"
artifact = artifacts[-1]
with zipfile.ZipFile(artifact) as archive:
    entries = set(archive.namelist())
    assert "refmap.yml" in entries
    assert "BlurFaces/main.py" in entries
    assert "BlurFaces/metainfo.yml" in entries
    assert not any(".gradle" in name or name.startswith("build/") or name.startswith("src/") for name in entries)
    assert not any(name.endswith("face_landmarker.task") or "/model/" in name for name in entries)
    binary_entries = {
        name for name in entries
        if name.endswith((".dex", ".so", ".task"))
    }
    assert binary_entries == {"BlurFaces/assets/" + name for name in MANDATORY}
    archive_urls = {}
    for name in entries:
        if not name.endswith((".py", ".yml", ".yaml", ".json", ".md", ".txt")):
            continue
        urls = re.findall(r'https://[^"\s]+', archive.read(name).decode("utf-8"))
        if urls:
            archive_urls[name] = urls
    assert archive_urls == {"BlurFaces/runtime.py": [MODEL_URL, LITE_MODEL_URL, FAR_MODEL_URL]}
    for name, generated in MANDATORY.items():
        entry = "BlurFaces/assets/" + name
        payload = archive.read(entry)
        assert payload == generated.read_bytes()
        assert digest(payload) == expected_hashes[name]
print(f"PASS: archive has three offline runtime payloads and downloads only the verified Google model: {artifact}")

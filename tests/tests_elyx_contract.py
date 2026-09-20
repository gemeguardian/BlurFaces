#!/usr/bin/env python3
import ast
import hashlib
import re
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TREE = ROOT / "BlurFaces"
MANDATORY = {
    "dex/core.dex": ROOT / "build/dex/core.dex",
    "jni/arm64-v8a/libblur_faces.so": ROOT / "libs/arm64-v8a/libblur_faces.so",
    "model/head_det.param": ROOT / "models/head_det.param",
    "model/head_det.bin": ROOT / "models/head_det.bin",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


meta = (TREE / "metainfo.yml").read_text(encoding="utf-8")
assert "id: blur_faces" in meta
assert 'version: "3.0.0"' in meta
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
assert not any("storage.googleapis.com" in url for url in all_urls)

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
assert len(selector_calls) == 3
assert all("subtext" not in {keyword.arg for keyword in call.keywords} for call in selector_calls)
assert 'key="detection_range"' in main_source
assert "DexRuntime" in main_source
assert "from .asset_hashes import ASSET_HASHES" in runtime_source
assert "http://" not in runtime_source
assert 'context.getDir("blur_faces_runtime_v3", 0)' in runtime_source
assert 'dex_class.getMethod("clearLogger").invoke(None)' in runtime_source
assert 'getMethod("setLogger", consumer_type).invoke(None, None)' not in runtime_source
assert "download_model" not in runtime_source
assert "delete_model" not in runtime_source
assert "is_model_downloaded" not in runtime_source
assert "switch_model" not in runtime_source
assert "ModelSettingsBridge" in bridge_source
assert "ModelRadioCell extends FrameLayout" in bridge_source
assert "CustomSetting.Factory<ModelRadioCell>" in bridge_source
assert "SimpleSettingFactory" not in main_source
assert "RadioCell" not in main_source
assert "ImageView" not in main_source
assert "LayoutHelper" not in main_source
assert "_model_factories" not in main_source
assert "PyObject.fromJava" not in main_source
assert "PythonPluginsEngine" not in main_source
assert "PythonPluginsEngine" not in bridge_source
assert "extends RadioCell" not in bridge_source
assert "import org.telegram.ui.Cells.RadioCell" not in bridge_source
assert "getChildAt" not in bridge_source
assert "msg_camera" not in bridge_source
assert "createActionMode()" not in main_source
assert "dynamic_proxy" not in main_source
assert '"load_" + uuid.uuid4().hex' not in runtime_source
assert '"runtime_" + RUNTIME_BUNDLE_ID' in runtime_source
assert '"core_" + CORE_BUNDLE_ID' in runtime_source
assert 'LOADER_ABI_SALT = "ncnn-native-v3"' in runtime_source
assert '_REGISTRY_KEY = "_blur_faces_ncnn_runtime_v3"' in runtime_source
assert 'if loaded_epoch > _MODULE_EPOCH:' in runtime_source
assert "_blur_faces_mediapipe_runtime" not in runtime_source
assert "os.makedirs(native_dir, exist_ok=True)" in runtime_source
assert "def _release_loaded_core(registry):" in runtime_source
assert 'registry["core_loader"] = None' in runtime_source
assert 'registry["core_load_token"] = _CORE_LOAD_TOKEN' in runtime_source

hash_namespace = {}
exec((TREE / "asset_hashes.py").read_text(encoding="ascii"), hash_namespace)
expected_hashes = hash_namespace["ASSET_HASHES"]
assert set(expected_hashes) == set(MANDATORY)
assert len(MANDATORY) == 4
for name, generated in MANDATORY.items():
    bundled = TREE / "assets" / name
    assert bundled.read_bytes() == generated.read_bytes()
    assert digest(bundled.read_bytes()) == expected_hashes[name]

artifacts = sorted((ROOT / "builds").glob("blur_faces-3.0.0*.elyx"), key=lambda path: path.stat().st_mtime_ns)
if artifacts:
    artifact = artifacts[-1]
    with zipfile.ZipFile(artifact) as archive:
        entries = set(archive.namelist())
        assert "refmap.yml" in entries
        assert "BlurFaces/main.py" in entries
        assert "BlurFaces/metainfo.yml" in entries
        assert not any(".gradle" in name or name.startswith("build/") or name.startswith("src/") for name in entries)
        binary_entries = {
            name for name in entries
            if name.endswith((".dex", ".so", ".param", ".bin"))
        }
        assert binary_entries == {"BlurFaces/assets/" + name for name in MANDATORY}
        for name, generated in MANDATORY.items():
            entry = "BlurFaces/assets/" + name
            payload = archive.read(entry)
            assert payload == generated.read_bytes()
            assert digest(payload) == expected_hashes[name]
    print(f"PASS: archive has offline native runtime and bundled NCNN head detector: {artifact}")
else:
    print("PASS: static checks passed; ready for ElyxBuilder packaging")

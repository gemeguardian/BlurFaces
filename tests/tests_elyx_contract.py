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
assert 'version: "1.0.0"' in meta
assert 'author: "@gemeguardian"' in meta
assert 'requirements: ""' in meta
refmap = (ROOT / "refmap.yml").read_text(encoding="utf-8")
assert "assets: BlurFaces/assets" in refmap

NETWORK_MODULES = {"urllib", "http", "socket", "ssl", "requests", "httpx", "aiohttp", "ftplib"}

python_files = list(TREE.glob("*.py"))
classes = []
for source_path in python_files:
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    # Zero-network invariant: no network stack imports and no remote endpoints.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots = {node.module.split(".")[0]}
        else:
            continue
        assert not roots & NETWORK_MODULES, f"{source_path.name} imports {roots & NETWORK_MODULES}"
    assert not re.findall(r'https?://[^"\s]+', source), f"{source_path.name} contains a remote URL"
    classes.extend(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(isinstance(base, ast.Name) and base.id == "BasePlugin" for base in node.bases)
    )
assert len(classes) == 1, "exactly one plugin entry point"

JAVA_NETWORK = re.compile(r"^\s*import\s+(java\.net|javax\.net|okhttp3|android\.net\.http)\b", re.M)
for java_path in (ROOT / "src/main/java").rglob("*.java"):
    assert not JAVA_NETWORK.search(java_path.read_text(encoding="utf-8")), f"{java_path.name} imports networking"

hash_namespace = {}
exec((TREE / "asset_hashes.py").read_text(encoding="ascii"), hash_namespace)
expected_hashes = hash_namespace["ASSET_HASHES"]
assert set(expected_hashes) == set(MANDATORY)
assert len(MANDATORY) == 4
for name, generated in MANDATORY.items():
    bundled = TREE / "assets" / name
    assert bundled.read_bytes() == generated.read_bytes()
    assert digest(bundled.read_bytes()) == expected_hashes[name]

artifacts = sorted((ROOT / "builds").glob("blur_faces-1.0.0*.elyx"), key=lambda path: path.stat().st_mtime_ns)
assert artifacts, "Build the installable .elyx before running the packaging contract"
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
        for source in python_files:
            assert archive.read("BlurFaces/" + source.name) == source.read_bytes()
    print(f"PASS: archive contains current Python, DEX, ELF and both hashed offline NCNN model files: {artifact}")

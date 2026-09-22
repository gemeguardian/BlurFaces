"""Execute production Python staging/UI code with only Android SDK boundaries replaced."""
import ast
import hashlib
import os
import re
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parent.parent


def functions_from(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class RuntimeBehaviorTest(unittest.TestCase):
    def test_offline_pair_staging_integrity_repair_and_missing_asset(self):
        build = ROOT / "build" / "python-tests"
        build.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build) as directory:
            root = Path(directory)
            source = root / "assets" / "model"
            source.mkdir(parents=True)
            payloads = {"head_det.param": b"7767517\nmodel", "head_det.bin": b"test weights"}
            for name, payload in payloads.items():
                (source / name).write_bytes(payload)
            hashes = {"model/" + name: hashlib.sha256(payload).hexdigest()
                      for name, payload in payloads.items()}
            namespace = functions_from(ROOT / "BlurFaces/runtime.py",
                {"_sha256", "_stage_asset", "cached_model_path"}, {
                    "hashlib": hashlib, "os": os, "shutil": shutil, "tempfile": tempfile,
                    "assets": SimpleNamespace(get=lambda name: SimpleNamespace(
                        path_str=str(root / "assets" / name))),
                    "ASSET_HASHES": hashes, "MODEL_FILENAME": "head_det.param",
                    "BIN_FILENAME": "head_det.bin", "RUNTIME_BUNDLE_ID": "test-bundle",
                })
            context = SimpleNamespace(getDir=lambda *_: SimpleNamespace(
                getCanonicalPath=lambda: str(root / "private")))
            stage = namespace["cached_model_path"]
            param = Path(stage(context))
            weights = param.with_suffix(".bin")
            self.assertNotEqual(param.parent, source)
            self.assertEqual(param.read_bytes(), payloads[param.name])
            self.assertEqual(weights.read_bytes(), payloads[weights.name])
            self.assertEqual(weights.stat().st_mode & 0o777, 0o444)
            original_time = weights.stat().st_mtime_ns
            self.assertEqual(stage(context), str(param))
            self.assertEqual(weights.stat().st_mtime_ns, original_time)
            weights.chmod(0o644)
            weights.write_bytes(b"corrupt cache")
            stage(context)
            self.assertEqual(weights.read_bytes(), payloads[weights.name])
            # Even a good cache cannot hide a corrupt or missing bundled model.
            (source / weights.name).write_bytes(b"corrupt bundle")
            with self.assertRaisesRegex(ValueError, "Bundled asset SHA-256 mismatch"):
                stage(context)
            (source / weights.name).unlink()
            with self.assertRaises(FileNotFoundError):
                stage(context)
            self.assertEqual(sorted(p.name for p in param.parent.iterdir()),
                             ["head_det.bin", "head_det.param"])

    def test_legacy_cleanup_only_removes_plugin_capture_files(self):
        build = ROOT / "build" / "python-tests"
        build.mkdir(parents=True, exist_ok=True)
        cleanup = functions_from(ROOT / "BlurFaces/runtime.py",
            {"_purge_legacy_debug_frames"}, {"os": os, "re": re})["_purge_legacy_debug_frames"]
        with tempfile.TemporaryDirectory(dir=build) as directory:
            root = Path(directory)
            captures = root / "blurfaces_storage"
            captures.mkdir()
            (captures / "frame_123_c1_s90_y50.jpg").write_bytes(b"old capture")
            (captures / "frame_124_c-1_s0_y0.jpg").write_bytes(b"old error capture")
            (captures / "keep.jpg").write_bytes(b"unrelated")
            (captures / "frame_125_c0_s0_y0.jpg").symlink_to(captures / "keep.jpg")
            cleanup(str(captures))
            self.assertEqual(sorted(p.name for p in captures.iterdir()),
                             ["frame_125_c0_s0_y0.jpg", "keep.jpg"])
            self.assertEqual((captures / "keep.jpg").read_bytes(), b"unrelated")
            empty = root / "empty"
            empty.mkdir()
            cleanup(str(empty))
            self.assertFalse(empty.exists())
            cleanup(str(empty))

    def test_self_test_callback_uses_current_fixed_mask_scale(self):
        path = ROOT / "BlurFaces/main.py"
        tree = ast.parse(path.read_text())
        # Keep the actual plugin class/constants; Android imports are unavailable
        # on the host. No plugin method under test is mocked or reimplemented.
        tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
        namespace = {"BasePlugin": type("BasePlugin", (), {"log": lambda *_: None})}
        exec(compile(tree, str(path), "exec"), namespace)
        plugin = namespace["BlurFacesPlugin"]()
        calls = []
        messages = []
        plugin.dex_loader = SimpleNamespace(run_privacy_self_test=lambda *args:
            calls.append(args) or "PASSED: synthetic-only")
        plugin.show_toast = messages.append
        plugin._on_self_test_click()
        self.assertEqual(calls, [(1.0, 0)])
        self.assertEqual(messages, ["PASSED: synthetic-only"])


if __name__ == "__main__":
    unittest.main()

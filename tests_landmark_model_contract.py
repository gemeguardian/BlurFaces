#!/usr/bin/env python3
"""Loader lifecycle and MediaPipe asset contracts."""
from pathlib import Path

root = Path(__file__).resolve().parent
plugin = (root / "loader/plugin.py").read_text()
utils = (root / "loader/utils.py").read_text()
main = (root / "src/main/java/com/makey/blurfaces/Main.java").read_text()
package = (root / "loader/build.py").read_text()
model = root / "model/face_landmarker.task"

assert model.is_file() and model.stat().st_size > 1_000_000
assert "if self.enabled:" in plugin
assert "run_on_queue(lambda: self._prepare_runtime(generation))" in plugin
assert "def _is_current(self, generation):" in plugin
assert "generation == self._generation" in plugin
assert "run_on_queue(self._stop_runtime)" in plugin
assert "self._generation += 1" in plugin
assert "GPU Face Landmarker unavailable; host camera left untouched" in main
assert "No CPU retry is permitted" in main
assert 'throw new IllegalStateException("GPU Face Landmarker initialization failed"' in main
assert "active.close()" in main
assert "input.close()" in main and "submitted.image.close()" in main
assert "prepare_native_load_copy(native_path, cache_dir)" in plugin
assert '"libmediapipe_tasks_vision_jni.so"' in utils
assert '"mediapipe_load_" + uuid.uuid4().hex' in utils
assert package.count("?sha256={sha256(") == 3
assert 'EMBEDDED_DEX_SHA256 = "{sha256(CORE_DEX)}"' in package
assert "MP_RUNTIME_DEX_SHA256" in package
assert "MP_DEX_SHA256" not in package
assert "actual_core_sha256 != EMBEDDED_DEX_SHA256" in (root / "loader/dex.py").read_text()
print("PASS: disabled/generation lifecycle and no-fallback GPU resource contracts")

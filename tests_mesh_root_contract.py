#!/usr/bin/env python3
"""Source contracts for the sole MediaPipe geometry pipeline."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "src/main/java/com/makey/blurfaces/Main.java").read_text()
loader = (root / "loader/plugin.py").read_text()
package = (root / "loader/build.py").read_text()

assert ".setDelegate(Delegate.GPU)" in source
assert ".setRunningMode(RunningMode.LIVE_STREAM)" in source
assert ".setResultListener(Main::onLandmarkerResult)" in source
assert "landmarker.detectAsync(image, frame.timestampMs)" in source
assert "Delegate.CPU" not in source
assert "NativeBridge" not in source
assert "SCRFD" not in source
assert "anchorsToAffine" not in source
assert "ovalPcaAffine" in source
assert "String source = currentSourceKey(thread)" in source
assert "activateSource(state, source, now)" in source
assert "FaceGeometry geometry = geometryFor(source, now)" in source
assert "SOURCE_BY_TEXTURE.get(textureId)" in source
assert "if (source == null) source = SOURCE_BY_SLOT.get(slot)" in source
assert "FaceGeometry geometry = geometryFor(source" in source
assert "\nNATIVE_URL =" not in package
assert "\nMODEL_PARAM_URL =" not in package
assert "MP_MODEL_URL" in package and "MP_NATIVE_URL" in package and "MP_RUNTIME_DEX_URL" in package
assert "loader/native.py" not in package
assert "download_file(MP_NATIVE_URL" in loader
print("PASS: GPU LIVE_STREAM dense mesh is the only preview/encoder geometry source")

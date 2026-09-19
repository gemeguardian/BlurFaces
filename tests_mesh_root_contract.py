#!/usr/bin/env python3
"""Source contracts for the native NCNN + ByteTrack geometry pipeline."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text()
loader = (root / "BlurFaces/main.py").read_text()
runtime = (root / "BlurFaces/runtime.py").read_text()

assert "NativeBridge.ensureLoaded" in source
assert "NativeBridge.init" in source
assert "NativeBridge.process" in source
assert "NativeBridge.reset" in source
assert "String source = currentSourceKey(thread)" in source
assert "activateSource(state, source, now)" in source
assert "FaceGeometry geometry = geometryFor(source, now)" in source
assert "SOURCE_BY_TEXTURE.get(textureId)" in source
assert "if (source == null) source = SOURCE_BY_SLOT.get(slot)" in source
assert "FaceGeometry geometry = geometryFor(source" in source
assert "java.net" not in loader
assert "storage.googleapis.com" not in runtime
assert "MODEL_URL" not in loader
assert "from elyx import assets" in runtime
print("PASS: native NCNN + ByteTrack runtime is the verified geometry source")

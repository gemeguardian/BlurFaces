#!/usr/bin/env python3
"""Source contracts for the sole MediaPipe geometry pipeline."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "src/main/java/com/makey/blurfaces/g2/Main.java").read_text()
loader = (root / "BlurFaces/main.py").read_text()
runtime = (root / "BlurFaces/runtime.py").read_text()

assert ".setDelegate(useGpu ? Delegate.GPU : Delegate.CPU)" in source
assert "Delegate.CPU" in source
assert ".setRunningMode(RunningMode.VIDEO)" in source
assert "new ByteBufferImageBuilder" in source
assert "landmarker.detectForVideo(image, frame.timestampMs)" in source
assert "detector.detectForVideo(image, frame.timestampMs)" in source
assert "Bitmap.createBitmap" not in source
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
assert "java.net" not in loader
assert "from java.net import URL" in runtime
assert "MODEL_URL" not in loader
assert "DexClassLoader" not in loader
assert "from elyx import assets" in runtime
print("PASS: bundled thread-affine VIDEO runtime remains the only geometry source")

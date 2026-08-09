#!/usr/bin/env python3
"""Contract tests for the stable-root plus optional dense-mesh design.

The 5-point privacy basis must continue to be the only geometry accepted by
preview and encoder shaders. A mesh can add effect detail but cannot select,
shrink, translate, or clear the privacy mask.
"""
from pathlib import Path

source = Path("src/main/java/com/makey/blurfaces/Main.java").read_text()

assert "private static final int MESH_STRIDE = 3" in source
assert "private static float[] meshForEffects" in source
assert "Never derive privacy axes, count, or emergency state from mesh." in source
assert "anchorsToAffine(anchors, i * FACE_ANCHOR_STRIDE" in source
assert "uploadFaceUniforms(s)" in source
assert "uploadFaceBasis(e.faceCount, e.facesData" in source
assert "private static void drainLatestMeshFrame()" in source
assert "Root publication never waits for mesh." in source
assert "MESH_WORKER_SCHEDULED" in source
assert "facialTransformationMatrixes" in source
assert "faceBlendshapes" in source
loader = Path("loader/dex.py").read_text()
plugin = Path("loader/plugin.py").read_text()
assert "DexClassLoader(" in loader
assert "mesh_library_dir" in loader
assert "filename=\"libmediapipe_tasks_vision_jni.so\"" in plugin
assert "MESH_DEX_URL" in plugin
assert "MESH_DEX_SHA256" in plugin
assert "mesh_dex_path" in loader
assert "os.chmod(core_path, 0o444)" in loader
utils = Path("loader/utils.py").read_text()
assert "os.chmod(path, 0o444)" in utils

# The first draw bootstraps EGL installation in the after-hook, but every visible
# subsequent draw must be armed in beforeHookedMethod. The later capture method
# legitimately queues optional mesh frames, so keep it outside this state slice.
preview = source[source.index("private static void prepareNextPreviewDraw"):source.index("private static void afterDraw")]
encoder = source[source.index("private static void beforeEncoderDraw"):source.index("private static float clamp")]
hook = source[source.index("private static void hookCameraGLThread"):source.index("private static void hookEncoderRenderer")]
assert "beforeHookedMethod" in hook
assert "prepareNextPreviewDraw(p.thisObject" in hook
assert "Preview replacement armed before visible host draw" in hook
assert "meshForEffects" not in preview
assert "meshForEffects" not in encoder
assert "LATEST_MESH" not in preview
assert "LATEST_MESH" not in encoder
print("PASS: mesh is a separate latest-frame worker; stable 5-point root remains preview/encoder privacy authority")

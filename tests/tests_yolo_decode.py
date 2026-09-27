"""The C++ YOLOv8 head decoder must reproduce the export's out0 exactly.

Needs the `ncnn` Python package (pip install ncnn) to run the bundled model on
the host; skipped without it. Model files are read from models/.
"""
import os
from pathlib import Path
import shlex
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent

try:
    import ncnn  # type: ignore
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover - optional host dependency
    ncnn = None

LEVELS = (("183", "202"), ("189", "208"), ("195", "214"))  # stride 8, 16, 32: box, cls


@unittest.skipIf(ncnn is None, "ncnn / numpy Python packages not installed")
class YoloDecodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.net = ncnn.Net()
        cls.net.opt.use_vulkan_compute = False
        assert cls.net.load_param(str(ROOT / "models/head_det.param")) == 0
        assert cls.net.load_model(str(ROOT / "models/head_det.bin")) == 0
        build = ROOT / "build" / "native-tests"
        build.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(dir=build)
        cls.exe = Path(cls.tmp.name) / "yolo-decode-check"
        subprocess.run(shlex.split(os.environ.get("CXX", "g++")) + [
            "-std=c++17", "-O1", "-fsanitize=undefined", "-fno-sanitize-recover=all",
            "-I" + str(ROOT / "src"), str(ROOT / "tests/yolo_decode_check.cpp"),
            "-o", str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def extract(self, size, seed):
        rng = np.random.default_rng(seed)
        # Smooth random image: upsampled noise, so the net sees structure.
        small = rng.integers(0, 256, (12, 12, 3), dtype=np.uint8)
        img = np.kron(small, np.ones((16, 16, 1), dtype=np.uint8))
        mat = ncnn.Mat.from_pixels_resize(img.tobytes(), ncnn.Mat.PixelType.PIXEL_RGB,
                                          192, 192, size, size)
        mat.substract_mean_normalize([0, 0, 0], [1 / 255.0] * 3)
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        blobs = {}
        for name in [b for level in LEVELS for b in level] + (["out0"] if size == 320 else []):
            ret, out = ex.extract(name)
            self.assertEqual(ret, 0, name)
            blobs[name] = np.array(out, dtype=np.float32)
        return blobs

    def decode(self, blobs):
        path = Path(self.tmp.name) / "levels.bin"
        with open(path, "wb") as f:
            for box, cls in LEVELS:
                b, c = blobs[box], blobs[cls]
                self.assertEqual(b.shape[0], 64)
                self.assertEqual(c.shape[0], 1)
                f.write(struct.pack("<ii", b.shape[2], b.shape[1]))
                f.write(np.ascontiguousarray(b).tobytes())
                f.write(np.ascontiguousarray(c).tobytes())
        out = subprocess.run([str(self.exe), str(path)], check=True, capture_output=True, text=True)
        return np.array([[float(v) for v in line.split()] for line in out.stdout.splitlines()])

    def test_matches_export_tail_at_320(self):
        for seed in (1, 2, 3):
            blobs = self.extract(320, seed)
            ours = self.decode(blobs)
            ref = blobs["out0"].T  # (2100, 5): cx, cy, w, h, conf
            self.assertEqual(ours.shape, ref.shape)
            np.testing.assert_allclose(ours[:, :4], ref[:, :4], rtol=1e-4, atol=2e-3)
            np.testing.assert_allclose(ours[:, 4], ref[:, 4], rtol=1e-4, atol=1e-5)

    def test_other_input_sizes_decode(self):
        for size in (256, 224, 192):
            ours = self.decode(self.extract(size, 4))
            cells = sum((size // s) ** 2 for s in (8, 16, 32))
            self.assertEqual(ours.shape, (cells, 5))
            self.assertTrue(np.isfinite(ours).all())
            self.assertTrue(((ours[:, 4] >= 0) & (ours[:, 4] <= 1)).all())


if __name__ == "__main__":
    unittest.main()

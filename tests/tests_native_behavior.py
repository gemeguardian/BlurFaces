"""Compile and execute production ByteTrack and the JNI result policy on the host."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent


class NativeBehaviorTest(unittest.TestCase):
    def test_production_tracker_and_result_policy(self):
        build = ROOT / "build" / "native-tests"
        build.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=build) as directory:
            executable = Path(directory) / "tracking-test"
            command = shlex.split(os.environ.get("CXX", "g++")) + [
                "-std=c++17", "-O1", "-g", "-fsanitize=undefined",
                "-fno-sanitize-recover=all",
                "-I" + str(ROOT / "src"),
                "-I" + str(ROOT / "ncnn-prebuilt/arm64-v8a/include"),
                str(ROOT / "tests/native_tracking_test.cpp"),
                str(ROOT / "src/bytetrack.cpp"), "-o", str(executable),
            ]
            subprocess.run(command, check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()

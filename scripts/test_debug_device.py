#!/usr/bin/env python3
"""Run synthetic debug-capture smoke tests over SSH; no camera or app-data access.

Build with ./build.sh first. Requires an authorized Android shell with app_process,
javac, SDK build-tools 36.0.0 and platform android-35. Uses a temporary device
folder, removes it afterwards, and never installs/reloads the plugin.
"""
import argparse
from pathlib import Path
import re
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="SSH host alias, e.g. phone")
    parser.add_argument("--sdk", type=Path, default=Path("/home/PluginDev/.android-sdk"))
    args = parser.parse_args()
    if args.host.startswith("-"):
        parser.error("host must not be an option")
    root = Path(__file__).resolve().parent.parent
    android = args.sdk / "platforms/android-35/android.jar"
    build = root / "build/device-test"
    classes, dex = build / "classes", build / "dex"
    classes.mkdir(parents=True, exist_ok=True)
    dex.mkdir(parents=True, exist_ok=True)
    jar = root / "build/dex-input/plugin-classes.jar"
    subprocess.run(["javac", "--release", "11", "-cp", f"{android}:{jar}", "-d", str(classes),
                    str(root / "tests/DebugCaptureDeviceTest.java")], check=True)
    subprocess.run([str(args.sdk / "build-tools/36.0.0/d8"), "--min-api", "26",
                    "--lib", str(android), "--classpath", str(jar), "--output", str(dex),
                    *[str(p) for p in classes.rglob("*.class")]], check=True)
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host]
    remote = subprocess.check_output(
        ssh + ["mktemp -d /data/local/tmp/blur-faces-debug-test.XXXXXX"], text=True).strip()
    if not re.fullmatch(r"/data/local/tmp/blur-faces-debug-test\.[A-Za-z0-9]+", remote):
        raise RuntimeError("Unexpected temporary device path")
    try:
        for path in (dex / "classes.dex", root / "build/dex/core.dex",
                     root / "libs/arm64-v8a/libblur_faces.so",
                     root / "models/head_det.param", root / "models/head_det.bin"):
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                            str(path), args.host + ":" + remote + "/"], check=True)
        command = (f"chmod 444 {remote}/*.dex && "
                   f"CLASSPATH={remote}/classes.dex:{remote}/core.dex "
                   f"app_process /system/bin com.makey.blurfaces.g2.DebugCaptureDeviceTest {remote}")
        result = subprocess.run(ssh + [command], text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=120)
        print(result.stdout)
        (build / "result.txt").write_text(result.stdout)
        result.check_returncode()
    finally:
        subprocess.run(ssh + ["rm -rf -- " + shlex.quote(remote)], check=True, timeout=20)


if __name__ == "__main__":
    main()

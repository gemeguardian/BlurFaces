#!/usr/bin/env python3
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent

def get_code_stats():
    res = subprocess.run(
        ["git", "ls-files", "src/*", "BlurFaces/*.py", "tests/*"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True
    )
    files = [ROOT / line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
    total_bytes = 0
    total_lines = 0
    for f in files:
        if f.exists() and f.is_file():
            data = f.read_bytes()
            total_bytes += len(data)
            total_lines += data.count(b"\n")
    return total_bytes, total_lines

def get_bundle_stats():
    zip_path = ROOT / "builds" / "blur_faces-1.0.0.elyx"
    if zip_path.exists():
        return zip_path.stat().st_size
    builds_dir = ROOT / "builds"
    elyx_files = sorted(builds_dir.glob("blur_faces-*.elyx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not elyx_files:
        return 0
    return elyx_files[0].stat().st_size

def update_readme():
    readme_path = ROOT / "README.md"
    if not readme_path.exists():
        return

    bytes_count, lines_count = get_code_stats()
    bundle_bytes = get_bundle_stats()

    kb = bytes_count / 1024.0
    loc_k = lines_count / 1000.0
    bundle_mb = bundle_bytes / (1024.0 * 1024.0)

    # Format badges:
    # [![Code Size](https://img.shields.io/badge/code%20size-422%20KB%20%7C%209.3k%20LoC-informational.svg)]()
    # [![Bundle Size](https://img.shields.io/badge/bundle-4.5%20MB-purple.svg)]()
    code_badge_text = f"code%20size-{kb:.0f}%20KB%20%7C%20{loc_k:.1f}k%20LoC"
    bundle_badge_text = f"bundle-{bundle_mb:.1f}%20MB"

    content = readme_path.read_text(encoding="utf-8")
    content = re.sub(
        r"badge/code%20size-[^)]+\.svg",
        f"badge/{code_badge_text}-informational.svg",
        content
    )
    content = re.sub(
        r"badge/bundle-[^)]+\.svg",
        f"badge/{bundle_badge_text}-purple.svg",
        content
    )
    readme_path.write_text(content, encoding="utf-8")
    print(f"Updated README badges: {kb:.0f} KB ({lines_count} LoC), bundle: {bundle_mb:.1f} MB")

if __name__ == "__main__":
    update_readme()

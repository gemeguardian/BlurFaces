#!/usr/bin/env python3
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def fix_zip_permissions():
    zip_path = ROOT / "builds" / "blur_faces-3.0.0.elyx"
    if not zip_path.exists():
        return
    temp_buf = io.BytesIO()
    with zipfile.ZipFile(zip_path, 'r') as zin:
        with zipfile.ZipFile(temp_buf, 'w') as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.is_dir() or item.filename.endswith('/'):
                    item.external_attr = 0o40755 << 16
                else:
                    item.external_attr = 0o100644 << 16
                zout.writestr(item, data, compress_type=item.compress_type)
    zip_path.write_bytes(temp_buf.getvalue())
    print(f"Normalized zip permissions for {zip_path.name}")

if __name__ == "__main__":
    fix_zip_permissions()

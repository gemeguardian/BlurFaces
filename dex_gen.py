"""Shared dex generation id for hot-swap safety.

Chaquopy caches Java class definitions by name inside the app process. When a
plugin update swaps core.dex without an app restart, new loaders define classes
with the SAME binary names, and every j2p conversion of a factory object fails
with "cannot create X proxy from X instance" because the cached definition no
longer matches the live class. Loading each generation under a unique package
name sidesteps the cache entirely.

Bump DEX_GEN on every change to the Java sources; keep it in sync with the
package rewrite in build.gradle and the class names in runtime.py.
"""

import re
from pathlib import Path

METAINFO = Path(__file__).parent / "BlurFaces" / "metainfo.yml"


def dex_gen() -> int:
    match = re.search(r'^dex_gen:\s*"?(\d+)"?\s*$', METAINFO.read_text(), re.M)
    if match is None:
        raise SystemExit("BlurFaces/metainfo.yml is missing the dex_gen key")
    return int(match.group(1))


def package() -> str:
    return f"com.makey.blurfaces.g{dex_gen()}"

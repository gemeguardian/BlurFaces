import os
import base64
import hashlib
import tempfile
import uuid
import glob
from java.net import URL
from java.io import File
from java.lang import System

from android_utils import log


def unique_load_path(canonical_path, cache_dir, prefix="libblur_faces_load"):
    """Copy the canonical .so to a fresh random-named file so System.load never
    hits 'already opened by ClassLoader' on a live process. Old copies are
    cleaned up so the cache dir does not grow indefinitely."""
    pat = os.path.join(cache_dir, prefix + "_*.so")
    for stale in glob.glob(pat):
        try:
            if os.path.abspath(stale) != os.path.abspath(canonical_path):
                os.unlink(stale)
        except OSError:
            pass
    unique_name = f"{prefix}_{uuid.uuid4().hex}.so"
    unique_path = os.path.join(cache_dir, unique_name)
    with open(canonical_path, "rb") as src, open(unique_path, "wb") as dst:
        dst.write(src.read())
    log(f"[BlurFaces] Unique load path: {unique_path}")
    return unique_path


def download_native(url, expected_sha256, cache_dir, filename="libblur_faces.so"):
    """Download the native payload once and verify it before replacing cache."""
    path = os.path.join(cache_dir, filename)

    try:
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == expected_sha256:
                log(f"[BlurFaces] Using cached native library: {path}")
                return path
    except OSError:
        pass

    connection = URL(url).openConnection()
    connection.setConnectTimeout(15000)
    connection.setReadTimeout(30000)
    stream = connection.getInputStream()
    fd, temp_path = tempfile.mkstemp(prefix=filename + ".", dir=cache_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            buffer = bytearray(65536)
            while True:
                count = stream.read(buffer)
                if count == -1:
                    break
                f.write(buffer[:count])
            f.flush()
            os.fsync(f.fileno())
        with open(temp_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"native SHA-256 mismatch: {actual_sha256}")
        os.replace(temp_path, path)
        log(f"[BlurFaces] Downloaded native library to {path}")
    finally:
        stream.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return path


def download_file(url, expected_sha256, path, label):
    """Download one verified payload and atomically replace its cache file."""
    try:
        with open(path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == expected_sha256:
                # A prior 1.7.1 cache can predate Android 14's read-only DEX gate.
                # Make a verified payload immutable before the caller gives it to ART.
                try:
                    os.chmod(path, 0o444)
                except OSError:
                    pass
                log(f"[BlurFaces] Using cached {label}: {path}")
                return False
    except OSError:
        pass

    connection = URL(url).openConnection()
    connection.setConnectTimeout(15000)
    connection.setReadTimeout(30000)
    stream = connection.getInputStream()
    fd, temp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "wb") as f:
            buffer = bytearray(65536)
            while True:
                count = stream.read(buffer)
                if count == -1:
                    break
                f.write(buffer[:count])
            f.flush()
            os.fsync(f.fileno())
        with open(temp_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"{label} SHA-256 mismatch: {actual_sha256}")
        os.replace(temp_path, path)
        # ART refuses writable dynamically loaded DEX files on Android 14+.
        # Read-only is also safe for immutable models and prevents accidental
        # mutation after their SHA-256 gate.
        try:
            os.chmod(path, 0o444)
        except OSError:
            pass
        log(f"[BlurFaces] Downloaded {label} to {path}")
    finally:
        stream.close()
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return True


def extract_model(param_url, param_sha256, bin_url, bin_sha256, mesh_url, mesh_sha256, cache_dir):
    """Download verified stable-root and optional dense-mesh assets privately."""
    param_path = os.path.join(cache_dir, "retinaface_mnet025_5kps.param")
    bin_path = os.path.join(cache_dir, "retinaface_mnet025_5kps.bin")
    mesh_path = os.path.join(cache_dir, "face_landmarker.task")

    param_downloaded = download_file(param_url, param_sha256, param_path, "landmark model param")
    bin_downloaded = download_file(bin_url, bin_sha256, bin_path, "landmark model bin")
    # Mesh is an additive effect layer. A failed download must not disable the
    # stable five-point privacy root or prevent the plugin from loading.
    try:
        mesh_downloaded = download_file(mesh_url, mesh_sha256, mesh_path, "dense face mesh model")
    except Exception as error:
        log(f"[BlurFaces] Dense mesh model unavailable; stable root continues: {error}")
        mesh_path = None
        mesh_downloaded = False

    return param_path, bin_path, mesh_path, param_downloaded or bin_downloaded or mesh_downloaded

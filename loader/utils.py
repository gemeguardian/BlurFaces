import os
import hashlib
import shutil
import tempfile
import uuid
from java.net import URL

from android_utils import log


def prepare_native_load_copy(canonical_path, cache_dir):
    """Give each DexClassLoader a distinct JNI path while preserving basename."""
    for name in os.listdir(cache_dir):
        if not name.startswith("mediapipe_load_"):
            continue
        try:
            shutil.rmtree(os.path.join(cache_dir, name))
        except OSError:
            pass
    load_dir = os.path.join(cache_dir, "mediapipe_load_" + uuid.uuid4().hex)
    os.makedirs(load_dir)
    load_path = os.path.join(load_dir, "libmediapipe_tasks_vision_jni.so")
    with open(canonical_path, "rb") as source, open(load_path, "wb") as target:
        shutil.copyfileobj(source, target)
    os.chmod(load_path, 0o444)
    log(f"[BlurFaces] MediaPipe JNI load directory: {load_dir}")
    return load_path


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

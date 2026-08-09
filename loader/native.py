import ctypes
import os
from android_utils import log


class NativeBridge:
    """ctypes bridge to the SCRFD native library."""

    def __init__(self):
        self._lib = None
        self._initialized = False

    def load_library(self, so_path):
        """Load the native .so library via ctypes."""
        if self._lib:
            return True
        try:
            self._lib = ctypes.CDLL(so_path)
            
            # blur_faces_init(const char* param_path, const char* bin_path) -> int
            self._lib.blur_faces_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
            self._lib.blur_faces_init.restype = ctypes.c_int
            
            # blur_faces_detect(unsigned char* rgba_pixels, int width, int height,
            #                   float* out_faces, int max_faces) -> int
            self._lib.blur_faces_detect.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),  # rgba_pixels
                ctypes.c_int,                     # width
                ctypes.c_int,                     # height
                ctypes.POINTER(ctypes.c_float),   # out_faces
                ctypes.c_int                      # max_faces
            ]
            self._lib.blur_faces_detect.restype = ctypes.c_int

            # blur_faces_blur_rgba(unsigned char* rgba_pixels, int width, int height,
            #                      float* faces, int face_count) -> int
            self._lib.blur_faces_blur_rgba.argtypes = [
                ctypes.POINTER(ctypes.c_ubyte),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            self._lib.blur_faces_blur_rgba.restype = ctypes.c_int
            
            # blur_faces_cleanup() -> void
            self._lib.blur_faces_cleanup.argtypes = []
            self._lib.blur_faces_cleanup.restype = None
            
            log(f"[BlurFaces] Native library loaded via ctypes: {so_path}")
            return True
        except Exception as e:
            log(f"[BlurFaces] Failed to load native library via ctypes: {e}")
            return False

    def init_model(self, param_path, bin_path):
        """Initialize the SCRFD model."""
        if not self._lib:
            log("[BlurFaces] Native library not loaded")
            return False
        try:
            result = self._lib.blur_faces_init(
                param_path.encode('utf-8'),
                bin_path.encode('utf-8')
            )
            self._initialized = (result == 0)
            log(f"[BlurFaces] Model init result: {result}")
            return self._initialized
        except Exception as e:
            log(f"[BlurFaces] Model init failed: {e}")
            return False

    def detect_faces(self, rgba_pixels, width, height, max_faces=10):
        """Detect faces in RGBA pixel buffer. Returns list of FaceRect or None."""
        if not self._initialized or not self._lib:
            return None
        try:
            # Create ctypes array from bytes/bytearray
            if isinstance(rgba_pixels, bytearray):
                pixels_array = (ctypes.c_ubyte * len(rgba_pixels)).from_buffer(rgba_pixels)
            else:
                pixels_array = (ctypes.c_ubyte * len(rgba_pixels)).from_buffer_copy(rgba_pixels)
            
            # Output array for face data
            out_faces = (ctypes.c_float * (max_faces * 5))()
            
            count = self._lib.blur_faces_detect(
                pixels_array,
                width,
                height,
                out_faces,
                max_faces
            )
            
            if count < 0:
                log("[BlurFaces] Detection returned error")
                return None
            
            # Convert to list of dicts
            faces = []
            for i in range(count):
                faces.append({
                    'x': out_faces[i * 5 + 0],
                    'y': out_faces[i * 5 + 1],
                    'w': out_faces[i * 5 + 2],
                    'h': out_faces[i * 5 + 3],
                    'prob': out_faces[i * 5 + 4]
                })
            
            return faces
        except Exception as e:
            log(f"[BlurFaces] Detection error: {e}")
            return None

    def blur_rgba(self, rgba_pixels, width, height, faces):
        """Pixelate detected face rectangles in-place in an RGBA buffer."""
        if not self._initialized or not self._lib or not faces:
            return False
        try:
            if not isinstance(rgba_pixels, bytearray):
                raise TypeError("rgba_pixels must be a bytearray")
            pixels_array = (ctypes.c_ubyte * len(rgba_pixels)).from_buffer(rgba_pixels)
            face_values = (ctypes.c_float * (len(faces) * 5))()
            for i, face in enumerate(faces):
                face_values[i * 5 + 0] = float(face['x'])
                face_values[i * 5 + 1] = float(face['y'])
                face_values[i * 5 + 2] = float(face['w'])
                face_values[i * 5 + 3] = float(face['h'])
                face_values[i * 5 + 4] = float(face['prob'])
            result = self._lib.blur_faces_blur_rgba(
                pixels_array, width, height, face_values, len(faces)
            )
            return result == 0
        except Exception as e:
            log(f"[BlurFaces] Native blur error: {e}")
            return False

    def cleanup(self):
        """Release native resources."""
        if self._lib:
            try:
                self._lib.blur_faces_cleanup()
            except Exception as e:
                log(f"[BlurFaces] Cleanup error: {e}")
            self._lib = None
        self._initialized = False

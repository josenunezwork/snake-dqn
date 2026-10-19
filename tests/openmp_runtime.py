"""Inspect actual dyld images without importing a numerical package."""

import ctypes
import sys
from pathlib import Path


def loaded_libomp_images() -> list[str]:
    """Return distinct loaded OpenMP image paths on macOS."""
    if sys.platform != "darwin":
        return []
    dyld = ctypes.CDLL(None)
    count = dyld._dyld_image_count
    count.argtypes = []
    count.restype = ctypes.c_uint32
    name = dyld._dyld_get_image_name
    name.argtypes = [ctypes.c_uint32]
    name.restype = ctypes.c_char_p
    images = set()
    for index in range(count()):
        raw = name(index)
        if raw:
            path = raw.decode("utf-8", errors="surrogateescape")
            if Path(path).name == "libomp.dylib":
                images.add(str(Path(path).resolve()))
    return sorted(images)


def assert_single_libomp() -> None:
    """Fail visibly instead of continuing threaded tests with conflicting runtimes."""
    images = loaded_libomp_images()
    if len(images) > 1:
        raise AssertionError("Multiple libomp images loaded in test process:\n" + "\n".join(images))

"""Stdlib-only import boundary, checked before loading the shared control policy."""

from __future__ import annotations

import hashlib
import sys
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path

# Existing unchanged operational-base bytes; these initializers run before the
# shared policy is loaded. They must not introduce unbound local work.
INIT_SHA256 = "272d8d65dbc74c10e9c977736886c0e2476b736fd9b8a289326b2f6551537ff1"
EXPORTS_SHA256 = "fb98467718f25c57cfacd395b3d5208d3f5e91c1eb47e070293f9fb5702cba1d"
PACKAGE_FILES = {
    "research/bl_labels/__init__.py": INIT_SHA256,
    "research/bl_labels/exports.py": EXPORTS_SHA256,
}


def verify(root: Path) -> None:
    """Bind namespace parents and existing package bytes without executing them."""
    root = root.resolve()
    for name, parent in (("research", root), ("research.compute", root / "research")):
        expected = root.joinpath(*name.split("."))
        spec = PathFinder.find_spec(name, [str(parent)])
        if (
            spec is None
            or spec.origin is not None
            or list(spec.submodule_search_locations or ()) != [str(expected)]
        ):
            raise RuntimeError("control namespace parent is not the pinned namespace")
        loaded = sys.modules.get(name)
        if loaded is not None:
            locations = {str(Path(path).resolve()) for path in loaded.__path__}
            if getattr(loaded, "__file__", None) is not None or locations != {str(expected)}:
                raise RuntimeError("control namespace was imported from another root")
        else:
            # Restrict the namespace to this root; a later installed regular
            # package must not override the approved cwd namespace during -m.
            sys.modules[name] = module_from_spec(spec)
    for relative, digest in PACKAGE_FILES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError("control package initializer/dependency differs from frozen bytes")

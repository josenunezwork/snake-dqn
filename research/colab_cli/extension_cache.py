"""Retention-only immutable PyTorch extension reuse across fresh job roots.

Only ordinary Python extension modules from the frozen repository are eligible.
Every other load fails closed in retention mode. Publication is create-only and
locked; a hit imports a verified private copy without invoking a compiler.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator

_PROCESS_MODULES: dict[tuple[str, str], Any] = {}
_MODULE_KEYS: dict[str, tuple[str, str]] = {}


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _key(name: str, kwargs: dict[str, Any], parent_key: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise RuntimeError("retention extension name is invalid")
    sources = kwargs.get("sources")
    if not isinstance(sources, (list, tuple)) or not sources:
        raise RuntimeError("retention extension requires explicit ordered sources")
    allowed = {
        "sources",
        "extra_cflags",
        "extra_cuda_cflags",
        "extra_ldflags",
        "extra_include_paths",
        "build_directory",
        "verbose",
        "with_cuda",
        "with_sycl",
        "is_python_module",
        "is_standalone",
        "keep_intermediates",
        "extra_sycl_cflags",
    }
    if set(kwargs) - allowed or kwargs.get("extra_include_paths") or kwargs.get("extra_ldflags"):
        raise RuntimeError("retention extension refuses unpinned external includes/linker inputs")
    if kwargs.get("is_python_module", True) is not True or kwargs.get("is_standalone", False):
        raise RuntimeError("retention extension only supports Python modules")
    if kwargs.get("with_sycl") or kwargs.get("extra_sycl_cflags"):
        raise RuntimeError("retention extension does not admit a SYCL toolchain")
    repo = (Path(os.environ["COLAB_CLI_ROOT"]) / "repo").resolve()
    rows = []
    for item in sources:
        source = Path(item).resolve()
        if not source.is_file() or repo not in source.parents:
            raise RuntimeError("retention extension source is outside the frozen repository")
        rows.append({"path": str(source.relative_to(repo)), "sha256": _digest(source)})
    flags = {}
    for field in ("extra_cflags", "extra_cuda_cflags"):
        values = kwargs.get(field) or []
        if not isinstance(values, (tuple, list)) or any(not isinstance(x, str) for x in values):
            raise RuntimeError("retention extension flags must be explicit string lists")
        if any(
            value.startswith(("-I", "-L", "-include", "--include-path", "-Wl")) for value in values
        ):
            raise RuntimeError("retention extension refuses undeclared external compiler inputs")
        flags[field] = list(values)
    identity = {
        "parent": parent_key,
        "name": name,
        "sources": rows,
        "flags": flags,
        "with_cuda": kwargs.get("with_cuda"),
        "environment": {
            key: os.environ.get(key, "")
            for key in (
                "CXXFLAGS",
                "CFLAGS",
                "LDFLAGS",
                "CC",
                "CXX",
                "CUDAHOSTCXX",
                "TORCH_CUDA_ARCH_LIST",
                "NVCC_PREPEND_FLAGS",
                "NVCC_APPEND_FLAGS",
            )
        },
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        until = time.monotonic() + 120
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= until:
                    raise RuntimeError("retention extension publication lock timed out")
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _publish(path: Path, data: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        path.chmod(0o444)
    finally:
        temp.unlink(missing_ok=True)


def _event(job: Path, record: dict[str, Any]) -> None:
    with (job / "extension-cache-events.jsonl").open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cached_load(
    original: Callable[..., Any],
    name: str,
    kwargs: dict[str, Any],
    *,
    job: Path,
    cache_root: Path,
    parent_key: str,
) -> Any:
    """Load a single exact-key immutable binary or compile and publish it once."""
    key = _key(name, kwargs, parent_key)
    identity = (parent_key, key)
    if identity in _PROCESS_MODULES:
        return _PROCESS_MODULES[identity]
    directory = cache_root / "frozen" / key
    # Locks live beside immutable entries, never inside a frozen artifact.
    with _locked(cache_root / "locks" / (key + ".lock")):
        if identity in _PROCESS_MODULES:
            return _PROCESS_MODULES[identity]
        directory.mkdir(parents=True, exist_ok=True)
        binary, meta, marker = (
            directory / "module.so",
            directory / "metadata.json",
            directory / ".complete",
        )
        if any(path.is_symlink() for path in (directory, binary, meta, marker)):
            raise RuntimeError("retention extension cache symlink refused")
        existing = [path.exists() for path in (binary, meta, marker)]
        if any(existing) and not all(existing):
            raise RuntimeError("partial immutable extension publication; no replacement")
        if all(existing):
            record = json.loads(meta.read_text())
            if (
                record.get("key") != key
                or record.get("requested_name") != name
                or record.get("size") != binary.stat().st_size
                or record.get("sha256") != _digest(binary)
                or marker.read_text().strip() != _digest(meta)
            ):
                raise RuntimeError("immutable extension binary/metadata digest mismatch")
            module_name = record["module_name"]
            if not re.fullmatch(re.escape(name) + r"(?:_v[0-9]+)?", module_name):
                raise RuntimeError("immutable extension module name mismatch")
            if module_name in sys.modules or (
                module_name in _MODULE_KEYS and _MODULE_KEYS[module_name] != identity
            ):
                raise RuntimeError("retention extension module name collision")
            restored = job / "torch_extensions" / key / "module.so"
            restored.parent.mkdir(parents=True, exist_ok=True)
            if restored.exists():
                raise RuntimeError("extension private restore path already used")
            shutil.copyfile(binary, restored)
            if _digest(restored) != record["sha256"]:
                raise RuntimeError("extension private copy digest mismatch")
            spec = importlib.util.spec_from_file_location(module_name, restored)
            if spec is None or spec.loader is None:
                raise RuntimeError("immutable extension cannot be imported")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODULE_KEYS[module_name] = identity
            _PROCESS_MODULES[identity] = module
            _event(job, {**record, "status": "HIT_VERIFIED_IMPORT"})
            return module
        private = job / "torch_extensions" / key / "build"
        private.mkdir(parents=True, exist_ok=True)
        compile_kwargs = dict(kwargs, build_directory=str(private))
        module = original(name=name, **compile_kwargs)
        built = Path(module.__file__).resolve()
        if private.resolve() not in built.parents or not built.is_file():
            raise RuntimeError("extension compiler output escaped its private build root")
        if module.__name__ in _MODULE_KEYS and _MODULE_KEYS[module.__name__] != identity:
            raise RuntimeError("compiled extension module name collision")
        record = {
            "schema": "snake-dqn-retention-extension/v1",
            "key": key,
            "requested_name": name,
            "module_name": module.__name__,
            "sha256": _digest(built),
            "size": built.stat().st_size,
        }
        _publish(binary, built.read_bytes())
        _publish(meta, (json.dumps(record, sort_keys=True) + "\n").encode())
        _publish(marker, (_digest(meta) + "\n").encode())
        _MODULE_KEYS[module.__name__] = identity
        _PROCESS_MODULES[identity] = module
        _event(job, {**record, "status": "MISS_COMPILED"})
        return module


def install() -> None:
    """Install only in an explicitly admitted retained-root interpreter."""
    parent = os.environ.get("COLAB_RETENTION_CACHE_KEY")
    if parent is None:
        return
    if not re.fullmatch(r"[0-9a-f]{64}", parent):
        raise RuntimeError("invalid retention parent cache key")
    import torch.utils.cpp_extension as extension

    if getattr(extension.load, "_snake_retention", False):
        return
    original = extension.load
    signature = inspect.signature(original)
    job = Path(os.environ["COLAB_CLI_ROOT"])
    cache = Path("/content/snake-cli-immutable-cache") / parent

    def load(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        options = dict(bound.arguments)
        name = options.pop("name")
        return cached_load(original, name, options, job=job, cache_root=cache, parent_key=parent)

    load._snake_retention = True  # type: ignore[attr-defined]
    extension.load = load

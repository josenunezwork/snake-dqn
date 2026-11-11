"""Device normalization, single-arch CUDA pre-init, phase stamps, and create-only IO."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import torch


def normalize_device(device: str | torch.device) -> torch.device:
    """Resolve a concrete device with explicit index where applicable.

    Args:
        device: Requested device specification (e.g. ``"cpu"``, ``"mps"``, ``"cuda"``).

    Returns:
        Canonical ``torch.device`` matching allocated tensor ``.device`` attributes
        (e.g. ``mps:0`` or ``cuda:0`` rather than unindexed ``mps``/``cuda``).
    """
    requested = torch.device(device)
    if requested.type == "cuda":
        torch.cuda.init()
        idx = 0 if requested.index is None else int(requested.index)
        resolved = torch.device("cuda", idx)
        torch.cuda.set_device(resolved)
        return resolved
    return torch.zeros((), device=requested).device


def ensure_cuda_initialized(device: torch.device) -> None:
    """Initialize CUDA context and bind current device before memory queries.

    Args:
        device: Normalized target device.
    """
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.set_device(device)


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous accelerator queues on CUDA or MPS.

    Args:
        device: Target execution device.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def configure_determinism(deterministic: bool) -> None:
    """Configure deterministic or throughput-oriented backend flags.

    Args:
        deterministic: When True, enable deterministic cuDNN and disable TF32 drift.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = not bool(deterministic)
        torch.backends.cudnn.allow_tf32 = not bool(deterministic)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not bool(deterministic)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)


def preload_native_sim_extensions(device: torch.device) -> dict[str, Any]:
    """Pre-compile and load native DeviceSim CUDA kernels for the present GPU only.

    Restricts ``TORCH_CUDA_ARCH_LIST`` to the active GPU's exact compute capability
    so ``nvcc`` never wastes wall time compiling unused SM targets or triggers lazy
    compilation inside timed rollout ticks or first-death events.

    Args:
        device: Normalized execution device.

    Returns:
        Metadata dictionary describing loaded native modules and target SM arch.
    """
    if device.type != "cuda":
        return {"device_type": device.type, "loaded": [], "arch": None}

    ensure_cuda_initialized(device)
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"{major}.{minor}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    base_ext_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if base_ext_dir:
        scoped_dir = Path(base_ext_dir) / f"sm_{major}{minor}"
    else:
        scoped_dir = Path(tempfile.gettempdir()) / f"arena_ppo_torch_ext_sm_{major}{minor}"
    scoped_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = str(scoped_dir)

    from research.gpu_sim_spike import event_cuda, move_cuda

    move_cuda._extension()
    if event_cuda._EXTENSION is None:
        from torch.utils.cpp_extension import load

        spike_root = Path(event_cuda.__file__).resolve().parent
        event_cuda._EXTENSION = load(
            name="snake_dqn_a0_event_v1",
            sources=[str(spike_root / "event_cuda.cpp"), str(spike_root / "event_cuda.cu")],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    synchronize_device(device)
    return {
        "device_type": "cuda",
        "device_name": torch.cuda.get_device_name(device),
        "arch": arch,
        "extensions_dir": str(scoped_dir),
        "loaded": ["move_cuda", "event_cuda"],
    }


def phase_stamp(phase: str, event: str, **fields: Any) -> None:
    """Emit a structured, UTC-timestamped phase START/END line to stdout.

    Args:
        phase: Logical phase identifier.
        event: Phase transition label (typically ``"START"`` or ``"END"``).
        **fields: Additional JSON-serializable telemetry key-values.
    """
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")
    record = {"ts": now, "phase": phase, "event": event, **fields}
    print(f"ARENA_PPO_PHASE {json.dumps(record, sort_keys=True)}", flush=True)


def sha256_file(path: Path | str) -> str:
    """Compute hex SHA-256 digest of a file on disk.

    Args:
        path: File path to hash.

    Returns:
        64-character lowercase hexadecimal SHA-256 string.
    """
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Flush directory entries to durable storage when supported by the OS."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def write_create_only_artifact(
    path: Path | str,
    writer: bytes | Callable[[Path], None],
) -> str:
    """Publish an immutable create-only file with ``.sha256`` and ``.complete`` last.

    Refuses with ``FileExistsError`` if the target payload, ``.sha256`` sidecar,
    or ``.complete`` marker already exists so completed artifacts are never rewritten.

    Args:
        path: Target payload path.
        writer: Raw ``bytes`` payload or a callable populating a temporary ``Path``.

    Returns:
        64-character lowercase SHA-256 hex digest of the published payload.
    """
    target = Path(path)
    sha_path = Path(f"{target}.sha256")
    complete_path = Path(f"{target}.complete")
    for existing in (target, sha_path, complete_path):
        if existing.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact path: {existing}")

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp.",
        dir=str(target.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if isinstance(writer, (bytes, bytearray, memoryview)):
            with tmp_path.open("wb") as handle:
                handle.write(bytes(writer))
                handle.flush()
                os.fsync(handle.fileno())
        else:
            writer(tmp_path)
            with tmp_path.open("rb") as handle:
                os.fsync(handle.fileno())

        digest = sha256_file(tmp_path)
        os.link(tmp_path, target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    sha_tmp = Path(f"{sha_path}.tmp.{os.getpid()}")
    try:
        with sha_tmp.open("w", encoding="utf-8") as handle:
            handle.write(f"{digest}  {target.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(sha_tmp, sha_path)
    finally:
        if sha_tmp.exists():
            sha_tmp.unlink()

    comp_tmp = Path(f"{complete_path}.tmp.{os.getpid()}")
    try:
        with comp_tmp.open("w", encoding="utf-8") as handle:
            handle.write(f"{digest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(comp_tmp, complete_path)
    finally:
        if comp_tmp.exists():
            comp_tmp.unlink()

    _fsync_directory(target.parent)
    return digest


def write_create_only_json(path: Path | str, payload: dict[str, Any]) -> str:
    """Serialize a JSON mapping with create-only ``.sha256`` and ``.complete`` markers.

    Args:
        path: Target JSON file path.
        payload: JSON-serializable dictionary.

    Returns:
        Hex SHA-256 digest of the written JSON file.
    """
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return write_create_only_artifact(path, encoded)


def verify_completed_artifact(path: Path | str) -> str:
    """Verify payload existence, ``.complete`` marker, and ``.sha256`` digest match.

    Args:
        path: Artifact payload path.

    Returns:
        Verified 64-character lowercase SHA-256 hex digest.
    """
    target = Path(path)
    sha_path = Path(f"{target}.sha256")
    complete_path = Path(f"{target}.complete")
    if not target.is_file():
        raise FileNotFoundError(f"Missing artifact payload: {target}")
    if not complete_path.is_file():
        raise FileNotFoundError(f"Missing .complete marker for {target}")
    if not sha_path.is_file():
        raise FileNotFoundError(f"Missing .sha256 sidecar for {target}")

    raw_sidecar = sha_path.read_text(encoding="utf-8").strip()
    expected = raw_sidecar.split()[0] if raw_sidecar else ""
    actual = sha256_file(target)
    if len(expected) != 64 or actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {target}: expected {expected!r}, observed {actual!r}"
        )
    return actual


def remove_completed_artifact(path: Path | str) -> None:
    """Remove a completed artifact payload together with its sidecars if present.

    Args:
        path: Target payload path whose ``.complete``, ``.sha256``, and payload
            should be removed (unlink order: ``.complete`` first so partial states
            never look complete).
    """
    target = Path(path)
    for candidate in (
        Path(f"{target}.complete"),
        Path(f"{target}.sha256"),
        target,
    ):
        candidate.unlink(missing_ok=True)


class SignalStopFlag:
    """Context manager installing ``SIGUSR1`` and ``SIGTERM`` cooperative stop flags."""

    def __init__(self) -> None:
        self.signal_name: str | None = None
        self.signal_number: int | None = None
        self.received_monotonic: float | None = None
        self._previous_handlers: dict[int, Any] = {}

    @property
    def stopped(self) -> bool:
        """Return True once ``SIGUSR1`` or ``SIGTERM`` has been received."""
        return self.signal_name is not None

    def _handle(self, signum: int, _frame: Any) -> None:
        if self.signal_name is None:
            try:
                self.signal_name = signal.Signals(signum).name
            except ValueError:
                self.signal_name = str(signum)
            self.signal_number = int(signum)
            self.received_monotonic = time.monotonic()

    def __enter__(self) -> "SignalStopFlag":
        signals_to_catch: list[int] = [signal.SIGTERM]
        if hasattr(signal, "SIGUSR1"):
            signals_to_catch.append(signal.SIGUSR1)
        for sig in signals_to_catch:
            self._previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for sig, prev in self._previous_handlers.items():
            signal.signal(sig, prev)
        self._previous_handlers.clear()

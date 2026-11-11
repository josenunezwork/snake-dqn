"""Device normalization, single-arch CUDA pre-init, phase stamps, and create-only IO."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

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

    comp_tmp = Path(f"{complete_path}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        with comp_tmp.open("w", encoding="utf-8") as handle:
            handle.write(f"{digest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if complete_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable artifact path: {complete_path}")
        os.replace(comp_tmp, complete_path)
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


def _publish_sidecars_for_existing_payload(target: Path, digest: str) -> None:
    """Publish create-only ``.sha256`` and ``.complete`` sidecars for an immutable payload."""
    sha_path = Path(f"{target}.sha256")
    complete_path = Path(f"{target}.complete")
    for existing in (sha_path, complete_path):
        if existing.exists():
            raise FileExistsError(f"Refusing to overwrite immutable sidecar path: {existing}")

    sha_tmp = Path(f"{sha_path}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        with sha_tmp.open("w", encoding="utf-8") as handle:
            handle.write(f"{digest}  {target.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(sha_tmp, sha_path)
    finally:
        if sha_tmp.exists():
            sha_tmp.unlink()

    comp_tmp = Path(f"{complete_path}.tmp.{os.getpid()}.{time.monotonic_ns()}")
    try:
        with comp_tmp.open("w", encoding="utf-8") as handle:
            handle.write(f"{digest}\n")
            handle.flush()
            os.fsync(handle.fileno())
        if complete_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable sidecar path: {complete_path}")
        os.replace(comp_tmp, complete_path)
    finally:
        if comp_tmp.exists():
            comp_tmp.unlink()


def publish_hardlinked_artifacts(
    source_path: Path | str,
    targets: Sequence[Path | str],
    *,
    staging_dir: Path | str | None = None,
) -> str:
    """Publish a work-dir file into one or more immutable create-only paths via hard links.

    Guarantees POSIX ``(st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns)`` stability for
    ``research.colab_cli.runtime._exports``:
    - All ``os.link`` operations complete BEFORE any ``.complete`` marker is created.
    - If ``source_path`` already has ``st_nlink > 1`` (for example, it was previously hard-linked
      into ``output/`` with a ``.complete`` marker), stages a fresh single-inode copy first so
      adding new links never mutates ``st_ctime_ns`` on an already-completed export.

    Args:
        source_path: Existing source payload outside ``output/``.
        targets: One or more create-only destination paths.
        staging_dir: Optional directory for fresh-inode staging when ``st_nlink > 1`` or ``EXDEV``.

    Returns:
        64-character lowercase SHA-256 hex digest of the published payload.
    """
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Missing source payload for hard-link publication: {src}")
    dst_list = [Path(t) for t in targets]
    if not dst_list:
        raise ValueError("targets must contain at least one destination path")
    if len({str(p.resolve()) if p.parent.exists() else str(p) for p in dst_list}) != len(dst_list):
        raise ValueError(f"Duplicate destination paths in targets: {dst_list}")

    for dst in dst_list:
        for existing in (dst, Path(f"{dst}.sha256"), Path(f"{dst}.complete")):
            if existing.exists():
                raise FileExistsError(f"Refusing to overwrite immutable artifact path: {existing}")

    for dst in dst_list:
        dst.parent.mkdir(parents=True, exist_ok=True)

    digest = sha256_file(src)
    src_stat = src.stat()
    linked_directly = False
    if src_stat.st_nlink == 1:
        created_direct: list[Path] = []
        try:
            for dst in dst_list:
                os.link(src, dst)
                created_direct.append(dst)
            linked_directly = True
        except OSError:
            for partial in created_direct:
                partial.unlink(missing_ok=True)
            linked_directly = False

    if not linked_directly:
        stage_parent = Path(staging_dir) if staging_dir is not None else dst_list[0].parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        fd, stage_name = tempfile.mkstemp(
            prefix=f".{dst_list[0].name}.stage.",
            dir=str(stage_parent),
        )
        os.close(fd)
        stage_path = Path(stage_name)
        created_staged: list[Path] = []
        try:
            shutil.copyfile(src, stage_path)
            with stage_path.open("rb") as handle:
                os.fsync(handle.fileno())
            for dst in dst_list:
                try:
                    os.link(stage_path, dst)
                except OSError:
                    fd_d, dst_tmp_name = tempfile.mkstemp(
                        prefix=f".{dst.name}.tmp.",
                        dir=str(dst.parent),
                    )
                    os.close(fd_d)
                    dst_tmp = Path(dst_tmp_name)
                    try:
                        shutil.copyfile(stage_path, dst_tmp)
                        with dst_tmp.open("rb") as h_d:
                            os.fsync(h_d.fileno())
                        os.link(dst_tmp, dst)
                    finally:
                        dst_tmp.unlink(missing_ok=True)
                created_staged.append(dst)
        except BaseException:
            for partial in created_staged:
                partial.unlink(missing_ok=True)
            raise
        finally:
            # Unlink stage_path BEFORE publishing .complete so st_ctime_ns is final.
            stage_path.unlink(missing_ok=True)

    for dst in dst_list:
        _publish_sidecars_for_existing_payload(dst, digest)

    seen_parents: set[Path] = set()
    for dst in dst_list:
        parent = dst.parent.resolve()
        if parent not in seen_parents:
            seen_parents.add(parent)
            _fsync_directory(dst.parent)
    return digest


def remove_completed_artifact(path: Path | str) -> None:
    """Remove a work-dir artifact payload together with its sidecars if unshared.

    Must ONLY be used in scratch/work directories outside ``$COLAB_CLI_ROOT/output``.
    Refuses to unlink payloads with ``st_nlink > 1`` so an inode hard-linked into
    ``output/`` never has its ``st_ctime_ns`` mutated.

    Args:
        path: Target work-dir payload path whose ``.complete``, ``.sha256``, and payload
            should be removed when unshared (`st_nlink == 1`).
    """
    target = Path(path)
    if target.exists() and target.stat().st_nlink > 1:
        return
    for candidate in (
        Path(f"{target}.complete"),
        Path(f"{target}.sha256"),
        target,
    ):
        candidate.unlink(missing_ok=True)


class RotatingCreateOnlyJSONLWriter:
    """Append-only rotating JSONL writer publishing immutable ``.complete`` segments.

    Each segment is created exclusively with ``open(..., "xb")`` under ``log_dir`` and
    receives one flushed JSON line per update without ever rewriting prior bytes. While a
    segment is open it has no ``.complete`` marker (so ``colab_cli.runtime._exports``
    ignores it); when rotated or closed, the segment is fsynced, closed permanently, and
    sealed with ``.sha256`` and ``.complete`` sidecars.
    """

    def __init__(
        self,
        log_dir: Path | str,
        *,
        prefix: str = "train_log",
        rotate_every_records: int = 64,
        rotate_every_seconds: float = 900.0,
    ) -> None:
        if int(rotate_every_records) < 1:
            raise ValueError(f"rotate_every_records must be >= 1, got {rotate_every_records}")
        if float(rotate_every_seconds) <= 0.0:
            raise ValueError(f"rotate_every_seconds must be > 0, got {rotate_every_seconds}")
        self.log_dir = Path(log_dir)
        self.prefix = str(prefix)
        self.rotate_every_records = int(rotate_every_records)
        self.rotate_every_seconds = float(rotate_every_seconds)

        self._next_segment_idx = 1
        self._active_path: Path | None = None
        self._active_handle: Any = None
        self._active_records = 0
        self._active_opened_mono = 0.0
        self.total_records = 0
        self.completed_segments: list[Path] = []

    def _open_next_segment(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        seg_path = self.log_dir / f"{self.prefix}_{self._next_segment_idx:05d}.jsonl"
        for existing in (seg_path, Path(f"{seg_path}.sha256"), Path(f"{seg_path}.complete")):
            if existing.exists():
                raise FileExistsError(f"Refusing to overwrite immutable log segment: {existing}")
        self._active_handle = seg_path.open("xb")
        self._active_path = seg_path
        self._active_records = 0
        self._active_opened_mono = time.monotonic()
        self._next_segment_idx += 1

    def append(self, record: dict[str, Any]) -> Path:
        """Write a single JSON line to the active segment, rotating when thresholds hit."""
        if self._active_handle is None:
            self._open_next_segment()
        assert self._active_path is not None
        current_path = self._active_path
        encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        self._active_handle.write(encoded)
        self._active_handle.flush()
        self._active_records += 1
        self.total_records += 1

        if (
            self._active_records >= self.rotate_every_records
            or (time.monotonic() - self._active_opened_mono) >= self.rotate_every_seconds
        ):
            self.seal_current_segment()
        return current_path

    def seal_current_segment(self) -> Path | None:
        """Flush, fsync, close, and publish ``.sha256`` + ``.complete`` for the active segment."""
        if self._active_handle is None or self._active_path is None:
            return None
        seg_path = self._active_path
        handle = self._active_handle
        self._active_handle = None
        self._active_path = None
        self._active_records = 0
        try:
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()

        digest = sha256_file(seg_path)
        _publish_sidecars_for_existing_payload(seg_path, digest)
        _fsync_directory(self.log_dir)
        self.completed_segments.append(seg_path)
        return seg_path

    def close(self) -> list[Path]:
        """Seal any open segment and return all completed segment paths."""
        self.seal_current_segment()
        return list(self.completed_segments)


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

#!/usr/bin/env python3
"""Verify and unpack the T16 inputs, then record the H100 pod host receipt.

This helper runs inside the pod only after the archive-only checkout and pinned T16
dependencies are in place.  It does not create, delete, or otherwise manage a pod;
``perf_podbench_20261013.pod_bench`` owns the session safety machinery.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.t16_colab import bundle, run_smoke  # noqa: E402

BUNDLE_SHA256 = "229be2949c99f198b27decdf403b83ec4085d8df2dcfe629b7eff9208593be8e"
EXTRACTED_ROOT = "t16-colab-inputs"
SHM_PATH = Path("/dev/shm")
MIN_SHM_BYTES = 12 * 1024**3


def _cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | None:
    """Return the first Linux CPU model name, if the host exposes one."""
    try:
        for line in cpuinfo.read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"}:
                return value.strip() or None
    except OSError:
        pass
    return None


def _vcpu_count() -> int | None:
    """Return the CPUs in this process's affinity mask, falling back to OS visibility."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count()


def _shared_memory_available(path: Path = SHM_PATH) -> int:
    """Return immediately available bytes on the file-backed ring filesystem."""
    try:
        info = path.statvfs()
    except OSError as error:
        raise RuntimeError(
            f"cannot inspect shared-memory ring directory {path}: {error}"
        ) from error
    return info.f_bavail * info.f_frsize


def p0_ring_bytes() -> dict[str, int]:
    """Return exact two-actor mmap-ring allocations for the committed P0 defaults."""
    from research.perf_learner_20261013 import bench_learner
    from research.redesign_m3c_20261008 import train_m3c
    from research.redesign_m3c_20261008.obs_pack import train_layout

    raw = train_m3c.ring_nbytes(40_000, 60)
    packed = train_m3c.ring_nbytes(40_000, 60, train_layout())
    return {
        "ring_rows_per_actor": 40_000,
        "actors": bench_learner.N_ACTORS,
        "raw_two_actor_bytes": raw * bench_learner.N_ACTORS,
        "packed_two_actor_bytes": packed * bench_learner.N_ACTORS,
    }


def _gpu_models(run: Any = subprocess.run) -> list[str]:
    """Read GPU model names through nvidia-smi without importing a CUDA package."""
    completed = run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        check=True,
        text=True,
    )
    models = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not models:
        raise RuntimeError("nvidia-smi reported no GPU models")
    return models


def host_info(run: Any = subprocess.run) -> dict[str, Any]:
    """Capture the host details needed to interpret T16/T17 throughput rows."""
    return {
        "gpu_models": _gpu_models(run),
        "vcpu_count": _vcpu_count(),
        "vcpu_model": _cpu_model(),
        "shared_memory_path": str(SHM_PATH),
        "shared_memory_available_bytes": _shared_memory_available(),
    }


def prepare(
    bundle_path: Path, destination: Path, output: Path, *, run: Any = subprocess.run
) -> dict[str, Any]:
    """Validate, unpack, and receipt the T16 input bundle for this pod.

    The outer digest is checked before extraction.  ``gcc`` is required because the
    T16 matrix includes compiled learner cells; absence fails before the smoke starts.
    """
    bundle_path, destination, output = Path(bundle_path), Path(destination), Path(output)
    actual_sha256 = bundle.sha256_file(bundle_path)
    if actual_sha256 != BUNDLE_SHA256:
        raise RuntimeError(
            f"T16 input bundle SHA-256 mismatch: expected {BUNDLE_SHA256}, got {actual_sha256}"
        )
    compiler = shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("gcc is required for the compiled T16 learner cells")
    shm_available = _shared_memory_available()
    if shm_available < MIN_SHM_BYTES:
        raise RuntimeError(
            f"{SHM_PATH} has {shm_available} bytes available; T18 requires at least "
            f"{MIN_SHM_BYTES} bytes for the P0 mmap-ring matrix"
        )
    # This is the authoritative T16 runtime gate: it executes a CUDA tensor operation and
    # verifies the exact NumPy/Numba/llvmlite plus torch-2.9 contract before P0 can silently
    # report a CUDA-unavailable skip as a successful benchmark invocation.
    runtime = run_smoke.environment_info()
    extracted = bundle.verify_and_extract(bundle_path, destination)
    expected = destination / EXTRACTED_ROOT
    if extracted.resolve() != expected.resolve() or not extracted.is_dir():
        raise RuntimeError(f"T16 bundle extracted to unexpected root: {extracted}")
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "t18-h100pod-prepare/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bundle": {
            "path": str(bundle_path),
            "sha256": actual_sha256,
            "extracted_root": str(extracted),
            "manifest_sha256": bundle.sha256_file(extracted / "manifest.json"),
        },
        "compiler": {"gcc": compiler},
        "runtime": runtime,
        "p0_rings": p0_ring_bytes(),
        "host": {**host_info(run), "shared_memory_available_bytes": shm_available},
    }
    receipt_path = output / "prepare_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    """Run pod-side preparation and print its receipt JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.bundle, args.destination, args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

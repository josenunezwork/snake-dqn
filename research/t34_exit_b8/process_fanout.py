"""Deterministic single-thread process fan-out for bounded T34 world batches."""

from __future__ import annotations

import hashlib
import importlib
import multiprocessing as mp
import os
import queue
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def default_nproc(*, device: str) -> int:
    """Return the host process count, respecting the Mac two-thread ceiling."""
    count = max(1, int(os.cpu_count() or 1))
    return min(count, 2) if device == "cpu" and os.uname().sysname == "Darwin" else count


def shard_rows(rows: Sequence[Mapping[str, Any]], nproc: int) -> list[list[dict[str, Any]]]:
    """Shard precomputed rows round-robin, preserving one global deterministic order."""
    if nproc < 1:
        raise ValueError("nproc must be positive")
    if not rows:
        raise ValueError("fanout requires at least one world row")
    if nproc > len(rows):
        raise ValueError("nproc cannot exceed the total world count")
    return [[dict(row) for row in rows[rank::nproc]] for rank in range(nproc)]


def _resolve(reference: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    module, name = reference.split(":", 1)
    candidate = getattr(importlib.import_module(module), name)
    if not callable(candidate):
        raise TypeError(f"fanout runner {reference!r} is not callable")
    return candidate


def _worker(reference: str, config: dict[str, Any], result_queue: Any) -> None:
    try:
        threads = int(config.get("threads", 1))
        for key in (
            "NUMBA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[key] = str(threads) if key == "NUMBA_NUM_THREADS" else "1"
        os.environ["SNAKE_DQN_DEVICE"] = str(config.get("device", "cpu"))
        import numba

        numba.config.reload_config()
        numba.set_num_threads(threads)
        result_queue.put((int(config["_fanout_rank"]), _resolve(reference)(config), None))
    except BaseException as exc:  # Parent reports a deterministic rank-specific failure.
        result_queue.put((int(config["_fanout_rank"]), None, f"{type(exc).__name__}: {exc}"))


def _sum_mapping(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    total: dict[str, float] = defaultdict(float)
    for row in rows:
        value = row.get(key, {})
        if isinstance(value, Mapping):
            for name, amount in value.items():
                if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                    total[str(name)] += float(amount)
    return dict(total)


def aggregate_results(
    results: Sequence[Mapping[str, Any]], *, nproc: int, elapsed: float
) -> dict[str, Any]:
    """Aggregate child receipts in rank order without changing world payload bytes."""
    if len(results) != nproc:
        raise ValueError("fanout result count does not match nproc")
    worlds = sum(int(row.get("worlds", 0)) for row in results)
    requested = sum(int(row.get("requested_worlds", 0)) for row in results)
    active = sum(int(row.get("active_row_steps", 0)) for row in results)
    resident = sum(int(row.get("resident_row_steps", 0)) for row in results)
    decisions = sum(int(row.get("decisions", 0)) for row in results)
    exports = sum(int(row.get("export_bytes", 0)) for row in results)
    completed_work = sum(int(row.get("completed_world_row_steps", 0)) for row in results)
    unfinished_work = sum(int(row.get("unfinished_row_steps", 0)) for row in results)
    profile = _sum_mapping(results, "teacher_profile")
    profile["active_row_steps"] = active
    profile["resident_row_steps"] = resident
    mixes: dict[str, dict[str, int]] = {}
    for row in results:
        for name, metrics in dict(row.get("mixes", {})).items():
            target = mixes.setdefault(str(name), {"searched": 0, "expanded": 0, "worlds": 0})
            for key in target:
                target[key] += int(dict(metrics).get(key, 0))
    return {
        "worlds": worlds,
        "requested_worlds": requested,
        "complete": all(bool(row.get("complete")) for row in results),
        "invalid_reason": next(
            (row.get("invalid_reason") for row in results if row.get("invalid_reason")), None
        ),
        "elapsed_seconds": elapsed,
        "decisions": decisions,
        "decisions_per_second": decisions / max(elapsed, 1e-9),
        "worlds_per_hour": worlds * 3600 / max(elapsed, 1e-9),
        "export_bytes": exports,
        "export_bytes_per_world": exports / worlds if worlds else None,
        "active_row_steps": active,
        "resident_row_steps": resident,
        "active_row_steps_per_second": active / max(elapsed, 1e-9),
        "active_resident_fraction": active / max(resident, 1),
        "executed_row_steps_per_world": completed_work / worlds if worlds else None,
        "completed_world_row_steps": completed_work,
        "unfinished_row_steps": unfinished_work,
        "unfinished_worlds": requested - worlds,
        "world_rate_scope": "completed worlds; work includes unfinished roots",
        "configured_root_capacity": sum(
            int(row.get("configured_root_capacity", 0)) for row in results
        ),
        "peak_root_worlds": sum(int(row.get("peak_root_worlds", 0)) for row in results),
        "search_batch_peak": max(
            (int(row.get("search_batch_peak", 0)) for row in results), default=0
        ),
        "root_inference_bucket_peak": max(
            (int(row.get("root_inference_bucket_peak", 0)) for row in results), default=0
        ),
        "branch_bucket_capacity": max(
            (int(row.get("branch_bucket_capacity", 0)) for row in results), default=0
        ),
        "teacher_profile": profile,
        "host_stage_seconds": _sum_mapping(results, "host_stage_seconds"),
        "gpu_utilization_samples_percent": [],
        "fanout": {
            "nproc": nproc,
            "worker_reports": list(results),
            "aggregation": "fixed process-rank order; disjoint round-robin world shards",
        },
    }


def run_fanout(
    config: Mapping[str, Any], *, runner: str, rows: Sequence[Mapping[str, Any]], nproc: int
) -> dict[str, Any]:
    """Run independent one-thread workers and aggregate their bounded receipts."""
    shards = shard_rows(rows, nproc)
    base_output = Path(str(config["output_dir"]))
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    children = []
    started = time.monotonic()
    # The registered wall bound starts inside the collector after process
    # import/JIT setup.  Reserve a bounded startup/export allowance so the
    # parent does not kill a cold CPU lane before it can emit its own receipt.
    deadline = (
        started
        + float(config.get("seconds", 0))
        + float(config.get("fanout_startup_grace_seconds", 300.0))
    )
    received: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    try:
        for rank, shard in enumerate(shards):
            worker_config = dict(config)
            worker_config.update(
                _fanout_worker=True,
                _fanout_rank=rank,
                nproc=1,
                rows=shard,
                row_indices=list(range(rank, len(rows), nproc)),
                max_worlds=len(shard),
                output_dir=base_output / f"process-{rank:03d}" / "worlds",
            )
            child = context.Process(target=_worker, args=(runner, worker_config, result_queue))
            child.start()
            children.append(child)
        while len(received) + len(failures) < len(children):
            try:
                rank, result, failure = result_queue.get(timeout=1)
            except queue.Empty:
                if time.monotonic() >= deadline:
                    break
                if any(child.exitcode not in (None, 0) for child in children):
                    break
                if all(not child.is_alive() for child in children):
                    break
                continue
            if failure is None:
                received[int(rank)] = dict(result)
            else:
                failures[int(rank)] = str(failure)
                break
    finally:
        for child in children:
            child.join(timeout=5)
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
            if child.is_alive():
                child.kill()
                child.join(timeout=5)
    failed_exit = [index for index, child in enumerate(children) if child.exitcode not in (0, None)]
    if failures or failed_exit or len(received) != nproc:
        raise RuntimeError(
            "fanout workers failed: "
            f"errors={failures}, exitcodes={[child.exitcode for child in children]}"
        )
    return aggregate_results(
        [received[index] for index in range(nproc)], nproc=nproc, elapsed=time.monotonic() - started
    )


def export_digest(output_dir: Path) -> str:
    """Digest payload bytes by world id, independent of fanout directory layout."""
    pairs = []
    for path in sorted(output_dir.rglob("world-*.json")):
        payload = path.read_bytes()
        import json

        pairs.append((str(json.loads(payload)["world_id"]), hashlib.sha256(payload).hexdigest()))
    return hashlib.sha256(repr(sorted(pairs)).encode()).hexdigest()

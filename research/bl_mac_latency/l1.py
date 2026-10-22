"""Measure frozen-root CPU/MPS diagnostics alongside CPU Watch6 paced at 24fps.

The full Y1 Mac receipt supplies an immutable semantic reference. Actual spawned
workers report admission-to-decoded-visibility latency, while the coordinator
runs ordinary game updates on a fixed integer tick schedule. These constructed
stress roots cannot prove production reachability or consumption; production
L1 qualification therefore remains NOT_READY with no nominated cell.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from research.bl_paced_identity.authorization import load_probe_authorization
from research.bl_served_search.exports import export_json, verify_export
from research.bl_served_search.ruler import (
    MIN_BATCHES,
)
from research.bl_served_search.ruler import RECEIPT_SCHEMA as RULER_RECEIPT_SCHEMA
from research.bl_served_search.ruler import (
    canonical,
    sha256_bytes,
    source_identity,
    validate_fixtures,
)
from research.mac_lease import DEFAULT_LEASE_PATH, held

SCHEMA = "bl-mac-l1-prime-diagnostic/v1"
PACED_STATUS = "NOT_READY"
MPS_STATUS = "NOT_MEASURED"
ROOT_PROVIDER_REASON = "INSTRUMENT_MISSING: production Watch6 offered-root provider is absent"

# The owner selected these bars after Astra 04.  They are retained in the
# receipt even though this diagnostic cannot evaluate them without real roots.
S_P99_BARS_SECONDS = {1: 1.458, 2: 0.830, 3: 1.250, 4: 1.458}
C_P99_BARS_SECONDS = {"2.4": 1 / 2.4 - 2 / 24, "1.6": 1 / 1.6 - 2 / 24}
BURST_BARS = {
    "12s": {"2.4": 29, "1.6": 20},
    "15s": {"2.4": 36, "1.6": 24},
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def diagnostic_source_identity() -> dict[str, Any]:
    """Bind this diagnostic layer separately from the frozen T3 math identity."""
    root = Path(__file__).resolve().parents[2]
    names = (
        "research/bl_mac_latency/l1.py",
        "research/bl_mac_latency/host.py",
        "research/bl_served_search/ruler_operator.py",
        "research/bl_paced_identity/adapter.py",
        "research/bl_paced_identity/transport.py",
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "git_head": head,
        "files": {name: _sha256_bytes((root / name).read_bytes()) for name in names},
    }


def nearest_rank(samples: Iterable[float], percentile: float) -> float | None:
    """Return the inclusive nearest-rank percentile, or ``None`` for no data."""
    values = sorted(float(value) for value in samples)
    if not values:
        return None
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    return values[math.ceil(percentile * len(values)) - 1]


def distribution(samples: Iterable[float]) -> dict[str, Any]:
    """Summarize raw seconds with the predeclared nearest-rank estimator."""
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("timing samples must be finite non-negative seconds")
    return {
        "quantile": "nearest-rank/inclusive/v1",
        "count": len(values),
        "p50_seconds": nearest_rank(values, 0.50),
        "p95_seconds": nearest_rank(values, 0.95),
        "p99_seconds": nearest_rank(values, 0.99),
        "max_seconds": max(values) if values else None,
    }


def fixed_cells() -> list[dict[str, Any]]:
    """Return the frozen requested order; K1 precedes every expanded cell."""
    cells = [{"cell_id": "cpu-w1-k1", "device": "cpu", "workers": 1, "max_pending": 1}]
    cells.extend(
        {
            "cell_id": f"cpu-w{workers}-k8",
            "device": "cpu",
            "workers": workers,
            "max_pending": 8,
        }
        for workers in range(1, 5)
    )
    cells.append({"cell_id": "mps-w1-k1", "device": "mps", "workers": 1, "max_pending": 1})
    cells.extend(
        {
            "cell_id": f"mps-w{workers}-k8",
            "device": "mps",
            "workers": workers,
            "max_pending": 8,
        }
        for workers in range(1, 5)
    )
    return cells


def _read_completed_json(path: Path) -> tuple[dict[str, Any], str]:
    verify_export(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    # ``exports.export_json`` deliberately owns its formatting.  The immutable
    # digest + completion sidecars above bind bytes; do not impose the paced
    # journal's compact-JSON convention on ruler artifacts.
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value, _sha256_bytes(raw)


def load_fixtures(path: Path) -> tuple[list[dict[str, Any]], str, str]:
    """Load the complete nine-root bank without rebuilding or modifying it."""
    value, file_sha = _read_completed_json(Path(path))
    return validate_fixtures(value), file_sha, sha256_bytes(canonical(value).encode())


def load_cpu_reference(path: Path, ruler_fixture_sha256: str) -> dict[str, Any]:
    """Validate and reduce the complete Y1 Mac ruler receipt.

    The output keeps each raw ``worker_wall_seconds`` value so an audit can
    recompute tails.  This is explicitly a dense-batch micro-timing receipt,
    not an IPC/visibility or consumption measurement.
    """
    receipt, receipt_sha = _read_completed_json(Path(path))
    if receipt.get("schema") != RULER_RECEIPT_SCHEMA:
        raise ValueError("cpu reference has an unexpected ruler receipt schema")
    if receipt.get("platform", {}).get("label") != "mac":
        raise ValueError("cpu reference must be a Mac ruler receipt")
    if receipt.get("source_identity") != source_identity():
        raise ValueError("cpu reference source identity differs from current exact ruler")
    if receipt.get("fixture_sha256") != ruler_fixture_sha256:
        raise ValueError("cpu reference used a different frozen fixture bank")
    if int(receipt.get("min_batches_per_fixture", 0)) < MIN_BATCHES:
        raise ValueError("cpu reference has fewer than 30 required batches per fixture")

    rows = receipt.get("records")
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError("cpu reference must contain exactly nine fixture records")
    by_fixture: dict[str, dict[str, Any]] = {}
    all_samples: list[float] = []
    for row in rows:
        fixture_id = row.get("fixture_id")
        batches = row.get("batches")
        if not isinstance(fixture_id, str) or fixture_id in by_fixture:
            raise ValueError("cpu reference fixture identities are invalid")
        if not isinstance(batches, list) or len(batches) < MIN_BATCHES:
            raise ValueError(f"cpu reference {fixture_id} has incomplete batches")
        samples = []
        for batch in batches:
            if not isinstance(batch, dict) or batch.get("semantic", {}).get("fired") is not True:
                raise ValueError(f"cpu reference {fixture_id} batch is not an actual dense firing")
            value = batch.get("worker_wall_seconds")
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"cpu reference {fixture_id} has invalid worker timing")
            samples.append(float(value))
        semantic_rows = {
            json.dumps(batch["semantic"], sort_keys=True, separators=(",", ":"))
            for batch in batches
        }
        if len(semantic_rows) != 1:
            raise ValueError(f"cpu reference {fixture_id} has semantic drift across batches")
        by_fixture[fixture_id] = {"samples_seconds": samples, "distribution": distribution(samples)}
        all_samples.extend(samples)
    return {
        "path": str(Path(path).resolve()),
        "sha256": receipt_sha,
        "fixture_sha256": ruler_fixture_sha256,
        "kind": "UNCHANGED_EXACT_T3_DENSE_BATCH_MICRO_TIMING",
        "not_latency_evidence": [
            "no production offered roots",
            "no IPC coordinator visibility timestamps",
            "no exact-state validation or consumption timestamps",
            "no paced Watch6 update/tick measurements",
        ],
        "all_samples_seconds": all_samples,
        "distribution": distribution(all_samples),
        "per_fixture": by_fixture,
    }


def _cell_result(cell: dict[str, Any], micro_timing: dict[str, Any]) -> dict[str, Any]:
    """Construct one fail-closed paced cell without manufacturing a root."""
    status = PACED_STATUS
    reason = ROOT_PROVIDER_REASON
    return {
        **cell,
        "status": status,
        "reason": reason,
        "configured_only": True,
        "executed_worker_processes": 0,
        "paced_execution": {
            "status": "NOT_MEASURED",
            "required_provider": "production Watch6 offered-root provider",
            "tick_rate_fps": 24,
            "five_decision_spacing": True,
            "S_seconds": [],
            "C_seconds": [],
            "worker_occupancy_seconds": [],
            "valid_roots": 0,
            "valid_consumptions": 0,
            "tick_overruns": None,
            "update_p95_seconds": None,
            "burst_coverage": [],
        },
        "bars": {
            "S_p99_seconds": S_P99_BARS_SECONDS[cell["workers"]],
            "C_p99_seconds": C_P99_BARS_SECONDS if cell["max_pending"] == 1 else None,
            "burst_valid_consumptions": BURST_BARS,
            "coverage_governing_roots_per_second": 2.4,
            "coverage_sensitivity_roots_per_second": 1.6,
            "zero_tick_overruns_required": True,
            "watch6_update_p95_seconds": 0.011688,
        },
        "micro_timing_reference": {
            "receipt_sha256": micro_timing["sha256"],
            "kind": micro_timing["kind"],
            "distribution": micro_timing["distribution"],
        },
        "bar_verdict": "NOT_EVALUABLE",
    }


_WORKER_OPERATOR: Any | None = None
_WORKER_STOP: Any | None = None
_WORKER_DEADLINE = float("inf")


def _move_torch_modules(policy: Any, device: str) -> list[dict[str, str]]:
    """Move only actual policy networks and attest every resulting placement."""
    import torch

    attestation = []
    for name in ("net", "network", "student"):
        module = getattr(policy, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(device)
            module.eval()
            parameter = next(module.parameters(), None)
            observed = str(parameter.device) if parameter is not None else "parameterless"
            attestation.append(
                {
                    "attribute": name,
                    "requested": device,
                    "observed": observed,
                    "dtype": None if parameter is None else str(parameter.dtype),
                    "synchronized": device == "mps",
                }
            )
    if hasattr(policy, "network") and getattr(policy, "network") is not None:
        network = getattr(policy, "network")
        parameter = next(network.parameters(), None)
        if parameter is not None and hasattr(policy, "device"):
            # Vector61SimdPolicy's input transfer is keyed to this cache.
            policy.device = parameter.device
    return attestation


class DeviceOperator:
    """The unchanged exact T3 operator with an explicit diagnostic device move.

    It delegates world construction and dense selection to ``ExactT3Operator``.
    CPU uses its normal initialization.  MPS is a diagnostic backend only: all
    moved module placements are returned with each result and synchronised on
    both sides of batch timing.  The frozen root bytes intentionally omit model
    devices, so a root digest remains a state identity rather than a placement
    assertion.
    """

    def __init__(self, device: str) -> None:
        from research.bl_served_search.ruler_operator import create_stop_operator

        if device not in {"cpu", "mps"}:
            raise ValueError("device must be cpu or mps")
        self.device = device
        self._exact = create_stop_operator()
        self.placement: dict[str, list[dict[str, str]]] = {}

    def _sync(self) -> None:
        if self.device == "mps":
            import torch

            torch.mps.synchronize()

    def _world(self, row: dict[str, Any]) -> Any:
        world = self._exact._world(row)
        attestation: list[dict[str, str]] = []
        if self.device == "mps":
            for label, policy in [
                ("hero", world.hero),
                *[(f"opponent-{i}", p) for i, p in enumerate(world.opponents)],
            ]:
                for value in _move_torch_modules(policy, "mps"):
                    attestation.append({"policy": label, **value})
            self._sync()
        else:
            for label, policy in [
                ("hero", world.hero),
                *[(f"opponent-{i}", p) for i, p in enumerate(world.opponents)],
            ]:
                for value in _move_torch_modules(policy, "cpu"):
                    attestation.append({"policy": label, **value})
        self.placement[str(row["fixture_id"])] = attestation
        return world

    def batch(self, fixture: dict[str, Any]) -> dict[str, Any]:
        # ExactT3Operator.batch calls _restore, so seed its private root cache
        # through our device-specific world constructor on a cache miss.
        self._ensure_root(fixture)
        self._sync()
        result = self._exact.batch(fixture)
        self._sync()
        return {**result, "placement": self.placement.get(str(fixture["fixture_id"]), [])}

    def _ensure_root(self, fixture: dict[str, Any]) -> None:
        fixture_id = str(fixture["fixture_id"])
        if fixture_id in self._exact._roots:
            return
        world = self._world(fixture)
        from research.bl_served_search.fixture_geometry import install
        from research.bl_served_search.ruler_operator import _clone_root, _root_bytes

        install(world, int(fixture["target_length"]))

        if _sha256_bytes(_root_bytes(world)) != fixture["root_sha256"]:
            raise RuntimeError("device adapter changed frozen root state")
        self._exact._roots[fixture_id] = _clone_root(world)

    def tick_world(self, fixture: dict[str, Any]) -> Any:
        """Restore a real Watch6 fixture for ordinary no-search paced ticks."""
        self._ensure_root(fixture)
        return self._exact._restore(fixture)


def _worker_init(device: str, stop_event: Any, deadline: float) -> None:
    global _WORKER_OPERATOR, _WORKER_STOP, _WORKER_DEADLINE
    _WORKER_STOP, _WORKER_DEADLINE = stop_event, deadline
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = device
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
    _WORKER_OPERATOR = DeviceOperator(device)


def _check_worker_stop() -> None:
    """Cooperate between rollout steps without terminating a process."""
    if (_WORKER_STOP is not None and _WORKER_STOP.is_set()) or time.monotonic() >= _WORKER_DEADLINE:
        raise RuntimeError("measurement stopped before next rollout step")


def _worker_batch(fixture: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_OPERATOR is None:
        raise RuntimeError("diagnostic worker was not initialized")
    from research.t3_rollout_labels_20261015 import bank_engine

    originals = {name: getattr(bank_engine, name) for name in ("choose_and_step", "step_prepared")}

    def guarded(original: Callable[..., Any]) -> Callable[..., Any]:
        def step(*args: Any, **kwargs: Any) -> Any:
            _check_worker_stop()
            return original(*args, **kwargs)

        return step

    started = time.monotonic_ns()
    _check_worker_stop()
    try:
        for name, original in originals.items():
            setattr(bank_engine, name, guarded(original))
        result = _WORKER_OPERATOR.batch(fixture)
        import numba
        import torch

        completed = time.monotonic_ns()
        return {
            "worker_started_ns": started,
            "worker_completed_ns": completed,
            "worker_pid": os.getpid(),
            "execution_environment": {
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": torch.get_num_interop_threads(),
                "requested_device": os.environ.get("SNAKE_DQN_DEVICE"),
                "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
                "intentional_cpu_work": (
                    "simulator, raster/v8 features, opponent veto, " "CPU-readable export"
                ),
                "warmup": "none; cold worker/root initialization included in diagnostic S",
            },
            "semantic": result["semantic"],
            "native_v8": result["native_v8"],
            "stage_seconds": result["stage_seconds"],
            "worker_wall_seconds": (completed - started) / 1e9,
            "worker_threads": {
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMBA_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
                "torch": torch.get_num_threads(),
                "numba": numba.get_num_threads(),
            },
            "placement": result.get("placement", []),
        }
    finally:
        for name, original in originals.items():
            setattr(bank_engine, name, original)


def _semantic_reference(cpu_reference: dict[str, Any]) -> dict[str, int]:
    """Compare chosen actions to all-branch Y1, allowing exact-stop skipped rows."""
    receipt, _ = _read_completed_json(Path(cpu_reference["path"]))
    return {
        str(row["fixture_id"]): int(row["batches"][0]["semantic"]["chosen"])
        for row in receipt["records"]
    }


def _paced_ticks(
    operator: DeviceOperator,
    fixture: dict[str, Any],
    duration_seconds: float,
    guard: Callable[[], dict[str, Any]],
    collect_ready: Callable[[], None],
    keep_running: Callable[[], bool] | None = None,
    deadline: float = float("inf"),
) -> dict[str, Any]:
    """Run ordinary Watch6 ticks at 24fps while worker work is observed off-tick."""
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step

    world = operator.tick_world(fixture)
    epoch_ns = time.monotonic_ns()
    tick_index = 0
    durations: list[float] = []
    tick_records: list[dict[str, Any]] = []
    overruns = 0
    guards = []
    stop_reason = None
    while (time.monotonic_ns() - epoch_ns) / 1e9 < duration_seconds or (
        keep_running is not None and keep_running()
    ):
        if time.monotonic() >= deadline:
            stop_reason = "measurement budget exhausted during paced work"
            break
        target_ns = epoch_ns + tick_index * 1_000_000_000 // 24
        now_ns = time.monotonic_ns()
        if now_ns < target_ns:
            time.sleep((target_ns - now_ns) / 1e9)
        iteration_started = time.monotonic_ns()
        try:
            collect_ready()
            tick_started = time.monotonic_ns()
            hero_alive_before = bool(world.sim.alive[0, 0])
            choose_and_step(world, capture_obs=False)
            update_finished = time.monotonic_ns()
            elapsed = (update_finished - tick_started) / 1e9
            durations.append(elapsed)
            tick_index += 1
            collect_ready()
            if tick_index % 24 == 0:
                guards.append(guard())
            finished = time.monotonic_ns()
            next_target_ns = epoch_ns + tick_index * 1_000_000_000 // 24
            overrun = finished >= next_target_ns
            overruns += int(overrun)
            tick_records.append(
                {
                    "tick": tick_index - 1,
                    "target_ns": target_ns,
                    "in_burst": target_ns < epoch_ns + int(duration_seconds * 1e9),
                    "iteration_started_ns": iteration_started,
                    "update_started_ns": tick_started,
                    "update_finished_ns": update_finished,
                    "finished_ns": finished,
                    "start_lateness_ns": max(0, iteration_started - target_ns),
                    "full_iteration_seconds": (finished - iteration_started) / 1e9,
                    "overrun": overrun,
                    "hero_alive_before": hero_alive_before,
                    "hero_alive": bool(world.sim.alive[0, 0]),
                    "hero_length": int(world.sim.length[0, 0]),
                }
            )
        except Exception as exc:
            stop_reason = f"{type(exc).__name__}: {exc}"
            break
    return {
        "epoch_ns": epoch_ns,
        "finished_ns": time.monotonic_ns(),
        "tick_count": len(durations),
        "completed_full_iterations": len(tick_records),
        "tick_durations_seconds": durations,
        "tick_records": tick_records,
        "tick_overruns": overruns,
        "alive_tick_count": sum(int(row["hero_alive"]) for row in tick_records),
        "terminal_hero_no_respawn": True,
        "hero_alive_end": bool(world.sim.alive[0, 0]),
        "hero_length_end": int(world.sim.length[0, 0]),
        "guards": guards,
        "stop_reason": stop_reason,
    }


def run_stress_cell(
    cell: dict[str, Any],
    fixture: dict[str, Any],
    cpu_reference: dict[str, Any],
    *,
    burst_seconds: float,
    guard: Callable[[], dict[str, Any]],
    deadline: float = float("inf"),
) -> dict[str, Any]:
    """Measure an explicitly non-qualifying diagnostic worker/tick probe.

    At most W roots are admitted: K is configured and attested but intentionally
    not saturated, because the roots are ineligible for production consumption.
    Visibility is the coordinator's first completed-future observation.  It is
    a real IPC timing but never called a valid root or consumption.
    """
    if (
        not isinstance(burst_seconds, (int, float))
        or not math.isfinite(burst_seconds)
        or burst_seconds <= 0
    ):
        raise ValueError("burst_seconds must be finite and positive")
    guard()
    device, workers = str(cell["device"]), int(cell["workers"])
    if device == "mps":
        import torch

        if not torch.backends.mps.is_available():
            return {"status": MPS_STATUS, "reason": "MPS_UNAVAILABLE", "samples": [], "tick": None}
    expected = _semantic_reference(cpu_reference)
    if fixture.get("mix") != "frozen":
        raise ValueError("paced diagnostic requires a frozen Watch6 fixture")
    chosen = [fixture for _ in range(workers)]
    samples: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    pool = concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(device, stop_event, deadline),
    )
    futures = {}
    admitted_count = 0

    def collect_ready() -> None:
        """Decode IPC payloads at the first coordinator observation, never later."""
        for future in list(futures):
            if not future.done():
                continue
            index, row, admitted_ns = futures.pop(future)
            item: dict[str, Any] = {
                "sample_index": index,
                "fixture_id": row["fixture_id"],
                "admitted_ns": admitted_ns,
            }
            try:
                payload = future.result()
                visible = time.monotonic_ns()
                item["visible_ns"] = visible
                from research.bl_served_search.native_ruler import validate_batch

                validate_batch(payload, row, "exact_stop")
                item.update(
                    worker_started_ns=payload["worker_started_ns"],
                    worker_completed_ns=payload["worker_completed_ns"],
                    placement=payload["placement"],
                    worker_pid=payload["worker_pid"],
                    execution_environment=payload["execution_environment"],
                    semantic_matches_cpu_reference=(
                        payload["semantic"]["chosen"] == expected[row["fixture_id"]]
                    ),
                    native_v8=payload["native_v8"],
                    S_diagnostic_seconds=(visible - admitted_ns) / 1e9,
                    worker_occupancy_seconds=(
                        payload["worker_completed_ns"] - payload["worker_started_ns"]
                    )
                    / 1e9,
                )
            except Exception as exc:
                item["visible_ns"] = time.monotonic_ns()
                item.update(error=f"{type(exc).__name__}: {exc}")
            samples.append(item)

    stop_reason = None
    tick: dict[str, Any] = {}
    try:
        try:
            for index, row in enumerate(chosen):
                admitted_ns = time.monotonic_ns()
                futures[pool.submit(_worker_batch, row)] = (index, row, admitted_ns)
                admitted_count += 1
            # The paced load remains CPU Watch6 physics/AI for every search backend;
            # only the off-tick diagnostic dense worker changes device.
            tick_operator = DeviceOperator("cpu")
            tick = _paced_ticks(
                tick_operator,
                fixture,
                burst_seconds,
                guard,
                collect_ready,
                keep_running=lambda: bool(futures),
                deadline=deadline,
            )
            stop_reason = tick.get("stop_reason")
        except Exception as exc:
            stop_reason = f"{type(exc).__name__}: {exc}"
        if stop_reason is not None:
            stop_event.set()
        # Draining is deliberately off-tick and never cancels a process.  These
        # roots may take far longer than the burst window; their actual tails
        # remain visible in the receipt.
        while futures:
            concurrent.futures.wait(
                futures, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
            )
            print(
                json.dumps(
                    {
                        "event": "l1_drain_progress",
                        "pending_workers": len(futures),
                        "completed_samples": len(samples),
                    }
                ),
                flush=True,
            )
            try:
                if time.monotonic() >= deadline:
                    raise RuntimeError("measurement budget exhausted")
                guard()
            except Exception as exc:
                stop_reason = f"{type(exc).__name__}: {exc}"
                stop_event.set()
                samples.append({"guard_failure_during_drain": stop_reason})
            collect_ready()
    finally:
        if futures:
            stop_event.set()
        pool.shutdown(wait=True, cancel_futures=False)
    completed = [row for row in samples if "S_diagnostic_seconds" in row]
    semantic_invalid = any(row.get("semantic_matches_cpu_reference") is False for row in completed)
    worker_errors = [row for row in samples if "error" in row]
    return {
        "status": (
            "STOPPED"
            if stop_reason
            else (
                "INVALID"
                if semantic_invalid
                else (
                    "INCOMPLETE"
                    if worker_errors or len(completed) != workers
                    else "DIAGNOSTIC_COMPLETE"
                )
            )
        ),
        "stop_reason": stop_reason,
        "configured_workers": workers,
        "configured_max_pending": int(cell["max_pending"]),
        "actual_max_pending": admitted_count,
        "fixture_id": fixture["fixture_id"],
        "target_length": int(fixture["target_length"]),
        "admitted_roots": admitted_count,
        "pending_capacity_saturated": False,
        "samples": samples,
        "S_diagnostic_distribution": distribution(row["S_diagnostic_seconds"] for row in completed),
        "worker_occupancy_distribution": distribution(
            row["worker_occupancy_seconds"] for row in completed
        ),
        "tick": {**tick, "absolute_clock": "T(n)=t0+floor(n*1e9/24)"},
        "tick_backend": "cpu",
        "valid_roots": 0,
        "valid_consumptions": 0,
        "watch6_load_continued_until_worker_visibility": True,
        "qualification": "STRESS_ROOT_DIAGNOSTIC_ONLY",
    }


def build_receipt(
    *,
    fixtures_path: Path,
    authorization_path: Path,
    cpu_reference_path: Path,
    budget_seconds: int,
    guard: Callable[[], dict[str, Any]],
    stress_probe: bool = False,
    burst_seconds: tuple[float, ...] = (12.0, 15.0),
    attempt_export_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the L1 diagnostic receipt after a host guard snapshot.

    ``guard`` is supplied by :mod:`research.bl_mac_latency.host`; it raises on
    AC/lid/thermal failure.  It is intentionally invoked before every read that
    could otherwise be followed by a long measurement in a future extension.
    """
    if type(budget_seconds) is not int or budget_seconds <= 0:
        raise ValueError("budget_seconds must be a positive integer")
    guard_snapshot = guard()
    fixtures, fixture_file_sha, fixture_ruler_sha = load_fixtures(Path(fixtures_path))
    guard_snapshot_after_fixtures = guard()
    authorization, authorization_sha = _read_completed_json(Path(authorization_path))
    probe_authorization = load_probe_authorization(Path(authorization_path))
    cpu_reference = load_cpu_reference(Path(cpu_reference_path), fixture_ruler_sha)
    guard_snapshot_after_reference = guard()
    cells = [_cell_result(cell, cpu_reference) for cell in fixed_cells()]
    for cell in cells:
        cell["probe_authorized"] = probe_authorization.permits(
            int(cell["workers"]), int(cell["max_pending"]), str(cell["device"])
        )
    stress_started = time.monotonic()
    deadline = stress_started + budget_seconds
    diagnostic_source_before = diagnostic_source_identity()
    stopped_reason: str | None = None
    if stress_probe:
        frozen_by_length = {
            int(row["target_length"]): row for row in fixtures if row.get("mix") == "frozen"
        }
        if set(frozen_by_length) != {800, 1200, 1500}:
            raise ValueError("stress probe requires frozen L800/L1200/L1500 fixtures")
        for result in cells:
            attempts = []
            if not result["probe_authorized"]:
                result["diagnostic_stress_attempts"] = [
                    {"status": "NOT_MEASURED", "reason": "PROBE_AUTHORIZATION_DENIED"}
                ]
                continue
            for target_length in (800, 1200, 1500):
                for duration in burst_seconds:
                    reserve_seconds = 300.0 + duration
                    if time.monotonic() >= deadline - reserve_seconds:
                        attempts.append(
                            {
                                "status": "NOT_MEASURED",
                                "reason": "BUDGET_EXHAUSTED",
                                "target_length": target_length,
                                "burst_seconds": duration,
                            }
                        )
                        continue
                    print(
                        json.dumps(
                            {
                                "event": "l1_attempt_start",
                                "cell": result["cell_id"],
                                "target_length": target_length,
                                "burst_seconds": duration,
                                "remaining_seconds": max(0.0, deadline - time.monotonic()),
                            }
                        ),
                        flush=True,
                    )
                    try:
                        result["configured_only"] = False
                        attempt = {
                            "burst_seconds": duration,
                            **run_stress_cell(
                                result,
                                frozen_by_length[target_length],
                                cpu_reference,
                                burst_seconds=duration,
                                guard=guard,
                                deadline=deadline,
                            ),
                        }
                    except Exception as exc:
                        stopped_reason = f"{type(exc).__name__}: {exc}"
                        attempt = {
                            "target_length": target_length,
                            "burst_seconds": duration,
                            "status": "STOPPED",
                            "reason": stopped_reason,
                        }
                    if attempt.get("stop_reason"):
                        stopped_reason = attempt["stop_reason"]
                    attempts.append(attempt)
                    if attempt_export_dir is not None:
                        export_json(
                            attempt_export_dir
                            / f"{result['cell_id']}-l{target_length}-{int(duration)}s.json",
                            attempt,
                        )
                    print(
                        json.dumps(
                            {
                                "event": "l1_attempt_end",
                                "cell": result["cell_id"],
                                "target_length": target_length,
                                "burst_seconds": duration,
                                "status": attempt["status"],
                                "remaining_seconds": max(0.0, deadline - time.monotonic()),
                            }
                        ),
                        flush=True,
                    )
                    if stopped_reason is not None:
                        break
                if stopped_reason is not None:
                    break
            result["diagnostic_stress_attempts"] = attempts
            # Stress fixtures do not become production roots even when their
            # device semantics agree with the Y1 CPU reference.
            result["bar_verdict"] = "NOT_EVALUABLE"
            if any(
                attempt.get("status") in {"DIAGNOSTIC_COMPLETE", "INVALID", "INCOMPLETE"}
                for attempt in attempts
            ):
                result["executed_worker_processes"] = int(result["workers"])
            if stopped_reason is not None:
                break
    return {
        "schema": SCHEMA,
        "status": "NOT_READY",
        "verdict": "NOT_READY(INSTRUMENT_MISSING)",
        "reason": ROOT_PROVIDER_REASON,
        "nomination": None,
        "budget_seconds": budget_seconds,
        "fixture_bank": {
            "path": str(Path(fixtures_path).resolve()),
            "file_sha256": fixture_file_sha,
            "ruler_sha256": fixture_ruler_sha,
            "fixture_ids": [row["fixture_id"] for row in fixtures],
        },
        "authorization": {
            "path": str(Path(authorization_path).resolve()),
            "sha256": authorization_sha,
            "owner_record": authorization,
            "parsed_probe_authorization": probe_authorization.evidence(),
            "scope": "PROBE_ONLY; never serving authority",
        },
        "guard_snapshots": [
            guard_snapshot,
            guard_snapshot_after_fixtures,
            guard_snapshot_after_reference,
        ],
        "source_identity": source_identity(),
        "diagnostic_source_identity_before": diagnostic_source_before,
        "diagnostic_source_identity_after": diagnostic_source_identity(),
        "diagnostic_source_identity_stable": diagnostic_source_before
        == diagnostic_source_identity(),
        "cpu_reference": cpu_reference,
        "cells": cells,
        "qualification_boundary": {
            "may_pass_l1_prime": False,
            "why": [
                ROOT_PROVIDER_REASON,
                "no production offered-root consumption trace",
                "zero valid consumptions cannot pass",
                "micro-timing worker wall time is not S or C",
            ],
        },
        "stress_probe_stopped_reason": stopped_reason if stress_probe else None,
    }


def run(
    *,
    fixtures_path: Path,
    out: Path,
    authorization_path: Path,
    cpu_reference_path: Path,
    budget_seconds: int,
    guard: Callable[[], dict[str, Any]],
    stress_probe: bool = False,
    burst_seconds: tuple[float, ...] = (12.0, 15.0),
    mac_lease: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Create-only write of a bounded diagnostic receipt and its sidecars."""
    receipt = build_receipt(
        fixtures_path=fixtures_path,
        authorization_path=authorization_path,
        cpu_reference_path=cpu_reference_path,
        budget_seconds=budget_seconds,
        guard=guard,
        stress_probe=stress_probe,
        burst_seconds=burst_seconds,
        attempt_export_dir=(
            Path(out).parent / (Path(out).stem + "-attempts") if stress_probe else None
        ),
    )
    if mac_lease is not None:
        receipt["mac_lease"] = mac_lease
    export_json(Path(out), receipt)
    return receipt


def _run(
    args: argparse.Namespace,
    check_guard: Callable[[], dict[str, Any]],
    mac_lease: dict[str, object],
) -> None:
    run(
        fixtures_path=args.fixtures,
        out=args.out,
        authorization_path=args.authorization,
        cpu_reference_path=args.cpu_reference,
        budget_seconds=args.budget_seconds,
        guard=check_guard,
        stress_probe=args.stress_probe,
        burst_seconds=tuple(args.burst_seconds or (12.0, 15.0)),
        mac_lease=mac_lease,
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, required=True)
    parser.add_argument(
        "--check", action="store_true", help="validate inputs only; never start workers"
    )
    parser.add_argument("--lease-owner", default="bl-mac-l1")
    parser.add_argument("--lease-path", type=Path, default=DEFAULT_LEASE_PATH)
    parser.add_argument(
        "--stress-probe",
        action="store_true",
        help="run diagnostic-only frozen-root workers and 24fps Watch6 ticks",
    )
    parser.add_argument(
        "--burst-seconds",
        action="append",
        type=float,
        default=None,
        help="diagnostic burst duration; repeat for 12 and 15 seconds",
    )
    args = parser.parse_args()
    from research.bl_mac_latency.host import check_guard

    if args.check:
        receipt = build_receipt(
            fixtures_path=args.fixtures,
            authorization_path=args.authorization,
            cpu_reference_path=args.cpu_reference,
            budget_seconds=args.budget_seconds,
            guard=check_guard,
        )
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "cells": len(receipt["cells"]),
                    "fixture_bank": receipt["fixture_bank"],
                    "authorization": receipt["authorization"]["parsed_probe_authorization"],
                    "diagnostic_source_identity": receipt["diagnostic_source_identity_before"],
                },
                sort_keys=True,
            )
        )
        return
    with held(
        args.lease_owner,
        "bl-mac-l1 paced diagnostic",
        seconds=args.budget_seconds,
        path=args.lease_path,
    ) as lease:
        _run(args, check_guard, lease.receipt)


if __name__ == "__main__":
    _main()

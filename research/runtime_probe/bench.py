"""Hardware inspection and Watch6 CPU benchmark (1 worker vs N=cpu_count workers)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from research.watch6_l1_confirm import engine as l1_engine
from research.watch6_l1_confirm import protocol as l1_protocol
from tools.ops.package_spec import publish_json, verify_immutable

ROOT = Path(__file__).resolve().parents[2]
PROBE_SCHEMA = "runtime-probe/v1"
CONTRACT_SCHEMA = "runtime-probe-contract/v1"
LEDGER_SCHEMA = "runtime-probe-ledger/v1"
UNIT_ID = "probe"

# Excluded fixture seed from Watch6-L1-V1 (outside confirm bank 34900001..34900100).
DEFAULT_SEED: int = int(l1_protocol.FIXTURE_SEEDS[0])
DEFAULT_HORIZON: int = 15
DEFAULT_INTERVAL: int = int(l1_protocol.TRIGGER_INTERVAL)
BENCH_ARMS: tuple[str, str] = ("bare", "l1")

THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OMP_THREAD_LIMIT",
    "TORCH_NUM_THREADS",
)


def _enforce_single_thread_env() -> None:
    for key in THREAD_ENV_VARS:
        os.environ[key] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    os.environ.setdefault("SNAKE_SERVE_PATHB", "off")
    os.environ.setdefault("SNAKE_MECHANICS_V2", "1")
    os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")
    import torch

    torch.set_num_threads(1)


def _configure_watch_paths(local: bool = False) -> tuple[str, str | None]:
    ckpt_dir = os.environ.get("SNAKE_CKPT_DIR")
    if not ckpt_dir:
        colab_input = os.environ.get("COLAB_INPUT_DIR")
        if colab_input and (Path(colab_input) / "checkpoints").is_dir():
            ckpt_dir = str((Path(colab_input) / "checkpoints").resolve())
        elif local or not colab_input:
            candidate = ROOT.parent / "snake-dqn/saved_snakes"
            if not candidate.is_dir():
                candidate = ROOT / "saved_snakes"
            ckpt_dir = str(candidate.resolve())
        else:
            raise ValueError("SNAKE_CKPT_DIR is not set and COLAB_INPUT_DIR/checkpoints is missing")
        os.environ["SNAKE_CKPT_DIR"] = ckpt_dir

    evidence_dir = os.environ.get("WATCH6_RELEASE_EVIDENCE")
    if not evidence_dir:
        colab_input = os.environ.get("COLAB_INPUT_DIR")
        if colab_input and (Path(colab_input) / "release-evidence").is_dir():
            evidence_dir = str((Path(colab_input) / "release-evidence").resolve())
            os.environ["WATCH6_RELEASE_EVIDENCE"] = evidence_dir
    return ckpt_dir, evidence_dir


def read_cpu_model_name(cpuinfo_path: Path = Path("/proc/cpuinfo")) -> str:
    """Read CPU model name from /proc/cpuinfo on Linux, with sysctl fallback on macOS."""
    if cpuinfo_path.is_file():
        try:
            text = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
            for key in ("model name", "Hardware", "Processor", "cpu model"):
                for line in text.splitlines():
                    if ":" in line:
                        lhs, rhs = line.split(":", 1)
                        if lhs.strip().lower() == key.lower() and rhs.strip():
                            return rhs.strip()
        except OSError:
            pass
    if shutil.which("sysctl"):
        for sysctl_key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", sysctl_key],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                ).strip()
                if out:
                    return out
            except (OSError, subprocess.SubprocessError):
                continue
    proc = platform.processor().strip()
    return proc if proc else "unknown"


def read_total_ram_bytes(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    """Read total physical RAM in bytes from /proc/meminfo or OS sysconf/sysctl."""
    if meminfo_path.is_file():
        try:
            for line in meminfo_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
        except (OSError, ValueError):
            pass
    if shutil.which("sysctl"):
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
            if out:
                return int(out)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and phys_pages > 0:
            return page_size * phys_pages
    except (AttributeError, ValueError, OSError):
        pass
    return 0


def read_gpu_name() -> str:
    """Query GPU name via nvidia-smi if present; return 'none' when no NVIDIA GPU exists."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return "none"
    try:
        out = subprocess.check_output(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        names = [line.strip() for line in out.splitlines() if line.strip()]
        return ", ".join(names) if names else "none"
    except (OSError, subprocess.SubprocessError):
        return "none"


def collect_system_info() -> dict[str, Any]:
    """Collect CPU count, sched_getaffinity count, CPU model, RAM, GPU, and Python/Torch info."""
    import torch

    cpu_count = int(os.cpu_count() or 1)
    sched_fn = getattr(os, "sched_getaffinity", None)
    if callable(sched_fn):
        try:
            affinity_count: int | None = len(sched_fn(0))
        except OSError:
            affinity_count = cpu_count
        sched_supported = True
    else:
        affinity_count = cpu_count
        sched_supported = False

    ram_bytes = read_total_ram_bytes()
    return {
        "cpu_count": cpu_count,
        "sched_getaffinity_count": int(affinity_count),
        "sched_getaffinity_supported": sched_supported,
        "cpu_model_name": read_cpu_model_name(),
        "total_ram_bytes": int(ram_bytes),
        "total_ram_gib": round(ram_bytes / (1024**3), 3),
        "gpu_name": read_gpu_name(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "thread_env": {
            **{k: os.environ.get(k, "") for k in THREAD_ENV_VARS},
            "SNAKE_DQN_DEVICE": os.environ.get("SNAKE_DQN_DEVICE", ""),
        },
    }


def _deterministic_arm_view(arm_result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in arm_result.items() if k != "timing"}


def run_watch6_slice(
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
    interval: int = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    """Run one deterministic Watch6 CPU slice covering bare + exact-stop (l1) search."""
    if horizon < 1 or interval < 1:
        raise ValueError("horizon and interval must be >= 1")
    bare_res = l1_engine.run_arm(
        int(seed),
        "bare",
        lambda: None,
        horizon=int(horizon),
    )
    l1_res = l1_engine.run_arm(
        int(seed),
        "l1",
        lambda: None,
        horizon=int(horizon),
    )
    canonical = json.dumps(
        {
            "bare": _deterministic_arm_view(bare_res),
            "l1": _deterministic_arm_view(l1_res),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "seed": int(seed),
        "horizon": int(horizon),
        "interval": int(interval),
        "arms": list(BENCH_ARMS),
        "bare_alive_frames": int(bare_res["alive_frames"]),
        "bare_mass_sum": int(bare_res["mass_sum"]),
        "l1_alive_frames": int(l1_res["alive_frames"]),
        "l1_mass_sum": int(l1_res["mass_sum"]),
        "l1_decision_count": int(l1_res["batches"]),
        "l1_frame_decisions": int(l1_res["decisions"]),
        "l1_intervention_count": int(l1_res["interventions"]),
        "l1_branches_evaluated": int(l1_res["branches_evaluated"]),
        "slice_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _parallel_worker_entry(
    worker_id: int,
    seed: int,
    horizon: int,
    interval: int,
    ckpt_dir: str,
    evidence_dir: str | None,
    ready_barrier: Any,
    start_barrier: Any,
    result_queue: Any,
) -> None:
    try:
        os.environ["SNAKE_CKPT_DIR"] = ckpt_dir
        if evidence_dir:
            os.environ["WATCH6_RELEASE_EVIDENCE"] = evidence_dir
        _enforce_single_thread_env()
        t_warm0 = time.perf_counter()
        run_watch6_slice(seed=seed, horizon=1, interval=interval)
        warmup_seconds = time.perf_counter() - t_warm0

        ready_barrier.wait(timeout=600)
        start_barrier.wait(timeout=600)

        t0 = time.perf_counter()
        slice_summary = run_watch6_slice(seed=seed, horizon=horizon, interval=interval)
        wall_seconds = time.perf_counter() - t0
        result_queue.put(
            {
                "worker_id": int(worker_id),
                "ok": True,
                "warmup_seconds": float(warmup_seconds),
                "wall_seconds": float(wall_seconds),
                "slice": slice_summary,
            }
        )
    except BaseException as exc:  # pragma: no cover - defensive worker error propagation
        try:
            ready_barrier.abort()
        except BaseException:
            pass
        try:
            start_barrier.abort()
        except BaseException:
            pass
        result_queue.put(
            {
                "worker_id": int(worker_id),
                "ok": False,
                "error": repr(exc),
            }
        )


def _run_parallel_workers(
    workers: int,
    seed: int,
    horizon: int,
    interval: int,
    ckpt_dir: str,
    evidence_dir: str | None,
) -> tuple[float, float, list[dict[str, Any]]]:
    ctx = mp.get_context("spawn")
    ready_barrier = ctx.Barrier(workers + 1)
    start_barrier = ctx.Barrier(workers + 1)
    result_queue: Any = ctx.Queue()

    procs = [
        ctx.Process(
            target=_parallel_worker_entry,
            args=(
                idx,
                seed,
                horizon,
                interval,
                ckpt_dir,
                evidence_dir,
                ready_barrier,
                start_barrier,
                result_queue,
            ),
        )
        for idx in range(workers)
    ]

    t_init0 = time.perf_counter()
    for proc in procs:
        proc.start()

    try:
        ready_barrier.wait(timeout=600)
        init_and_warmup_seconds = time.perf_counter() - t_init0

        t_start = time.perf_counter()
        start_barrier.wait(timeout=600)

        results: list[dict[str, Any]] = []
        for _ in range(workers):
            results.append(result_queue.get(timeout=600))
        parallel_wall_seconds = time.perf_counter() - t_start
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():  # pragma: no cover
                proc.terminate()
                proc.join(timeout=5)

    failures = [row for row in results if not row.get("ok")]
    if failures:
        raise RuntimeError(f"parallel worker failure: {failures[0].get('error')}")
    for proc in procs:
        if proc.exitcode != 0:
            raise RuntimeError(f"parallel worker exited with code {proc.exitcode}")
    results.sort(key=lambda item: int(item["worker_id"]))
    return init_and_warmup_seconds, parallel_wall_seconds, results


def run_benchmark(
    *,
    local: bool = False,
    seed: int = DEFAULT_SEED,
    horizon: int = DEFAULT_HORIZON,
    interval: int = DEFAULT_INTERVAL,
    workers_override: int | None = None,
) -> dict[str, Any]:
    """Collect system info and benchmark 1 worker and N=cpu_count workers in parallel."""
    _enforce_single_thread_env()
    ckpt_dir, evidence_dir = _configure_watch_paths(local=local)
    sys_info = collect_system_info()
    parallel_workers = (
        int(workers_override) if workers_override is not None else int(sys_info["cpu_count"])
    )
    if parallel_workers < 1:
        raise ValueError("parallel_workers must be >= 1")

    t_warm0 = time.perf_counter()
    run_watch6_slice(seed=seed, horizon=1, interval=interval)
    warmup_seconds = time.perf_counter() - t_warm0

    t_single0 = time.perf_counter()
    single_slice = run_watch6_slice(seed=seed, horizon=horizon, interval=interval)
    single_wall_seconds = time.perf_counter() - t_single0

    init_and_warmup_seconds, parallel_wall_seconds, worker_rows = _run_parallel_workers(
        workers=parallel_workers,
        seed=seed,
        horizon=horizon,
        interval=interval,
        ckpt_dir=ckpt_dir,
        evidence_dir=evidence_dir,
    )

    for row in worker_rows:
        if row["slice"]["slice_sha256"] != single_slice["slice_sha256"]:
            raise RuntimeError(
                f"non-deterministic slice digest on worker {row['worker_id']}: "
                f"{row['slice']['slice_sha256']} != {single_slice['slice_sha256']}"
            )

    decision_count = max(1, int(single_slice["l1_decision_count"]))
    worker_times = [float(row["wall_seconds"]) for row in worker_rows]
    single_slices_per_hour = 3600.0 / single_wall_seconds
    aggregate_slices_per_hour = (parallel_workers * 3600.0) / parallel_wall_seconds
    speedup = aggregate_slices_per_hour / single_slices_per_hour
    per_worker_efficiency = speedup / parallel_workers

    payload: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "mode": "local" if local else "colab",
        **sys_info,
        "benchmark": {
            "workload": "watch6_l1_confirm_bare_plus_exact_stop_l1",
            "seed": int(seed),
            "horizon": int(horizon),
            "interval": int(interval),
            "arms": list(BENCH_ARMS),
            "decision_points_per_slice": int(single_slice["l1_decision_count"]),
            "warmup_seconds": round(warmup_seconds, 6),
            "single_worker": {
                "workers": 1,
                "wall_seconds": round(single_wall_seconds, 6),
                "slices_per_hour": round(single_slices_per_hour, 4),
                "decisions_per_second": round(decision_count / single_wall_seconds, 6),
                "seconds_per_decision": round(single_wall_seconds / decision_count, 6),
            },
            "parallel_cpu_count": {
                "workers": int(parallel_workers),
                "init_and_warmup_seconds": round(init_and_warmup_seconds, 6),
                "wall_seconds": round(parallel_wall_seconds, 6),
                "worker_seconds_min": round(min(worker_times), 6),
                "worker_seconds_mean": round(sum(worker_times) / len(worker_times), 6),
                "worker_seconds_max": round(max(worker_times), 6),
                "aggregate_slices_per_hour": round(aggregate_slices_per_hour, 4),
                "aggregate_decisions_per_second": round(
                    (parallel_workers * decision_count) / parallel_wall_seconds,
                    6,
                ),
                "speedup_vs_single_worker": round(speedup, 4),
                "per_worker_efficiency": round(per_worker_efficiency, 4),
            },
            "slice_verification": single_slice,
        },
    }
    validate_probe_output(payload)
    return payload


def validate_probe_output(payload: dict[str, Any]) -> None:
    """Validate the output/probe.json schema and invariants."""
    if not isinstance(payload, dict):
        raise ValueError("probe output must be a JSON object")
    if payload.get("schema") != PROBE_SCHEMA:
        raise ValueError(f"schema must be {PROBE_SCHEMA!r}")
    if payload.get("mode") not in ("local", "colab"):
        raise ValueError("mode must be 'local' or 'colab'")

    for int_field in ("cpu_count", "sched_getaffinity_count", "total_ram_bytes"):
        val = payload.get(int_field)
        if type(val) is not int or val < 1:
            raise ValueError(f"{int_field} must be a positive int")
    if type(payload.get("sched_getaffinity_supported")) is not bool:
        raise ValueError("sched_getaffinity_supported must be a bool")

    for str_field in (
        "cpu_model_name",
        "gpu_name",
        "python_version",
        "torch_version",
        "platform",
        "system",
        "machine",
    ):
        val = payload.get(str_field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{str_field} must be a non-empty string")

    ram_gib = payload.get("total_ram_gib")
    if (
        isinstance(ram_gib, bool)
        or not isinstance(ram_gib, (int, float))
        or not math.isfinite(ram_gib)
        or ram_gib <= 0
    ):
        raise ValueError("total_ram_gib must be a positive finite number")

    bench = payload.get("benchmark")
    if not isinstance(bench, dict):
        raise ValueError("benchmark must be an object")
    for int_field in ("seed", "horizon", "interval", "decision_points_per_slice"):
        val = bench.get(int_field)
        if type(val) is not int or val < 1:
            raise ValueError(f"benchmark.{int_field} must be a positive int")
    if bench.get("arms") != list(BENCH_ARMS):
        raise ValueError(f"benchmark.arms must equal {list(BENCH_ARMS)!r}")

    single = bench.get("single_worker")
    if not isinstance(single, dict) or single.get("workers") != 1:
        raise ValueError("benchmark.single_worker must have workers == 1")
    for num_field in (
        "wall_seconds",
        "slices_per_hour",
        "decisions_per_second",
        "seconds_per_decision",
    ):
        val = single.get(num_field)
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(val)
            or val <= 0
        ):
            raise ValueError(f"benchmark.single_worker.{num_field} must be > 0")

    parallel = bench.get("parallel_cpu_count")
    if not isinstance(parallel, dict):
        raise ValueError("benchmark.parallel_cpu_count must be an object")
    if type(parallel.get("workers")) is not int or parallel["workers"] < 1:
        raise ValueError("benchmark.parallel_cpu_count.workers must be a positive int")
    for num_field in (
        "init_and_warmup_seconds",
        "wall_seconds",
        "worker_seconds_min",
        "worker_seconds_mean",
        "worker_seconds_max",
        "aggregate_slices_per_hour",
        "aggregate_decisions_per_second",
        "speedup_vs_single_worker",
        "per_worker_efficiency",
    ):
        val = parallel.get(num_field)
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(val)
            or val <= 0
        ):
            raise ValueError(f"benchmark.parallel_cpu_count.{num_field} must be > 0")

    ver = bench.get("slice_verification")
    if not isinstance(ver, dict):
        raise ValueError("benchmark.slice_verification must be an object")
    sha = ver.get("slice_sha256")
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("benchmark.slice_verification.slice_sha256 must be 64-char hex")


def _check_resume_contract(
    contract_path: Path,
    ledger_path: Path,
) -> tuple[dict[str, Any], bool]:
    verify_immutable(contract_path)
    verify_immutable(ledger_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"invalid contract schema: {contract.get('schema')!r}")
    if ledger.get("schema") != LEDGER_SCHEMA:
        raise ValueError(f"invalid ledger schema: {ledger.get('schema')!r}")
    if ledger.get("namespace") != contract.get("namespace") or ledger.get(
        "allocation"
    ) != contract.get("allocation"):
        raise ValueError("foreign namespace or allocation in verified-ledger.json")
    verified_units = ledger.get("verified_units")
    if not isinstance(verified_units, list):
        raise ValueError("verified_units must be a list")
    if any(u != UNIT_ID for u in verified_units):
        raise ValueError(f"unexpected unit in verified_units: {verified_units!r}")
    return contract, (UNIT_ID in verified_units)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run benchmark locally on the Mac using local checkpoints/pins",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path for probe.json (defaults to $COLAB_OUTPUT_DIR/probe.json when set)",
    )
    parser.add_argument("--contract", type=Path, help="Optional immutable probe-contract.json")
    parser.add_argument("--ledger", type=Path, help="Optional immutable verified-ledger.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional override for parallel worker count (defaults to os.cpu_count())",
    )
    args = parser.parse_args(argv)

    seed = int(args.seed)
    horizon = int(args.horizon)
    interval = int(args.interval)

    if args.contract is not None or args.ledger is not None:
        if args.contract is None or args.ledger is None:
            raise ValueError("--contract and --ledger must be provided together")
        contract, already_verified = _check_resume_contract(args.contract, args.ledger)
        seed = int(contract["seed"])
        horizon = int(contract["horizon"])
        interval = int(contract["interval"])
        if already_verified:
            print(json.dumps({"status": "SKIPPED_VERIFIED", "unit": UNIT_ID}), flush=True)
            return 0

    out_path = args.out
    if out_path is None and os.environ.get("COLAB_OUTPUT_DIR"):
        out_path = Path(os.environ["COLAB_OUTPUT_DIR"]) / "probe.json"

    payload = run_benchmark(
        local=bool(args.local),
        seed=seed,
        horizon=horizon,
        interval=interval,
        workers_override=args.workers,
    )
    if out_path is not None:
        publish_json(out_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

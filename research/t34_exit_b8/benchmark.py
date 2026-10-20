"""A5 bounded, real-root ExIt benchmark.

This module deliberately builds all requested prepared roots before one
``ExitTeacher.choose`` call.  It must never turn the registered W=256 A5 shape
into a serial collection and still describe it as the production kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping


def _runtime_hardware() -> dict[str, int | str | None]:
    """Describe a CPU runtime without embedding a host path in an export."""
    total: int | None = None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * pages
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "kind": "cpu-runtime-probe",
        "nproc": int(os.cpu_count() or 1),
        "ram_bytes": total,
    }


def _kernel_runner(kernel: str) -> Any:
    """Select a governed teacher kernel without changing v1 replay semantics."""
    if kernel == "v1":
        from research.t34_exit_b8.trajectory import run_worlds

        return run_worlds
    if kernel == "v2":
        from research.t34_exit_b8.teacher_v2 import run_worlds

        return run_worlds
    raise ValueError("kernel must be v1 or v2")


def _cpu_ceiling(threads: int) -> None:
    """Set the Numba ceiling after legacy imports have finished mutating env."""
    if threads < 1:
        raise ValueError("threads must be positive")
    for key, value in {
        "NUMBA_NUM_THREADS": str(threads),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }.items():
        os.environ[key] = value
    import numba

    # These modules rewrite NUMBA_NUM_THREADS at import time.  Import them
    # before set_num_threads launches the pool, then restore the registered
    # ceiling before a compiler config reload can observe a 5->1 mismatch.
    import research.phase0_search_20261019.teacher  # noqa: F401
    import research.t33_batched_teacher.teacher  # noqa: F401

    os.environ["NUMBA_NUM_THREADS"] = str(threads)
    numba.config.reload_config()
    numba.set_num_threads(threads)


def _gpu_utilization() -> float | None:
    """Best-effort one-shot GPU utilisation sample; timing is authoritative."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            text=True,
            timeout=3,
        )
        return float(result.stdout.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _profile_value(profile: Mapping[str, Any], name: str) -> int:
    value = profile.get(name, 0)
    if not isinstance(value, (int, float)):
        raise TypeError(f"teacher profile {name} must be numeric")
    return int(value)


def _batch_timing(result: Mapping[str, Any], *, steady_state_seconds: float) -> dict[str, Any]:
    """Copy the v2 collector's cold-fill and post-fill timing receipt.

    The collector owns the boundaries: cold fill ends only after all resident
    roots have their initial shadow, and the steady-state interval begins at
    that boundary.  Keeping these values separate from whole-run throughput
    prevents a cold bootstrap from being represented as the production rate.
    """
    numeric = (
        "cold_fill_elapsed_seconds",
        "cold_fill_active_row_steps",
        "cold_fill_resident_row_steps",
        "first_decision_elapsed_seconds",
        "steady_state_elapsed_seconds",
        "steady_state_active_row_steps",
        "steady_state_resident_row_steps",
        "steady_state_root_row_steps",
        "steady_state_shadow_row_steps",
        "steady_state_decisions",
    )
    timing: dict[str, Any] = {"steady_state_window_seconds": steady_state_seconds}
    for name in numeric:
        value = result.get(name)
        if value is None:
            timing[name] = None
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        ):
            timing[name] = value
        else:
            raise TypeError(f"teacher timing {name} must be a finite nonnegative number or null")
    stage_seconds = result.get("steady_state_host_stage_seconds")
    if stage_seconds is None:
        timing["host_stage_seconds"] = None
    elif isinstance(stage_seconds, Mapping) and all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
        for value in stage_seconds.values()
    ):
        timing["host_stage_seconds"] = dict(stage_seconds)
    else:
        raise TypeError(
            "teacher timing steady_state_host_stage_seconds must be a numeric mapping or null"
        )
    complete = result.get("steady_state_window_complete")
    if complete is not None and not isinstance(complete, bool):
        raise TypeError("teacher timing steady_state_window_complete must be boolean or null")
    timing["window_complete"] = complete
    elapsed = timing["steady_state_elapsed_seconds"]
    if elapsed is None or elapsed <= 0:
        timing["active_row_steps_per_second"] = None
        timing["decisions_per_second"] = None
    else:
        timing["active_row_steps_per_second"] = (
            timing["steady_state_active_row_steps"] / elapsed
            if timing["steady_state_active_row_steps"] is not None
            else None
        )
        timing["decisions_per_second"] = (
            timing["steady_state_decisions"] / elapsed
            if timing["steady_state_decisions"] is not None
            else None
        )
    return timing


def _load_policy(config: Mapping[str, Any]) -> Any | None:
    """Load an opt-in future student adapter; default remains the B8 policy."""
    reference = config.get("policy_factory")
    if not reference:
        if config.get("student_checkpoint"):
            raise ValueError("--student-checkpoint requires --policy-factory")
        return None
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError("--policy-factory must be module:callable")
    module, name = reference.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module), name)
        policy = factory(
            device=str(config["device"]),
            checkpoint=Path(str(config.get("student_checkpoint") or config["checkpoint"])),
        )
        if not callable(getattr(policy, "evaluate", None)) or not callable(
            getattr(policy, "evaluate_rows", None)
        ):
            raise TypeError(
                "policy must implement evaluate(world) and evaluate_rows(sim, rows, net)"
            )
        return policy
    except Exception as exc:
        raise RuntimeError(f"unsupported rollout-policy adapter {reference!r}") from exc


def run_benchmark(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the shared resident-root scheduler; it owns complete-world metrics."""
    kernel = str(config.get("kernel", "v1"))
    threads = int(config.get("threads", 5))
    worlds_per_process = int(config.get("worlds", 256))
    max_worlds = int(config.get("max_worlds", worlds_per_process))
    preflight = bool(config.get("preflight", False))
    mac_full_round = bool(config.get("mac_full_round", False))
    seconds = float(config.get("seconds", 1500))
    cpu_profile = bool(config.get("cpu_profile", False))
    nproc = int(config.get("nproc", 1))
    device = str(config.get("device", "cuda"))
    if not math.isfinite(seconds) or not 0 < seconds <= (600 if preflight else 3600):
        raise ValueError("benchmark wall bound must be positive and at most 3600s (600s preflight)")
    if nproc < 1:
        raise ValueError("nproc must be positive")
    if os.uname().sysname == "Darwin" and device == "cpu" and nproc * threads > 2:
        raise ValueError("Mac fanout is capped at two total CPU threads")
    configured_steady_state = config.get("steady_state_seconds")
    steady_state_seconds = float(
        min(1200, seconds) if configured_steady_state is None else configured_steady_state
    )
    if not math.isfinite(steady_state_seconds) or not 0 < steady_state_seconds <= seconds:
        raise ValueError(
            "steady-state window must be positive and no longer than the benchmark wall bound"
        )
    cpu_runtime_probe = bool(config.get("cpu_runtime_probe", False))
    if cpu_runtime_probe and kernel != "v2":
        raise ValueError("CPU runtime probe is available only for the v2 kernel")
    if cpu_runtime_probe and preflight:
        raise ValueError("CPU runtime probe is a Colab-only descriptive variant")
    if mac_full_round and (kernel != "v2" or preflight or cpu_runtime_probe):
        raise ValueError("Mac full rounds require the non-preflight v2 kernel")
    if (
        not preflight
        and not cpu_runtime_probe
        and not mac_full_round
        and device == "cuda"
        and nproc == 1
        and threads != 5
    ):
        raise ValueError("A5 requires exactly five Numba threads per worker")
    if preflight and int(config.get("frames", 5000)) > 500:
        raise ValueError("A6 benchmark preflight is capped at 500 frames")
    if (
        not preflight
        and not cpu_runtime_probe
        and not mac_full_round
        and device == "cuda"
        and nproc == 1
        and worlds_per_process != 256
    ):
        raise ValueError("A5 production benchmark requires exactly W=256 root worlds")
    if preflight and not 1 <= worlds_per_process <= 2:
        raise ValueError("Mac A6 preflight is limited to one or two root worlds")
    if max_worlds < worlds_per_process:
        raise ValueError("max_worlds cannot be smaller than the requested root batch")
    if int(config.get("horizon", 300)) != 300:
        raise ValueError("A5 benchmark requires deterministic H=300")
    if (
        not preflight
        and not cpu_runtime_probe
        and not mac_full_round
        and nproc == 1
        and device not in {"cuda", "cpu"}
    ):
        raise ValueError("A5 recipe requires CUDA on an L4")
    if cpu_runtime_probe and device != "cpu":
        raise ValueError("CPU runtime probe requires CPU")
    if cpu_runtime_probe and threads > 2:
        raise ValueError("CPU runtime probe is capped at two CPU threads")
    if mac_full_round and (device != "cpu" or not 1 <= threads <= 2):
        raise ValueError("Mac full rounds require CPU and one or two CPU threads")
    if preflight and device != "cpu":
        raise ValueError("A6 preflight must stay on Mac CPU")
    if preflight and threads > 1:
        raise ValueError("A6 preflight permits one CPU thread per process")
    if nproc > 1 and (device != "cpu" or threads != 1):
        raise ValueError("process fanout requires CPU and one thread per process")
    if nproc > 1 and (config.get("policy_factory") or config.get("student_checkpoint")):
        raise ValueError("process fanout does not support a rollout-policy adapter")
    if nproc > 1 and preflight and nproc * worlds_per_process > 2:
        raise ValueError("A6 preflight keeps its total two-root cap under fanout")
    if cpu_profile and (device != "cpu" or threads != 1 or seconds > 60):
        raise ValueError("CPU profile requires CPU, one thread per process, and at most 60 seconds")
    mixes = config.get("mixes", ("scripted", "frozen", "mixed"))
    if isinstance(mixes, str):
        mixes = tuple(piece for piece in mixes.split(",") if piece)
    if tuple(mixes) != ("scripted", "frozen", "mixed"):
        raise ValueError("A5 needs all three mixes in scripted,frozen,mixed order")
    triggers = dict(config.get("triggers", {}))
    for name in (
        "confirm_interval",
        "confirm_min_length",
        "low_length_probability",
        "death_window",
        "max_rewinds",
    ):
        if config.get(name) is not None:
            triggers[name] = config[name]
    if nproc > 1 and not bool(config.get("_fanout_worker", False)):
        from research.t34_exit_b8.process_fanout import run_fanout
        from research.t34_exit_b8.trajectory import make_world_rows

        rows = list(config.get("rows") or make_world_rows({**config, "mixes": mixes}, max_worlds))
        effective_nproc = min(nproc, max_worlds)
        result = run_fanout(
            {
                **config,
                "threads": 1,
                "device": "cpu",
                "triggers": triggers,
                "mixes": mixes,
                "steady_state_seconds": steady_state_seconds,
                "worlds_per_process": worlds_per_process,
            },
            runner=(
                "research.t34_exit_b8.teacher_v2:run_worlds"
                if kernel == "v2"
                else "research.t34_exit_b8.trajectory:run_worlds"
            ),
            rows=rows,
            nproc=effective_nproc,
        )
        nproc = effective_nproc
        result["runtime_hardware"] = _runtime_hardware()
        result["runtime_hardware"]["configured_nproc"] = nproc
        result["runtime_hardware"]["ram_per_process_bytes"] = [
            int(row.get("rss_bytes", 0)) for row in result["fanout"]["worker_reports"]
        ]
    else:
        result = None
    if result is None:
        _cpu_ceiling(threads)
        import numba

        os.environ["SNAKE_DQN_DEVICE"] = device
        numba.set_num_threads(threads)
        started = time.monotonic()
        policy = _load_policy(config)
        result = _kernel_runner(kernel)(
            {
                **config,
                "policy": policy,
                "triggers": triggers,
                "gpu_utilization_sampler": _gpu_utilization if device == "cuda" else None,
                "worlds_per_process": worlds_per_process,
                "max_worlds": max_worlds,
                "mixes": mixes,
                "steady_state_seconds": steady_state_seconds,
                "output_dir": config["output_dir"],
            }
        )
        elapsed = time.monotonic() - started
    else:
        elapsed = float(result["elapsed_seconds"])
    profile = dict(result.get("teacher_profile", {}))
    active = _profile_value(profile, "active_row_steps")
    resident = _profile_value(profile, "resident_row_steps")
    sample = _gpu_utilization() if device == "cuda" else None
    by_mix = {
        name: {**metrics, "expansion_fraction": metrics["expanded"] / max(metrics["searched"], 1)}
        for name, metrics in dict(result["mixes"]).items()
    }
    result = {
        "schema": f"t34-a5-benchmark/{kernel}",
        "kernel": kernel,
        "complete": bool(result["complete"]),
        "preflight": preflight,
        "mac_full_round": mac_full_round,
        "cpu_runtime_probe": cpu_runtime_probe,
        "preflight_total_root_cap": 2 if preflight else None,
        "recipe": {
            "workers": 1,
            "nproc": nproc,
            "threads_per_worker": threads,
            "root_worlds_per_worker": worlds_per_process,
            "horizon": 300,
            "device": device,
        },
        "namespace": str(config.get("namespace", "")),
        "elapsed_seconds": elapsed,
        "mixes": by_mix,
        "active_row_steps": active,
        "resident_row_steps": resident,
        "executed_row_steps_per_world": result.get(
            "executed_row_steps_per_world",
            active / int(result["worlds"]) if int(result["worlds"]) else None,
        ),
        "active_row_steps_per_second": active / max(elapsed, 1e-9),
        "active_resident_fraction": active / max(resident, 1),
        "completed_worlds": result["worlds"],
        "worlds_per_hour": result["worlds_per_hour"],
        "decisions": result["decisions"],
        "decisions_per_second": result["decisions_per_second"],
        "export_bytes": result["export_bytes"],
        "export_bytes_per_world": result["export_bytes_per_world"],
        "configured_root_capacity": result["configured_root_capacity"],
        "peak_root_worlds": result["peak_root_worlds"],
        "search_batch_peak": result["search_batch_peak"],
        "ram_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if os.uname().sysname == "Darwin" else 1024),
        "gpu_utilization_samples_percent": list(result.get("gpu_utilization_samples_percent", []))
        + ([] if sample is None else [sample]),
        "teacher_profile": profile,
        "host_stage_seconds": dict(
            result.get("host_stage_seconds", profile.get("host_stage_seconds", {}))
        ),
        "worker_rate_descriptive_only": True,
        **(
            {"batch_timing": _batch_timing(result, steady_state_seconds=steady_state_seconds)}
            if kernel == "v2"
            else {}
        ),
        **({"runtime_hardware": _runtime_hardware()} if cpu_runtime_probe else {}),
        **(
            {"runtime_hardware": result["runtime_hardware"]} if "runtime_hardware" in result else {}
        ),
        **({"fanout": result["fanout"]} if "fanout" in result else {}),
        **{
            key: result[key]
            for key in (
                "invalid_reason",
                "unfinished_worlds",
                "world_rate_scope",
                "root_inference_bucket_peak",
                "branch_bucket_capacity",
            )
            if key in result
        },
    }
    return result


def _write_once(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    with path.with_name(path.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(hashlib.sha256(payload.encode()).hexdigest() + "\n")
    with path.with_name(path.name + ".complete").open("x", encoding="utf-8") as stream:
        stream.write("complete\n")


def aggregate_lane_reports(paths: list[Path]) -> dict[str, Any]:
    """Produce the only A5 lane-level go rule from both concurrent workers."""
    if len(paths) != 2:
        raise ValueError("A5 aggregate requires exactly two worker reports")
    rows = [json.loads(path.read_text()) for path in paths]
    if any(
        row.get("schema") not in {"t34-a5-benchmark/v1", "t34-a5-benchmark/v2"}
        or row.get("preflight")
        or row.get("cpu_runtime_probe")
        for row in rows
    ):
        raise ValueError("aggregate inputs must be production A5 worker reports")
    kernels = {row.get("kernel", "v1") for row in rows}
    if len(kernels) != 1:
        raise ValueError("aggregate inputs must use one teacher kernel")
    if kernels == {"v2"}:
        expected_recipe = {
            "workers": 1,
            "threads_per_worker": 5,
            "root_worlds_per_worker": 256,
            "horizon": 300,
            "device": "cuda",
        }

        def is_cuda_recipe(recipe: Any) -> bool:
            if not isinstance(recipe, Mapping):
                return False
            normalized = dict(recipe)
            nproc = normalized.pop("nproc", 1)
            return nproc == 1 and normalized == expected_recipe

        if any(
            row.get("mac_full_round")
            or not is_cuda_recipe(row.get("recipe"))
            or not isinstance(row.get("namespace"), str)
            or not row["namespace"]
            for row in rows
        ):
            raise ValueError("v2 aggregate requires two CUDA L4 A5 worker recipes")
        if paths[0].resolve() == paths[1].resolve() or rows[0]["namespace"] == rows[1]["namespace"]:
            raise ValueError("v2 aggregate requires distinct worker report paths and namespaces")
    wall = max(float(row["elapsed_seconds"]) for row in rows)
    active = sum(int(row["active_row_steps"]) for row in rows)
    resident = sum(int(row["resident_row_steps"]) for row in rows)
    rate = active / max(wall, 1e-9)
    timings = [row.get("batch_timing") for row in rows]
    if any(timing is not None and not isinstance(timing, dict) for timing in timings):
        raise ValueError("worker batch timing must be an object when present")
    first_decisions = [
        float(timing["first_decision_elapsed_seconds"])
        for timing in timings
        if isinstance(timing, dict) and timing.get("first_decision_elapsed_seconds") is not None
    ]
    return {
        "schema": (
            "t34-a5-lane-aggregate/v2"
            if all(row.get("kernel") == "v2" for row in rows)
            else "t34-a5-lane-aggregate/v1"
        ),
        "complete": all(bool(row["complete"]) for row in rows),
        "workers": 2,
        "worker_reports": [path.name for path in paths],
        "elapsed_seconds": wall,
        "active_row_steps": active,
        "resident_row_steps": resident,
        "active_row_steps_per_second": rate,
        "active_resident_fraction": active / max(resident, 1),
        "go_rule": {
            "active_row_steps_per_second": 3000,
            "met": all(bool(row["complete"]) for row in rows) and rate >= 3000,
            "enforced": False,
        },
        **(
            {
                "batch_timing": {
                    "first_decision_elapsed_seconds": min(first_decisions, default=None),
                    "workers": timings,
                    "aggregation": (
                        "per-worker cold-fill and steady-state windows are retained separately; "
                        "only the earliest first decision is summarized"
                    ),
                }
            }
            if kernels == {"v2"}
            else {}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--student-checkpoint", type=Path)
    parser.add_argument("--policy-factory")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--max-worlds", type=int, default=256)
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--seconds", type=float, default=1500)
    parser.add_argument("--steady-state-seconds", type=float)
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--nproc",
        type=int,
        default=(
            min(2, max(1, int(os.cpu_count() or 1)))
            if os.uname().sysname == "Darwin"
            else max(1, int(os.cpu_count() or 1))
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--kernel", choices=("v1", "v2"), default="v1")
    parser.add_argument("--cpu-runtime-probe", action="store_true")
    parser.add_argument("--cpu-profile", action="store_true")
    parser.add_argument("--mac-full-round", action="store_true")
    parser.add_argument("--mixes", default="scripted,frozen,mixed")
    parser.add_argument("--namespace", default="t34-exit-a5/v1")
    parser.add_argument("--confirm-interval", type=int)
    parser.add_argument("--confirm-min-length", type=int)
    parser.add_argument("--low-length-probability", type=float)
    parser.add_argument("--death-window", type=int)
    parser.add_argument("--max-rewinds", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate", type=Path, nargs=2, metavar=("LANE0", "LANE1"))
    args = parser.parse_args()
    if args.aggregate:
        _write_once(args.out, aggregate_lane_reports(args.aggregate))
        return
    if args.checkpoint is None or args.output_dir is None:
        parser.error("--checkpoint and --output-dir are required unless --aggregate is used")
    _write_once(args.out, run_benchmark(vars(args)))


if __name__ == "__main__":
    main()

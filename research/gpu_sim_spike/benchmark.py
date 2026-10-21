"""Bounded S0 dense decision benchmark and independent CPU label rescoring.

The CPU comparator always runs the unchanged production reference. CUDA device
cost excludes neither host launch overhead nor tensor transfer. Startup/export
and provider rate must be added from the supervisor receipt for a final verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import queue
import re
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike.parity import compare_decision, summarize
from research.gpu_sim_spike.reference import cpu_decision, fixture_worlds, rescore_label
from research.gpu_sim_spike.rollout import dense_decision

PACKAGE = Path(__file__).resolve().parent


def source_commit() -> str:
    """Resolve pinned identity even in the supervisor's verified sparse fallback."""
    pinned = os.environ.get("GPU_SIM_SPIKE_COMMIT")
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=PACKAGE.parents[1],
        timeout=10,
    )
    actual = result.stdout.strip() if result.returncode == 0 else None
    if pinned is not None:
        if not re.fullmatch(r"[0-9a-f]{40}", pinned):
            raise ValueError("GPU_SIM_SPIKE_COMMIT must be a full lowercase SHA")
        if actual is not None and pinned != actual:
            raise ValueError("runtime source commit differs from manifest pin")
        return pinned
    if actual is None:
        raise ValueError("verified sparse source requires GPU_SIM_SPIKE_COMMIT")
    return actual


def write_receipt(path: Path, result: dict) -> None:
    """Create immutable JSON + checksum + completion sidecars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with path.with_name(path.name + ".sha256").open("x") as stream:
        stream.write(f"{digest}  {path.name}\n")
    with path.with_name(path.name + ".complete").open("x") as stream:
        stream.write("complete\n")


def _reference_job(job: tuple) -> dict:
    """One process owns one complete paired world and every reference branch."""
    case_name, checkpoint, offset, depth = job
    torch.set_num_threads(1)
    built = time.perf_counter()
    case, world = fixture_worlds(Path(checkpoint), [case_name])[0]
    _advance(world, offset)
    build_seconds = time.perf_counter() - built
    result = cpu_decision(world, depth)
    return {"case": case["case"], "offset": offset, "build_seconds": build_seconds, **result}


def _reference_worker(jobs: Any, results: Any) -> None:
    """Consume indexed work; the parent owns and bounds every spawned process."""
    while True:
        entry = jobs.get()
        if entry is None:
            return
        index, job = entry
        try:
            results.put((index, _reference_job(job), None))
        except Exception:
            results.put((index, None, traceback.format_exc()))
            return


def _reference_phase(jobs: list[tuple], workers: int, deadline: float) -> list[dict]:
    """Bound CPU reference work, including cold imports/JIT, without retries."""
    context = mp.get_context("spawn")
    work, results = context.Queue(), context.Queue()
    children = [
        context.Process(target=_reference_worker, args=(work, results))
        for _ in range(min(workers, len(jobs)))
    ]
    for index, job in enumerate(jobs):
        work.put((index, job))
    for _ in children:
        work.put(None)
    received: dict[int, dict] = {}
    heartbeat = time.perf_counter()
    try:
        for child in children:
            child.start()
        while len(received) < len(jobs):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("CPU reference exhausted spike wall budget; no retry")
            try:
                index, value, error = results.get(timeout=min(5.0, remaining))
            except queue.Empty:
                if all(not child.is_alive() for child in children):
                    raise RuntimeError("reference workers exited without complete results")
                if time.perf_counter() - heartbeat >= 110:
                    print(
                        json.dumps(
                            {
                                "event": "cpu_reference_progress",
                                "done": len(received),
                                "total": len(jobs),
                            }
                        ),
                        flush=True,
                    )
                    heartbeat = time.perf_counter()
                continue
            if error:
                raise RuntimeError(error)
            received[index] = value
        return [received[index] for index in range(len(jobs))]
    finally:
        # Only these explicitly owned processes can be terminated. This also
        # prevents timed-out pool work from surviving the bounded parent.
        for child in children:
            if child.pid is not None and child.is_alive():
                child.terminate()
        for child in children:
            if child.pid is not None:
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()
                    child.join(timeout=5)
        work.close()
        results.close()


def _advance(world: Any, offset: int) -> None:
    from research.t34_exit_b8.engine import ExitTeacher
    from research.t34_exit_b8.trajectory import prepare_world

    teacher = ExitTeacher(device="cpu")
    for _ in range(offset):
        if not world.sim.alive[0, 0]:
            raise ValueError("fixture hero died before requested dense decision")
        prepare_world(world)
        action = teacher.evaluate(world).greedy_action()
        world.sim.step(teacher.actions(world, action))
        world.runtime.observe_step(world.sim)
    prepare_world(world)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _candidate_json(result: dict, *, depth: int, snakes: int) -> dict:
    """Validate transferred values before they can enter a parity receipt."""
    downloaded = {key: value.detach().cpu() for key, value in result.items()}
    shapes = {
        "action": (1,),
        "q": (1, 6),
        "legal": (1, 6),
        "alive_steps": (1, 6),
        "mass_integral": (1, 6),
        "action_trace": (1, 6, depth, snakes),
    }
    for key, shape in shapes.items():
        if key not in downloaded or tuple(downloaded[key].shape) != shape:
            raise ValueError(f"candidate {key} has invalid shape (expected {shape})")
    for key in ("q", "mass_integral"):
        if not bool(torch.isfinite(downloaded[key]).all()):
            raise ValueError(f"candidate {key} contains nonfinite values")
    if downloaded["legal"].dtype != torch.bool:
        raise ValueError("candidate legal mask must be boolean")
    for key in ("action", "action_trace", "alive_steps"):
        if downloaded[key].dtype not in (torch.int32, torch.int64):
            raise ValueError(f"candidate {key} must be integer")
    for key in ("action", "action_trace"):
        if bool(((downloaded[key] < 0) | (downloaded[key] >= 6)).any()):
            raise ValueError(f"candidate {key} has out-of-range action")
    if bool(((downloaded["alive_steps"] < 0) | (downloaded["alive_steps"] > depth)).any()):
        raise ValueError("candidate survival count is outside the full horizon")
    if bool((downloaded["mass_integral"] < 0).any()):
        raise ValueError("candidate mass integral is negative")
    values = {key: value.tolist()[0] for key, value in downloaded.items()}
    return {
        "action": values["action"],
        "q": values["q"],
        "legal": values["legal"],
        "action_trace": values["action_trace"],
        "outcomes": {
            action: {
                "alive_steps": values["alive_steps"][action],
                "mass_integral": values["mass_integral"][action],
            }
            for action in range(6)
        },
    }


def run(args: argparse.Namespace) -> dict:
    """Run a measured CPU phase then candidate phase; never allocate a runtime."""
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.speed_s0_baseline.harness import fixture_cases

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable (never silently substitute CPU)")
    if args.cpu_workers > 2 and platform.system() == "Darwin":
        raise ValueError("Mac probes must stay within two CPU threads")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    deadline = started + args.seconds
    names = (
        [row["case"] for row in fixture_cases()]
        if args.fixtures == "all"
        else (
            ["tiny-scripted", "tiny-frozen", "tiny-mixed"]
            if args.fixtures == "tiny"
            else args.fixtures.split(",")
        )
    )
    jobs = [
        (name, str(args.checkpoint), 5 * sample, min(args.horizon, 5000 - 5 * sample))
        for name in names
        for sample in range(args.decisions_per_case)
    ]
    reference_started = time.perf_counter()
    refs = []
    cpu_case_baselines = {}
    for name in names:
        selected = [job for job in jobs if job[0] == name]
        # Fill all requested CPU slots using repeated frozen inputs. Repeats
        # count only for throughput, never as extra independent parity samples.
        repeats = max(1, (args.cpu_workers + len(selected) - 1) // len(selected))
        baseline_jobs = selected * repeats
        case_started = time.perf_counter()
        measured = _reference_phase(baseline_jobs, args.cpu_workers, deadline)
        wall = time.perf_counter() - case_started
        refs.extend(measured[: len(selected)])
        cpu_case_baselines[name] = {
            "wall_seconds": wall,
            "executed_decisions": len(measured),
            "unique_parity_inputs": len(selected),
            "workers": min(args.cpu_workers, len(measured)),
            "seconds_per_dense_decision": wall / len(measured),
            "units_per_decision_at_session_rate": (
                wall * args.rate_units_per_hour / 3600 / len(measured)
                if args.device == "cuda"
                else None
            ),
            "repeats_are_throughput_only": True,
        }
    reference_seconds = time.perf_counter() - reference_started
    write_receipt(
        args.output.with_name("cpu-reference-before-device.json"),
        {
            "schema": "gpu-sim-spike-cpu-before-device/v1",
            "source_commit": source_commit(),
            "frozen_baseline_sha256": hashlib.sha256(
                (PACKAGE / "BASELINE.json").read_bytes()
            ).hexdigest(),
            "per_case": cpu_case_baselines,
            "references": refs,
            "qualification": (
                "same-runtime CPU reference measured and persisted before device setup; "
                "A100 CPU is not L4 CPU"
            ),
        },
    )
    rows = []
    host = {
        "fixture_build_seconds": 0.0,
        "upload_and_policy_setup_seconds": 0.0,
        "download_seconds": 0.0,
        "parity_cpu_search_seconds": reference_seconds,
        "label_cpu_rescoring_seconds": 0.0,
    }
    candidate_pipeline_started = time.perf_counter()
    runtime = 0.0
    host_launch_cpu_seconds = 0.0
    limitations: set[str] = set()
    for index, (job, reference) in enumerate(zip(jobs, refs)):
        if time.perf_counter() >= deadline:
            raise TimeoutError("spike wall budget exhausted; no automatic retry")
        name, _, offset, depth = job
        stage = time.perf_counter()
        _, world = fixture_worlds(args.checkpoint, [name])[0]
        _advance(world, offset)
        host["fixture_build_seconds"] += time.perf_counter() - stage
        stage = time.perf_counter()
        sim = DeviceSim.from_reference(world.sim, device=args.device)
        policy = DevicePolicy.from_world(
            world, device=args.device, backend=args.policy_backend, sim=sim
        )
        for owner in (sim, policy):
            declared = getattr(owner, "port_limitations", ())
            limitations.update([declared] if isinstance(declared, str) else declared)
        _sync(args.device)
        host["upload_and_policy_setup_seconds"] += time.perf_counter() - stage
        # Warmup is explicit, outside steady timing but inside end-to-end cost.
        warm_started = time.perf_counter()
        dense_decision(sim.clone(), policy, depth=1, deadline=deadline)
        _sync(args.device)
        warm_seconds = time.perf_counter() - warm_started
        stage = time.perf_counter()
        cpu_started = time.process_time()
        values = dense_decision(sim, policy, depth, deadline=deadline)
        _sync(args.device)
        elapsed = time.perf_counter() - stage
        launch_cpu = time.process_time() - cpu_started
        host_launch_cpu_seconds += launch_cpu
        runtime += elapsed
        stage = time.perf_counter()
        candidate = _candidate_json(values, depth=depth, snakes=sim.S)
        host["download_seconds"] += time.perf_counter() - stage
        label_score = rescore_label(world, candidate["action_trace"][candidate["action"]], depth)
        host["label_cpu_rescoring_seconds"] += label_score["wall_seconds"]
        comparison = compare_decision(reference, candidate)
        rows.append(
            {
                "case": name,
                "offset": offset,
                "reference": reference,
                "candidate": candidate,
                "comparison": comparison,
                "cpu_rescored_label": label_score,
                "device_loop_seconds": elapsed,
                "host_launch_cpu_seconds": launch_cpu,
                "warmup_seconds": warm_seconds,
                "host_stages": getattr(policy, "host_stages", []),
                "policy_timing": dict(getattr(policy, "host_stages", {})),
            }
        )
        write_receipt(args.output.parent / f"decision-{index:04d}.json", rows[-1])
        print(
            json.dumps(
                {
                    "event": "gpu_spike_progress",
                    "done": index + 1,
                    "total": len(jobs),
                    "case": name,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    elapsed = time.perf_counter() - started
    candidate_pipeline_seconds = time.perf_counter() - candidate_pipeline_started
    baseline = json.loads((PACKAGE / "BASELINE.json").read_text())
    parity = summarize([row["comparison"] for row in rows])
    host_stages = sorted({stage for row in rows for stage in row["host_stages"]})
    complete = not host_stages
    units_per_decision = (
        candidate_pipeline_seconds * args.rate_units_per_hour / 3600 / max(1, len(rows))
    )
    return {
        "schema": "gpu-sim-spike-benchmark/v1",
        "device": args.device,
        "status": (
            "PREFLIGHT"
            if args.horizon != 300 or args.fixtures == "tiny"
            else "MEASURED_PENDING_PROVIDER_RECEIPT"
        ),
        "verdict": "PENDING_OPERATOR_GPU_DATA" if args.device == "cpu" else "STOP_OR_PENDING_AUDIT",
        "full_device_loop": complete,
        "interpretation": (
            "whole rollout resident; parity/provider qualification pending"
            if complete
            else "device-portion speed upper bound; incomplete port"
        ),
        "remaining_host_rollout_stages": host_stages,
        "port_limitations": sorted(limitations),
        "host_time_seconds": host,
        "elapsed_seconds": elapsed,
        "candidate_pipeline_seconds": candidate_pipeline_seconds,
        "study_units_at_planning_rate": (
            elapsed * args.rate_units_per_hour / 3600 if args.device == "cuda" else None
        ),
        "device_loop_seconds": runtime,
        "host_launch_cpu_seconds": host_launch_cpu_seconds,
        "host_launch_time_scope": (
            "process CPU during rollout; overlaps device wall; never add to that wall "
            "(CPU fallback includes tensor compute)"
        ),
        "device_seconds_per_dense_decision": runtime / max(1, len(rows)),
        "pipeline_units_per_dense_decision_at_planning_rate": (
            units_per_decision if args.device == "cuda" else None
        ),
        "provider_rate_units_per_hour": args.rate_units_per_hour,
        "provider_rate_status": "planning ceiling; replace with measured rate in operator RESULT",
        "cost_scope": (
            "selected-label CPU action replay + fixture creation + setup/transfer + warmup "
            "+ rollout + readback; exhaustive CPU parity search is study overhead; "
            "supervisor bootstrap/export/debit pending"
        ),
        "cpu_reference": {
            "workers": args.cpu_workers,
            "wall_seconds": reference_seconds,
            "decisions": len(refs),
            "units_per_decision_at_session_planning_rate": (
                reference_seconds
                * args.rate_units_per_hour
                / 3600
                / max(1, sum(case["executed_decisions"] for case in cpu_case_baselines.values()))
                if args.device == "cuda"
                else None
            ),
            "actual_workers": max(case["workers"] for case in cpu_case_baselines.values()),
            "rate_scope": "same allocated runtime; A100 CPU timings are not L4 CPU data",
            "per_case": cpu_case_baselines,
            "twelve_process_measured": all(
                case["workers"] == 12 for case in cpu_case_baselines.values()
            ),
        },
        "frozen_baseline_sha256": hashlib.sha256(
            (PACKAGE / "BASELINE.json").read_bytes()
        ).hexdigest(),
        "frozen_cost_baseline": baseline["frozen_cost_comparator_units_per_dense_decision"],
        "parity": parity,
        "rows": rows,
        "argv": vars(args) | {"checkpoint": str(args.checkpoint), "output": str(args.output)},
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "source_commit": source_commit(),
        "source_sha256": {
            str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PACKAGE.glob("*.py"))
        },
        "threads": args.threads,
        "pid": os.getpid(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-backend", choices=("device", "reference"), default="device")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixtures", default="all")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--decisions-per-case", type=int, default=3)
    parser.add_argument("--cpu-workers", type=int, default=1)
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--rate-units-per-hour", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.horizon, args.decisions_per_case, args.cpu_workers, args.seconds) <= 0:
        parser.error("horizon/decisions/workers/seconds must be positive")
    if args.decisions_per_case > 1000:
        parser.error("dense sample offsets must remain inside H5000")
    if args.output.exists():
        parser.error("output already exists; immutable receipts cannot be overwritten")
    entrypoint_started = time.perf_counter()
    try:
        receipt = run(args)
    except TimeoutError as exc:
        receipt = {
            "schema": "gpu-sim-spike-benchmark/v1",
            "status": "INCOMPLETE",
            "verdict": "NO_CONTINUE",
            "reason": str(exc),
            "device": args.device,
            "source_commit": source_commit(),
            "note": (
                "budget timeout, no automatic retry; inspect immutable "
                "completed decision receipts"
            ),
        }
    receipt["entrypoint_run_wall_seconds"] = time.perf_counter() - entrypoint_started
    persist_started = time.perf_counter()
    write_receipt(args.output, receipt)
    persist_seconds = time.perf_counter() - persist_started
    write_receipt(
        args.output.with_name("host-export.json"),
        {
            "schema": "gpu-sim-spike-host-export/v1",
            "benchmark_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            "benchmark_json_bytes": args.output.stat().st_size,
            "json_serialization_hash_and_local_write_seconds": persist_seconds,
            "entrypoint_run_wall_seconds": receipt["entrypoint_run_wall_seconds"],
            "note": (
                "Local benchmark JSON, SHA and completion writes; "
                "remote transfer/bootstrap/stop are measured by the supervisor receipt"
            ),
        },
    )


if __name__ == "__main__":
    main()

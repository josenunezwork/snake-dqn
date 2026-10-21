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
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike.parity import compare_decision, summarize
from research.gpu_sim_spike.reference import cpu_decision, fixture_worlds
from research.gpu_sim_spike.rollout import dense_decision

PACKAGE = Path(__file__).resolve().parent


def write_receipt(path: Path, result: dict) -> None:
    """Create immutable JSON + checksum + completion sidecars."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
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
    from research.t3_rollout_labels_20261015.bank_engine import choose_and_step
    from research.t34_exit_b8.trajectory import prepare_world

    for _ in range(offset):
        if not world.sim.alive[0, 0]:
            raise ValueError("fixture hero died before requested dense decision")
        choose_and_step(world, capture_obs=False)
    prepare_world(world)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _candidate_json(result: dict) -> dict:
    values = {key: value.detach().cpu().tolist()[0] for key, value in result.items()}
    return {
        "action": values["action"],
        "q": values["q"],
        "legal": values["legal"],
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
        else args.fixtures.split(",")
    )
    jobs = [
        (name, str(args.checkpoint), 5 * sample, args.horizon)
        for name in names
        for sample in range(args.decisions_per_case)
    ]
    reference_started = time.perf_counter()
    refs = _reference_phase(jobs, args.cpu_workers, deadline)
    reference_seconds = time.perf_counter() - reference_started
    rows = []
    host = {
        "fixture_build_seconds": 0.0,
        "upload_and_policy_setup_seconds": 0.0,
        "download_seconds": 0.0,
        "cpu_rescoring_seconds": reference_seconds,
    }
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
        policy = DevicePolicy.from_world(world, device=args.device)
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
        candidate = _candidate_json(values)
        host["download_seconds"] += time.perf_counter() - stage
        comparison = compare_decision(reference, candidate)
        rows.append(
            {
                "case": name,
                "offset": offset,
                "reference": reference,
                "candidate": candidate,
                "comparison": comparison,
                "device_loop_seconds": elapsed,
                "host_launch_cpu_seconds": launch_cpu,
                "warmup_seconds": warm_seconds,
                "host_stages": getattr(policy, "host_stages", []),
                "policy_timing": dict(getattr(policy, "host_stages", {})),
            }
        )
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
    baseline = json.loads((PACKAGE / "BASELINE.json").read_text())
    parity = summarize([row["comparison"] for row in rows])
    host_stages = sorted({stage for row in rows for stage in row["host_stages"]})
    complete = not host_stages and not limitations
    units_per_decision = elapsed * args.rate_units_per_hour / 3600 / max(1, len(rows))
    return {
        "schema": "gpu-sim-spike-benchmark/v1",
        "device": args.device,
        "status": "PREFLIGHT" if args.horizon != 300 else "MEASURED_PENDING_PROVIDER_RECEIPT",
        "verdict": "PENDING_OPERATOR_GPU_DATA" if args.device == "cpu" else "STOP_OR_PENDING_AUDIT",
        "full_device_loop": complete,
        "interpretation": (
            "end-to-end candidate"
            if complete
            else "device-portion speed upper bound; incomplete port"
        ),
        "remaining_host_rollout_stages": host_stages,
        "port_limitations": sorted(limitations),
        "host_time_seconds": host,
        "elapsed_seconds": elapsed,
        "device_loop_seconds": runtime,
        "host_launch_cpu_seconds": host_launch_cpu_seconds,
        "host_launch_time_scope": (
            "process CPU during rollout; overlaps device wall; never add to that wall "
            "(CPU fallback includes tensor compute)"
        ),
        "device_seconds_per_dense_decision": runtime / max(1, len(rows)),
        "pipeline_units_per_dense_decision_at_planning_rate": units_per_decision,
        "provider_rate_units_per_hour": args.rate_units_per_hour,
        "provider_rate_status": "planning ceiling; replace with measured rate in operator RESULT",
        "cost_scope": (
            "CPU rescoring + fixture creation + setup/transfer + warmup + rollout + readback; "
            "supervisor bootstrap/export/debit pending"
        ),
        "cpu_reference": {
            "workers": args.cpu_workers,
            "wall_seconds": reference_seconds,
            "decisions": len(refs),
            "units_per_decision": reference_seconds * 2 / 3600 / max(1, len(refs)),
            "twelve_process_measured": args.cpu_workers == 12,
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
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(PACKAGE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(PACKAGE.glob("*.py"))
        },
        "threads": args.threads,
        "pid": os.getpid(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    if args.output.exists():
        parser.error("output already exists; immutable receipts cannot be overwritten")
    write_receipt(args.output, run(args))


if __name__ == "__main__":
    main()

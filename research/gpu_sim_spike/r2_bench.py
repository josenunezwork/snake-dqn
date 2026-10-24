"""R2 A100 ruler: the unchanged I0 resident tick, I (pinned engine) versus I' (R2).

Authority: ``JUDGMENT_POST_I0S_DIRECTION_20261024.md`` step 4.  Both arms are
built by the admitted I0 engine factory (``integration_engine._case(..., "I")``:
MOVE2 indexed movement, native geometry, static flood, ordinary v8); arm
``I_prime`` additionally registers its simulator and reset template with
:mod:`r2_cache`.  Arm ``I`` therefore executes the pinned runtime code paths,
byte for byte (``test_r2_engine_identity.py`` proves this against the pinned
source on the Mac).

Per cell, before any clock: a pure-optimization identity proof runs
``PROOF_TICKS`` progressing ticks per arm with ``I_prime`` in R2 debug mode
(every cached value re-derived with the pinned full-capacity code) and compares
the ordered digests of every observable output plus the complete persistent
state inventory (padding included) after every tick.  Timing then follows the
I0 ruler: warmups, interleaved reset-inclusive plain repeats (the scored clock),
separate exclusive-stage diagnostic ticks, and for E1024 three interleaved
32-tick continuous blocks; every arm/repeat digest must agree across arms.
Exception cells (``exc-*``) apply the fixed exception-heavy fixture
(``r2_identity.exception_fixture`` plus wrapped rings) to both arms.
Plain timing is the sole scored clock; traces are diagnostic only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import integration_bench as i0
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import integration_native, r2_cache, r2_identity
from research.gpu_sim_spike import r2_units as units
from research.gpu_sim_spike.benchmark import source_commit, source_hashes

ARMS = ("I", "I_prime")
ORDERS = (ARMS, ("I_prime", "I"), ARMS)
WARMUPS = 3
REPEATS = 3
CONTINUOUS_TICKS = 32
PROOF_TICKS = 3
SCHEMA = "gpu-sim-spike-r2-bench/v1"


class R2ParityError(ValueError):
    """An I versus I' observation differed: the R2 package is INVALID."""


def build_case(checkpoint: Path, kind: str, mix: str, worlds: int, device: str, arm: str) -> tuple:
    """One arm's six-tuple; both arms use the admitted I0 engine factory."""
    if arm not in ARMS:
        raise ValueError(f"unknown R2 arm: {arm}")
    sim, policy, template, policy_template, collection, _ = engine._case(
        checkpoint, mix, worlds, device, "I"
    )
    if kind == "exc":
        for target in (sim, template):
            r2_identity.exception_fixture(target)
            r2_identity.wrap_rings(target)
    if arm == "I_prime":
        for target in (sim, template):
            r2_cache.enable(target, debug=False)
    return sim, policy, template, policy_template, collection, arm


def _tick(case: tuple, device: str, timer: Any = None, *, digest: bool = True) -> tuple:
    sim, policy, template, policy_template, collection, _arm = case
    with engine._arm("I"), torch.inference_mode():
        i0._sync(device)
        started = time.perf_counter()
        if timer is None:
            ruler._reset(sim, policy, template, policy_template)
            ruler._full_tick(sim, policy, collection, 0)
        else:
            with i0.instrument(sim, timer), timer.stage("unscoped"):
                with timer.stage("reset"):
                    ruler._reset(sim, policy, template, policy_template)
                ruler._full_tick(sim, policy, collection, 0)
        i0._sync(device)
    return time.perf_counter() - started, engine.digest(case) if digest else ""


def prove(cases: dict[str, tuple], device: str, ticks: int = PROOF_TICKS) -> dict[str, Any]:
    """Pure-optimization identity before timing; I' runs in R2 debug mode."""
    engine.assert_oracle_separation(cases["I"], cases["I_prime"])
    candidate = cases["I_prime"][0]
    r2_cache.enable(candidate, debug=True)
    try:
        records = {arm: r2_identity.record(cases[arm], ticks) for arm in ARMS}
        i0._sync(device)
        stats = r2_cache.stats(candidate)
    finally:
        r2_cache.enable(candidate, debug=False)
    old, new = records["I"], records["I_prime"]
    for tick, (left, right) in enumerate(zip(old, new)):
        if left["calls"] != right["calls"]:
            raise R2ParityError(f"tick {tick}: an observable output differs between I and I'")
        if left["state_sha256"] != right["state_sha256"]:
            raise R2ParityError(f"tick {tick}: persistent state differs between I and I'")
    if len(old) != ticks or len(new) != ticks or not stats.get("debug_checks"):
        raise R2ParityError("R2 identity proof is incomplete")
    return {
        "status": "PASS",
        "ticks": ticks,
        "per_tick": [
            {
                "state_sha256": row["state_sha256"],
                "observed_outputs": len(row["calls"]),
                "outputs_sha256": hashlib.sha256(json.dumps(row["calls"]).encode()).hexdigest(),
                "alive": row["alive"],
                "deaths": row["deaths"],
                "corpse_food": row["corpse_food"],
                "boosted": row["boosted"],
                "max_seg_count": row["max_seg_count"],
            }
            for row in old
        ],
        "candidate_debug": stats,
        "comparator": "pinned-mode I arm, same process, same initial state and policies",
    }


def _continuous(cases: dict[str, tuple], device: str, deadline: float) -> dict[str, Any]:
    """I0's interleaved progressing blocks: reset once, every collector position used."""
    rows: dict[str, list] = {name: [] for name in ARMS}
    collections = {
        name: ruler._collection(CONTINUOUS_TICKS, case[0].E, case[0].device)
        for name, case in cases.items()
    }
    for name, case in cases.items():
        sim, policy, template, policy_template = case[:4]
        with engine._arm("I"), torch.inference_mode():
            ruler._reset(sim, policy, template, policy_template)
            for index in range(WARMUPS):
                ruler._full_tick(sim, policy, collections[name], index)
    for repeat, order in enumerate(ORDERS):
        for name in order:
            sim, policy, template, policy_template = cases[name][:4]
            if time.monotonic() >= deadline:
                raise TimeoutError("R2 deadline before continuous block")
            tick_seconds = []
            with engine._arm("I"), torch.inference_mode():
                i0._sync(device)
                started = time.perf_counter()
                ruler._reset(sim, policy, template, policy_template)
                for index in range(CONTINUOUS_TICKS):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("R2 deadline during continuous block")
                    began = time.perf_counter()
                    ruler._full_tick(sim, policy, collections[name], index)
                    i0._sync(device)
                    tick_seconds.append(time.perf_counter() - began)
                i0._sync(device)
                elapsed = time.perf_counter() - started
            final_case = (*cases[name][:4], collections[name], name)
            rows[name].append(
                {
                    "repeat": repeat,
                    "completed_world_steps": sim.E * CONTINUOUS_TICKS,
                    "wall_seconds": elapsed,
                    "tick_seconds": tick_seconds,
                    "complete_world_steps_per_second": sim.E * CONTINUOUS_TICKS / elapsed,
                    "final_digest": engine.digest(final_case),
                    "collector_positions": list(range(CONTINUOUS_TICKS)),
                }
            )
    for repeat in range(len(ORDERS)):
        if rows["I"][repeat]["final_digest"] != rows["I_prime"][repeat]["final_digest"]:
            raise R2ParityError("continuous block final state differs between I and I'")
    return {
        "status": "COMPLETE",
        "ticks": CONTINUOUS_TICKS,
        "orders": [list(order) for order in ORDERS],
        "arms": {name: {"repeats": values} for name, values in rows.items()},
        "reset_charged_once_per_repeat": True,
        "warmups": WARMUPS,
    }


def _trace(case: tuple, device: str, output: Path, unit: str) -> dict[str, Any]:
    """One profiler tick per arm, outside scored samples (diagnostic only)."""
    if device != "cuda":
        return {"status": "NOT_RUN_CPU"}
    destination = output / "traces" / f"{unit}-{case[-1]}.json.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    raw = destination.with_suffix(".json")
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            _tick(case, device, digest=False)
        summary = i0.a1_prof.profiler_summary(list(prof.events()), device)
        prof.export_chrome_trace(str(raw))
        payload = gzip.compress(raw.read_bytes(), mtime=0)
        if len(payload) > 16 * 1024**2:
            raise RuntimeError("R2 trace exceeds compact export limit")
        from research.gpu_sim_spike.a1_prof_units import _atomic_create

        _atomic_create(destination, payload)
        digest = hashlib.sha256(payload).hexdigest()
        _atomic_create(
            Path(str(destination) + ".sha256"), f"{digest}  {destination.name}\n".encode()
        )
        _atomic_create(Path(str(destination) + ".complete"), b"complete\n")
        return {
            "status": "PASS",
            "path": str(destination.relative_to(output)),
            "sha256": digest,
            "size": len(payload),
            "kernel_host_inventory": summary,
        }
    finally:
        raw.unlink(missing_ok=True)


def measure(
    checkpoint: Path,
    unit: str,
    device: str,
    deadline: float,
    output: Path,
    *,
    repeats: int,
    warmups: int,
    continuous_ticks: int,
    proof_ticks: int,
    trace: bool,
) -> dict[str, Any]:
    kind, mix, worlds = units.parse(unit)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    cases = {arm: build_case(checkpoint, kind, mix, worlds, device, arm) for arm in ARMS}
    try:
        proof = prove(cases, device, proof_ticks)
        arms = {name: {"plain_seconds": [], "digests": [], "stages": []} for name in ARMS}
        for case in cases.values():
            for _ in range(warmups):
                if time.monotonic() >= deadline:
                    raise TimeoutError("R2 deadline during warmup")
                _tick(case, device)
        orders = ORDERS[:repeats]
        for order in orders:
            for name in order:
                if time.monotonic() >= deadline:
                    raise TimeoutError("R2 deadline during scored repeat")
                plain, digest = _tick(cases[name], device)
                arms[name]["plain_seconds"].append(plain)
                arms[name]["digests"].append(digest)
        digests = {digest for name in ARMS for digest in arms[name]["digests"]}
        if len(digests) != 1:
            raise R2ParityError("restored plain tick state differs across arms or repeats")
        if kind == "cell":
            for repeat, order in enumerate(orders):
                for name in order:
                    timer = i0.IntegrationTiming(device)
                    event, event_digest = _tick(cases[name], device, timer)
                    if event_digest not in digests:
                        raise R2ParityError("plain and diagnostic tick state differs")
                    attribution = timer.result()
                    stages = (
                        attribution["exclusive_cuda_elapsed_seconds"]
                        if device == "cuda"
                        else attribution["exclusive_host_seconds"]
                    )
                    total = sum(stages.values())
                    reconciled = math.isfinite(total) and total > 0
                    reconciled = reconciled and abs(total - event) <= event * 0.10
                    arms[name]["stages"].append(
                        {
                            "repeat": repeat,
                            "wall_seconds": event,
                            "reconciled_within_10_percent": reconciled,
                            **attribution,
                        }
                    )
        continuous = (
            _continuous(cases, device, deadline) if continuous_ticks else {"status": "NOT_REQUIRED"}
        )
        if continuous_ticks:
            for name in ARMS:
                finals = [row["final_digest"] for row in continuous["arms"][name]["repeats"]]
                if len(set(finals)) != 1:
                    raise R2ParityError("nondeterministic continuous block: " + name)
        sim = cases["I_prime"][0]
        return {
            "unit": unit,
            "kind": kind,
            "mix": mix,
            "worlds": worlds,
            "repeats": repeats,
            "warmups": warmups,
            "orders": [list(x) for x in orders],
            "identity_proof": proof,
            "arms": arms,
            "continuous": continuous,
            "pure_optimization_identity": True,
            "r2_window": r2_cache.window(sim),
            "capacity": int(sim.cap),
            "max_seg_count": int(sim.seg_count.max()),
            "body_inventory": {
                name: {
                    "active_body_segments": int(case[0].seg_count[case[0].alive].sum().cpu()),
                    "body_storage_bytes": case[0].bodies.numel() * case[0].bodies.element_size(),
                }
                for name, case in cases.items()
            },
            "device": device,
            "memory": i0._memory(device),
            "traces": (
                {name: _trace(cases[name], device, output, unit) for name in ARMS}
                if trace
                else {"status": "NOT_RUN"}
            ),
        }
    finally:
        cases.clear()
        ruler._release_cell(device)


def r2_native(output: Path, device: str) -> dict[str, Any]:
    """R2 CUDA twin-identity tests on the allocated GPU (CPU runs record PREFLIGHT)."""
    select = "cuda" if device == "cuda" else "window_rule"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "research/gpu_sim_spike/test_r2_padding.py",
            "-k",
            select,
            "-rA",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    lines = result.stdout.splitlines()
    passed = sum(line.startswith("PASSED ") for line in lines)
    failed = any(line.startswith(("FAILED ", "ERROR ")) for line in lines)
    status = "PREFLIGHT" if device != "cuda" else "PASS"
    if result.returncode != 0 or failed or passed == 0 or (device == "cuda" and passed < 5):
        status = "INVALID"
    value = {
        "schema": "gpu-sim-spike-r2-native-parity/v1",
        "status": status,
        "device": device,
        "selection": select,
        "pytest_exit": result.returncode,
        "passed_rows": passed,
        "stdout_tail": result.stdout[-6000:],
        "commit": source_commit(),
        "source_sha256": source_hashes(),
    }
    units.publish_json(output, value)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise ValueError("R2 requires A100")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= (420 if args.canary else 2400):
        raise ValueError("R2 requires a finite positive bounded producer deadline")
    if args.device == "cpu" and not args.smoke:
        raise ValueError("CPU is tiny smoke only")
    if args.smoke and args.canary:
        raise ValueError("smoke and canary are exclusive")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    deadline = started + args.seconds
    commit = source_commit()
    cells = units.allocation(smoke=args.smoke, canary=args.canary)
    ns = units.namespace(commit, smoke=args.smoke, canary=args.canary)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(110):
            print(
                json.dumps(
                    {"event": "r2_heartbeat", "elapsed_seconds": time.monotonic() - started}
                ),
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()

    def finish(result: dict[str, Any]) -> dict[str, Any]:
        units.publish_json(output / "r2-bench.json", result)
        stop_heartbeat.set()
        return result

    base = {"schema": SCHEMA, "commit": commit, "source_sha256": source_hashes()}
    # Native admission precedes every cell: I0's MOVE2/geometry proof, then R2's.
    native = integration_native.run(output / "native-parity.json", device=args.device)
    r2_native_value = (
        r2_native(output / "r2-native-parity.json", args.device)
        if native["status"] in ("PASS", "PREFLIGHT")
        else {"status": "NOT_RUN"}
    )
    if native["status"] not in ("PASS", "PREFLIGHT") or r2_native_value["status"] not in (
        "PASS",
        "PREFLIGHT",
    ):
        invalid = "INVALID" in (native["status"], r2_native_value["status"])
        return finish(
            {
                **base,
                "status": "INVALID" if invalid else "INCOMPLETE",
                "reason": "NATIVE_PARITY_REQUIRED",
                "completed_units": [],
                "native_parity": "native-parity.json",
                "r2_native_parity": "r2-native-parity.json",
            }
        )
    prior = units.load_ledger(args.ledger, ns, cells)
    completed: list[str] = []
    skipped: list[str] = []
    try:
        for unit in cells:
            if unit in prior:
                skipped.append(unit)
                completed.append(unit)
                print(json.dumps({"event": "unit_skipped_verified", "unit": unit}), flush=True)
                continue
            kind, _mix, worlds = units.parse(unit)
            light = args.smoke or args.canary
            value = measure(
                args.checkpoint,
                unit,
                args.device,
                deadline,
                output,
                repeats=1 if light else REPEATS,
                warmups=1 if light else WARMUPS,
                continuous_ticks=(
                    CONTINUOUS_TICKS if not light and kind == "cell" and worlds == 1024 else 0
                ),
                proof_ticks=2 if args.smoke else PROOF_TICKS,
                trace=kind == "cell" and (worlds == 1024 or (args.canary and worlds == 64)),
            )
            units.publish_unit(output, ns, cells, unit, value)
            completed.append(unit)
            print(
                json.dumps(
                    {
                        "event": "unit_done",
                        "unit": unit,
                        "done": len(completed),
                        "total": len(cells),
                    }
                ),
                flush=True,
            )
    except R2ParityError as exc:
        units.build_ledger(
            output,
            output / "verified-ledger.json",
            commit=commit,
            smoke=args.smoke,
            canary=args.canary,
        )
        failure = {
            **base,
            "status": "INVALID",
            "reason": str(exc),
            "failure_class": "R2_PURE_OPTIMIZATION_MISMATCH",
            "completed_units": completed,
            "skipped_units": skipped,
        }
        units.publish_json(output / "r2-failure.json", failure)
        return finish(failure)
    except TimeoutError as exc:
        return finish(
            {
                **base,
                "status": "INCOMPLETE",
                "reason": str(exc),
                "completed_units": completed,
                "skipped_units": skipped,
                "native_parity": "native-parity.json",
                "r2_native_parity": "r2-native-parity.json",
            }
        )
    ledger = units.build_ledger(
        output, output / "verified-ledger.json", commit=commit, smoke=args.smoke, canary=args.canary
    )
    return finish(
        {
            **base,
            "status": "MEASURED_PENDING_MAC_REVIEW",
            "device": args.device,
            "elapsed_seconds": time.monotonic() - started,
            "precision": {"dtype": "fp32", "matmul_tf32": False, "cudnn_tf32": False},
            "namespace": ledger["namespace"],
            "completed_units": completed,
            "skipped_units": skipped,
            "native_parity": "native-parity.json",
            "r2_native_parity": "r2-native-parity.json",
            "ledger": "verified-ledger.json",
            "component_flags": {
                "flood": "static",
                "v8_bundle": "off",
                "move": "indexed",
                "geometry": "native",
                "arms": {"I": "pinned I0 runtime paths", "I_prime": "R2 L1+L2+L3"},
            },
            "cpu_closed_loop_identity": "UNPROVED",
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--canary", action="store_true")
    a = p.parse_args()
    result = run(a)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "MEASURED_PENDING_MAC_REVIEW":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

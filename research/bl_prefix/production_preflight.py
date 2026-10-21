"""Closer-to-production Mac timing for the BL prefix E6/E7 workers.

This harness is deliberately separate from the frozen E6/E7 entrypoints.  It
uses their real world, continuation, and worker functions, but binds only the
burned preflight audit and local fixture R/F endpoints.  Every payload is
``PREFLIGHT_ONLY`` and ``NOT_SCORED_PREFLIGHT``; no scientific conclusion is
computed.  The worker calls below omit the package preflight ``max_steps=16``
limit so the measured continuation reaches the original H5000 horizon (or a
natural death).

Run under the Mac operational wrapper, for example::

    nice -n 10 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 caffeinate -dims \
      ./venv/bin/python -m research.bl_prefix.production_preflight ...

The wrapper is responsible for AC/lid checks and the two slot locks.  This
module never contacts Colab or the allocator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

HORIZON = 5000
MIXES = ("scripted", "frozen", "mixed")
DEFAULT_AUDIT = Path(__file__).with_name("SEED_AUDIT.json")
DEFAULT_B8 = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "bl-e6e7/local-runtime-01/inputs/checkpoints/b8_s3_t7_20261016.pth"
)
DEFAULT_FIXTURES = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "bl-e6e7/local-runtime-01/inputs"
)
DEFAULT_OUTPUT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "bl-e6e7/production-preflight-v2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_json(path: Path, value: Any) -> dict[str, Any]:
    """Publish one canonical immutable JSON payload with completion sidecars."""

    from research.bl_ppo.exports import export_json

    return export_json(path, value)


def _progress(event: str, **fields: Any) -> None:
    """Flush one operational progress record for long-running guarded jobs."""

    print(json.dumps({"progress": event, **fields}, sort_keys=True), flush=True)


def _verify_audit(path: Path) -> tuple[dict[str, Any], str]:
    from research.bl_ppo.exports import verify_export
    from research.bl_prefix import seeds

    digest = verify_export(path)
    value = json.loads(path.read_text())
    seeds.verify_receipt(value, round_index=int(value.get("round_index", 0)))
    allocation = value.get("allocation")
    if not isinstance(allocation, Mapping) or set(allocation) != {
        "source",
        "future",
        "exploration",
        "preflight",
    }:
        raise ValueError("SEED_AUDIT.json does not contain the frozen four namespaces")
    preflight = allocation["preflight"]
    if not isinstance(preflight, Mapping) or len(preflight.get("seeds", ())) < 3:
        raise ValueError("SEED_AUDIT.json has fewer than three burned preflight seeds")
    return value, digest


def _source_with_roots(
    plan: Mapping[str, Any], b8_path: Path, root_frames: Sequence[int]
) -> tuple[dict[str, Any], dict[int, Any]]:
    """Run one B8 source through H5000 while capturing late living roots."""

    import numpy as np

    from research.bl_prefix.controllers import B8Controller
    from research.bl_prefix.engine import (
        PrefixRecord,
        capture_root,
        clone_world,
        initial_world,
        step,
    )

    world = initial_world(arm="s3", plan=plan, b8_path=b8_path)
    b8 = B8Controller(b8_path)
    history: list[PrefixRecord] = []
    targets = tuple(sorted(set(int(frame) for frame in root_frames)))
    roots: dict[int, Any] = {}
    started = time.perf_counter()
    death_frame: int | None = None
    death_length: int | None = None
    steps = 0
    clone_seconds: list[float] = []
    while int(world.sim.frame[0]) < HORIZON and bool(world.sim.alive[0, 0]):
        result = step(world, b8)
        history.append(
            PrefixRecord(
                result.observation,
                result.action,
                np.asarray(result.legal, dtype=bool),
                np.asarray(result.advisory, dtype=bool),
            )
        )
        steps += 1
        if result.died:
            death_frame = result.frame
            death_length = result.pre_length
        frame = int(result.frame)
        if frame in targets and result.alive and frame not in roots:
            roots[frame] = capture_root(world, history)
            clone_began = time.perf_counter()
            clone_world(world)
            clone_seconds.append(time.perf_counter() - clone_began)
    elapsed = time.perf_counter() - started
    return (
        {
            "seed": int(plan["seed"]),
            "mix": str(plan["mix"]),
            "steps": steps,
            "elapsed_seconds": elapsed,
            "seconds_per_world_step": elapsed / steps if steps else None,
            "terminal_alive": bool(world.sim.alive[0, 0]),
            "terminal_frame": int(world.sim.frame[0]),
            "death_frame": death_frame,
            "last_living_length": death_length,
            "captured_root_frames": sorted(roots),
            "root_lengths": {
                str(frame): int(root.snapshot["sim"]["length"][0, 0])
                for frame, root in sorted(roots.items())
            },
            "history_decisions_per_root": 128,
            "clone_seconds": clone_seconds,
            "horizon": HORIZON,
        },
        roots,
    )


def _continuation_timing(root: Any, b8_path: Path, future_seed: int) -> dict[str, Any]:
    from research.bl_prefix.controllers import B8Controller
    from research.bl_prefix.engine import rollout

    began = time.perf_counter()
    result = rollout(
        root,
        B8Controller(b8_path),
        B8Controller(b8_path),
        prefix_steps=0,
        future_seed=int(future_seed),
    )
    elapsed = time.perf_counter() - began
    return {
        "root_frame": int(root.frame),
        "root_length": int(root.snapshot["sim"]["length"][0, 0]),
        "requested_horizon": HORIZON,
        "steps": int(result.steps),
        "elapsed_seconds": elapsed,
        "seconds_per_world_step": elapsed / result.steps if result.steps else None,
        "natural_dead_stop": bool(not result.terminal_survival),
        "truncated": bool(result.truncated),
        "scientific_metrics": "omitted",
    }


def _run_e6_workers(
    roots: Sequence[tuple[Path, Mapping[str, Any]]],
    b8_path: Path,
    fixture_r: Path,
    fixture_f: Path,
    future_seeds: Sequence[int],
    workers: int,
) -> dict[str, Any]:
    """Time the real E6 root worker with full continuations and fixture R/F."""

    tasks = [
        (
            str(root_path),
            {"stratum": "early", "source_world": str(meta["seed"]), "relocations": {}},
            {"B8": str(b8_path), "R": str(fixture_r), "F": str(fixture_f)},
            tuple(int(value) for value in future_seeds[:2]),
            False,
            0,
        )
        for root_path, meta in roots
    ]
    began = time.perf_counter()
    row_count = 0
    world_steps = 0
    rollouts = 0
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            for index, (rows, counts) in enumerate(executor.map(_e6_worker_instrumented, tasks)):
                row_count += len(rows)
                world_steps += int(counts["world_steps"])
                rollouts += int(counts["rollouts"])
                _progress(
                    "e6_worker_complete",
                    worker_index=index + 1,
                    workers=len(tasks),
                    rows_completed=len(rows),
                    rollouts_completed=int(counts["rollouts"]),
                    world_steps=int(counts["world_steps"]),
                )
        status = "PREFLIGHT_ONLY"
        reason = None
    except Exception as exc:  # record a diagnosable timing failure in the export
        status = "INCOMPLETE"
        reason = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - began
    return {
        "status": status,
        "reason": reason,
        "worker_path": "research.bl_prefix.e6._root_rows",
        "roots": len(tasks),
        "workers": workers,
        "full_horizon": True,
        "max_steps": None,
        "rows_completed": row_count,
        "rollouts_completed": rollouts,
        "world_steps": world_steps,
        "elapsed_seconds": elapsed,
        "scientific_metrics": "omitted",
    }


def _run_e7_workers(
    source_rows: Sequence[tuple[Mapping[str, Any], Mapping[int, Any]]],
    b8_path: Path,
    future_seeds: Sequence[int],
    exploration_seeds: Sequence[int],
    workers: int,
) -> dict[str, Any]:
    """Time the real E7 worker with full continuations on synthetic late roots."""

    tasks = []
    for index, (meta, roots) in enumerate(source_rows):
        selected = sorted(roots.items())[-3:]
        if len(selected) != 3:
            return {
                "status": "INCOMPLETE",
                "reason": "fewer than three late living roots for E7 timing",
                "worker_path": "research.bl_prefix.e7._source_rows",
                "roots": len(selected),
                "workers": workers,
                "full_horizon": True,
                "max_steps": None,
                "scientific_metrics": "omitted",
            }
        source = {
            "source_world": str(meta["seed"]),
            "roots": {ancestor: root for ancestor, (_, root) in zip((100, 150, 200), selected)},
        }
        start = index * 24
        tasks.append(
            (
                source,
                str(b8_path),
                {},
                tuple(int(value) for value in future_seeds[:3]),
                tuple(int(value) for value in exploration_seeds[start : start + 24]),
                False,
                0,
            )
        )
    began = time.perf_counter()
    selections = confirmations = 0
    world_steps = rollouts = calibrations = calibration_seconds_ns = 0
    try:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            for index, (selected_rows, confirmation_rows, counts) in enumerate(
                executor.map(_e7_worker_instrumented, tasks)
            ):
                selections += len(selected_rows)
                confirmations += len(confirmation_rows)
                world_steps += int(counts["world_steps"])
                rollouts += int(counts["rollouts"])
                calibrations += int(counts["calibrations"])
                calibration_seconds_ns += int(counts["calibration_seconds_ns"])
                _progress(
                    "e7_source_complete",
                    source_index=index + 1,
                    sources=len(tasks),
                    source_world=str(tasks[index][0]["source_world"]),
                    selection_rows=len(selected_rows),
                    confirmation_rows=len(confirmation_rows),
                    rollouts_completed=int(counts["rollouts"]),
                    world_steps=int(counts["world_steps"]),
                    calibrations=int(counts["calibrations"]),
                )
        status = "PREFLIGHT_ONLY"
        reason = None
    except Exception as exc:  # record a diagnosable timing failure in the export
        status = "INCOMPLETE"
        reason = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - began
    return {
        "status": status,
        "reason": reason,
        "worker_path": "research.bl_prefix.e7._source_rows",
        "sources": len(tasks),
        "ancestors_per_source": 3,
        "workers": workers,
        "full_horizon": True,
        "max_steps": None,
        "selection_rows_completed": selections,
        "confirmation_rows_completed": confirmations,
        "rollouts_completed": rollouts,
        "world_steps": world_steps,
        "calibrations": calibrations,
        "calibration_seconds": calibration_seconds_ns / 1_000_000_000,
        "elapsed_seconds": elapsed,
        "scientific_metrics": "omitted",
    }


def _e6_worker_instrumented(task: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Run the frozen E6 worker while counting its actual rollout steps."""

    import research.bl_prefix.e6 as module

    original = module.rollout
    counts = {"world_steps": 0, "rollouts": 0}

    def counted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        counts["world_steps"] += int(result.steps)
        counts["rollouts"] += 1
        return result

    module.rollout = counted
    try:
        rows = module._root_rows(task)
    finally:
        module.rollout = original
    return rows, counts


def _e7_worker_instrumented(
    task: tuple[Any, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Run the frozen E7 worker while counting its actual rollout steps."""

    import research.bl_prefix.e7 as module

    original = module.rollout
    counts = {"world_steps": 0, "rollouts": 0, "calibrations": 0}
    calibration_seconds = 0.0

    def counted(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        counts["world_steps"] += int(result.steps)
        counts["rollouts"] += 1
        return result

    original_calibrate = module.calibrate_b8_temperature

    def counted_calibrate(*args: Any, **kwargs: Any) -> Any:
        nonlocal calibration_seconds
        started = time.perf_counter()
        value = original_calibrate(*args, **kwargs)
        calibration_seconds += time.perf_counter() - started
        counts["calibrations"] += 1
        return value

    module.rollout = counted
    module.calibrate_b8_temperature = counted_calibrate
    try:
        selections, confirmations = module._source_rows(task)
    finally:
        module.rollout = original
        module.calibrate_b8_temperature = original_calibrate
    counts["calibration_seconds_ns"] = int(calibration_seconds * 1_000_000_000)
    return selections, confirmations, counts


def run(args: argparse.Namespace) -> dict[str, Any]:
    from research.bl_prefix import bank

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"fresh output directory required: {output}")
    output.mkdir(parents=True, exist_ok=True)
    audit, audit_sha = _verify_audit(Path(args.audit))
    b8_path = Path(args.b8).resolve(strict=True)
    if _sha256(b8_path) != "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8":
        raise ValueError("B8 checkpoint does not match the pinned s3 identity")
    fixture_r = Path(args.fixtures, "fixture-R.pth").resolve(strict=True)
    fixture_f = Path(args.fixtures, "fixture-F.pth").resolve(strict=True)
    for fixture in (fixture_r, fixture_f):
        if not fixture.is_file():
            raise ValueError(f"missing fixture endpoint: {fixture}")
    registration = Path(__file__).with_name("REGISTRATION_E10.md")
    registration_sha = _sha256(registration)
    preflight_seeds = list(audit["allocation"]["preflight"]["seeds"])
    preflight = preflight_seeds[: args.sources]
    if args.sources != 3:
        raise ValueError("this closer production preflight is frozen to exactly three sources")
    plans = bank._plans(preflight)
    if tuple(str(plan["mix"]) for plan in plans) != MIXES:
        raise ValueError("first three audited preflight seeds do not cover scripted/frozen/mixed")
    root_frames = tuple(int(value) for value in args.root_frames.split(",") if value.strip())
    if len(root_frames) < 3 or any(not 128 <= frame < HORIZON for frame in root_frames):
        raise ValueError("need at least three root frames in [128, 4999]")
    began = time.perf_counter()
    startup_began = time.perf_counter()
    from research.bl_prefix.controllers import B8Controller

    B8Controller(b8_path)
    startup_seconds = time.perf_counter() - startup_began
    source_rows: list[tuple[dict[str, Any], dict[int, Any]]] = []
    all_root_rows: list[tuple[Path, Mapping[str, Any]]] = []
    for source_index, plan in enumerate(plans):
        meta, roots = _source_with_roots(plan, b8_path, root_frames)
        source_rows.append((meta, roots))
        for frame, root in sorted(roots.items()):
            root_path = output / f"root-{meta['mix']}-{meta['seed']}-f{frame}.npz"
            from research.bl_prefix.engine import save_root

            save_root(root, root_path)
            all_root_rows.append((root_path, meta))
        _progress(
            "source_h5000_complete",
            source_index=source_index + 1,
            sources=len(plans),
            seed=int(meta["seed"]),
            mix=str(meta["mix"]),
            steps=int(meta["steps"]),
            elapsed_seconds=float(meta["elapsed_seconds"]),
            captured_root_frames=list(meta["captured_root_frames"]),
        )
    # Keep every operational identity inside the burned preflight namespace.
    # Scientific future/exploration namespaces are never consumed by this run.
    future_seeds = preflight_seeds[3:6]
    exploration_seeds = preflight_seeds[6:78]
    if len(future_seeds) != 3 or len(exploration_seeds) != 72:
        raise ValueError("burned preflight namespace lacks the timing RNG budget")
    continuation_rows = []
    for meta, roots in source_rows:
        if not roots:
            continue
        frame, root = sorted(roots.items())[-1]
        continuation = _continuation_timing(root, b8_path, future_seeds[0])
        continuation.update({"seed": meta["seed"], "mix": meta["mix"], "root_frame": frame})
        continuation_rows.append(continuation)
    if len(all_root_rows) < 3:
        raise ValueError("fewer than three 128-history late living roots were captured")
    latest_roots = []
    for meta, roots in source_rows:
        if roots:
            frame = sorted(roots)[-1]
            latest_roots.append((output / f"root-{meta['mix']}-{meta['seed']}-f{frame}.npz", meta))
    e6 = _run_e6_workers(
        latest_roots,
        b8_path,
        fixture_r,
        fixture_f,
        future_seeds,
        args.workers,
    )
    _progress(
        "e6_phase_complete",
        status=str(e6.get("status")),
        roots=int(e6.get("roots") or 0),
        rollouts_completed=int(e6.get("rollouts_completed") or 0),
        world_steps=int(e6.get("world_steps") or 0),
        elapsed_seconds=e6.get("elapsed_seconds"),
    )
    e7 = _run_e7_workers(
        source_rows,
        b8_path,
        future_seeds,
        exploration_seeds,
        args.workers,
    )
    _progress(
        "e7_phase_complete",
        status=str(e7.get("status")),
        sources=int(e7.get("sources") or 0),
        rollouts_completed=int(e7.get("rollouts_completed") or 0),
        world_steps=int(e7.get("world_steps") or 0),
        elapsed_seconds=e7.get("elapsed_seconds"),
    )
    elapsed = time.perf_counter() - began
    source_steps = sum(int(meta["steps"]) for meta, _ in source_rows)
    continuation_steps = sum(int(row["steps"]) for row in continuation_rows)
    e6_steps = int(e6.get("world_steps") or 0)
    e7_steps = int(e7.get("world_steps") or 0)
    step_rates = [
        float(meta["elapsed_seconds"]) / int(meta["steps"])
        for meta, _ in source_rows
        if int(meta["steps"]) > 0
    ]
    step_rates.extend(
        float(row["elapsed_seconds"]) / int(row["steps"])
        for row in continuation_rows
        if int(row["steps"]) > 0
    )
    for worker in (e6, e7):
        if int(worker.get("world_steps") or 0) > 0 and worker.get("elapsed_seconds"):
            step_rates.append(
                float(worker["elapsed_seconds"])
                * int(worker["workers"])
                / int(worker["world_steps"])
            )
    overall_status = (
        "PREFLIGHT_ONLY"
        if e6.get("status") == "PREFLIGHT_ONLY" and e7.get("status") == "PREFLIGHT_ONLY"
        else "INCOMPLETE"
    )
    timing = {
        "schema": "bl-prefix-production-preflight/v2",
        "status": overall_status,
        "scientific": False,
        "decision": "NOT_SCORED_PREFLIGHT",
        "scientific_metrics": "omitted",
        "audit": {"path": str(Path(args.audit).resolve()), "sha256": audit_sha},
        "registration_e10_sha256": registration_sha,
        "b8": {"path": str(b8_path), "sha256": _sha256(b8_path)},
        "sources": [meta for meta, _ in source_rows],
        "root_count": len(all_root_rows),
        "root_frames_requested": list(root_frames),
        "root_history_decisions": 128,
        "continuations": continuation_rows,
        "worker_runs": {"e6": e6, "e7": e7},
        "timing_seconds": {
            "b8_startup": startup_seconds,
            "clone_total": sum(
                sum(float(value) for value in meta.get("clone_seconds", ()))
                for meta, _ in source_rows
            ),
            "e7_calibration_total": float(e7.get("calibration_seconds") or 0.0),
            "source_h5000_total": sum(float(meta["elapsed_seconds"]) for meta, _ in source_rows),
            "continuation_total": sum(float(row["elapsed_seconds"]) for row in continuation_rows),
            "e6_full_worker": e6.get("elapsed_seconds"),
            "e7_full_worker": e7.get("elapsed_seconds"),
            "conservative_max_seconds_per_world_step": max(step_rates) if step_rates else None,
            "wall_total": elapsed,
        },
        "workload": {
            "source_h5000_worlds": 3,
            "source_world_steps": source_steps,
            "continuation_world_steps": continuation_steps,
            "e6_worker_world_steps": e6_steps,
            "e7_worker_world_steps": e7_steps,
            "e7_calibrations": int(e7.get("calibrations") or 0),
            "captured_root_clone_count": sum(
                len(meta.get("clone_seconds", ())) for meta, _ in source_rows
            ),
            "all_measured_world_steps": source_steps + continuation_steps + e6_steps + e7_steps,
            "dead_stops_included": True,
            "reacting_opponents": True,
            "workers": args.workers,
            "thread_policy": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
        "no_inference": "These burned-fixture timings do not estimate scientific outcomes.",
        "command": " ".join(sys.argv),
    }
    _publish_json(output / "timings.json", timing)
    _publish_json(
        output / "worker_timing.json",
        {"schema": "bl-prefix-production-preflight-workers/v1", "e6": e6, "e7": e7},
    )
    _publish_json(
        output / "root_inventory.json",
        {
            "schema": "bl-prefix-production-preflight-roots/v1",
            "status": "PREFLIGHT_ONLY",
            "scientific": False,
            "roots": [
                {
                    "path": str(path),
                    "seed": int(meta["seed"]),
                    "mix": meta["mix"],
                    "root_frame": int(path.stem.rsplit("-f", 1)[1]),
                    "root_length": int(meta["root_lengths"][path.stem.rsplit("-f", 1)[1]]),
                    "history_decisions": 128,
                }
                for path, meta in all_root_rows
            ],
        },
    )
    return timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--b8", type=Path, default=DEFAULT_B8)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sources", type=int, choices=(3,), default=3)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--root-frames",
        default="2000,2500,3000,3500",
        help="comma-separated late living decision boundaries to capture",
    )
    args = parser.parse_args()
    value = run(args)
    print(json.dumps(value, sort_keys=True))
    return 0 if value.get("status") == "PREFLIGHT_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())

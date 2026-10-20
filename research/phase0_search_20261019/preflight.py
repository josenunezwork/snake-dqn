"""Bounded disjoint CPU timing/calibration; never evaluates selection worlds."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import platform
import statistics
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from research.phase0_search_20261019.teacher import (
    SearchConfig,
    SearchTeacher,
    build_world,
    configure_cpu,
    step_world,
)

B8 = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth"
)


def fixture(row: dict, checkpoint: Path, length: int) -> Any:
    """Strict-profile sim and real roster with valid, synthetic long bodies."""
    from src.simd_env.grid_scenarios import inject_serpentines

    world = build_world(row, checkpoint)
    inject_serpentines([world.sim], [length, 300, 400, 200, 100, 100], box_height=11)
    world.sim.check_grid_invariants()
    return world


def timing_task(payload: tuple) -> dict:
    """One single-thread search including cloning, features and inference."""
    row, checkpoint, length, config, *execution = payload
    implementation, device = execution or ("batched", "cpu")
    configure_cpu()
    from research.phase0_search_20261019.runner import operational_guard

    operational_guard()
    world = fixture(row, Path(checkpoint), length)
    teacher = SearchTeacher(
        replace(SearchConfig(**config), length_gate=0), implementation=implementation, device=device
    )
    started = time.perf_counter()
    diag = step_world(world, teacher)
    return {
        "world_id": row["world_id"],
        "mix": row["mix"],
        "length": length,
        "seconds": time.perf_counter() - started,
        "legal_actions": len(diag.get("scores", {})),
        "searched": diag["searched"],
        "action": diag["action"],
        "scores": diag["scores"],
        "implementation": implementation,
        "device": device,
    }


def calibrate(rows: list, checkpoint: Path, horizon: int) -> dict:
    """Fit a strict '< threshold' margin trigger from bare-B8 worlds only."""
    import numpy as np

    margins, lengths, worlds = [], [], []
    for row in rows:
        from research.phase0_search_20261019.runner import operational_guard

        operational_guard()
        world = build_world(row, checkpoint)
        count = 0
        offset = len(lengths)
        started = time.perf_counter()
        while count < horizon and world.sim.alive[0, 0]:
            if time.perf_counter() - started > 90:
                raise RuntimeError("calibration world exceeded 90-second local preflight cap")
            length = int(world.sim.length[0, 0])
            diag = step_world(world)
            margins.append(float(diag["margin"]))
            lengths.append(length)
            count += 1
        worlds.append(
            {
                "world_id": row["world_id"],
                "decisions": count,
                "mandatory_length_decisions": sum(x >= 600 for x in lengths[offset:]),
                "proposal_length800_decisions": sum(x >= 800 for x in lengths[offset:]),
                "seconds": time.perf_counter() - started,
            }
        )
        print(
            json.dumps(
                {"event": "calibration_complete", "world_id": row["world_id"], "decisions": count}
            ),
            flush=True,
        )
    m, long = np.asarray(margins), np.asarray(lengths) >= 600
    target = int(round(len(m) * 0.10))
    additional = max(0, target - int(long.sum()))
    eligible = np.sort(m[~long & np.isfinite(m)])
    threshold = 0.0
    if additional and len(eligible):
        k = min(additional, len(eligible))
        threshold = float(np.nextafter(eligible[k - 1], np.inf))
    admitted = long | (m < threshold)
    return {
        "worlds": worlds,
        "horizon": horizon,
        "decisions": len(m),
        "mandatory_length_fraction": float(long.mean()),
        "searched_fraction": float(admitted.mean()),
        "proposal_length800_fraction": float((np.asarray(lengths) >= 800).mean()),
        "margin_threshold": threshold,
        "target_fraction": 0.10,
        "method": "all alive bare-B8 decisions pooled; mandatory L>=600 first; "
        "strict margin threshold fills remaining target, ties admitted together",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=B8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("calibrate", "timing", "scaling"), required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--implementation", choices=("batched", "reference"), default="batched")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 7))
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--rounds", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    if args.device != "cpu" and args.workers != 1:
        parser.error("GPU timing uses one process; CPU process scaling is a separate measurement")
    if args.device != "cpu" and args.implementation == "reference":
        parser.error("the reference teacher requires CPU")
    configure_cpu()
    from research.phase0_search_20261019.runner import operational_guard

    operational_guard()
    bank = json.loads(args.bank.read_text())
    audit = json.loads(args.bank.with_name("SEED_AUDIT.json").read_text())
    if (
        audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != bank.get("allocation_semantic_sha256")
        or audit.get("world_bank_file", {}).get("sha256")
        != hashlib.sha256(args.bank.read_bytes()).hexdigest()
    ):
        raise ValueError("preflight requires a passing hash-bound bank audit")
    config = asdict(SearchConfig()) if args.config is None else json.loads(args.config.read_text())
    config = config.get("search", config)
    rows = bank["fixtures"]["calibration" if args.mode == "calibrate" else "timing"]
    start = time.perf_counter()
    if args.mode == "calibrate":
        result = calibrate(rows, args.checkpoint, args.horizon)
    else:
        # Warm one tiny real fixture outside measurement to compile numba kernels.
        timing_task(
            (
                rows[0],
                str(args.checkpoint),
                200,
                {**config, "rollouts": 1, "horizon": 2},
                args.implementation,
                args.device,
            )
        )
        payloads = [
            (row, str(args.checkpoint), length, config, args.implementation, args.device)
            for _ in range(args.rounds)
            for length in (200, 800, 1500)
            for row in rows
        ]
        if args.mode == "scaling":
            payloads = [payloads[-1]] * 6
        clock = time.perf_counter()
        if args.workers == 1:
            results = [timing_task(p) for p in payloads]
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
            ) as pool:
                results = list(pool.map(timing_task, payloads))
        wall = time.perf_counter() - clock
        result = {
            "rows": results,
            "workers": args.workers,
            "rounds": args.rounds if args.mode == "timing" else 1,
            "batch_wall_seconds": wall,
            "search_seconds_median": statistics.median(r["seconds"] for r in results),
            "decisions_per_wall_second": len(results) / wall,
            "note": "Synthetic self-avoiding body placement; real strict-profile roster. "
            "Costs include early branch deaths; not a representative gameplay sample.",
        }
    result.update(
        mode=args.mode,
        command=[sys.executable, *sys.argv],
        platform=platform.platform(),
        implementation=args.implementation,
        device=args.device,
        config=config,
        total_seconds=time.perf_counter() - start,
        bank_sha256=hashlib.sha256(args.bank.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_sha256={
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("preflight.py", "teacher.py")
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

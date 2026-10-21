"""Bounded real O1 entrypoint: excluded timing/parity before the one-look bank."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from research.t34_o1_harvest.registration import (
    B8_SHA,
    CAPS,
    require_registration,
    sha256,
    write_once,
)


class CapExceeded(TimeoutError):
    """An operational cap was reached; this is never a scientific KILL."""


def check_cap(deadline: float) -> None:
    """Stop on the first registered time cap."""
    if time.monotonic() >= deadline:
        raise CapExceeded("O1 ordinary/timing wall cap expired")


@contextlib.contextmanager
def runtime_inputs(catalog: str | None, input_root: str | None) -> Iterator[None]:
    """Use existing verified public paths on Colab, local checkpoints otherwise."""
    if bool(catalog) != bool(input_root):
        raise ValueError("catalog and input-root must be supplied together")
    if catalog:
        from research.t34_pathb.unpaced_colab import _public_paths, _runtime_checkpoint_paths

        with _runtime_checkpoint_paths(_public_paths(Path(catalog), Path(input_root))):
            yield
    else:
        yield


def step(world: Any, policy: Any, forced: int | None = None) -> list[int]:
    """Take a prepared step with fresh reacting opponents and private global RNG."""
    from research.t34_exit_b8.trajectory import prepare_world

    outer = random.getstate()
    random.setstate(world._o1_rng)
    try:
        prepare_world(world)
        action = policy.evaluate(world).greedy_action() if forced is None else forced
        grid = policy.action_grid(world, action)
        world.sim.step(grid)
        world.runtime.observe_step(world.sim)
        world._o1_rng = random.getstate()
        return [int(value) for value in grid[0]]
    finally:
        random.setstate(outer)


def digest(world: Any, observer: Any | None = None) -> str:
    """Hash physics, food order, policy carry and every RNG without observer code."""
    from research.t34_shadow.shadow import full_digest

    with observer.detached() if observer else contextlib.nullcontext():
        return full_digest(world, checkpoint=B8_SHA, global_rng_state=world._o1_rng)


def snapshot(world: Any, observer: Any) -> tuple[Any, Any]:
    """Clone all mutable state without copying observer closures into dynamics."""
    from research.t34_shadow.shadow import clone_world_exact

    with observer.detached():
        clone = clone_world_exact(world)
    clone._o1_rng = world._o1_rng
    copied = observer.clone_for(clone.sim)
    copied.begin()
    return clone, copied


def reservoir_priority(seed: int, decision: int) -> str:
    """Fixed outcome-independent uniform-hash reservoir priority."""
    return hashlib.sha256(f"o1-event-v1:{seed}:{decision}".encode()).hexdigest()


def rank_branches(branches: dict[int, dict[str, Any]], bare: int) -> list[int]:
    """Mass ranking only, exact ties to bare, then action index."""
    return sorted(branches, key=lambda a: (-branches[a]["mass_sum"], a != bare, a))


def evaluate_root(
    world: Any, observer: Any, policy: Any, horizon: int, deadline: float
) -> dict[str, Any]:
    """Enumerate advisory-legal first actions, H-step B8, ledger on every branch."""
    import numpy as np

    before = digest(world, observer)
    evaluation = policy.evaluate(world)
    bare = evaluation.greedy_action()
    mask = np.asarray(evaluation.select, dtype=bool).copy()
    if not mask.any():
        mask[:3] = evaluation.legal[:3]
    if not mask[bare]:
        raise ValueError("bare is outside its advisory/fallback domain")
    branches = {}
    for action in np.flatnonzero(mask):
        branch, ledger = snapshot(world, observer)
        mass, alive_steps, rows = 0, 0, 0
        started = time.monotonic()
        for tick in range(horizon):
            check_cap(deadline)
            step(branch, policy, int(action) if tick == 0 else None)
            ledger.assert_accounting()
            rows += 1
            if bool(branch.sim.alive[0, 0]):
                mass += int(branch.sim.length[0, 0])
                alive_steps += 1
            else:
                break
        branches[int(action)] = {
            "action": int(action),
            "mass_sum": mass,
            "alive_steps": alive_steps,
            "row_steps": rows,
            "survives": alive_steps == horizon,
            "ledger": ledger.ledger(),
            "elapsed_seconds": time.monotonic() - started,
            "final_digest": digest(branch, ledger),
        }
        ledger.uninstall()
    if digest(world, observer) != before:
        raise ValueError("branching changed source state/carry/RNG")
    baseline = branches[bare]["mass_sum"]
    for value in branches.values():
        value["delta_mass_sum"] = value["mass_sum"] - baseline
        value["delta_A500"] = (value["mass_sum"] - baseline) / horizon
    return {
        "bare_action": bare,
        "advisory_legal": mask.tolist(),
        "illegal_actions": [a for a in range(6) if not mask[a]],
        "source_digest": before,
        "source_preserved": True,
        "horizon": horizon,
        "q": [float(q) for q in evaluation.q],
        "ranked_actions": rank_branches(branches, bare),
        "branches": {str(a): value for a, value in branches.items()},
    }


def collect(
    row: dict[str, Any],
    checkpoint: str,
    *,
    frames: int,
    horizon: int,
    time_roots: tuple[int, ...],
    event_limit: int,
    deadline: float,
    parity: bool = False,
    catalog: str | None = None,
    input_root: str | None = None,
) -> dict[str, Any]:
    """One deterministic world per process; never log outcomes during collection."""
    import numpy as np
    import torch

    from research.t34_exit_b8.policy import B8Policy
    from research.t34_exit_b8.trajectory import build_world, prepare_world
    from research.t34_o1_harvest.provenance import ProvenanceObserver
    from research.t34_shadow.shadow import clone_world_exact

    torch.set_num_threads(1)
    started = time.monotonic()
    with runtime_inputs(catalog, input_root):
        world = build_world(row, checkpoint)
    world._o1_rng = random.Random(int(row["world_seed"])).getstate()
    if world.sim.cfg.mechanics_version != 2:
        raise ValueError("O1 constructor must execute mechanics v2")
    policy = B8Policy(device="cpu")
    plain = clone_world_exact(world) if parity else None
    if plain:
        plain._o1_rng = world._o1_rng
    observer = ProvenanceObserver(world.sim)
    time_saved: dict[int, tuple[Any, Any]] = {}
    events, event_saved = [], {}
    last_event, base_seconds, base_steps = -64, 0.0, 0
    parity_chain = hashlib.sha256()
    last_progress = time.monotonic()
    for decision in range(frames + 1):
        check_cap(deadline)
        if not bool(world.sim.alive[0, 0]):
            break
        random_outer = random.getstate()
        random.setstate(world._o1_rng)
        prepare_world(world)
        world._o1_rng = random.getstate()
        random.setstate(random_outer)
        if decision in time_roots:
            time_saved[decision] = snapshot(world, observer)
        pending = bool(events and events[-1]["decision"] == decision and events[-1]["eligible"])
        if pending and event_limit:
            priority = reservoir_priority(int(row["world_seed"]), decision)
            if len(event_saved) < event_limit or priority < max(event_saved):
                event_saved[priority] = (decision, snapshot(world, observer))
                if len(event_saved) > event_limit:
                    worst = max(event_saved)
                    removed = event_saved.pop(worst)[1][1]
                    removed.uninstall()
        if decision == frames:
            break
        prior_alive = world.sim.alive.copy()
        tick_started = time.monotonic()
        actions = step(world, policy)
        observer.assert_accounting()
        base_seconds += time.monotonic() - tick_started
        base_steps += 1
        if parity:
            plain_actions = step(plain, policy)
            observed_digest, plain_digest = digest(world, observer), digest(plain)
            if actions != plain_actions or observed_digest != plain_digest:
                raise ValueError(f"observer parity mismatch at {decision}")
            parity_chain.update(bytes.fromhex(observed_digest))
        died = prior_alive[0] & ~world.sim.alive[0]
        near = []
        if bool(world.sim.alive[0, 0]):
            heads = world.sim.heads()[0]
            for slot in np.flatnonzero(died):
                if slot and int(np.abs(heads[int(slot)] - heads[0]).sum()) <= 30:
                    near.append(int(slot))
        if near:
            eligible = decision + 1 - last_event >= 64
            events.append({"decision": decision + 1, "slots": near, "eligible": eligible})
            if eligible:
                last_event = decision + 1
        if time.monotonic() - last_progress >= 100:
            print(
                json.dumps(
                    {
                        "event": "o1_progress",
                        "world_id": row["world_id"],
                        "base_steps": base_steps,
                        "elapsed": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            last_progress = time.monotonic()
    roots = dict(time_saved)
    for decision, saved in event_saved.values():
        if decision in roots:
            saved[1].uninstall()
        else:
            roots[decision] = saved
    eligible_count = sum(e["eligible"] for e in events)
    event_decisions = {decision for decision, _ in event_saved.values()}
    for event in events:
        event["selected"] = event["decision"] in event_decisions
        event["inclusion_probability"] = (
            min(1.0, event_limit / eligible_count) if event["eligible"] and eligible_count else 0.0
        )
    result_roots = []
    for decision, (root, root_observer) in sorted(roots.items()):
        root_result = evaluate_root(root, root_observer, policy, horizon, deadline)
        root_result.update(
            decision=decision,
            time_root=decision in time_saved,
            event_root=decision in event_decisions,
        )
        result_roots.append(root_result)
        root_observer.uninstall()
    observer.assert_accounting()
    result = {
        "schema": "t34-o1-world/v1",
        "row": row,
        "cfg": asdict(world.sim.cfg),
        "checkpoint_sha256": B8_SHA,
        "base_steps": base_steps,
        "base_seconds": base_seconds,
        "base_ledger": observer.ledger(),
        "base_final_digest": digest(world, observer),
        "base_survives": bool(world.sim.alive[0, 0]),
        "base_death_cause": int(world.sim.get_death_cause()[0, 0]),
        "events": events,
        "roots": result_roots,
        "frames": frames,
        "horizon": horizon,
        "complete": True,
        "accounting": True,
        "elapsed_seconds": time.monotonic() - started,
        "parity": (
            {"pass": True, "frames": base_steps, "chain": parity_chain.hexdigest()}
            if parity
            else None
        ),
    }
    observer.uninstall()
    return result


def _task(task: dict[str, Any]) -> dict[str, Any]:
    payload = dict(task)
    output = payload.pop("_output", None)
    result = collect(**payload)
    if output is not None:
        write_once(Path(output), result)
    return result


def collect_many(
    rows: list[dict[str, Any]],
    checkpoint: str,
    workers: int,
    payload_dir: Path | None = None,
    prefix: str = "world",
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Deterministically merge independent worlds in bank order."""
    tasks = [
        dict(
            row=row,
            checkpoint=checkpoint,
            **kwargs,
            _output=str(payload_dir / f"{prefix}-{i:03d}.json") if payload_dir else None,
        )
        for i, row in enumerate(rows)
    ]
    if workers == 1:
        return [_task(task) for task in tasks]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        return list(pool.map(_task, tasks))


def fixture_forecast(worlds: list[dict[str, Any]], workers: int) -> dict[str, Any]:
    """Worst-case base/branch work from measured fixture component costs."""
    base_rate = max(w["base_seconds"] / max(1, w["base_steps"]) for w in worlds)
    branches = [b for w in worlds for r in w["roots"] for b in r["branches"].values()]
    if not branches:
        raise ValueError("timing fixture produced no branches")
    branch_rate = max(b["elapsed_seconds"] / max(1, b["row_steps"]) for b in branches)
    setup_per_world = max(
        0,
        max(
            w["elapsed_seconds"]
            - w["base_seconds"]
            - sum(b["elapsed_seconds"] for r in w["roots"] for b in r["branches"].values())
            for w in worlds
        ),
    )
    # Five world waves bound 48+3 parity tasks at 11 workers; parity costs twice base.
    import math

    seconds = math.ceil(48 / workers) * (
        5000 * base_rate + 8 * 6 * 500 * branch_rate + setup_per_world
    ) + math.ceil(3 / workers) * (10000 * base_rate + setup_per_world)
    # 2x platform/contention caution; fixed 20 minutes for setup/export is separate.
    seconds *= 2
    return {
        "base_seconds_per_step": base_rate,
        "branch_seconds_per_row_step": branch_rate,
        "setup_seconds_per_world": setup_per_world,
        "workers": workers,
        "conservative_multiplier": 2,
        "remaining_work_seconds": seconds,
        "l4_forecast_seconds_including_setup_export": seconds + 1200,
        "l4_forecast_units_at_2_per_hour": (seconds + 1200) / 1800,
        "scope": "extrapolation only; live disjoint L4 fixture required before bank admission",
    }


def ordinary_remaining(runtime_root: Path, now: float | None = None) -> float:
    """Deduct all already charged setup and the 20% reserve from broker time."""
    current = time.time() if now is None else now
    state = json.loads((runtime_root / "state.json").read_text())
    spec = json.loads((runtime_root / "spec.json").read_text())
    started, timeout = float(state["started_at"]), float(spec["timeout_seconds"])
    if (
        not all(math.isfinite(x) for x in (current, started, timeout))
        or started > current
        or not 0 < timeout <= CAPS["runtime_seconds"]
    ):
        raise ValueError("invalid supervisor session clock/budget")
    # The broker timeout already subtracts allocation/upload time. Keep 48 min
    # reserve plus 60 seconds for analyzer/export/stop INSIDE the session cap.
    return timeout - (current - started) - (CAPS["runtime_seconds"] - CAPS["ordinary_seconds"]) - 60


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    registration = require_registration(args.registration)
    if sha256(args.checkpoint) != B8_SHA:
        raise ValueError("B8 checkpoint identity differs")
    if args.workers < 1 or args.workers > CAPS["workers"]:
        raise ValueError("worker count exceeds registration")
    if args.out.exists():
        raise FileExistsError("output must be fresh")
    if args.mode == "campaign" and (
        not os.environ.get("COLAB_CLI_ROOT") or not args.catalog or args.workers != CAPS["workers"]
    ):
        raise ValueError("study execution requires supervised L4/public inputs/fixed worker count")
    if args.mode == "preflight" and (args.workers != 1 or args.frames != 40 or args.horizon != 8):
        raise ValueError("Mac preflight is fixed at 2 worlds,40 frames,H8,one worker")
    args.out.mkdir(parents=True)
    write_once(
        args.out / "start.json",
        {
            "schema": "t34-o1-start/v1",
            "mode": args.mode,
            "registration_sha256": sha256(args.registration),
            "source_commit": registration["source_commit"],
            "command": __import__("sys").argv,
            "platform": platform.platform(),
            "niceness": os.getpriority(os.PRIO_PROCESS, 0),
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    )
    bank = registration["bank"]
    kwargs = dict(
        catalog=str(args.catalog) if args.catalog else None,
        input_root=str(args.input_root) if args.input_root else None,
    )
    try:
        allowed = (
            180.0
            if args.mode == "preflight"
            else min(
                float(CAPS["runner_seconds"]),
                ordinary_remaining(Path(os.environ["COLAB_CLI_ROOT"])),
            )
        )
        deadline = started + allowed
        check_cap(deadline)
        if args.mode == "preflight":
            worlds = collect_many(
                bank["excluded_preflight_rows"],
                str(args.checkpoint),
                1,
                payload_dir=args.out,
                frames=40,
                horizon=8,
                time_roots=(10, 20),
                event_limit=0,
                deadline=deadline,
                **kwargs,
            )
            parity_worlds, admission = [], None
        else:
            timing_start = time.monotonic()
            timing = collect_many(
                bank["excluded_timing_rows"],
                str(args.checkpoint),
                args.workers,
                payload_dir=args.out,
                prefix="timing-world",
                frames=5000,
                horizon=500,
                time_roots=(1000, 3000),
                event_limit=0,
                deadline=min(deadline, timing_start + CAPS["timing_seconds"]),
                **kwargs,
            )
            if any(len(w["roots"]) != 2 for w in timing):
                raise CapExceeded("excluded timing fixture did not supply two roots per world")
            forecast = fixture_forecast(timing, args.workers)
            admission = {"timing_seconds": time.monotonic() - timing_start, "forecast": forecast}
            write_once(args.out / "timing.json", {"worlds": timing, **admission})
            if forecast["remaining_work_seconds"] > deadline - time.monotonic():
                raise CapExceeded("L4 timing forecast does not fit remaining ordinary cap")
            parity_worlds = collect_many(
                bank["excluded_parity_rows"],
                str(args.checkpoint),
                args.workers,
                payload_dir=args.out,
                prefix="parity-world",
                frames=5000,
                horizon=500,
                time_roots=(),
                event_limit=0,
                deadline=deadline,
                parity=True,
                **kwargs,
            )
            write_once(args.out / "parity.json", parity_worlds)
            worlds = collect_many(
                bank["rows"],
                str(args.checkpoint),
                args.workers,
                payload_dir=args.out,
                frames=5000,
                horizon=500,
                time_roots=(1000, 2000, 3000, 4000),
                event_limit=4,
                deadline=deadline,
                **kwargs,
            )
        for index, world in enumerate(worlds):
            if not (args.out / f"world-{index:03d}.json").exists():
                write_once(args.out / f"world-{index:03d}.json", world)
        require_registration(args.registration)
        receipt = {
            "schema": "t34-o1-run/v1",
            "status": "COMPLETE",
            "mode": args.mode,
            "registration_sha256": sha256(args.registration),
            "worlds": [
                {"path": f"world-{i:03d}.json", "sha256": sha256(args.out / f"world-{i:03d}.json")}
                for i in range(len(worlds))
            ],
            "parity_pass": len(parity_worlds) == 3
            and all(w["parity"]["pass"] for w in parity_worlds),
            "timing_admitted": admission is not None,
            "elapsed_seconds": time.monotonic() - started,
            "forecast": (
                fixture_forecast(worlds, CAPS["workers"])
                if args.mode == "preflight"
                else admission["forecast"]
            ),
        }
        if args.mode == "campaign":
            receipt["parity_sha256"] = sha256(args.out / "parity.json")
            receipt["timing_sha256"] = sha256(args.out / "timing.json")
        write_once(args.out / "receipt.json", receipt)
        return receipt
    except Exception as exc:
        write_once(
            args.out / "failure.json",
            {
                "status": "INCOMPLETE" if isinstance(exc, CapExceeded) else "INVALID",
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("campaign", "preflight"), required=True)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--input-root", type=Path)
    args = parser.parse_args()
    result = run(args)
    print(
        json.dumps({key: result[key] for key in ("status", "mode", "elapsed_seconds", "forecast")}),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Replay all 46 late T1 LADs on CPU, retaining every decision through D100."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402

from research.b8_death_census_20261014 import census  # noqa: E402
from research.t1_commitment_20261015.replay import (  # noqa: E402
    ARTIFACTS,
    INPUTS,
    Replay,
    assert_same_world,
    build_replay,
    configure_cpu,
    load_inputs,
    state_fields,
)
from research.t9_divergence_20261017.metrics import measure  # noqa: E402

DEFAULT_OUT = census.ARTIFACTS / "t9-divergence-20261017"


def selected_rows(path: Path) -> list[dict]:
    """Require the complete, unique source cohort before starting a run."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [r for r in rows if r["length"] >= 800 and r["lad"] is not None]
    assert len(selected) == 46, f"expected 46 late LADs, got {len(selected)}"
    assert len({(r["arm"], r["env"]) for r in selected}) == 46
    return sorted(selected, key=lambda r: (r["arm"], r["env"]))


def trace_branch(prepared: Replay, row: dict, series: dict, fatal: bool) -> dict:
    """Measure D0..D100; extra unmeasured steps validate the T1 endpoint."""
    lad = row["lad"]
    start, env = int(lad["decision"]), int(row["env"])
    first = int(lad["chosen_action"] if fatal else lad["best_surviving_action"])
    branch = prepared.clone(prepared=True)
    assert_same_world(prepared, branch)
    trace: list[dict] = []
    checked = 0

    def capture(r: Replay, offset: int, action: int) -> None:
        nonlocal checked
        assert r.sim.alive[0, 0]
        assert int(r.sim.frame[0]) == start + offset + 1
        if fatal:
            assert int(r.sim.length[0, 0]) == int(series["length"][env, start + offset])
            checked += 1
        if offset <= 100:
            trace.append({"offset": offset, "action": action, **measure(r.sim, action)})

    capture(branch, 0, first)
    branch.first(first)
    executed = 1
    horizon = row["fatal_t"] - start + 1 if fatal else lad["outcomes"][str(first)]["horizon"]
    while executed < horizon and branch.sim.alive[0, 0]:
        forced = int(series["action"][env, start + executed]) if fatal else None

        def observe(r: Replay) -> None:
            # Hero inference is stateless. Read it once for the diagnostic and
            # let the canonical dispatcher choose the same action for stepping.
            if forced is None:
                mask = r.sim.get_advisory_action_mask()[0, 0].copy()
                if not mask.any():
                    mask[:3] = True
                action = int(np.where(mask, r.q().astype(np.float64), -np.inf).argmax())
            else:
                action = forced
            capture(r, executed, action)

        branch.step(None if forced is None else [forced], observe)
        executed += 1
    alive = bool(branch.sim.alive[0, 0])
    death = None if alive else start + executed - 1
    if fatal:
        assert death == row["fatal_t"], ("fatal replay mismatch", death, row["fatal_t"])
    else:
        assert alive and executed == horizon, ("T1 survivor no longer survives", row["arm"], env)
    return {
        "trace": trace,
        "first_action": first,
        "death_decision": death,
        "validated_transitions": executed,
        "endpoint_alive": alive,
        "length_values_checked": checked,
    }


def study_row(row: dict, series: dict, context: Any) -> dict:
    """Recreate one prepared LAD, verify its Q and geometry, then clone twice."""
    started = time.perf_counter()
    env, lad = row["env"], row["lad"]
    replay = build_replay(row["arm"], [env], context)
    assert replay.seeds[0] == row["world_seed"]
    snapshots = []
    for decision in range(lad["decision"] + 1):

        def observe(r: Replay) -> None:
            assert r.sim.alive[0, 0]
            assert int(r.sim.length[0, 0]) == int(series["length"][env, decision])
            if decision == lad["decision"]:
                np.testing.assert_array_equal(r.q(), np.asarray(lad["q"], dtype=np.float32))
                assert int(series["action"][env, decision]) == lad["chosen_action"]
                best = max(lad["surviving_actions"], key=lambda a: (lad["q"][a], -a))
                assert best == lad["best_surviving_action"]
                from research.t1_commitment_20261015.topology import describe_action

                assert describe_action(r.sim, best) == lad["best_surviving_geometry"]
                assert describe_action(r.sim, lad["chosen_action"]) == lad["chosen_geometry"]
                snapshots.append(r.clone(prepared=True))

        replay.step([int(series["action"][env, decision])], observe)
    prepared = snapshots[0]
    before = state_fields(prepared.sim, prepared.opponents)
    fatal = trace_branch(prepared, row, series, True)
    surviving = trace_branch(prepared, row, series, False)
    assert state_fields(prepared.sim, prepared.opponents) == before, "clone source mutated"
    return {
        "arm": row["arm"],
        "env": env,
        "world_seed": row["world_seed"],
        "length": row["length"],
        "lad_decision": lad["decision"],
        "fatal_t": row["fatal_t"],
        "H_LAD": row["H_LAD"],
        "fatal": fatal,
        "surviving": surviving,
        "warmup_length_values_checked": lad["decision"] + 1,
        "lad_q_and_action_geometry_exact": True,
        "source_clone_unchanged": True,
        "prepared_state_sha256_fields": before,
        "wall_seconds": time.perf_counter() - started,
    }


def bindings(rows_path: Path) -> dict:
    """Pin source and external inputs used by this replay, excluding report code."""
    paths = [rows_path, Path("docs/research/specs/t9_divergence_spec_2026-10-17.md")]
    paths += sorted(Path("src").rglob("*.py"))
    paths += [p for p in Path("research").rglob("*.py") if p.name != "report.py"]
    for arm in census.B8_ARMS:
        paths += [INPUTS / f"{arm}.json", INPUTS / f"{arm}_series.npz", census.b8_path(arm)]
    return {str(p): census.sha256_file(p) for p in sorted(set(paths))}


def run(out: Path, limit: int | None = None) -> None:
    """Use one process and one numerical thread, with an immutable launch manifest."""
    configure_cpu()
    import torch

    from src.core.device_manager import DeviceManager

    DeviceManager.override_device(torch.device("cpu"))
    out.mkdir(parents=True, exist_ok=False)
    rows_path = ARTIFACTS / "rows.jsonl"
    selected = selected_rows(rows_path)
    manifest = {
        "schema": "t9-divergence/v1",
        "authority": "diagnostic only; not gate evidence",
        "pid": os.getpid(),
        "device": "cpu",
        "processes": 1,
        "threads": 1,
        "launch_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_source_sha256": bindings(rows_path),
        "expected_cohort": len(selected),
        "limit": limit,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    context = census.load_context()
    cached: dict = {}
    completed = 0
    with (out / "rows.jsonl").open("x") as stream:
        for row in selected[:limit]:
            arm = row["arm"]
            if arm not in cached:
                _, cached[arm] = load_inputs(arm)
            result = study_row(row, cached[arm], context)
            stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            completed += 1
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "arm": arm,
                        "env": row["env"],
                        "seconds": round(result["wall_seconds"], 2),
                    }
                ),
                flush=True,
            )
    assert bindings(rows_path) == manifest["input_source_sha256"], "inputs/source changed"
    receipt = {
        "status": "PASS",
        "completed": completed,
        "full_cohort": completed == 46,
        "wall_seconds": time.perf_counter() - started,
        "rows_sha256": census.sha256_file(out / "rows.jsonl"),
        "manifest_sha256": census.sha256_file(out / "manifest.json"),
        "inputs_source_unchanged": True,
    }
    (out / "completion.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, help="pilot only; omitted for the complete cohort")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run(args.out, args.limit)


if __name__ == "__main__":
    main()

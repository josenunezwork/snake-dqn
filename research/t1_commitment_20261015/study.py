"""Exact-action interventions with B8's own greedy continuation (no veto)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.parity import write_json
from research.t1_commitment_20261015.replay import (
    ARTIFACTS,
    Replay,
    build_replay,
    configure_cpu,
    load_inputs,
)


def coarse_decisions(fatal: int) -> list:
    """Stride ten through the last 300 decisions, including f-10 when clamped."""
    start, end = max(0, fatal - 300), max(0, fatal - 10)
    return sorted(set(range(start, end + 1, 10)) | {end})


def rollout(prepared: Replay, action: int, decision: int, fatal: Optional[int] = None) -> dict:
    """300-transition endpoint, with an extended death-reproduction control."""
    branch = prepared.clone(prepared=True)
    horizon = min(300, census.HORIZON - decision)
    branch.first(action)
    executed = 1
    alive_transitions = int(branch.sim.alive[0, 0])
    while executed < horizon and branch.sim.alive[0, 0]:
        branch.step()
        executed += 1
        alive_transitions += int(branch.sim.alive[0, 0])
    survived = bool(branch.sim.alive[0, 0]) and executed == horizon
    endpoint_executed = executed
    if fatal is not None:
        # At d=f-300 the fatal transition is the 301st. Its reproduction is
        # checked separately from the specified 300-transition endpoint.
        while branch.sim.alive[0, 0] and decision + executed <= fatal:
            branch.step()
            executed += 1
        actual_fatal = None if branch.sim.alive[0, 0] else decision + executed - 1
        assert actual_fatal == fatal, (
            "recorded control death mismatch",
            decision,
            fatal,
            actual_fatal,
        )
    else:
        actual_fatal = None if branch.sim.alive[0, 0] else decision + executed - 1
    return {
        "action": action,
        "survives": survived,
        "horizon": horizon,
        "frames_survived": alive_transitions,
        "frames_executed": endpoint_executed,
        "death_decision": actual_fatal,
        "control_reproduced": True if fatal is not None else None,
        "env_frames": branch.env_frames,
    }


def evaluate_decision(prepared: Replay, chosen: int, decision: int, fatal: int) -> dict:
    """Evaluate all advisory-legal actions plus the recorded-action control."""
    sim = prepared.sim
    legal = [int(a) for a in np.flatnonzero(sim.get_advisory_action_mask()[0, 0])]
    q = prepared.q()
    fallback_mask = np.zeros(6, bool)
    fallback_mask[legal] = True
    if not legal:
        fallback_mask[:3] = True
    greedy = int(np.where(fallback_mask, q.astype(np.float64), -np.inf).argmax())
    assert chosen == greedy, ("recorded action != own greedy", decision, chosen, greedy)
    outcomes = {}
    for action in sorted(set(legal) | {chosen}):
        outcomes[str(action)] = rollout(
            prepared, action, decision, fatal if action == chosen else None
        )
    survivors = [a for a in legal if a != chosen and outcomes[str(a)]["survives"]]
    best = max(survivors, key=lambda a: (float(q[a]), -a)) if survivors else None
    return {
        "decision": decision,
        "frame": int(sim.frame[0]),
        "hero_length": int(sim.length[0, 0]),
        "chosen_action": chosen,
        "chosen_is_boost": chosen >= 3,
        "advisory_legal_actions": legal,
        "q": [float(x) for x in q],
        "surviving_actions": survivors,
        "any_surviving_boost": any(a >= 3 for a in survivors),
        "best_surviving_action": best,
        "q_gap": float(q[chosen]) - float(q[best]) if best is not None else None,
        "outcomes": outcomes,
    }


def add_geometry(replay: Replay, measurement: dict) -> dict:
    from research.t1_commitment_20261015.topology import component_geometry, describe_action

    result = dict(measurement)
    geometry = component_geometry(replay.sim)
    chosen = result["chosen_action"]
    best = result["best_surviving_action"]
    result["chosen_geometry"] = describe_action(replay.sim, chosen, geometry=geometry)
    result["surviving_geometry"] = {
        str(a): describe_action(replay.sim, a, geometry=geometry)
        for a in result["surviving_actions"]
    }
    result["best_surviving_geometry"] = (
        result["surviving_geometry"][str(best)] if best is not None else None
    )
    return result


def study_death(arm: str, death: dict, series: Any, context: Any) -> dict:
    from research.t1_commitment_20261015.topology import head_region, sealer_identity

    start = time.perf_counter()
    env, fatal = int(death["env"]), int(death["fatal_t"])
    replay = build_replay(arm, [env], context)
    coarse = set(coarse_decisions(fatal))
    onset = death.get("trap_onset")
    t0 = int(onset["t0"]) if onset else None
    scans, regions = {}, []
    last_positive, last_snapshot, onset_measurement, sealer = None, None, None, None
    before_onset = None
    lengths_checked = 0
    env_frames = 0
    for t in range(fatal + 1):

        def observe(r: Replay) -> None:
            nonlocal last_positive, last_snapshot, before_onset
            nonlocal onset_measurement, sealer, lengths_checked, env_frames
            actual = int(r.sim.length[0, 0]) if r.sim.alive[0, 0] else 0
            assert actual == int(series["length"][env, t]), (arm, env, t, actual)
            assert r.sim.alive[0, 0], ("hero died before recorded fatal decision", arm, env, t)
            lengths_checked += 1
            if t0 is not None and t == t0 - 1:
                before_onset = r.clone(prepared=True)
            if t >= max(0, fatal - 59):
                regions.append({"decision": t, **head_region(r.sim)})
            if t in coarse or t == t0:
                measured = evaluate_decision(r, int(series["action"][env, t]), t, fatal)
                env_frames += sum(o["env_frames"] for o in measured["outcomes"].values())
                scans[t] = measured
                if max(0, fatal - 300) <= t < fatal and measured["surviving_actions"]:
                    last_positive, last_snapshot = t, r.clone(prepared=True)
                if t == t0:
                    onset_measurement = add_geometry(r, measured)
                    onset_measurement["head_region"] = head_region(r.sim)
                    sealer = (
                        sealer_identity(before_onset.sim, r.sim, r.kinds)
                        if before_onset is not None
                        else {"owner": "unresolved", "reason": "no previous decision"}
                    )

        replay.step([int(series["action"][env, t])], observe)
    assert not replay.sim.alive[0, 0], ("recorded replay did not die", arm, env, fatal)
    lad = None
    refinement_env_frames = 0
    if last_positive is not None:
        # Refine every decision in the last positive stride-10 window, even
        # when survival is nonmonotonic inside that window.
        lad = add_geometry(last_snapshot, scans[last_positive])
        refined = last_snapshot.clone(prepared=True)
        refined.first(int(series["action"][env, last_positive]))
        for t in range(last_positive + 1, min(last_positive + 10, fatal)):

            def refine(r: Replay) -> None:
                nonlocal lad, env_frames
                measured = scans.get(t)
                if measured is None:
                    measured = evaluate_decision(r, int(series["action"][env, t]), t, fatal)
                    env_frames += sum(o["env_frames"] for o in measured["outcomes"].values())
                    scans[t] = measured
                if measured["surviving_actions"]:
                    lad = add_geometry(r, measured)

            refined.step([int(series["action"][env, t])], refine)
        refinement_env_frames = refined.env_frames
    assert all(m["outcomes"][str(m["chosen_action"])]["control_reproduced"] for m in scans.values())
    return {
        "schema": "t1a-death/v1",
        "authority": "development diagnostic, not gate evidence",
        "arm": arm,
        "env": env,
        "world_seed": int(death["world_seed"]),
        "fatal_t": fatal,
        "frame": int(death["frame"]),
        "length": int(death["length"]),
        "stratum": "late" if death["length"] >= 800 else "early",
        "trap_onset_census": onset,
        "lad": lad,
        "H_LAD": fatal - lad["decision"] if lad is not None else None,
        "disposition": "LAD found" if lad is not None else "none within 300",
        "onset": onset_measurement,
        "sealer": sealer,
        "region_series": regions,
        "scans": [scans[k] for k in sorted(scans)],
        "recorded_death_reproduced": True,
        "length_values_compared": lengths_checked,
        "rollout_env_frames": env_frames,
        "recorded_replay_env_frames": replay.env_frames,
        "refinement_replay_env_frames": refinement_env_frames,
        "env_frames": env_frames + replay.env_frames + refinement_env_frames,
        "wall_seconds": time.perf_counter() - start,
    }


def run_arm(arm: str, out: Path, limit: Optional[int] = None) -> dict:
    configure_cpu()
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    receipt = {"arm": arm, "pid": os.getpid(), "status": "RUNNING", "deaths_completed": 0}
    write_json(out / f"{arm}.status.json", receipt)
    try:
        data, series = load_inputs(arm)
        context = census.load_context()
        with (out / f"{arm}.jsonl").open("w") as stream:
            for death in data["deaths"][:limit]:
                row = study_death(arm, death, series, context)
                stream.write(json.dumps(census._jsonable(row), sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                receipt["deaths_completed"] += 1
                receipt["wall_seconds"] = time.perf_counter() - started
                receipt["last_env"] = death["env"]
                write_json(out / f"{arm}.status.json", receipt)
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "env": death["env"],
                            "H_LAD": row["H_LAD"],
                            "seconds": round(row["wall_seconds"], 2),
                        }
                    ),
                    flush=True,
                )
        receipt["status"] = "PASS"
    except BaseException:
        receipt["status"] = "FAIL"
        receipt["failure"] = traceback.format_exc()
        raise
    finally:
        receipt["wall_seconds"] = time.perf_counter() - started
        write_json(out / f"{arm}.status.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--arms", default=",".join(census.B8_ARMS))
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    precondition = json.loads((ARTIFACTS / "parity.json").read_text())
    assert precondition["status"] == "PASS", "Step 0 must pass before Step 1"
    arms = args.arms.split(",")
    assert len(arms) == len(set(arms)) and set(arms) <= set(census.B8_ARMS)
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        jobs = [pool.submit(run_arm, arm, args.out / "rows", args.limit) for arm in arms]
        receipts = [job.result() for job in jobs]
    write_json(
        args.out / "run.json",
        {
            "status": "PASS",
            "arms": receipts,
            "workers": args.workers,
            "wall_seconds": time.perf_counter() - started,
            "partial_pilot": args.limit is not None,
            "references_run": False,
        },
    )


if __name__ == "__main__":
    main()

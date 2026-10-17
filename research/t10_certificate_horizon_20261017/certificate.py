"""Replay T3 dense and B8 trajectories and audit optimistic TTL certificates."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

from research.b8_death_census_20261014 import census  # noqa: E402
from research.t1_commitment_20261015.replay import (INPUTS,  # noqa: E402
                                                    build_replay,
                                                    configure_cpu, load_inputs)
from research.t3_rollout_labels_20261015.ceiling import _greedy  # noqa: E402
from research.t9_divergence_20261017.metrics import _arrival  # noqa: E402

ROOT = census.ARTIFACTS / "t10-certificate-horizon-20261017"
DENSE = census.ARTIFACTS / "t3-rollout-labels-20261015/dense/ceiling"
ARMS = ("b8_s0", "b8_s3")
KS = (1, 5, 10, 25, 50, 100, 200, 300)


@njit(cache=True)
def occupancy(
    bodies: np.ndarray,
    pointers: np.ndarray,
    counts: np.ndarray,
    lengths: np.ndarray,
    alive: np.ndarray,
    width: int,
    height: int,
) -> tuple:
    """Mirror T1/T9 unique-cell TTL construction without Python segment loops."""
    owners = np.full((height, width), -1, np.int32)
    offsets = np.full((height, width), -1, np.int32)
    ttl = np.zeros((height, width), np.int32)
    cap = bodies.shape[1]
    for slot in range(len(alive)):
        if not alive[slot]:
            continue
        for offset in range(counts[slot]):
            x, y = bodies[slot, (pointers[slot] - offset) % cap]
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            release = lengths[slot] - offset
            if owners[y, x] < 0 or offset < offsets[y, x]:
                owners[y, x], offsets[y, x], ttl[y, x] = slot, offset, release
            elif owners[y, x] != slot or offsets[y, x] != offset:
                ttl[y, x] = max(ttl[y, x], release)
    return owners, ttl


def measure(sim: Any) -> dict:
    """Report current-tail arrival by L and 128, plus T9 strict race-128 area.

    TTL waiting is optimistic: this does not certify a feasible Snake path.
    Current-tail arrival within L is an explicit length-scale deadline convention.
    """
    assert sim.alive[0, 0]
    length = int(sim.length[0, 0])
    owners, ttl = occupancy(
        sim.bodies[0],
        sim.head_ptr[0],
        sim.seg_count[0],
        sim.length[0],
        sim.alive[0],
        int(sim.grid_w),
        int(sim.grid_h),
    )
    heads = [
        tuple(map(int, sim.bodies[0, s, sim.head_ptr[0, s]]))
        for s in range(sim.S if hasattr(sim, "S") else len(sim.alive[0]))
    ]
    tail = tuple(
        map(int, sim.bodies[0, 0, (sim.head_ptr[0, 0] - sim.seg_count[0, 0] + 1) % sim.cap])
    )
    release = np.where(owners >= 0, ttl, 0).astype(np.int32)
    blocked = np.zeros_like(owners, dtype=bool)
    hero = _arrival([heads[0]], blocked, release, horizon=max(length, 128))
    opponents = _arrival(
        [h for s, h in enumerate(heads) if s and sim.alive[0, s]], blocked, release, horizon=128
    )
    arrival = int(hero[tail[1], tail[0]])
    race = int(
        np.count_nonzero((hero >= 0) & (hero <= 128) & ((opponents < 0) | (hero < opponents)))
    )
    return {
        "tail_arrival": arrival if arrival >= 0 else None,
        "tail_L": bool(0 <= arrival <= length),
        "tail_128": bool(0 <= arrival <= 128),
        "race_128": race,
    }


def episode(policy: str, arm: str, env: int, series: dict, context: Any) -> dict:
    """Replay saved actions/interventions; reject any endpoint discrepancy."""
    dense = json.loads((DENSE / f"{arm}_{env}.json").read_text())
    assert dense["status"] == "PASS" and dense["trigger_mode"] == "dense"
    assert dense["checkpoint"]["sha256"] == census.sha256_file(census.b8_path(arm))
    trigger = {r["decision"]: r for r in dense["trigger_rows"]}
    assert len(trigger) == len(dense["trigger_rows"])
    replay = build_replay(arm, [env], context)
    assert int(replay.seeds[0]) == dense["world_seed"]
    rows, checks = [], {"baseline_lengths": 0, "dense_triggers": 0}
    mass = alive_frames = changes = 0
    death = death_length = None
    for decision in range(5000):
        selected = {}

        def selector(sim: Any) -> np.ndarray:
            nonlocal changes
            length = int(sim.length[0, 0])
            assert sim.alive[0, 0] and int(sim.frame[0]) == decision + 1
            if policy == "b8":
                action = int(series["action"][env, decision])
            else:
                greedy = _greedy(replay.q(), sim.get_advisory_action_mask()[0, 0])
                row = trigger.get(decision)
                action = greedy if row is None else int(row["chosen"])
                if row is not None:
                    assert greedy == row["greedy"] and length == row["length"]
                    checks["dense_triggers"] += 1
                changes += int(action != greedy)
            if policy == "b8" or changes == 0:
                assert length == int(series["length"][env, decision])
                assert action == int(series["action"][env, decision])
                checks["baseline_lengths"] += 1
            selected["length"] = length
            if length >= 800:
                rows.append({"decision": decision, "length": length, **measure(sim)})
            return replay.actions([action])

        replay.sim.step_with_policy(selector)
        alive = bool(replay.sim.alive[0, 0])
        alive_frames += int(alive)
        mass += int(replay.sim.length[0, 0]) if alive else 0
        if not alive:
            death, death_length = decision, selected["length"]
            break
    survived = bool(replay.sim.alive[0, 0])
    expected = dense if policy == "dense" else dense["baseline"]
    assert survived == bool(expected["episode_survival" if policy == "dense" else "survived"])
    assert mass / 5000 == expected["mass_integral"]
    assert (
        alive_frames / 5000
        == expected["alive_frame_fraction" if policy == "dense" else "survival_fraction"]
    )
    if policy == "dense":
        assert death == dense["death_t"] and death_length == dense["death_length"]
        assert changes == dense["action_changes"]
        assert checks["dense_triggers"] == len(trigger)
    return {
        "policy": policy,
        "arm": arm,
        "env": env,
        "world_seed": dense["world_seed"],
        "survived": survived,
        "death_decision": death,
        "death_length": death_length,
        "mass_integral": mass / 5000,
        "alive_frame_fraction": alive_frames / 5000,
        "checks": checks,
        "trace": rows,
    }


def summarize(episodes: list[dict]) -> dict:
    """Decision fractions, censored certificate loss, and conditional survival.

    k consecutive means every actual decision is length-eligible and certified.
    Each episode enters once at its first qualifying run; survival is at H5000.
    Also retain decision-landmark denominators (not independent observations).
    """
    result = {}
    for policy in ("b8", "dense"):
        cohort = [e for e in episodes if e["policy"] == policy]
        traces = [r for e in cohort for r in e["trace"]]
        n = len(traces)
        counts = {
            "tail_L": sum(r["tail_L"] for r in traces),
            "tail_128": sum(r["tail_128"] for r in traces),
            "race_above_L2": sum(r["race_128"] > r["length"] / 2 for r in traces),
            "race_above_L": sum(r["race_128"] > r["length"] for r in traces),
        }
        deaths, conditionals = [], {}
        for field in ("tail_L", "tail_128"):
            conditionals[field] = {}
            for k in KS:
                eligible = survivors = landmarks = surviving_landmarks = 0
                for e in cohort:
                    streak = 0
                    previous = -2
                    matches = 0
                    for r in e["trace"]:
                        streak = (
                            (streak + 1 if r["decision"] == previous + 1 else 1) if r[field] else 0
                        )
                        matches += int(streak >= k)
                        previous = r["decision"]
                    eligible += int(matches > 0)
                    survivors += int(matches > 0 and e["survived"])
                    landmarks += matches
                    surviving_landmarks += matches * int(e["survived"])
                conditionals[field][str(k)] = {
                    "episodes": eligible,
                    "survivors": survivors,
                    "survival": survivors / eligible if eligible else None,
                    "decision_landmarks": landmarks,
                    "surviving_decision_landmarks": surviving_landmarks,
                }
        for e in cohort:
            if e["survived"]:
                continue
            row = {k: e[k] for k in ("arm", "env", "world_seed", "death_decision", "death_length")}
            row["eligible_decisions"] = len(e["trace"])
            for field in ("tail_L", "tail_128"):
                true = [r["decision"] for r in e["trace"] if r[field]]
                last = max(true) if true else None
                # D-d counts decisions after the last true state through fatal D.
                row[field] = {
                    "last_true_decision": last,
                    "decisions_before_death": (
                        e["death_decision"] - last if last is not None else None
                    ),
                    "status": (
                        "observed"
                        if last is not None
                        else "never_held_in_eligible_states" if e["trace"] else "no_L800_states"
                    ),
                }
            deaths.append(row)
        loss = {}
        for field in ("tail_L", "tail_128"):
            values = [
                d[field]["decisions_before_death"]
                for d in deaths
                if d[field]["decisions_before_death"] is not None
            ]
            loss[field] = {
                "observed": len(values),
                "unobserved": len(deaths) - len(values),
                "quartiles": (
                    np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist() if values else None
                ),
            }
        result[policy] = {
            "episodes": len(cohort),
            "survivors": sum(e["survived"] for e in cohort),
            "eligible_decisions": n,
            "counts": counts,
            "fractions": {k: v / n if n else None for k, v in counts.items()},
            "deaths": deaths,
            "loss_duration": loss,
            "conditional_survival": conditionals,
        }
    return result


def bindings() -> dict:
    """Hash replay inputs and executable dependencies; never mutate sources."""
    paths = list(Path("src").rglob("*.py"))
    for directory in (
        "b8_death_census_20261014",
        "t1_commitment_20261015",
        "t3_rollout_labels_20261015",
        "t9_divergence_20261017",
    ):
        paths += list((Path("research") / directory).glob("*.py"))
    paths += [
        Path(__file__),
        Path("docs/research/specs/t10_certificate_horizon_spec_2026-10-17.md"),
    ]
    for arm in ARMS:
        paths += [INPUTS / f"{arm}.json", INPUTS / f"{arm}_series.npz", census.b8_path(arm)]
        paths += [DENSE / f"{arm}_{e}.json" for e in range(32)]
    return {str(p.resolve()): census.sha256_file(p) for p in sorted(set(paths))}


def run(out: Path, limit: int | None) -> None:
    configure_cpu()
    import torch

    from src.core.device_manager import DeviceManager

    DeviceManager.override_device(torch.device("cpu"))
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": "t10-certificate/v1",
        "device": "cpu",
        "processes": 1,
        "threads": 1,
        "pid": os.getpid(),
        "limit": limit,
        "input_source_sha256": bindings(),
        "authority": "diagnostic only",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    context, episodes = census.load_context(), []
    started = time.perf_counter()
    with (out / "episodes.jsonl").open("x") as stream:
        for arm in ARMS:
            _, series = load_inputs(arm)
            for env in range(32 if limit is None else limit):
                for policy in ("b8", "dense"):
                    row = episode(policy, arm, env, series, context)
                    episodes.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                "completed": len(episodes),
                                "arm": arm,
                                "env": env,
                                "policy": policy,
                                "seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
    assert bindings() == manifest["input_source_sha256"], "source/input changed during execution"
    result = {
        "status": "PASS",
        "full_cohort": len(episodes) == 128,
        "wall_seconds": time.perf_counter() - started,
        "summary": summarize(episodes),
        "episodes_sha256": census.sha256_file(out / "episodes.jsonl"),
        "manifest_sha256": census.sha256_file(out / "manifest.json"),
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "summary"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "certificate")
    parser.add_argument("--limit", type=int, choices=range(1, 33))
    args = parser.parse_args()
    run(args.out, args.limit)


if __name__ == "__main__":
    main()

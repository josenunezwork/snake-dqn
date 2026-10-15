"""Part B B8-lookahead-shadow ceiling experiment (development diagnostic)."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.replay import build_replay, configure_cpu, load_inputs
from research.t3_rollout_labels_20261015.engine import batched_rollout

ARMS = ("b8_s0", "b8_s3")
HORIZON = 5000
_CACHE: dict[str, tuple[Any, Any, Any, dict[int, dict]]] = {}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_identity() -> dict[str, str]:
    """Hash every implementation/baseline input that governs a resume receipt."""
    repo = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__),
        repo / "research/t3_rollout_labels_20261015/engine.py",
        repo / "research/t1_commitment_20261015/replay.py",
        repo / "research/t1_commitment_20261015/study.py",
        repo / "research/b8_death_census_20261014/census.py",
        repo / "src/simd_env/batch_sim.py",
        repo / "src/simd_env/grid_sim.py",
        repo / "src/simd_env/ego2s_policy.py",
        repo / "src/simd_env/eval_engine.py",
        repo / "src/simd_env/fast_anchors.py",
    )
    identity = {str(path): _sha(path) for path in paths}
    for arm in ARMS:
        path = census.OUT_ROOT / "runs" / f"{arm}.json"
        identity[str(path)] = _sha(path)
    return identity


def _baseline(arm: str) -> dict[int, dict]:
    path = census.OUT_ROOT / "runs" / f"{arm}.json"
    data = json.loads(path.read_text())
    return {int(row["world_seed"]): row for row in data["episodes"]}


def _inputs(arm: str) -> tuple[Any, Any, Any, dict[int, dict]]:
    if arm not in _CACHE:
        _data, series = load_inputs(arm)
        _CACHE[arm] = (series, census.load_context(), census.b8_path(arm), _baseline(arm))
    return _CACHE[arm]


def _greedy(q: np.ndarray, advisory: np.ndarray) -> int:
    allowed = np.asarray(advisory, dtype=bool).copy()
    if not allowed.any():
        allowed[:3] = True
    return int(np.where(allowed, q.astype(np.float64), -np.inf).argmax())


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def run_episode(
    arm: str, env: int, threshold: float, source_sha: str, identity: dict[str, str]
) -> dict:
    """Run one exact census world, applying at most one shadow batch per five decisions."""
    configure_cpu()
    series, context, checkpoint, baselines = _inputs(arm)
    replay = build_replay(arm, [env], context)
    seed = int(replay.seeds[0])
    started = time.perf_counter()
    mass = alive_frames = triggers = eligible_decisions = batches = changes = core_frames = 0
    core_wall_seconds = 0.0
    raw_triggers = decisions = 0
    trigger_rows = []
    last_batch = -5
    prefix_ok = True
    death_t = death_length = None
    for decision in range(HORIZON):
        selected: dict[str, Any] = {}

        def choose(current: Any) -> None:
            nonlocal triggers, raw_triggers, eligible_decisions, batches, changes
            nonlocal last_batch
            nonlocal core_frames, core_wall_seconds
            sim = current.sim
            q = current.q()
            advisory = sim.get_advisory_action_mask()[0, 0]
            chosen = _greedy(q, advisory)
            greedy_before = chosen
            length = int(sim.length[0, 0]) if sim.alive[0, 0] else 0
            baseline_action = int(series["action"][env, decision])
            baseline_length = int(series["length"][env, decision])
            if not changes:
                assert chosen == baseline_action and length == baseline_length, (env, decision)
            in_scope = bool(sim.alive[0, 0] and length >= 600)
            eligible_decisions += int(in_scope)
            eligible = bool(in_scope and -float(q.max()) > threshold)
            if eligible:
                triggers += 1
                raw_triggers += 1
            if eligible and decision - last_batch >= 5 and bool(advisory.any()):
                actions = [int(a) for a in np.flatnonzero(advisory)]
                outcomes = batched_rollout(
                    current.clone(prepared=True), actions, decision, depth=300
                )
                receipt = outcomes.pop("_batch_receipt")
                core_frames += int(receipt["env_frames"])
                core_wall_seconds += float(receipt["wall_seconds"])
                best_frames = max(outcomes[action]["frames_survived"] for action in actions)
                tied = [a for a in actions if outcomes[a]["frames_survived"] == best_frames]
                chosen = chosen if chosen in tied else max(tied, key=lambda a: (float(q[a]), -a))
                trigger_rows.append(
                    {
                        "decision": decision,
                        "length": length,
                        "greedy": _greedy(q, advisory),
                        "chosen": chosen,
                        "risk": -float(q.max()),
                        "actions": actions,
                        "outcomes": outcomes,
                    }
                )
                batches += 1
                last_batch = decision
                changes += int(chosen != greedy_before)
            selected["action"] = chosen
            selected["length"] = length

        def selector(_sim: Any) -> np.ndarray:
            choose(replay)
            return replay.actions([selected["action"]])

        replay.sim.step_with_policy(selector)
        replay.env_frames += 1
        decisions += 1
        post_alive = bool(replay.sim.alive[0, 0])
        alive_frames += int(post_alive)
        mass += int(replay.sim.length[0, 0]) if post_alive else 0
        if not post_alive:
            death_t, death_length = decision, int(selected["length"])
            break
    survived = bool(replay.sim.alive[0, 0])
    result = {
        "schema": "t3-ceiling/v1",
        "status": "PASS",
        "arm": arm,
        "env": env,
        "world_seed": seed,
        "threshold": float(threshold),
        "threshold_source_sha256": source_sha,
        "source_identity": identity,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
        "episode_survival": survived,
        "alive_frame_fraction": alive_frames / HORIZON,
        "mass_integral": mass / HORIZON,
        "death_t": death_t,
        "death_length": death_length,
        "trigger_eligible": triggers,
        "raw_trigger_count": raw_triggers,
        "trigger_scope_decisions": eligible_decisions,
        "trigger_rate": triggers / eligible_decisions if eligible_decisions else 0.0,
        "batch_count": batches,
        "action_changes": changes,
        "decisions": decisions,
        "trigger_rows": trigger_rows,
        "core_rollout_env_frames": core_frames,
        "core_rollout_wall_seconds": core_wall_seconds,
        "wall_seconds": time.perf_counter() - started,
        "prefix_identity": prefix_ok,
        "baseline": baselines[seed],
    }
    if changes == 0:
        assert result["episode_survival"] == bool(baselines[seed]["survived"])
        assert result["alive_frame_fraction"] == float(baselines[seed]["survival_fraction"])
        assert result["mass_integral"] == float(baselines[seed]["mass_integral"])
    result["paired_delta"] = {
        "episode_survival": int(survived) - int(bool(baselines[seed]["survived"])),
        "alive_frame_fraction": result["alive_frame_fraction"]
        - float(baselines[seed]["survival_fraction"]),
        "mass_integral": result["mass_integral"] - float(baselines[seed]["mass_integral"]),
    }
    return result


def _worker(job: tuple[str, int, float, str, dict[str, str]]) -> tuple[str, int, dict]:
    arm, env, threshold, source_sha, identity = job
    return arm, env, run_episode(arm, env, threshold, source_sha, identity)


def summarize(out: Path, worlds: int, total_wall_seconds: float) -> dict:
    """Aggregate completed per-world receipts without treating missing rows as results."""
    rows = []
    for arm in ARMS:
        for env in range(worlds):
            path = out / f"{arm}_{env}.json"
            if not path.exists():
                raise RuntimeError(f"incomplete ceiling run: missing {path.name}")
            rows.append(json.loads(path.read_text()))
    return {
        "schema": "t3-ceiling-summary/v1",
        "status": "PASS",
        "episodes": len(rows),
        "pooled": {
            key: float(np.mean([row[key] for row in rows]))
            for key in ("episode_survival", "alive_frame_fraction", "mass_integral", "trigger_rate")
        },
        "paired_delta_mean": {
            key: float(np.mean([row["paired_delta"][key] for row in rows]))
            for key in ("episode_survival", "alive_frame_fraction", "mass_integral")
        },
        "deaths": [
            {"arm": row["arm"], "env": row["env"], "length": row["death_length"]}
            for row in rows
            if not row["episode_survival"]
        ],
        "trigger_eligible": int(sum(row["trigger_eligible"] for row in rows)),
        "batch_count": int(sum(row["batch_count"] for row in rows)),
        "action_changes": int(sum(row["action_changes"] for row in rows)),
        "core_rollout_env_frames": int(sum(row["core_rollout_env_frames"] for row in rows)),
        "wall_seconds_sum": float(sum(row["wall_seconds"] for row in rows)),
        "wall_seconds": total_wall_seconds,
        "total_decisions": int(sum(row["decisions"] for row in rows)),
        "raw_trigger_rate": float(sum(row["raw_trigger_count"] for row in rows))
        / max(1, sum(row["trigger_scope_decisions"] for row in rows)),
        "capped_trigger_rate": float(sum(row["batch_count"] for row in rows))
        / max(1, sum(row["trigger_scope_decisions"] for row in rows)),
        "go": bool(worlds == 32 and np.mean([row["episode_survival"] for row in rows]) >= 0.65),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--worlds", type=int, default=32, choices=range(1, 33))
    args = parser.parse_args()
    trigger = json.loads(args.threshold_path.read_text())
    if trigger.get("status") != "PASS":
        raise RuntimeError("trigger calibration must PASS")
    threshold = float(trigger["threshold"])
    source_sha = _sha(args.threshold_path)
    identity = source_identity()
    started = time.perf_counter()
    out = args.out / "ceiling"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in ARMS:
        checkpoint_sha = _sha(census.b8_path(arm))
        for env in range(args.worlds):
            path = out / f"{arm}_{env}.json"
            if path.exists():
                prior = json.loads(path.read_text())
                if (
                    prior.get("threshold_source_sha256") != source_sha
                    or prior.get("checkpoint", {}).get("sha256") != checkpoint_sha
                    or prior.get("source_identity") != identity
                ):
                    raise RuntimeError(f"resume receipt mismatch: {path}")
                continue
            jobs.append((arm, env, threshold, source_sha, identity))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            arm, env, result = future.result()
            _atomic_json(out / f"{arm}_{env}.json", result)
            print(json.dumps({"arm": arm, "env": env, "status": "PASS"}), flush=True)
    _atomic_json(out / "summary.json", summarize(out, args.worlds, time.perf_counter() - started))


if __name__ == "__main__":
    main()

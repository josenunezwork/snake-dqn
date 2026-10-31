"""R6 Watch6-v2 L<600 safe-harvest counterfactual & OOF separability census runner.

Runs 64 fresh Watch6-v2 sessions (6,000 frames, 5x frp3-s12+v8, mechanics_version=2,
continuous respawn) on Mac CPU (<=2 threads total, nice 15), captures 6 deduplicated
(>=64 frames apart) L<600 roots/world (384 roots), evaluates all <=6 advisory-legal
1-step H=500 continuations with K-1 native v8 and ProvenanceObserver mass-balance
verification, and scores 5-fold world-disjoint OOF probes against R6's kill criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.bl_k1.native import RowStats, branch_scope, configure
from research.bl_served_search.ruler_operator import _clone_root, _root_bytes
from research.r6_l600_census import B8_SHA256, FRP3_SHA256, NAMESPACE, R6_KILL_THRESHOLDS
from research.r6_l600_census.features import extract_root_features
from research.r6_l600_census.probes import run_world_disjoint_oof_probes
from research.r6_l600_census.seeds import ROOT, audit_seeds, derive_world_seeds
from research.t34_o1_harvest.provenance import DEATH_DROP, SOURCES, ProvenanceObserver
from research.watch_replay.product import open_watch
from src.simd_env.eval_engine import _dispatch_actions
from src.simd_env.vector61_policy import Vector61SimdPolicy
from tools.ops.package_spec import digest, publish_json


def _priority(kind: str, seed: int, decision: int) -> str:
    return hashlib.sha256(f"{NAMESPACE}:{kind}:{seed}:{decision}".encode("utf-8")).hexdigest()


def build_watch6_v2_world(seed: int) -> tuple[Any, str, str]:
    """Construct exact Watch6-v2 world with 1 shared v8-vetoed FRP3 policy across slots 1..5."""
    session = open_watch({"seed": int(seed)}, 24, "bare")
    world = session.world
    b8_path = str(session.live.served_checkpoint["path"])
    frp3_path = str(session.live.watch_opponent_checkpoint["path"])
    session.close()

    if digest(Path(b8_path)) != B8_SHA256 or digest(Path(frp3_path)) != FRP3_SHA256:
        raise ValueError("Watch6-v2 checkpoint SHA mismatch")
    if world.sim.cfg.mechanics_version != 2 or not bool(world.sim.allow_respawn):
        raise ValueError("Watch6-v2 census requires mechanics_version=2 and allow_respawn=True")

    shared_opp = Vector61SimdPolicy(
        frp3_path,
        world.runtime,
        veto_slots=(1, 2, 3, 4, 5),
        forward="rowwise",
        veto_variant="v8",
        veto_lambda=8.0,
    )
    world.opponents = [shared_opp] * 5
    return world, b8_path, frp3_path


def _snapshot_root(world: Any, observer: ProvenanceObserver) -> tuple[Any, ProvenanceObserver]:
    with observer.detached():
        cloned_world = _clone_root(world)
    cloned_obs = observer.clone_for(cloned_world.sim)
    cloned_obs.begin()
    return cloned_world, cloned_obs


def evaluate_l600_root(
    root_world: Any,
    root_observer: ProvenanceObserver,
    *,
    horizon: int = 500,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract t=0 features and run all advisory-legal 1-step H-frame B8 continuations."""
    if not bool(root_world.sim.alive[0, 0]):
        raise ValueError("captured root has dead hero")
    hero_length = int(root_world.sim.length[0, 0])
    if hero_length >= 600:
        raise ValueError(f"captured root length {hero_length} violates L<600 filter")

    root_observer.assert_accounting()
    feat = extract_root_features(root_world, root_observer)
    q_b8 = feat["q_b8"]

    advisory = np.asarray(root_world.sim.get_advisory_action_mask()[0, 0], dtype=bool).copy()
    if not advisory.any():
        advisory[:3] = np.asarray(root_world.sim.get_resolved_action_mask()[0, 0, :3], dtype=bool)
    legal_actions = [int(a) for a in np.flatnonzero(advisory)]
    if not legal_actions:
        raise ValueError("empty legal mask at L<600 root")
    bare = min(legal_actions, key=lambda a: (-float(q_b8[a]), a))

    with root_observer.detached():
        before_bytes = _root_bytes(root_world)

    branches: dict[str, Any] = {}
    for act in legal_actions:
        br_world, br_obs = _snapshot_root(root_world, root_observer)
        mass_sum = 0
        alive_steps = 0
        row_steps = 0
        t0 = time.perf_counter()
        for tick in range(horizon):
            if tick > 0:
                br_world.runtime.prepare(br_world.sim)
            acts = np.ones((1, br_world.sim.S), dtype=np.int64)
            _dispatch_actions(
                br_world.sim,
                br_world.sim.get_resolved_action_mask(),
                acts,
                [br_world.hero],
                [br_world.opponents],
            )
            if tick == 0:
                if int(acts[0, 0]) != bare:
                    raise ValueError(f"bare dispatch {acts[0, 0]} != masked greedy Q {bare}")
                acts[0, 0] = act
            br_world.sim.step(acts)
            br_world.runtime.observe_step(br_world.sim)
            row_steps += 1
            if bool(br_world.sim.alive[0, 0]):
                mass_sum += int(br_world.sim.length[0, 0])
                alive_steps += 1
            else:
                break

        survives = alive_steps == horizon
        if survives:
            br_obs.assert_accounting()
        else:
            br_obs.assert_food_accounting()
        led = br_obs.ledger()
        env0 = led["per_env"][0]
        if survives:
            expected_mass = (
                horizon * int(env0["initial_stored_length"])
                + sum(int(env0["weighted_cumulative_intake"][s]) for s in SOURCES)
                - int(env0["weighted_cumulative_boost_cost"])
            )
            if mass_sum != expected_mass:
                raise AssertionError(
                    f"O1 integer mass decomposition failed: {mass_sum} != {expected_mass}"
                )
        branches[str(act)] = {
            "action": act,
            "mass_sum": int(mass_sum),
            "alive_steps": int(alive_steps),
            "row_steps": int(row_steps),
            "survives": bool(survives),
            "initial_stored_length": int(env0["initial_stored_length"]),
            "final_stored_length": int(env0["final_stored_length"]),
            "cumulative_intake": dict(env0["cumulative_intake"]),
            "weighted_cumulative_intake": dict(env0["weighted_cumulative_intake"]),
            "boost_cost": int(env0["boost_cost"]),
            "weighted_cumulative_boost_cost": int(env0["weighted_cumulative_boost_cost"]),
            "elapsed_seconds": round(time.perf_counter() - t0, 4),
        }
        br_obs.uninstall()

    with root_observer.detached():
        after_bytes = _root_bytes(root_world)
    if after_bytes != before_bytes:
        raise AssertionError("branching mutated parent root state/carry/RNG")

    bare_br = branches[str(bare)]
    bare_mass = int(bare_br["mass_sum"])
    bare_death_drop = int(bare_br["weighted_cumulative_intake"][DEATH_DROP])
    bare_survives = bool(bare_br["survives"])

    for a_str, br in branches.items():
        act = int(a_str)
        d_mass = int(br["mass_sum"]) - bare_mass
        d_death = int(br["weighted_cumulative_intake"][DEATH_DROP]) - bare_death_drop
        d_a500 = float(d_mass) / float(horizon)
        dd_a500 = float(d_death) / float(horizon)
        qual = bool(
            act != bare
            and bool(br["survives"])
            and bare_survives
            and dd_a500 >= d_a500 >= R6_KILL_THRESHOLDS["min_qualifying_delta_A500"]
        )
        br["delta_mass_sum"] = d_mass
        br["delta_A500"] = d_a500
        br["death_delta_mass_sum"] = d_death
        br["death_delta_A500"] = dd_a500
        br["qualifies"] = qual
        br["lethal_flip"] = bool(act != bare and bare_survives and not bool(br["survives"]))

    summary = {
        "hero_length": hero_length,
        "bare_action": bare,
        "advisory_legal": advisory.tolist(),
        "horizon": horizon,
        "source_preserved": True,
        "q_b8": [float(x) for x in q_b8.tolist()],
        "corpse_fifo_scalars": [float(x) for x in feat["corpse_fifo_scalars"].tolist()],
        "corpse_fifo_aux": [float(x) for x in feat["corpse_fifo_aux"].tolist()],
        "qualifies": any(bool(b["qualifies"]) for b in branches.values()),
        "branches": branches,
    }
    feature_record = {
        **feat,
        "bare_action": bare,
        "advisory_legal": advisory,
        "branches": branches,
    }
    return summary, feature_record


def collect_watch6_v2_world(
    *,
    world_idx: int,
    world_seed: int,
    frames: int = 6000,
    roots_per_world: int = 6,
    max_event_roots: int = 4,
    min_root_gap: int = 64,
    horizon: int = 500,
) -> dict[str, Any]:
    """Run one Watch6-v2 world and evaluate ``roots_per_world`` deduplicated L<600 roots."""
    torch.set_num_threads(1)
    t_start = time.perf_counter()
    world, _, _ = build_watch6_v2_world(world_seed)
    observer = ProvenanceObserver(world.sim, hero=0)
    stats = RowStats()

    event_reservoir: dict[str, tuple[int, tuple[Any, ProvenanceObserver]]] = {}
    time_candidates: dict[str, tuple[int, tuple[Any, ProvenanceObserver]]] = {}
    time_cap = max(24, 4 * roots_per_world)
    last_time_snap_decision = -min_root_gap
    last_eligible_event = -min_root_gap
    pending_event_decision = -1
    was_alive = bool(world.sim.alive[0, 0])
    hero_deaths = 0
    l600_alive_frames = 0
    l600_plus_alive_frames = 0

    with configure("kernel", stats), branch_scope("mixed"):
        t_base0 = time.perf_counter()
        for decision in range(frames):
            world.runtime.prepare(world.sim)
            now_alive = bool(world.sim.alive[0, 0])
            if now_alive != was_alive:
                observer.begin()
                was_alive = now_alive

            hero_len = int(world.sim.length[0, 0]) if now_alive else 0
            if now_alive:
                if hero_len < 600:
                    l600_alive_frames += 1
                else:
                    l600_plus_alive_frames += 1

            if now_alive and hero_len < 600:
                if pending_event_decision == decision and max_event_roots > 0:
                    ep = _priority("event", world_seed, decision)
                    if len(event_reservoir) < max_event_roots or ep < max(event_reservoir):
                        event_reservoir[ep] = (decision, _snapshot_root(world, observer))
                        if len(event_reservoir) > max_event_roots:
                            worst = max(event_reservoir)
                            event_reservoir.pop(worst)[1][1].uninstall()
                elif decision - last_time_snap_decision >= min_root_gap:
                    tp = _priority("time", world_seed, decision)
                    if len(time_candidates) < time_cap or tp < max(time_candidates):
                        time_candidates[tp] = (decision, _snapshot_root(world, observer))
                        last_time_snap_decision = decision
                        if len(time_candidates) > time_cap:
                            worst = max(time_candidates)
                            time_candidates.pop(worst)[1][1].uninstall()

            prior_alive = world.sim.alive.copy()
            acts = np.ones((1, world.sim.S), dtype=np.int64)
            _dispatch_actions(
                world.sim,
                world.sim.get_resolved_action_mask(),
                acts,
                [world.hero],
                [world.opponents],
            )
            world.sim.step(acts)
            world.runtime.observe_step(world.sim)
            post_alive = bool(world.sim.alive[0, 0])
            if was_alive and not post_alive:
                hero_deaths += 1
            if post_alive != was_alive:
                observer.begin()
                was_alive = post_alive

            died_slots = prior_alive[0] & ~world.sim.alive[0]
            if post_alive and np.any(died_slots[1:]):
                heads = world.sim.heads()[0]
                near_kill = any(
                    int(np.abs(heads[int(s)] - heads[0]).sum()) <= 30
                    for s in np.flatnonzero(died_slots)
                    if int(s) > 0
                )
                if near_kill and (decision + 1 - last_eligible_event >= min_root_gap):
                    last_eligible_event = decision + 1
                    pending_event_decision = decision + 1

        observer.assert_accounting()
        base_seconds = time.perf_counter() - t_base0

        selected: list[tuple[int, bool, tuple[Any, ProvenanceObserver]]] = []
        chosen_decisions: list[int] = []

        for _, (dec, snap_pair) in sorted(event_reservoir.items(), key=lambda kv: kv[0]):
            if len(selected) < min(max_event_roots, roots_per_world) and all(
                abs(dec - prev) >= min_root_gap for prev in chosen_decisions
            ):
                selected.append((dec, True, snap_pair))
                chosen_decisions.append(dec)
            else:
                snap_pair[1].uninstall()

        for _, (dec, snap_pair) in sorted(time_candidates.items(), key=lambda kv: kv[0]):
            if len(selected) < roots_per_world and all(
                abs(dec - prev) >= min_root_gap for prev in chosen_decisions
            ):
                selected.append((dec, False, snap_pair))
                chosen_decisions.append(dec)
            else:
                snap_pair[1].uninstall()

        if len(selected) != roots_per_world:
            for _, _, sp in selected:
                sp[1].uninstall()
            observer.uninstall()
            raise RuntimeError(
                f"world {world_idx} (seed={world_seed}) yielded {len(selected)} < "
                f"{roots_per_world} deduplicated L<600 roots "
                f"(l600_alive_frames={l600_alive_frames})"
            )

        selected.sort(key=lambda item: item[0])
        t_br0 = time.perf_counter()
        root_summaries: list[dict[str, Any]] = []
        root_features: list[dict[str, Any]] = []
        branch_row_steps = 0
        for dec, is_event, (rw, ro) in selected:
            r_sum, r_feat = evaluate_l600_root(rw, ro, horizon=horizon)
            r_sum.update(
                world_idx=world_idx,
                world_seed=world_seed,
                decision=dec,
                event_root=is_event,
                time_root=not is_event,
            )
            r_feat.update(
                world_idx=world_idx,
                world_seed=world_seed,
                decision=dec,
                event_root=is_event,
            )
            branch_row_steps += sum(int(b["row_steps"]) for b in r_sum["branches"].values())
            root_summaries.append(r_sum)
            root_features.append(r_feat)
            ro.uninstall()
        branch_seconds = time.perf_counter() - t_br0

    observer.uninstall()
    counters = stats.as_dict()
    if (
        counters["kernel_rows"] <= 0
        or counters["stock_rows"] != 0
        or counters["fallback_rows"] != 0
    ):
        raise AssertionError(f"K-1 native v8 coverage failed: {counters}")

    return {
        "world_summary": {
            "schema": "t34-r6-watch6-v2-world/v1",
            "world_idx": world_idx,
            "world_seed": world_seed,
            "frames": frames,
            "horizon": horizon,
            "hero_deaths": hero_deaths,
            "l600_alive_frames": l600_alive_frames,
            "l600_plus_alive_frames": l600_plus_alive_frames,
            "num_roots": len(root_summaries),
            "event_roots": sum(1 for r in root_summaries if r["event_root"]),
            "qualifying_roots": sum(1 for r in root_summaries if r["qualifies"]),
            "branch_row_steps": branch_row_steps,
            "base_seconds": round(base_seconds, 3),
            "branch_seconds": round(branch_seconds, 3),
            "elapsed_seconds": round(time.perf_counter() - t_start, 3),
            "native_v8": counters,
            "roots": root_summaries,
        },
        "root_features": root_features,
    }


def _worker_entry(task: dict[str, Any]) -> dict[str, Any]:
    ckpt_dir = task.pop("ckpt_dir")
    os.environ["SNAKE_CKPT_DIR"] = ckpt_dir
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    out_world_json = task.pop("out_world_json", None)
    res = collect_watch6_v2_world(**task)
    if out_world_json is not None:
        publish_json(Path(out_world_json), res["world_summary"])
    return res


def run_census(
    *,
    out_dir: Path,
    num_worlds: int = 64,
    frames: int = 6000,
    roots_per_world: int = 6,
    max_event_roots: int = 4,
    min_root_gap: int = 64,
    horizon: int = 500,
    workers: int = 2,
    cv_folds: int = 5,
    ckpt_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the full R6 Watch6-v2 L<600 census and 5-fold world-disjoint OOF evaluation."""
    if not 1 <= workers <= 2:
        raise ValueError("Mac CPU workers must be 1 or 2 (<=2 threads rule)")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir = out_dir / "worlds"
    if worlds_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing census worlds dir: {worlds_dir}")

    thresholds_path = out_dir / "PRE_RUN_THRESHOLDS.json"
    if not thresholds_path.is_file():
        publish_json(
            thresholds_path,
            {
                "schema": "t34-r6-watch6-v2-l600-census-thresholds/v1",
                "frozen_before_run": True,
                "kill_criteria": dict(R6_KILL_THRESHOLDS),
            },
        )
    thresholds_sha256 = digest(thresholds_path)

    resolved_ckpt = (
        Path(ckpt_dir).resolve()
        if ckpt_dir is not None
        else Path(
            os.environ.get("SNAKE_CKPT_DIR", str(ROOT.parent / "snake-dqn" / "saved_snakes"))
        ).resolve()
    )
    seeds = derive_world_seeds(NAMESPACE, count=num_worlds)
    seed_audit = audit_seeds(seeds, repo=ROOT)
    publish_json(out_dir / "SEED_AUDIT.json", {"seeds": seeds, "audit": seed_audit})

    tasks = [
        {
            "world_idx": idx,
            "world_seed": int(seed),
            "frames": frames,
            "roots_per_world": roots_per_world,
            "max_event_roots": max_event_roots,
            "min_root_gap": min_root_gap,
            "horizon": horizon,
            "ckpt_dir": str(resolved_ckpt),
            "out_world_json": str(worlds_dir / f"world-{idx:03d}.json"),
        }
        for idx, seed in enumerate(seeds)
    ]

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    if workers == 1:
        for i, task in enumerate(tasks, 1):
            results.append(_worker_entry(task))
            if i % 8 == 0 or i == len(tasks):
                elapsed = time.perf_counter() - t0
                print(
                    f"[R6 census] completed {i}/{len(tasks)} worlds in {elapsed:.1f}s",
                    flush=True,
                )
    else:
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            for i, res in enumerate(pool.map(_worker_entry, tasks), 1):
                results.append(res)
                if i % 8 == 0 or i == len(tasks):
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[R6 census] completed {i}/{len(tasks)} worlds in {elapsed:.1f}s",
                        flush=True,
                    )

    wall_seconds = time.perf_counter() - t0
    world_summaries = [r["world_summary"] for r in results]
    all_root_features = [rf for r in results for rf in r["root_features"]]

    # Save compact root feature archive (.npz) for reproducibility.
    np.savez_compressed(
        out_dir / "roots_features.npz",
        world_idx=np.array([r["world_idx"] for r in all_root_features], dtype=np.int64),
        world_seed=np.array([r["world_seed"] for r in all_root_features], dtype=np.int64),
        decision=np.array([r["decision"] for r in all_root_features], dtype=np.int64),
        hero_length=np.array([r["hero_length"] for r in all_root_features], dtype=np.int64),
        bare_action=np.array([r["bare_action"] for r in all_root_features], dtype=np.int64),
        phi=np.stack([r["phi"] for r in all_root_features], axis=0),
        q_b8=np.stack([r["q_b8"] for r in all_root_features], axis=0),
        scalars_bv8=np.stack([r["scalars_bv8"] for r in all_root_features], axis=0),
        corpse_trail_planes=np.stack([r["corpse_trail_planes"] for r in all_root_features], axis=0),
        corpse_trail_pooled=np.stack([r["corpse_trail_pooled"] for r in all_root_features], axis=0),
        corpse_fifo_scalars=np.stack([r["corpse_fifo_scalars"] for r in all_root_features], axis=0),
        corpse_fifo_aux=np.stack([r["corpse_fifo_aux"] for r in all_root_features], axis=0),
    )

    oof_report = run_world_disjoint_oof_probes(
        all_root_features,
        n_folds=cv_folds,
        thresholds=R6_KILL_THRESHOLDS,
    )

    total_event_roots = sum(int(w["event_roots"]) for w in world_summaries)
    total_l600_frames = sum(int(w["l600_alive_frames"]) for w in world_summaries)
    total_l600_plus_frames = sum(int(w["l600_plus_alive_frames"]) for w in world_summaries)
    total_alive_frames = total_l600_frames + total_l600_plus_frames
    total_deaths = sum(int(w["hero_deaths"]) for w in world_summaries)
    total_branch_steps = sum(int(w["branch_row_steps"]) for w in world_summaries)

    final_result = {
        "schema": "t34-r6-watch6-v2-l600-census-result/v1",
        "namespace": NAMESPACE,
        "pre_run_thresholds_sha256": thresholds_sha256,
        "frozen_kill_thresholds": dict(R6_KILL_THRESHOLDS),
        "b8_sha256": B8_SHA256,
        "frp3_sha256": FRP3_SHA256,
        "seed_audit": seed_audit,
        "execution": {
            "workers": workers,
            "wall_seconds": round(wall_seconds, 2),
            "wall_minutes": round(wall_seconds / 60.0, 2),
            "base_env_steps": num_worlds * frames,
            "branch_env_steps": total_branch_steps,
            "total_env_steps": num_worlds * frames + total_branch_steps,
        },
        "base_trajectory_stats": {
            "num_worlds": num_worlds,
            "frames_per_world": frames,
            "hero_deaths": total_deaths,
            "deaths_per_1k_frames": round(1000.0 * total_deaths / (num_worlds * frames), 4),
            "l600_alive_fraction": round(total_l600_frames / max(1, total_alive_frames), 4),
            "l600_plus_alive_fraction": round(
                total_l600_plus_frames / max(1, total_alive_frames), 4
            ),
            "total_roots": len(all_root_features),
            "event_roots": total_event_roots,
            "time_roots": len(all_root_features) - total_event_roots,
        },
        "oof_separability": oof_report,
    }
    publish_json(out_dir / "RESULT.json", final_result)
    return final_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-worlds", type=int, default=64)
    parser.add_argument("--frames", type=int, default=6000)
    parser.add_argument("--roots-per-world", type=int, default=6)
    parser.add_argument("--max-event-roots", type=int, default=4)
    parser.add_argument("--min-root-gap", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    args = parser.parse_args()
    res = run_census(
        out_dir=args.out_dir,
        num_worlds=args.num_worlds,
        frames=args.frames,
        roots_per_world=args.roots_per_world,
        max_event_roots=args.max_event_roots,
        min_root_gap=args.min_root_gap,
        horizon=args.horizon,
        workers=args.workers,
        cv_folds=args.cv_folds,
        ckpt_dir=args.ckpt_dir,
    )
    print(
        json.dumps(
            {
                "verdict": res["oof_separability"]["verdict"],
                "wall_minutes": res["execution"]["wall_minutes"],
                "num_roots": res["oof_separability"]["num_roots"],
                "qualifying_roots": res["oof_separability"]["qualifying_roots"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Part D dense label-bank collector (only run after the dense ceiling GO receipt)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from research.b8_death_census_20261014 import census
from research.t1_commitment_20261015.replay import configure_cpu
from research.t3_rollout_labels_20261015.bank_engine import (
    DEPTH,
    HORIZON,
    NAMESPACE,
    CapturedWorld,
    allocate_unseen_seeds,
    classify,
    counterweight_trim,
    cycle_jobs,
    dense_eligible,
    manifest,
    planned_rosters,
    rollout_all,
    sha256,
    summarize_rows,
    validate_b8_controls,
    verify_fresh_namespace,
    write_rows,
    write_world_rows,
)
from research.t3_rollout_labels_20261015.trigger import DenseTrigger


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def _row(
    obs: dict[str, np.ndarray],
    action: int,
    surviving: np.ndarray,
    decisive: bool,
    decision: int,
    frame: int,
) -> dict:
    return {
        "local": np.asarray(obs["local"][0], dtype=np.uint8).copy(),
        "global": np.asarray(obs["global"][0], dtype=np.uint8).copy(),
        "scalars": np.asarray(obs["scalars"][0], dtype=np.float32).copy(),
        "mask": np.asarray(obs["mask"][0], dtype=bool).copy(),
        "action": int(action),
        "surviving": np.asarray(surviving, dtype=bool).copy(),
        "decisive": bool(decisive),
        "decision": int(decision),
        "frame": int(frame),
    }


def collect_world(
    job: dict[str, Any], plan: dict[str, Any], out: Path, run_identity: dict[str, Any]
) -> dict[str, Any]:
    """Collect one resumable world; raw rows are trimmed only after global merge."""
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    configure_cpu()
    seed = int(plan["seed"])
    if str(plan["mix"]) != str(job["mix"]):
        raise ValueError("job mix and materialized roster mix differ")
    key = f"{job['arm']}-{job['mix']}-c{job['cycle']}-i{job['index']}-s{seed}"
    receipt_path = out / "worlds" / f"{key}.json"
    rows_path = out / "worlds" / f"{key}.npz"
    if receipt_path.exists() or rows_path.exists():
        if not (receipt_path.exists() and rows_path.exists()):
            raise RuntimeError(f"incomplete prior world receipt for {key}")
        prior = json.loads(receipt_path.read_text())
        expected = {
            "status": "PASS",
            "key": key,
            "arm": job["arm"],
            "mix": job["mix"],
            "cycle": job["cycle"],
            "seed": int(seed),
            "roster": plan,
            "run_identity": run_identity,
        }
        if any(prior.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"prior world receipt identity mismatch for {key}")
        if prior.get("rows_sha256") != sha256(rows_path):
            raise RuntimeError(f"prior world shard digest mismatch for {key}")
        return prior
    world: CapturedWorld = build_world(job["arm"], plan)
    trigger = DenseTrigger(min_length=700)
    rows, batches, eligible = [], 0, 0
    controls: list[tuple[int, bool, int]] = []
    started = time.perf_counter()
    for decision in range(HORIZON):
        from research.t3_rollout_labels_20261015.bank_engine import choose_and_step

        # ``choose_and_step`` owns the Watch seam.  It prepares Vector61 state
        # before B8/opponent dispatch; clone only after that preparation so a
        # mixed/frozen branch starts from the exact decision-time cache.
        selected: dict[str, Any] = {}
        original_prepare = world.runtime.prepare

        def prepared(sim: Any) -> None:
            original_prepare(sim)
            frame, alive = int(sim.frame[0]), bool(sim.alive[0, 0])
            length = int(sim.length[0, 0]) if alive else 0
            selected["fire"] = dense_eligible(trigger, decision, frame, length, alive) and bool(
                sim.get_advisory_action_mask()[0, 0].any()
            )
            selected["frame"] = frame
            if selected["fire"]:
                selected["branch"] = world.clone()
                from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

                selected["obs"] = build_ego_raster(
                    sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
                )
            return None

        world.runtime.prepare = prepared
        try:
            actions, advisory, _ = choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare
        b8_action = int(actions[0, 0])
        if selected.get("fire") and bool(advisory.any()):
            legal = [int(value) for value in np.flatnonzero(advisory)]
            outcomes = rollout_all(selected["branch"], legal, DEPTH, decision)
            controls.append(
                (
                    decision,
                    bool(outcomes[b8_action]["survives"]),
                    int(outcomes[b8_action]["frames_executed"]),
                )
            )
            decisive, surviving = classify(outcomes, b8_action)
            if bool(surviving.any()):
                rows.append(
                    _row(
                        selected["obs"], b8_action, surviving, decisive, decision, selected["frame"]
                    )
                )
            batches += 1
            eligible += 1
        if not bool(world.sim.alive[0, 0]):
            break
    death_decision = None if bool(world.sim.alive[0, 0]) else decision
    control_checks = validate_b8_controls(controls, death_decision)
    write_world_rows(rows_path, rows)
    receipt = {
        "schema": "t3-label-world/v1",
        "status": "PASS",
        "key": key,
        "arm": job["arm"],
        "mix": job["mix"],
        "cycle": job["cycle"],
        "seed": int(seed),
        "roster": plan,
        "run_identity": run_identity,
        "rows": len(rows),
        "decisive_rows": int(sum(bool(row["decisive"]) for row in rows)),
        "counterweight_candidates": int(sum(not bool(row["decisive"]) for row in rows)),
        "batches": batches,
        "eligible_batches": eligible,
        "decisions": decision + 1,
        "death_decision": death_decision,
        "b8_control_checks": control_checks,
        "wall_seconds": time.perf_counter() - started,
    }
    receipt["rows_sha256"] = sha256(rows_path)
    _atomic_json(receipt_path, receipt)
    return receipt


def merge_world_rows(
    out: Path, receipts: list[dict[str, Any]], run_identity: dict[str, Any], *, packed: bool = False
) -> dict[str, Any]:
    """Globally cap counterweights, write a loader-compatible final shard and manifest."""
    rows = []
    unique = {str(receipt["key"]): receipt for receipt in receipts}
    for key, receipt in sorted(unique.items()):
        if receipt.get("status") != "PASS" or receipt.get("run_identity") != run_identity:
            raise RuntimeError(f"untrusted world receipt during merge: {key}")
        path = out / "worlds" / f"{key}.npz"
        if not path.exists() or receipt.get("rows_sha256") != sha256(path):
            raise RuntimeError(f"world shard digest changed before merge: {key}")
        with np.load(path) as data:
            # World shards are deliberately raw so they can retain decisive flags.
            decisive = (
                data["decisive"]
                if "decisive" in data.files
                else np.zeros(len(data["action"]), bool)
            )
            for index in range(len(data["action"])):
                rows.append(
                    {
                        key: data[key][index].copy()
                        for key in ("local", "global", "scalars", "mask", "surviving", "action")
                    }
                    | {"decisive": bool(decisive[index])}
                    | {
                        "decision": int(data["decision"][index]),
                        "frame": int(data["frame"][index]),
                        "source_key": key,
                    }
                )
    selected = counterweight_trim(rows)
    write_rows(out / "bank-00000.npz", selected, packed=packed)
    with (out / "bank-00000.provenance.jsonl").open("w") as stream:
        for index, row in enumerate(selected):
            stream.write(
                json.dumps(
                    {
                        "row": index,
                        "source_key": row["source_key"],
                        "decision": row["decision"],
                        "frame": row["frame"],
                        "decisive": bool(row["decisive"]),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    stats = summarize_rows(selected)
    stats["raw_rows"] = len(rows)
    stats["rollout_batches"] = sum(int(receipt["batches"]) for receipt in unique.values())
    stats["b8_control_checks"] = sum(
        int(receipt.get("b8_control_checks", 0)) for receipt in unique.values()
    )
    stats["retained_rows_per_batch"] = len(selected) / max(1, stats["rollout_batches"])
    stats["play_decisions"] = sum(int(receipt.get("decisions", 0)) for receipt in unique.values())
    stats["retained_rows_per_decision"] = len(selected) / max(1, stats["play_decisions"])
    stats["decisive_rows_per_batch"] = stats["decisive_rows"] / max(1, stats["rollout_batches"])
    stats["raw_counterweight_candidates"] = int(sum(not bool(row["decisive"]) for row in rows))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cycles", type=int, required=True, help="maximum balanced 40-world cycles per B8 final"
    )
    parser.add_argument("--workers", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--target-decisive", type=int, default=20_000)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--dense-go-receipt", type=Path, required=True)
    args = parser.parse_args()
    go = json.loads(args.dense_go_receipt.read_text())
    if (
        go.get("schema") != "t3-dense-ceiling-result/v1"
        or go.get("status") != "PASS"
        or go.get("verdict") != "GO"
        or go.get("gate", {}).get("episodes") != 64
    ):
        raise SystemExit("refusing Part D: dense 64-episode GO result is required")
    if args.target_decisive < 20_000:
        raise SystemExit("Part D requires target decisive rows >= 20,000")
    jobs = cycle_jobs(args.cycles)
    namespace_path = args.out / "namespace.json"
    if namespace_path.exists():
        prior = json.loads(namespace_path.read_text())
        jobs, seeds, plans = prior.get("jobs"), prior.get("seeds"), prior.get("plans")
        if (
            prior.get("schema") != "t3-label-namespace/v1"
            or prior.get("namespace") != NAMESPACE
            or not isinstance(prior.get("freshness"), dict)
            or jobs != cycle_jobs(args.cycles)
            or not isinstance(seeds, list)
            or not isinstance(plans, list)
            or len(seeds) != len(jobs)
            or len(plans) != len(jobs)
        ):
            raise SystemExit("existing namespace receipt does not match requested balanced cycles")
    else:
        if args.out.exists() and any(args.out.iterdir()):
            raise SystemExit(
                "new Part-D output directory must be empty before namespace allocation"
            )
        repo = Path(__file__).resolve().parents[2]
        artifact_root = census.ARTIFACTS
        seeds, allocation = allocate_unseen_seeds(len(jobs), (repo, artifact_root))
        declarations = (
            repo / "research/t3_rollout_labels_20261015/bank_engine.py",
            repo / "research/t3_rollout_labels_20261015/collect.py",
            repo / "docs/research/specs/t3_rollout_labels_spec_2026-10-15.md",
            repo / "docs/research/specs/t3_ceiling_dense_spec_2026-10-15.md",
        )
        repo_freshness = verify_fresh_namespace(repo, seeds, declaration_exclusions=declarations)
        artifact_declarations = (
            artifact_root / "t3-rollout-labels-20261015/dense/ACTIVE_HANDOFF.md",
            artifact_root / "codex-t3b.err",
        )
        artifact_freshness = verify_fresh_namespace(
            artifact_root, seeds, declaration_exclusions=artifact_declarations
        )
        freshness = {
            "seed_allocation": allocation,
            "repository": repo_freshness,
            "artifacts": artifact_freshness,
        }
        from research.redesign_scope_20261007 import grid_h5000_identity as gid

        plans = planned_rosters(gid._context(1), seeds, [job["mix"] for job in jobs])
        args.out.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            namespace_path,
            {
                "schema": "t3-label-namespace/v1",
                "namespace": NAMESPACE,
                "freshness": freshness,
                "jobs": jobs,
                "seeds": seeds,
                "plans": plans,
            },
        )
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    profile = gid._context(1)["profile"]
    repo = Path(__file__).resolve().parents[2]
    source_paths = (
        repo / "research/t3_rollout_labels_20261015/collect.py",
        repo / "research/t3_rollout_labels_20261015/bank_engine.py",
        repo / "research/t3_rollout_labels_20261015/engine.py",
        repo / "research/t3_rollout_labels_20261015/trigger.py",
        repo / "research/t1_commitment_20261015/replay.py",
        repo / "research/b8_death_census_20261014/census.py",
        repo / "research/apex_safety_20260926/dev_screen.py",
        repo / "research/redesign_scope_20261007/grid_h5000_identity.py",
        repo / "research/redesign_m2_20261008/ni_spec.py",
        repo / "src/evaluation/strict_promotion.py",
        repo / "src/simd_env/eval_engine.py",
        repo / "src/simd_env/batch_sim.py",
        repo / "src/simd_env/grid_sim.py",
        repo / "src/simd_env/grid_sim_nb.py",
        repo / "src/simd_env/fast_anchors.py",
        repo / "src/simd_env/vector61_policy.py",
        repo / "src/simd_env/vector61_featurizer.py",
        repo / "src/simd_env/vector61_nb.py",
        repo / "src/simd_env/ego_raster.py",
        repo / "src/simd_env/ego_raster_bv8.py",
    )
    run_identity = {
        "namespace": NAMESPACE,
        "horizon": HORIZON,
        "depth": DEPTH,
        "dense_go_receipt_sha256": sha256(args.dense_go_receipt),
        "profile": profile.name,
        "profile_digest": profile.digest,
        "packed": bool(args.packed),
        "b8_final_sha256": {arm: sha256(census.b8_path(arm)) for arm in census.B8_ARMS},
        "source_sha256": {path.relative_to(repo).as_posix(): sha256(path) for path in source_paths},
        "roster_checkpoint_sha256": {
            str(spec[1]): sha256(Path(spec[1]))
            for plan in plans
            for spec in plan["specs"]
            if spec[0] == "checkpoint"
        },
    }
    started = time.perf_counter()
    receipts = []
    stats: dict[str, Any] = {"decisive_rows": 0}
    # A cycle has every final and the exact 27:9:4 mix.  Stop only between
    # complete cycles, retaining the training distribution at every handoff.
    for cycle in range(args.cycles):
        cycle_rows = [(job, plan) for job, plan in zip(jobs, plans) if job["cycle"] == cycle]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(collect_world, job, plan, args.out, run_identity)
                for job, plan in cycle_rows
            ]
            for finished, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                try:
                    receipt = future.result()
                except BaseException as exc:
                    for pending in futures:
                        pending.cancel()
                    pool.shutdown(wait=False, cancel_futures=True)
                    print(json.dumps({"status": "FAIL", "error": str(exc)}), flush=True)
                    raise
                receipts.append(receipt)
                print(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "completed": finished,
                            "of": len(futures),
                            "key": receipt["key"],
                            "rows": receipt["rows"],
                        }
                    ),
                    flush=True,
                )
        stats = merge_world_rows(args.out, receipts, run_identity, packed=args.packed)
        print(json.dumps({"cycle": cycle, "bank": stats}), flush=True)
        if stats["decisive_rows"] >= args.target_decisive:
            break
    if stats["decisive_rows"] < args.target_decisive:
        raise SystemExit(
            f"completed balanced cycles but only {stats['decisive_rows']} decisive rows"
        )
    provenance = {
        "namespace": NAMESPACE,
        "horizon": HORIZON,
        "depth": DEPTH,
        "trigger": "dense every fifth L>=700 frame>=2000",
        "workers": args.workers,
        "dense_go_receipt": str(args.dense_go_receipt),
        "run_identity": run_identity,
        "worlds": len(receipts),
        "stats": stats,
        "counterweight_sampling": "uniform without replacement; seed 20261015; cap = decisive rows",
        "wall_seconds": time.perf_counter() - started,
    }
    value = manifest(args.out, provenance)
    (args.out / "README.md").write_text(
        "# T3 dense rollout label bank\n\n```json\n"
        + json.dumps({**provenance, "manifest": value["shards"]}, indent=2, sort_keys=True)
        + "\n```\n"
    )
    print(json.dumps(provenance, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

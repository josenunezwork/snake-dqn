"""Part D dense label-bank collector (only run after the dense ceiling GO receipt)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import subprocess
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
    prepared_q,
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
    length: int,
    q: np.ndarray,
) -> dict:
    q_values = np.asarray(q, dtype=np.float32)
    if q_values.shape != (6,) or not np.isfinite(q_values).all():
        raise ValueError("prepared B8 Q vector must be six finite float32 values")
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
        "length": int(length),
        "q": q_values.copy(),
    }


def _key(job: dict[str, Any], plan: dict[str, Any]) -> str:
    """Return the immutable namespace key for a materialized world plan."""
    return f"{job['arm']}-{job['mix']}-c{job['cycle']}-i{job['index']}-s{int(plan['seed'])}"


def _legacy_entries(path: Path) -> dict[str, dict[str, Any]]:
    """Read the explicit, hash-bound legacy-resume receipt allowlist."""
    if not path.exists():
        return {}
    value = json.loads(path.read_text())
    if value.get("schema") != "t3-legacy-resume/v1":
        raise RuntimeError("legacy resume allowlist has an unknown schema")
    entries = value.get("entries")
    if not isinstance(entries, dict) or not all(
        isinstance(item, dict) for item in entries.values()
    ):
        raise RuntimeError("legacy resume allowlist entries must be a key-to-receipt mapping")
    return entries


def _quarantine_orphan(rows_path: Path, out: Path) -> None:
    """Preserve an unreceipted shard outside the active namespace before rerunning it."""
    stamp = str(time.time_ns())
    target = out / "quarantine" / stamp / rows_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(rows_path, target)


def _trusted_existing_receipt(
    job: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    run_identity: dict[str, Any],
    legacy: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Verify a completed world before parent-side resume skipping.

    A source change may only retain an old world when every byte and its old
    run identity are named in the dedicated allowlist.  An orphaned NPZ has no
    receipt authority, so it is quarantined and recomputed.
    """
    key = _key(job, plan)
    receipt_path = out / "worlds" / f"{key}.json"
    rows_path = out / "worlds" / f"{key}.npz"
    if not receipt_path.exists() and not rows_path.exists():
        return None
    if rows_path.exists() and not receipt_path.exists():
        _quarantine_orphan(rows_path, out)
        return None
    if receipt_path.exists() and not rows_path.exists():
        raise RuntimeError(
            f"completed receipt is missing its shard for {key}; receipt was retained"
        )
    prior = json.loads(receipt_path.read_text())
    expected = {
        "status": "PASS",
        "key": key,
        "arm": job["arm"],
        "mix": job["mix"],
        "cycle": job["cycle"],
        "seed": int(plan["seed"]),
        "roster": plan,
    }
    if any(prior.get(name) != value for name, value in expected.items()):
        raise RuntimeError(f"prior world receipt metadata or identity mismatch for {key}")
    rows_digest = sha256(rows_path)
    if prior.get("rows_sha256") != rows_digest:
        raise RuntimeError(f"prior world shard digest mismatch for {key}")
    if prior.get("run_identity") == run_identity:
        return prior | {"_resume_kind": "current"}
    allowed = legacy.get(key)
    if not allowed:
        raise RuntimeError(f"prior world receipt identity mismatch for {key}")
    if (
        allowed.get("receipt_sha256") != sha256(receipt_path)
        or allowed.get("rows_sha256") != rows_digest
        or allowed.get("run_identity") != prior.get("run_identity")
    ):
        raise RuntimeError(f"legacy resume allowlist mismatch for {key}")
    return prior | {"_resume_kind": "legacy"}


def collect_world(
    job: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    run_identity: dict[str, Any],
    threshold: float = float("-inf"),
) -> dict[str, Any]:
    """Collect one resumable world; raw rows are trimmed only after global merge."""
    from research.t3_rollout_labels_20261015.bank_engine import build_world

    configure_cpu()
    seed = int(plan["seed"])
    if str(plan["mix"]) != str(job["mix"]):
        raise ValueError("job mix and materialized roster mix differ")
    key = _key(job, plan)
    receipt_path = out / "worlds" / f"{key}.json"
    rows_path = out / "worlds" / f"{key}.npz"
    if receipt_path.exists() or rows_path.exists():
        _trusted_existing_receipt(job, plan, out, run_identity, {})
        raise RuntimeError(
            f"parent failed to resolve an existing world before worker launch: {key}"
        )
    world: CapturedWorld = build_world(job["arm"], plan)
    trigger = DenseTrigger(min_length=700)
    rows, batches, eligible = [], 0, 0
    dense_candidates, prefilter_skipped, q_action_mismatches = 0, 0, 0
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
            selected["length"] = length
            if selected["fire"]:
                from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

                selected["obs"] = build_ego_raster(
                    sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
                )
                selected["q"] = np.asarray(prepared_q(world, selected["obs"]), dtype=np.float32)
                selected["risk"] = float(-np.max(selected["q"]))
                selected["pass_prefilter"] = selected["risk"] > threshold
                if selected["pass_prefilter"]:
                    # Clone only after both raster capture and Q scoring.
                    selected["branch"] = world.clone()
            return None

        world.runtime.prepare = prepared
        try:
            actions, advisory, _ = choose_and_step(world, capture_obs=False)
        finally:
            world.runtime.prepare = original_prepare
        b8_action = int(actions[0, 0])
        if selected.get("fire"):
            dense_candidates += 1
            if not selected["pass_prefilter"]:
                prefilter_skipped += 1
        if selected.get("fire") and selected["pass_prefilter"] and bool(advisory.any()):
            legal = [int(value) for value in np.flatnonzero(advisory)]
            if not bool(advisory[b8_action]):
                raise RuntimeError(f"B8 selected an advisory-illegal action at decision {decision}")
            q_advisory = np.where(np.asarray(advisory, dtype=bool), selected["q"], -np.inf)
            if b8_action != int(np.argmax(q_advisory)):
                q_action_mismatches += 1
                raise RuntimeError(f"B8 Q/advisory action mismatch at decision {decision}")
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
                        selected["obs"],
                        b8_action,
                        surviving,
                        decisive,
                        decision,
                        selected["frame"],
                        selected["length"],
                        selected["q"],
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
        "dense_candidate_batches": dense_candidates,
        "prefilter_skipped_batches": prefilter_skipped,
        "prefilter_passed_batches": dense_candidates - prefilter_skipped,
        "q_action_mismatches": q_action_mismatches,
        "decisions": decision + 1,
        "death_decision": death_decision,
        "b8_control_checks": control_checks,
        "wall_seconds": time.perf_counter() - started,
    }
    receipt["rows_sha256"] = sha256(rows_path)
    _atomic_json(receipt_path, receipt)
    return receipt


def merge_world_rows(
    out: Path,
    receipts: list[dict[str, Any]],
    run_identity: dict[str, Any],
    legacy_enrichment: dict[str, str],
    *,
    packed: bool = False,
    counterweight_ratio: float = 1.0,
) -> dict[str, Any]:
    """Globally cap counterweights, write a loader-compatible final shard and manifest."""
    rows = []
    receipt_keys = [str(receipt["key"]) for receipt in receipts]
    if len(receipt_keys) != len(set(receipt_keys)):
        raise RuntimeError("duplicate completed world receipts are not mergeable")
    unique = {str(receipt["key"]): receipt for receipt in receipts}
    row_keys: set[tuple[str, int]] = set()
    for key, receipt in sorted(unique.items()):
        if receipt.get("status") != "PASS" or receipt.get("_resume_kind") not in {
            "current",
            "legacy",
        }:
            raise RuntimeError(f"untrusted world receipt during merge: {key}")
        if receipt["_resume_kind"] == "current" and receipt.get("run_identity") != run_identity:
            raise RuntimeError(f"current world receipt identity mismatch during merge: {key}")
        path = out / "worlds" / f"{key}.npz"
        if not path.exists() or receipt.get("rows_sha256") != sha256(path):
            raise RuntimeError(f"world shard digest changed before merge: {key}")
        with np.load(path) as data:
            required = {"local", "global", "scalars", "mask", "surviving", "action"}
            if not required.issubset(data.files):
                raise RuntimeError(f"world shard has an incomplete row schema: {key}")
            enriched: dict[str, np.ndarray] = {}
            if key in legacy_enrichment:
                enrichment_path = out / "calibration" / f"{key}.npz"
                expected_enrichment = legacy_enrichment.get(key)
                if not expected_enrichment or not enrichment_path.exists():
                    raise RuntimeError(f"legacy world lacks an allowlisted Q enrichment: {key}")
                if sha256(enrichment_path) != expected_enrichment:
                    raise RuntimeError(f"legacy Q enrichment digest mismatch: {key}")
                with np.load(enrichment_path) as sidecar:
                    for field in ("q", "length", "decision", "frame"):
                        if field not in sidecar.files:
                            raise RuntimeError(f"legacy Q enrichment lacks {field}: {key}")
                        enriched[field] = sidecar[field].copy()
                if (
                    enriched["q"].dtype != np.float32
                    or enriched["q"].shape != (len(data["action"]), 6)
                    or enriched["length"].shape != (len(data["action"]),)
                    or not np.isfinite(enriched["q"]).all()
                ):
                    raise RuntimeError(f"legacy Q enrichment schema mismatch: {key}")
                for field in ("decision", "frame"):
                    if field in data.files and not np.array_equal(data[field], enriched[field]):
                        raise RuntimeError(f"legacy Q enrichment row keys mismatch: {key}")
            else:
                if "q" not in data.files or "length" not in data.files:
                    raise RuntimeError(f"current world shard lacks Q/length metadata: {key}")
                enriched = {"q": data["q"], "length": data["length"]}
                if (
                    enriched["q"].dtype != np.float32
                    or enriched["q"].shape != (len(data["action"]), 6)
                    or enriched["length"].shape != (len(data["action"]),)
                    or not np.isfinite(enriched["q"]).all()
                ):
                    raise RuntimeError(f"current world Q metadata schema mismatch: {key}")
            # World shards are deliberately raw so they can retain decisive flags.
            decisive = (
                data["decisive"]
                if "decisive" in data.files
                else np.zeros(len(data["action"]), bool)
            )
            for index in range(len(data["action"])):
                row_key = (key, int(data["decision"][index]))
                if row_key in row_keys:
                    raise RuntimeError(f"duplicate source decision row during merge: {row_key}")
                row_keys.add(row_key)
                rows.append(
                    {
                        key: data[key][index].copy()
                        for key in ("local", "global", "scalars", "mask", "surviving", "action")
                    }
                    | {"decisive": bool(decisive[index])}
                    | {
                        "decision": int(data["decision"][index]),
                        "frame": int(data["frame"][index]),
                        "length": int(enriched["length"][index]),
                        "q": np.asarray(enriched["q"][index], dtype=np.float32).copy(),
                        "source_key": key,
                    }
                )
    selected = counterweight_trim(rows, ratio=counterweight_ratio)
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
    stats["dense_candidate_batches"] = sum(
        int(receipt.get("dense_candidate_batches", receipt.get("eligible_batches", 0)))
        for receipt in unique.values()
    )
    stats["prefilter_skipped_batches"] = sum(
        int(receipt.get("prefilter_skipped_batches", 0)) for receipt in unique.values()
    )
    stats["prefilter_passed_batches"] = sum(
        int(receipt.get("prefilter_passed_batches", receipt.get("batches", 0)))
        for receipt in unique.values()
    )
    stats["q_action_mismatches"] = sum(
        int(receipt.get("q_action_mismatches", 0)) for receipt in unique.values()
    )
    stats["b8_control_checks"] = sum(
        int(receipt.get("b8_control_checks", 0)) for receipt in unique.values()
    )
    stats["retained_rows_per_batch"] = len(selected) / max(1, stats["rollout_batches"])
    stats["play_decisions"] = sum(int(receipt.get("decisions", 0)) for receipt in unique.values())
    stats["retained_rows_per_decision"] = len(selected) / max(1, stats["play_decisions"])
    stats["decisive_rows_per_batch"] = stats["decisive_rows"] / max(1, stats["rollout_batches"])
    stats["raw_counterweight_candidates"] = int(sum(not bool(row["decisive"]) for row in rows))
    return stats


def _prefilter(path: Path, legacy_path: Path) -> tuple[float, dict[str, Any]]:
    """Validate the declared calibration before it changes collection admission."""
    value = json.loads(path.read_text())
    coverage = float(value.get("coverage", 0.0))
    if (
        value.get("schema") != "t3-prefilter/v1"
        or value.get("status") != "PASS"
        or not isinstance(value.get("threshold"), (int, float))
        or not np.isfinite(float(value["threshold"]))
        or not np.isfinite(coverage)
        or not 0.95 <= coverage <= 1.0
    ):
        raise SystemExit("refusing filtered Part D: PASS prefilter coverage >= 0.95 is required")
    if value.get("legacy_resume_sha256") != sha256(legacy_path):
        raise SystemExit("prefilter is not bound to the supplied legacy-resume allowlist")
    return float(value["threshold"]), value


def _write_progress(
    out: Path, receipts: list[dict[str, Any]], stats: dict[str, Any], cycles_done: list[int]
) -> None:
    """Persist a restart-safe, small resume receipt after every completed world."""
    current = [item for item in receipts if item.get("_resume_kind") == "current"]
    legacy = [item for item in receipts if item.get("_resume_kind") == "legacy"]
    _atomic_json(
        out / "progress.json",
        {
            "schema": "t3-label-progress/v1",
            "keys_done": sorted(str(item["key"]) for item in receipts),
            "cycles_done": sorted(cycles_done),
            "completed_worlds": len(receipts),
            "retained_legacy_worlds": len(legacy),
            "new_worlds": len(current),
            "decisive_rows": int(sum(int(item.get("decisive_rows", 0)) for item in receipts)),
            "counterweight_candidates": sum(
                int(item.get("counterweight_candidates", 0)) for item in receipts
            ),
        },
    )
    (out / "README.md").write_text(
        "# T3 dense rollout label bank\n\n"
        "Collection is resumable.  The machine-readable completed key list and "
        "balanced cycles are in `progress.json`.\n\n```json\n"
        + json.dumps(
            {
                "cycles_done": sorted(cycles_done),
                "keys_done": sorted(str(item["key"]) for item in receipts),
                "completed_worlds": len(receipts),
                "retained_legacy_worlds": len(legacy),
                "new_worlds": len(current),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n```\n"
    )


def _require_ac_for_twelve_workers(workers: int) -> None:
    """Require observed AC power before the addendum's 12-worker collection."""
    if workers != 12:
        return
    result = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False)
    if result.returncode != 0 or "AC Power" not in result.stdout:
        raise SystemExit("refusing 12-worker collection unless pmset reports AC Power")


def _validate_death_window_receipt(path: Path, out: Path, repo: Path) -> dict[str, Any]:
    """Bind the five-key replay proof to the code and preserved bank being launched."""
    value = json.loads(path.read_text())
    keys = value.get("keys", [])
    if (
        value.get("schema") != "t3-death-window-validation/v1"
        or value.get("status") != "PASS"
        or len(keys) != 5
        or len(set(keys)) != 5
    ):
        raise SystemExit("death-window validation requires five unique passing keys")
    sources = value.get("source_sha256", {})
    required = {
        f"research/t3_rollout_labels_20261015/{name}.py"
        for name in (
            "death_window",
            "bank_engine",
            "collect",
            "engine",
            "trigger",
            "validate_death_window",
        )
    }
    if not required <= set(sources):
        raise SystemExit("death-window validation does not bind required sources")
    for name, digest in sources.items():
        if sha256(repo / name) != digest:
            raise SystemExit(f"death-window validation source changed: {name}")
    for name in ("namespace", "prefilter"):
        if value.get(f"{name}_sha256") != sha256(out / f"{name}.json"):
            raise SystemExit(f"death-window validation {name} changed")
    checkpoints = value.get("checkpoint_sha256", {})
    if not set(census.B8_ARMS) <= set(checkpoints):
        raise SystemExit("death-window validation does not bind all B8 checkpoints")
    for name, digest in checkpoints.items():
        checkpoint = census.b8_path(name) if name in census.B8_ARMS else Path(name)
        if sha256(checkpoint) != digest:
            raise SystemExit(f"death-window validation checkpoint changed: {name}")
    bindings = value.get("legacy_bindings", {})
    if set(bindings) != set(keys):
        raise SystemExit("death-window validation key bindings differ")
    selected = value.get("selected", [])
    if {item.get("key") for item in selected} != set(keys) or not all(
        item.get("comparison", {}).get("pass") is True for item in selected
    ):
        raise SystemExit("death-window validation has missing or failing comparisons")
    for key, binding in bindings.items():
        for suffix, field in (("json", "receipt_sha256"), ("npz", "rows_sha256")):
            if sha256(out / "worlds" / f"{key}.{suffix}") != binding.get(field):
                raise SystemExit(f"death-window validation preserved source changed: {key}")
    return value


def _target_met(stats: dict[str, Any], target: int, ratio: float) -> bool:
    """Require both the decisive target and the declared counterweight complement."""
    decisive = int(stats.get("decisive_rows", 0))
    return decisive >= target and int(stats.get("counterweight_rows", 0)) >= int(decisive * ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--cycles", type=int, required=True, help="maximum balanced 40-world cycles per B8 final"
    )
    parser.add_argument("--workers", type=int, default=12, choices=range(1, 13))
    parser.add_argument("--target-decisive", type=int, default=10_000)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--dense-go-receipt", type=Path, required=True)
    parser.add_argument("--prefilter", type=Path, required=True)
    parser.add_argument("--legacy-resume", type=Path)
    parser.add_argument("--death-window", action="store_true")
    parser.add_argument("--retained-resume", type=Path)
    parser.add_argument("--death-window-validation", type=Path)
    args = parser.parse_args()
    go = json.loads(args.dense_go_receipt.read_text())
    if (
        go.get("schema") != "t3-dense-ceiling-result/v1"
        or go.get("status") != "PASS"
        or go.get("verdict") != "GO"
        or go.get("gate", {}).get("episodes") != 64
    ):
        raise SystemExit("refusing Part D: dense 64-episode GO result is required")
    if args.target_decisive < 10_000:
        raise SystemExit("Part D requires target decisive rows >= 10,000")
    _require_ac_for_twelve_workers(args.workers)
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
            repo / "docs/research/specs/t3_bank_speedup_spec_2026-10-15.md",
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

    legacy_path = args.legacy_resume or args.out / "legacy-resume.json"
    legacy = _legacy_entries(legacy_path)
    threshold, prefilter = _prefilter(args.prefilter, legacy_path)
    counterweight_ratio = 0.5 if args.death_window else 1.0
    if args.death_window:
        resume_path = args.out / "death-window-resume.json"
        retained = _legacy_entries(resume_path)
        if not retained or any(retained.get(key) != value for key, value in legacy.items()):
            raise SystemExit("death-window resume must preserve the complete original allowlist")
        if args.retained_resume is not None:
            extended = _legacy_entries(args.retained_resume)
            if any(extended.get(key) != value for key, value in retained.items()):
                raise SystemExit("retained resume must preserve all pre-death-window entries")
            retained = extended
        legacy = retained
        validation_path = args.death_window_validation
        if validation_path is None:
            raise SystemExit("death-window mode requires the five-key validation receipt")
        _validate_death_window_receipt(
            validation_path, args.out, Path(__file__).resolve().parents[2]
        )
    profile = gid._context(1)["profile"]
    repo = Path(__file__).resolve().parents[2]
    source_paths = (
        repo / "research/t3_rollout_labels_20261015/collect.py",
        repo / "research/t3_rollout_labels_20261015/bank_engine.py",
        repo / "research/t3_rollout_labels_20261015/engine.py",
        repo / "research/t3_rollout_labels_20261015/trigger.py",
        repo / "docs/research/specs/t3_bank_speedup_spec_2026-10-15.md",
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
    if args.death_window:
        source_paths += (
            repo / "research/t3_rollout_labels_20261015/death_window.py",
            repo / "research/t3_rollout_labels_20261015/validate_death_window.py",
            repo / "docs/research/specs/t3_death_window_spec_2026-10-15.md",
        )
    run_identity = {
        "namespace": NAMESPACE,
        "horizon": HORIZON,
        "depth": DEPTH,
        "dense_go_receipt_sha256": sha256(args.dense_go_receipt),
        "profile": profile.name,
        "profile_digest": profile.digest,
        "packed": bool(args.packed),
        "prefilter_sha256": sha256(args.prefilter),
        "prefilter_threshold": threshold,
        "b8_final_sha256": {arm: sha256(census.b8_path(arm)) for arm in census.B8_ARMS},
        "source_sha256": {path.relative_to(repo).as_posix(): sha256(path) for path in source_paths},
        "roster_checkpoint_sha256": {
            str(spec[1]): sha256(Path(spec[1]))
            for plan in plans
            for spec in plan["specs"]
            if spec[0] == "checkpoint"
        },
    }
    if args.death_window:
        run_identity.update(
            {
                "collection_mode": "death-window",
                "counterweight_ratio": counterweight_ratio,
                "death_window_resume_sha256": sha256(resume_path),
                "death_window_validation_sha256": sha256(validation_path),
                "window_maxlen": 60,
                "window_depth": DEPTH,
                "window_strict": True,
                "counter_reservoir": 96,
                "death_window_validation_file": validation_path.name,
                "b8_control_every": 20,
            }
        )
        if args.retained_resume is not None:
            if args.retained_resume.resolve().parent != args.out.resolve():
                raise SystemExit("retained resume must reside in the bank directory")
            run_identity.update(
                {
                    "retained_resume_path": args.retained_resume.name,
                    "retained_resume_sha256": sha256(args.retained_resume),
                }
            )
    started = time.perf_counter()
    receipts: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"decisive_rows": 0}
    legacy_enrichment = prefilter.get("enrichment_sha256", {})
    if not isinstance(legacy_enrichment, dict):
        raise SystemExit("prefilter enrichment_sha256 must be a key-to-digest mapping")
    known: dict[str, dict[str, Any]] = {}
    for job, plan in zip(jobs, plans):
        prior = _trusted_existing_receipt(job, plan, args.out, run_identity, legacy)
        if prior is not None:
            known[str(prior["key"])] = prior
    receipts = list(known.values())
    if receipts:
        stats = merge_world_rows(
            args.out,
            receipts,
            run_identity,
            legacy_enrichment,
            packed=args.packed,
            counterweight_ratio=counterweight_ratio,
        )
    _write_progress(args.out, receipts, stats, [])
    worker = collect_world
    if args.death_window:
        from research.t3_rollout_labels_20261015.death_window import collect_world_death_window

        worker = collect_world_death_window
    # A cycle has every final and the exact 27:9:4 mix.  Stop only between
    # complete cycles, retaining the training distribution at every handoff.
    for cycle in range(args.cycles):
        cycle_rows = [(job, plan) for job, plan in zip(jobs, plans) if job["cycle"] == cycle]
        pending = [(job, plan) for job, plan in cycle_rows if _key(job, plan) not in known]
        if not pending:
            _write_progress(args.out, receipts, stats, list(range(cycle + 1)))
            if _target_met(stats, args.target_decisive, counterweight_ratio):
                break
            continue
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(worker, job, plan, args.out, run_identity, threshold)
                for job, plan in pending
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
                            "of": len(pending),
                            "key": receipt["key"],
                            "rows": receipt["rows"],
                        }
                    ),
                    flush=True,
                )
                receipt["_resume_kind"] = "current"
                known[str(receipt["key"])] = receipt
                _write_progress(args.out, receipts, stats, list(range(cycle)))
        stats = merge_world_rows(
            args.out,
            receipts,
            run_identity,
            legacy_enrichment,
            packed=args.packed,
            counterweight_ratio=counterweight_ratio,
        )
        _write_progress(args.out, receipts, stats, list(range(cycle + 1)))
        print(json.dumps({"cycle": cycle, "bank": stats}), flush=True)
        if _target_met(stats, args.target_decisive, counterweight_ratio):
            break
    if not _target_met(stats, args.target_decisive, counterweight_ratio):
        raise SystemExit(
            f"completed balanced cycles but only {stats['decisive_rows']} decisive rows"
        )
    provenance = {
        "namespace": NAMESPACE,
        "horizon": HORIZON,
        "depth": DEPTH,
        "trigger": (
            "dense every fifth L>=700 frame>=2000; unfiltered death window <300; "
            "random prefiltered out-of-window counters"
            if args.death_window
            else "dense every fifth L>=700 frame>=2000 and -max(Q)>threshold"
        ),
        "collection_mode": "death-window" if args.death_window else "filtered",
        "counterweight_ratio": counterweight_ratio,
        "prefilter": {
            "path": str(args.prefilter),
            "threshold": threshold,
            "coverage": prefilter["coverage"],
        },
        "workers": args.workers,
        "dense_go_receipt": str(args.dense_go_receipt),
        "run_identity": run_identity,
        "worlds": len(receipts),
        "retained_legacy_worlds": sum(item.get("_resume_kind") == "legacy" for item in receipts),
        "new_worlds": sum(item.get("_resume_kind") == "current" for item in receipts),
        "keys_done": sorted(str(item["key"]) for item in receipts),
        "cycles_done": [
            cycle
            for cycle in range(args.cycles)
            if all(
                _key(job, plan) in known
                for job, plan in zip(jobs, plans)
                if int(job["cycle"]) == cycle
            )
        ],
        "stats": stats,
        "counterweight_sampling": (
            "uniform without replacement; seed 20261015; "
            f"cap = floor({counterweight_ratio} * decisive rows)"
        ),
        "wall_seconds": time.perf_counter() - started,
        "legacy_worker_seconds": sum(
            float(item.get("wall_seconds", 0.0))
            for item in receipts
            if item.get("_resume_kind") == "legacy"
        ),
        "new_worker_seconds": sum(
            float(item.get("wall_seconds", 0.0))
            for item in receipts
            if item.get("_resume_kind") == "current"
        ),
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

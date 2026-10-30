"""Mac-only, one-look pilot aggregation from the verified immutable unit union.

Remote summaries are never admitted as evidence. All endpoint units, the final
actor, the seed inventory and the Mac latency receipt must agree before readout.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.sandbox_a_resident.training_rosters import training_rosters
from research.spatial_student_pilot.collect import select_dataset
from research.spatial_student_pilot.config import PilotConfig, audit_seeds
from research.spatial_student_pilot.model import load_checkpoint
from research.spatial_student_pilot.readout import evaluate_readout
from research.spatial_student_pilot.resume import UnitStore, validate_ledger
from research.spatial_student_pilot.run import digest, publish_json
from research.spatial_student_pilot.search import _freeze_hash, select_mass_actions


def expected_units(cfg: PilotConfig) -> set[str]:
    """Frozen complete endpoint inventory, including every fit trajectory boundary."""
    counts = {
        "collect": math.ceil(cfg.worlds / cfg.world_batch_size),
        "search": math.ceil((cfg.train_late + cfg.train_ordinary) / cfg.search_chunk),
        "assess": math.ceil((cfg.assessment_late + cfg.assessment_ordinary) / cfg.search_chunk),
        "gameplay": math.ceil(cfg.gameplay_worlds / cfg.world_batch_size),
    }
    result = {f"{stage}-{index:04d}" for stage, count in counts.items() for index in range(count)}
    final = cfg.bc_updates + cfg.improvement_updates
    snapshots = set(range(cfg.checkpoint_every, final + 1, cfg.checkpoint_every))
    snapshots.update((cfg.bc_updates, final))
    result.update(f"fit-{index:07d}" for index in snapshots)
    return result


def _object(value: Mapping[str, Any] | Path | str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    path = Path(value)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024**2:
        raise ValueError("missing, symlink or oversized JSON evidence")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("JSON evidence must be an object")
    return data


def _decision(verdict: str, reason: str, **details: Any) -> dict[str, Any]:
    return {
        "schema": "spatial-student-aggregate/v1",
        "verdict": verdict,
        "reasons": [reason],
        **details,
    }


def _verify_receipts(ledger: Mapping[str, Any]) -> None:
    """Recheck the supervisor files; a mutable ledger boolean cannot admit exports."""
    bound: set[str] = set()
    for binding in ledger["receipts"]:
        path = Path(binding["path"])
        if path.is_symlink() or digest(path) != binding["sha256"]:
            raise ValueError("supervisor receipt digest mismatch")
        receipt = _object(path)
        if (
            receipt.get("stop_verified") is not True
            or receipt.get("cleanup_error")
            or receipt.get("elapsed_seconds") != binding["elapsed_seconds"]
            or receipt.get("generation") != binding["generation"]
        ):
            raise ValueError("receipt stop/allocation binding mismatch")
        manifest_path = path.parent / "manifest.json"
        if digest(manifest_path) != receipt.get("manifest_sha256"):
            raise ValueError("supervisor manifest digest mismatch")
        manifest = _object(manifest_path)
        if receipt.get("commit") != manifest.get("commit"):
            raise ValueError("supervisor source commit mismatch")
        exports = receipt.get("exports")
        if not isinstance(exports, list) or len({e["path"] for e in exports}) != len(exports):
            raise ValueError("malformed supervisor export inventory")
        for entry in ledger["units"]:
            matches = [
                e
                for e in exports
                if e["path"] == entry["path"] or e["path"].endswith("/" + entry["path"])
            ]
            if len(matches) > 1:
                raise ValueError("ambiguous supervisor export binding")
            if matches:
                if (
                    matches[0].get("sha256") != entry["sha256"]
                    or matches[0].get("size") != entry["size"]
                ):
                    raise ValueError("unit differs from supervisor export binding")
                bound.add(entry["unit_id"])
    if bound != {e["unit_id"] for e in ledger["units"]}:
        raise ValueError("unit union lacks supervisor export bindings")


def _verify_seed_audit(cfg: PilotConfig, evidence: Mapping[str, Any]) -> None:
    """Rebuild the allocation and protected inventory without loading outcomes."""
    if (
        evidence.get("config_sha256") != cfg.sha256
        or evidence.get("namespace") != cfg.namespace
        or evidence.get("allocation_index") != cfg.allocation_index
    ):
        raise ValueError("foreign seed audit configuration")
    current = audit_seeds(Path(__file__).resolve().parents[2], cfg)
    # This also checks the protected registry and every retained JSON inventory;
    # collision_count alone is never treated as an admission credential.
    for key in (
        "schema",
        "namespace",
        "allocation_index",
        "config_sha256",
        "effective_keys",
        "protected_count",
        "inventories",
        "collision_count",
        "registry_sha256",
    ):
        if evidence.get(key) != current.get(key):
            raise ValueError(f"seed audit differs from independently rebuilt {key}")


def _same_tree(left: Any, right: Any) -> bool:
    if isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left, right)
        )
    if isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and np.array_equal(left, right)
    if isinstance(right, dict):
        return (
            isinstance(left, dict)
            and set(left) == set(right)
            and all(_same_tree(left[key], value) for key, value in right.items())
        )
    if isinstance(right, (tuple, list)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_same_tree(a, b) for a, b in zip(left, right))
        )
    return left == right


def _exact_rows(
    rows: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]], *, assessment: bool
) -> None:
    expected = {r["uid"]: r for r in selected}
    if len(expected) != len(selected) or len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate selected or exported root UID")
    if {r["uid"] for r in rows} != set(expected):
        raise ValueError("exported root UID set differs from frozen selected dataset")
    for row in rows:
        original = expected[row["uid"]]
        for key in ("world_id", "partition", "kind", "mix", "b8_action"):
            if row[key] != original[key]:
                raise ValueError("exported root metadata differs from selected root")
        if assessment and list(row["scoring_suffixes"]) != [4, 5]:
            raise ValueError("foreign independent scoring suffixes")
        if not assessment and any(
            not _same_tree(row.get(key), value) for key, value in original.items()
        ):
            raise ValueError("search input payload differs from selected training root")


def aggregate(
    cfg: PilotConfig,
    ledger: Mapping[str, Any] | Path | str,
    input_dir: Path | str,
    candidate: Path | str,
    latency: Mapping[str, Any] | Path | str | None,
    seed_audit: Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    """Admit only a complete, hash-verified endpoint, then perform the sole look."""
    if platform.system() != "Darwin" or cfg.mode != "pilot" or cfg.tiny:
        return _decision(
            "INVALID", "scientific aggregation requires the Mac and frozen pilot config"
        )
    try:
        verified = _object(ledger)
        validate_ledger(verified, cfg)
        expected = expected_units(cfg)
        actual = {entry["unit_id"] for entry in verified["units"]}
        if actual - expected:
            return _decision("INVALID", "foreign endpoint unit identities")
        if expected - actual:
            return _decision(
                "INCOMPLETE",
                "missing immutable endpoint units",
                missing_units=sorted(expected - actual),
            )
        if verified["consumed_seconds"] > 6 * 3600:
            return _decision(
                "INCOMPLETE", "cumulative verified allocation exceeded six-hour endpoint"
            )
        _verify_receipts(verified)
        _verify_seed_audit(cfg, _object(seed_audit))
        with tempfile.TemporaryDirectory(prefix="spatial-final-read-") as scratch:
            store = UnitStore(Path(scratch), cfg, verified, Path(input_dir))
            batches = [store.get(name) for name in sorted(actual) if name.startswith("collect-")]
            partitions = cfg.partitions()
            collection_rosters = training_rosters(cfg.worlds)
            for index, batch in enumerate(batches):
                if (
                    batch.get("schema") != "spatial-collection/v1"
                    or batch.get("ruler_partial")
                    or sorted(batch["world_ids"])
                    != list(
                        range(
                            index * cfg.world_batch_size,
                            min(cfg.worlds, (index + 1) * cfg.world_batch_size),
                        )
                    )
                ):
                    raise ValueError("collection unit differs from frozen world allocation")
                for row in batch["roots"]:
                    world = row["world_id"]
                    if (
                        row["partition"] != partitions[world]
                        or row["roster_index"] != world
                        or row["mix"] != collection_rosters[world].mix
                        or not row["uid"].startswith(f"{cfg.namespace}/a{cfg.allocation_index}/")
                    ):
                        raise ValueError("collection root namespace/partition/roster mismatch")
                if any(
                    row["partition"] != "train" or partitions[row["world_id"]] != "train"
                    for row in batch["demos"]
                ):
                    raise ValueError("assessment-world ordinary demonstrations")
            try:
                dataset = select_dataset(cfg, batches)
            except RuntimeError as error:
                return _decision("INCOMPLETE", str(error))
            train = [r for r in dataset["roots"] if r["partition"] == "train"]
            assessment = [r for r in dataset["roots"] if r["partition"] == "assessment"]
            labels = []
            for name in sorted(actual):
                if not name.startswith("search-"):
                    continue
                chunk = store.get(name)
                _exact_rows(
                    chunk,
                    train[
                        int(name.split("-")[1])
                        * cfg.search_chunk : (int(name.split("-")[1]) + 1)
                        * cfg.search_chunk
                    ],
                    assessment=False,
                )
                masks = np.asarray([r["mask"] for r in chunk], dtype=bool)
                masses = np.asarray([r["selection_mass"] for r in chunk], dtype=float)
                if masses.shape != (len(chunk), 6, 2):
                    raise ValueError("malformed two-suffix selection returns")
                chosen = select_mass_actions(masses, masks, [r["b8_action"] for r in chunk])
                if chosen.tolist() != [r["target_action"] for r in chunk]:
                    raise ValueError("search target differs from mass-maximizing frozen label")
                frozen = _freeze_hash(chunk, chosen)
                if any(r["selection_freeze_sha256"] != frozen for r in chunk):
                    raise ValueError("search selection freeze digest mismatch")
                labels.extend(chunk)
            _exact_rows(labels, train, assessment=False)
            rows = [
                r for name in sorted(actual) if name.startswith("assess-") for r in store.get(name)
            ]
            _exact_rows(rows, assessment, assessment=True)
            worlds = [
                r
                for name in sorted(actual)
                if name.startswith("gameplay-")
                for r in store.get(name)
            ]
            if sorted(r["world_id"] for r in worlds) != list(range(cfg.gameplay_worlds)):
                raise ValueError("incomplete or duplicate gameplay allocation")
            rosters = training_rosters(cfg.gameplay_worlds)
            if any(r["mix"] != rosters[r["world_id"]].mix for r in worlds):
                raise ValueError("gameplay roster/mix identity mismatch")
            final_update = cfg.bc_updates + cfg.improvement_updates
            final = store.get(f"fit-{final_update:07d}")
            if (
                final["update"] != final_update
                or final["seed"] != cfg.seed
                or final.get("final") is not True
            ):
                raise ValueError("final fit trajectory identity mismatch")
            actor_path = Path(candidate)
            if actor_path.is_symlink():
                raise ValueError("symlink candidate")
            actor, metadata = load_checkpoint(actor_path)
            for key, expected_value in (
                ("config_sha256", cfg.sha256),
                ("seed", cfg.seed),
                ("updates", final_update),
                ("namespace", cfg.namespace),
            ):
                if metadata.get(key) != expected_value:
                    raise ValueError("candidate metadata differs from final fit")
            tensors = actor.state_dict()
            if set(tensors) != set(final["model"]) or any(
                not torch.equal(value.cpu(), final["model"][key].cpu())
                for key, value in tensors.items()
            ):
                raise ValueError("candidate tensors differ from final fit snapshot")
            actor_hash = digest(actor_path)
            sidecar = Path(str(actor_path) + ".sha256")
            marker = Path(str(actor_path) + ".complete")
            if not marker.is_file() or sidecar.read_text().split()[0] != actor_hash:
                raise ValueError("candidate publication is incomplete or digest differs")
            if latency is None or (
                not isinstance(latency, Mapping) and not Path(latency).is_file()
            ):
                return _decision("INCOMPLETE", "missing checkpoint-bound Mac latency measurement")
            probe = _object(latency)
            late_uids = {r["uid"] for r in assessment if r["kind"] == "late"}
            probe_uids = probe.get("root_uids", [])
            if (
                probe.get("threshold_qualified") is not True
                or probe.get("checkpoint_sha256") != actor_hash
                or probe.get("config_sha256") != cfg.sha256
                or probe.get("namespace") != cfg.namespace
                or probe.get("allocation_index") != cfg.allocation_index
                or probe.get("checkpoint_metadata") != metadata
                or len(probe_uids) != len(late_uids)
                or set(probe_uids) != late_uids
            ):
                raise ValueError(
                    "latency receipt does not bind candidate/config/exact late assessment roots"
                )
            p95 = probe.get("statistics", {}).get("p95_ms")
            if isinstance(p95, bool) or not isinstance(p95, (int, float)) or not math.isfinite(p95):
                raise ValueError("nonfinite or malformed latency measurement")
            decision = evaluate_readout(
                rows,
                worlds,
                support={
                    "collision_audit": True,
                    "resume_verified": True,
                    "namespace": cfg.namespace,
                },
                latency_p95_ms=float(p95),
                complete=True,
            )
            decision["provenance"] = {
                "config_sha256": cfg.sha256,
                "candidate_sha256": actor_hash,
                "unit_ids": sorted(actual),
                "allocated_seconds": verified["consumed_seconds"],
                "seed_audit_rebuilt": True,
                "remote_summaries_used": False,
            }
            return decision
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError) as error:
        return _decision("INVALID", str(error))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--latency", type=Path)
    parser.add_argument("--seed-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = PilotConfig.from_dict(_object(args.config))
    decision = aggregate(
        cfg, args.ledger, args.input_dir, args.candidate, args.latency, args.seed_audit
    )
    publish_json(args.output, decision)
    print(json.dumps({"verdict": decision["verdict"], "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

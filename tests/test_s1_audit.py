"""Independent saved-look audit detects missing coverage and changed bytes."""

import copy
import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import audit


def saved_look(root: Path) -> dict:
    source = root / "source.py"
    source.write_text("# frozen source\n")
    worlds = {}
    number = 0
    for seed in audit.SEEDS:
        worlds[str(seed)] = []
        for mix, count in audit.MIX_COUNTS.items():
            for _ in range(count):
                number += 1
                worlds[str(seed)].append(
                    {
                        "world_seed": number,
                        "mix": mix,
                        "horizon": 5000,
                        "terminal_hero_slot": 0,
                        "world_semantic_sha256": f"{number:064x}",
                    }
                )
    bank = {
        "schema": "s1-mature-world-bank/v1",
        "training_seeds": list(audit.SEEDS),
        "mix_counts": audit.MIX_COUNTS,
        "profile": {"horizon": 5000, "veto": False, "hero_mode": "terminal"},
        "worlds": worlds,
        "allocation_semantic_sha256": audit.canonical_sha(worlds),
    }
    bank_path = root / "bank.json"
    bank_path.write_text(json.dumps(bank))
    bank_sha = audit.canonical_sha(bank)
    collision = {
        "pass": True,
        "world_bank_semantic_sha256": bank["allocation_semantic_sha256"],
        "world_bank_file": {"path": str(bank_path), "sha256": audit.digest(bank_path)},
    }
    shards = []
    for seed in audit.SEEDS:
        for arm in audit.ARMS:
            checkpoint = audit.B8 if arm == "B8" else audit.canonical_sha([seed, arm])
            provenance = {
                "schema": "s1-training-provenance/v1",
                "arm": arm,
                "training_seed": seed,
                "endpoint": 8_000_000,
                "checkpoint_sha256": checkpoint,
                "clean_exit": True,
                "tripwire_triggered": False,
            }
            rows = [
                {
                    **world,
                    "schema": "s1-eval-row/v1",
                    "arm": arm,
                    "training_seed": seed,
                    "checkpoint_step": 8_000_000,
                    "checkpoint_sha256": checkpoint,
                    "b8_checkpoint_sha256": audit.B8,
                    "world_bank_sha256": bank_sha,
                    "frames_stepped": 5000,
                    "clean_exit": True,
                    "mass_integral": 100.0,
                    "alive_frame_fraction": 1.0,
                    "no_death_episode": True,
                }
                for world in worlds[str(seed)]
            ]
            shards.append(
                {
                    "schema": "s1-eval-shard/v1",
                    "status": "completed",
                    "clean_exit": True,
                    "identity": {
                        "training_seed": seed,
                        "arm": arm,
                        "endpoint": 8_000_000,
                        "checkpoint_sha256": checkpoint,
                        "b8_sha256": audit.B8,
                        "bank_sha256": bank_sha,
                        "world_hashes": [w["world_semantic_sha256"] for w in worlds[str(seed)]],
                        "source_manifest": {"source.py": audit.digest(source)},
                    },
                    "training_provenance": None if arm == "B8" else provenance,
                    "rows": rows,
                    "rows_sha256": audit.canonical_sha(rows),
                }
            )
    return {
        "schema": "s1-analysis-input/v1",
        "bank": bank,
        "bank_audit": collision,
        "shards": shards,
    }


def test_saved_look_requires_complete_paired_coverage(tmp_path: Path) -> None:
    value = saved_look(tmp_path)
    assert audit.audit(value, tmp_path)["policy_episodes"] == 1920
    value["shards"].pop()
    with pytest.raises(ValueError, match="incomplete final look"):
        audit.audit(value, tmp_path)


def test_audit_rejects_tampering_even_with_recomputed_row_hash(tmp_path: Path) -> None:
    value = saved_look(tmp_path)
    changed = copy.deepcopy(value)
    shard = changed["shards"][0]
    shard["rows"][0]["world_seed"] += 1
    shard["rows_sha256"] = audit.canonical_sha(shard["rows"])
    with pytest.raises(ValueError, match="row identity"):
        audit.audit(changed, tmp_path)
    (tmp_path / "source.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        audit.audit(value, tmp_path)


def test_claimed_tripwire_without_bound_evidence_cannot_excuse_missing_finals(
    tmp_path: Path,
) -> None:
    value = saved_look(tmp_path)
    value["training_outcomes"] = [
        {
            "schema": "s1-training-outcome/v1",
            "status": "TRIPWIRE_KILL",
            "arm": "N1",
            "training_seed": audit.SEEDS[0],
            "b8_sha256": audit.B8,
            "reason": "non-finite loss",
            "raw_summary_halt": "non-finite loss",
        }
    ]
    value["shards"] = [s for s in value["shards"] if s["identity"]["arm"] != "N1"]
    with pytest.raises(KeyError, match="evidence"):
        audit.audit(value, tmp_path)

#!/usr/bin/env python3
"""Independently audit the complete saved S1 look, using only the standard library.

Run with ``python -I``. This auditor imports no producer and reports no outcome
statistics. It checks the fixed design before the separately implemented analyzer
can inspect the single registered look.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

SEEDS = (223213833, 3055723932, 1783343308, 1125234883, 1809856951)
MIX_COUNTS = {"scripted": 64, "frozen": 32, "mixed": 32}
ARMS = ("N1", "N3", "B8")
B8 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verified_artifact(entry: dict[str, str]) -> Path:
    """Open a bound local artifact; a claimed digest alone is insufficient."""
    path = Path(entry["path"])
    check(not path.is_symlink() and path.is_file(), "missing or linked evidence artifact")
    check(digest(path) == entry["sha256"], "evidence artifact hash mismatch")
    return path


def validated_halts(value: dict[str, Any], repo: Path) -> set[str]:
    """Authenticate the registered safety exception to missing final coverage."""
    outcomes = value.get("training_outcomes", [])
    check(isinstance(outcomes, list), "training outcomes must be a list")
    halted: set[str] = set()
    markers = (
        "non-finite loss",
        "nonfinite loss",
        "non-finite q",
        "nonfinite q",
        "non-finite gradient",
        "nonfinite gradient",
        "action collapse",
        "overall greedy boost rate",
        "greedy boost rate below length 700",
        "illegal action",
    )
    for outcome in outcomes:
        arm, seed = outcome["arm"], outcome["training_seed"]
        check(arm in ("N1", "N3") and arm not in halted and seed in SEEDS, "halt identity")
        check(outcome["schema"] == "s1-training-outcome/v1", "halt schema")
        check(
            outcome["status"] == "TRIPWIRE_KILL" and outcome["b8_sha256"] == B8,
            "halt status/initializer",
        )
        evidence = outcome["evidence"]
        summary_path = verified_artifact(evidence["summary"])
        log_path = verified_artifact(evidence["log"])
        receipt_path = verified_artifact(evidence["supervisor_receipt"])
        check(digest(summary_path) == outcome["summary_sha256"], "halt summary binding")
        check(digest(log_path) == outcome["log_sha256"], "halt log binding")
        check(
            digest(repo / "research/redesign_m3c_20261008/train_m3c.py")
            == outcome["trainer_source_sha256"],
            "halt trainer source",
        )
        summary, receipt = json.loads(summary_path.read_text()), json.loads(
            receipt_path.read_text()
        )
        reason = summary["halt"]
        check(
            isinstance(reason, str) and reason == outcome["raw_summary_halt"], "halt reason binding"
        )
        check(
            any(marker in reason.lower() for marker in markers), "unregistered halt is incomplete"
        )
        check(outcome["reason"] == reason, "halt reason differs from raw summary")
        check(
            receipt["status"] == "PASS"
            and receipt["stop_verified"] is True
            and receipt.get("cleanup_error") is None
            and receipt.get("error") is None
            and receipt["job_id"] == f"s1-mature-{arm.lower()}-{seed}"
            and bool(receipt["endpoint"]),
            "halt lacks verified transport release",
        )
        actors, seats = (3, 1) if arm == "N1" else (6, 3)
        namespace = (
            f"q1-task-aligned-n1-20261017/s1-mature-20261018/n1/seed-{seed}"
            if arm == "N1"
            else f"s1-mature-20261018/train/n3/seed-{seed}"
        )
        args = summary["args"]
        expected_args = {
            "seed": seed,
            "worlds": 128,
            "actors": actors,
            "learner_seats": seats,
            "transitions": 8_000_000,
            "schedule_package": "mature",
            "collection_profile": "task-aligned-n1" if arm == "N1" else "legacy",
            "namespace": namespace,
        }
        check(
            all(str(args.get(k)) == str(v) for k, v in expected_args.items()), "halt recipe changed"
        )
        check(
            summary["actor_exitcodes"] == [0] * actors and summary["ring_overrun_rows"] == 0,
            "halt has unclean actor/ring evidence",
        )
        check(summary["nstep_accounting_reconciled"] is True, "halt counter reconciliation")
        halted.add(arm)
    return halted


def audit(value: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Validate all saved units against the fixed design without selecting an arm."""
    check(value.get("schema") == "s1-analysis-input/v1", "wrong input schema")
    halted = validated_halts(value, repo)
    bank = value["bank"]
    check(bank["schema"] == "s1-mature-world-bank/v1", "wrong bank schema")
    check(tuple(bank["training_seeds"]) == SEEDS, "training seed blocks changed")
    check(bank["mix_counts"] == MIX_COUNTS, "mix allocation changed")
    check(bank["profile"]["horizon"] == 5000, "horizon changed")
    check(bank["profile"]["veto"] is False, "active veto in evaluation")
    check(bank["profile"]["hero_mode"] == "terminal", "nonterminal hero")
    collision = value["bank_audit"]
    check(collision.get("pass") is True, "collision audit did not pass")
    check(not collision.get("registered_collision_seeds"), "registered seed collision")
    check(not any(collision.get("namespace_collision_files", {}).values()), "namespace collision")
    check(
        collision["world_bank_semantic_sha256"] == bank["allocation_semantic_sha256"],
        "collision audit does not bind allocation",
    )
    # The bank file itself and its create-only collision receipt are authoritative
    # byte bindings; parsing the input must reproduce those exact JSON objects.
    bank_file = Path(collision["world_bank_file"]["path"])
    check(digest(bank_file) == collision["world_bank_file"]["sha256"], "bank bytes changed")
    check(json.loads(bank_file.read_text()) == bank, "input bank differs from audited bytes")
    bank_sha = canonical_sha(bank)
    expected: dict[int, list[dict[str, Any]]] = {}
    all_world_seeds: set[int] = set()
    for seed in SEEDS:
        worlds = bank["worlds"][str(seed)]
        check(len(worlds) == 128, "world count differs")
        check(
            {mix: sum(w["mix"] == mix for w in worlds) for mix in MIX_COUNTS} == MIX_COUNTS,
            "per-seed mix counts differ",
        )
        for world in worlds:
            check(world["world_seed"] not in all_world_seeds, "duplicate world seed")
            all_world_seeds.add(world["world_seed"])
            check(world["horizon"] == 5000 and world["terminal_hero_slot"] == 0, "world mode")
        expected[seed] = worlds
    check(len(all_world_seeds) == 640, "unique world coverage differs")
    fixtures = bank.get("timing_worlds", []) + bank.get("parity_worlds", [])
    check(not all_world_seeds.intersection(w["world_seed"] for w in fixtures), "fixture leakage")
    seen: set[tuple[int, str]] = set()
    row_count = 0
    for shard in value["shards"]:
        check(shard.get("schema") == "s1-eval-shard/v1", "wrong shard schema")
        check(
            shard.get("status") == "completed" and shard.get("clean_exit") is True, "unclean shard"
        )
        identity = shard["identity"]
        seed, arm = identity["training_seed"], identity["arm"]
        check(
            seed in SEEDS and arm in ARMS and (seed, arm) not in seen,
            "duplicate/unregistered identity",
        )
        seen.add((seed, arm))
        check(arm not in halted, "halted arm unexpectedly supplied efficacy rows")
        check(identity["endpoint"] == 8_000_000, "wrong checkpoint endpoint")
        check(
            identity["bank_sha256"] == bank_sha and identity["b8_sha256"] == B8,
            "bank/comparator binding",
        )
        check(
            identity["world_hashes"] == [w["world_semantic_sha256"] for w in expected[seed]],
            "world order",
        )
        sources = identity["source_manifest"]
        check(bool(sources), "missing source closure")
        for relative, expected_hash in sources.items():
            source = (repo / relative).resolve()
            check(source.is_relative_to(repo.resolve()), "source path escape")
            check(digest(source) == expected_hash, f"source changed: {relative}")
        if arm == "B8":
            check(identity["checkpoint_sha256"] == B8, "wrong B8 comparator bytes")
            check(shard.get("training_provenance") is None, "B8 has candidate provenance")
        else:
            provenance = shard["training_provenance"]
            required = {
                "schema": "s1-training-provenance/v1",
                "arm": arm,
                "training_seed": seed,
                "endpoint": 8_000_000,
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "clean_exit": True,
                "tripwire_triggered": False,
            }
            check(all(provenance.get(k) == v for k, v in required.items()), "candidate provenance")
        rows = shard["rows"]
        check(len(rows) == 128 and canonical_sha(rows) == shard["rows_sha256"], "row hash/count")
        for row, world in zip(rows, expected[seed]):
            required = {
                "schema": "s1-eval-row/v1",
                "arm": arm,
                "training_seed": seed,
                "checkpoint_step": 8_000_000,
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "b8_checkpoint_sha256": B8,
                "world_bank_sha256": bank_sha,
                "world_semantic_sha256": world["world_semantic_sha256"],
                "world_seed": world["world_seed"],
                "mix": world["mix"],
                "horizon": 5000,
                "frames_stepped": 5000,
                "clean_exit": True,
            }
            check(all(row.get(k) == v for k, v in required.items()), "row identity/order mismatch")
            for metric in ("mass_integral", "alive_frame_fraction"):
                number = row[metric]
                check(
                    type(number) in (int, float) and math.isfinite(number) and number >= 0,
                    "nonfinite/negative metric",
                )
            check(row["alive_frame_fraction"] <= 1, "invalid alive fraction")
            check(type(row["no_death_episode"]) is bool, "invalid no-death indicator")
        row_count += len(rows)
    expected_identities = {(seed, arm) for seed in SEEDS for arm in ARMS if arm not in halted}
    check(seen == expected_identities, "incomplete final look")
    check(row_count == len(expected_identities) * 128, "episode coverage mismatch")
    return {
        "status": "PASS",
        "worlds": 640,
        "policy_episodes": row_count,
        "identities": len(seen),
        "tripwire_killed_arms": sorted(halted),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    check(not args.out.exists(), "audit output already exists")
    result: dict[str, Any] = {
        "schema": "s1-independent-audit/v1",
        "analysis_input_sha256": digest(args.input),
        "auditor_sha256": digest(Path(__file__)),
    }
    try:
        result.update(audit(json.loads(args.input.read_text()), args.repo))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result.update(status="INVALID", error=str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"status": result["status"], "audit": str(args.out)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

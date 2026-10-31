"""Offline unit tests for the A100 DeviceSim mechanics-v2 parity Colab package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from research.colab_cli import canary as colab_canary
from research.devicesim_v2_parity.run import run_parity_job
from research.devicesim_v2_parity.units import (
    SCHEMA_UNIT,
    aggregate_attempt_dirs,
    allocation_for_lane,
    build_ledger_from_attempt_dirs,
    empty_ledger,
    load_verified_ledger,
    namespace_for,
    publish_immutable_json,
    specs_for_lane,
    validate_ledger_document,
    verify_published_json,
)
from tools.ops import calendar_lint

ROOT = Path(__file__).resolve().parents[1]


def _fake_rollout_report(*, num_envs: int, ticks: int, unexplained: int = 0) -> dict[str, Any]:
    return {
        "device": "cpu",
        "move_backend": "indexed",
        "num_envs": num_envs,
        "ticks": ticks,
        "world_ticks": num_envs * ticks,
        "snake_ticks": num_envs * 6 * ticks,
        "policy_mode": "product_b8_frp3_v8",
        "check_ct_sim": True,
        "discrete_mismatches": unexplained,
        "ct_discrete_mismatches": 0,
        "ego_obs_mismatches": 0,
        "total_live_decisions": num_envs * 6 * ticks,
        "tie_explained_decisions": 1,
        "unexplained_mismatches": unexplained,
        "policy_unexplained_mismatches": 0,
        "events": {
            "ambient_eats_and_instep_refills": 12,
            "corpse_eats": 5,
            "boost_steps": 40,
            "boost_trail_burns": 10,
            "boost_burns_at_full_450_cap": 3,
            "deaths_total": 4,
            "deaths_hero": 1,
            "deaths_opponent": 3,
            "deaths_wall": 1,
            "deaths_self": 0,
            "deaths_head": 1,
            "deaths_body": 2,
            "headon_size_wins": 1,
            "respawns_total": 4,
            "respawns_hero": 1,
            "respawns_opponent": 3,
            "mid_boost_deaths_followed_by_respawn": 1,
            "rng_draws_matched": 50,
        },
        "tie_samples": [{"tick": 3, "env": 0, "slot": 0, "kind": "hero_b8_q_float32_tie"}],
        "mismatch_samples": [],
        "verdict": "PASS" if unexplained == 0 else "FAIL",
    }


def test_atomic_publish_and_tamper_refusal(tmp_path: Path) -> None:
    commit = "a" * 40
    ns = namespace_for(commit, "main")
    shard_path = tmp_path / "shards" / "shard-00.json"
    doc = {
        "schema": SCHEMA_UNIT,
        "namespace": ns,
        "commit": commit,
        "lane": "main",
        "unit_id": "shard-00",
        "shard_index": 0,
        "seeds": [1, 2],
        "elapsed_seconds": 1.25,
        "report": _fake_rollout_report(num_envs=64, ticks=1000),
    }
    entry = publish_immutable_json(
        shard_path,
        doc,
        unit_id="shard-00",
        namespace=ns,
        relpath="shards/shard-00.json",
    )
    assert (tmp_path / "shards" / "shard-00.json.sha256").is_file()
    assert (tmp_path / "shards" / "shard-00.json.complete").is_file()

    # Create-only refuses overwrite.
    with pytest.raises(FileExistsError):
        publish_immutable_json(shard_path, doc, unit_id="shard-00", namespace=ns)

    verified_entry, verified_doc = verify_published_json(
        shard_path,
        expected_unit_id="shard-00",
        expected_namespace=ns,
        expected_relpath="shards/shard-00.json",
    )
    assert verified_entry == entry
    assert verified_doc["unit_id"] == "shard-00"

    # Foreign namespace refused.
    with pytest.raises(ValueError, match="foreign namespace"):
        verify_published_json(
            shard_path,
            expected_unit_id="shard-00",
            expected_namespace=namespace_for("b" * 40, "main"),
        )

    # Corrupted payload refused.
    shard_path.write_text(shard_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_published_json(shard_path, expected_unit_id="shard-00", expected_namespace=ns)


def test_verified_ledger_path_size_hash_only_and_retry_skip(tmp_path: Path) -> None:
    commit = "c" * 40
    ns = namespace_for(commit, "main")
    b8 = tmp_path / "b8.pth"
    frp3 = tmp_path / "frp3.pth"
    b8.write_bytes(b"stub-b8")
    frp3.write_bytes(b"stub-frp3")

    # Attempt 1 completes only shard-00 and shard-01 before simulated mechanical interruption.
    attempt1_dir = tmp_path / "attempt-1"
    (attempt1_dir / "shards").mkdir(parents=True)
    for spec in specs_for_lane("main")[:2]:
        doc = {
            "schema": SCHEMA_UNIT,
            "namespace": ns,
            "commit": commit,
            "lane": "main",
            "unit_id": spec.unit_id,
            "shard_index": spec.shard_index,
            "seeds": [10, 11],
            "elapsed_seconds": 2.0,
            "report": _fake_rollout_report(num_envs=spec.num_envs, ticks=spec.ticks),
        }
        publish_immutable_json(
            attempt1_dir / "shards" / f"{spec.unit_id}.json",
            doc,
            unit_id=spec.unit_id,
            namespace=ns,
            relpath=f"shards/{spec.unit_id}.json",
        )
    # Also drop an incomplete partial file for shard-02 WITHOUT .complete marker: must be ignored!
    (attempt1_dir / "shards" / "shard-02.json").write_text('{"partial": true}\n', encoding="utf-8")

    # Incomplete aggregation must fail closed.
    with pytest.raises(ValueError, match="incomplete allocation"):
        aggregate_attempt_dirs([attempt1_dir], commit=commit, lane="main")

    # Build Attempt 2 verified ledger from path/size/hash only.
    ledger2_doc = build_ledger_from_attempt_dirs([attempt1_dir], namespace=ns, lane="main")
    assert [u["unit_id"] for u in ledger2_doc["verified_units"]] == ["shard-00", "shard-01"]
    for u in ledger2_doc["verified_units"]:
        assert set(u) == {"unit_id", "payload_relpath", "sha256", "size"}

    # Extra outcome keys in ledger must be refused.
    polluted = json.loads(json.dumps(ledger2_doc))
    polluted["verified_units"][0]["verdict"] = "PASS"
    with pytest.raises(ValueError, match="path/size/hash keys only"):
        validate_ledger_document(polluted, expected_namespace=ns, expected_lane="main")

    ledger2_path = tmp_path / "ledger-attempt2.json"
    ledger2_path.write_text(json.dumps(ledger2_doc, indent=2) + "\n", encoding="utf-8")

    # Attempt 2 runs ONLY shard-02 and shard-03, never recomputing shard-00 or shard-01.
    called_seed_firsts: list[int] = []

    def _stub_rollout(**kwargs: Any) -> dict[str, Any]:
        called_seed_firsts.append(int(kwargs["seeds"][0]))
        return _fake_rollout_report(num_envs=len(kwargs["seeds"]), ticks=int(kwargs["ticks"]))

    attempt2_dir = tmp_path / "attempt-2"
    with patch("research.devicesim_v2_parity.run.run_v2_parity_rollout", side_effect=_stub_rollout):
        attempt2_summary = run_parity_job(
            commit=commit,
            lane="main",
            device="cpu",
            b8_checkpoint=b8,
            frp3_checkpoint=frp3,
            ledger_path=ledger2_path,
            output_dir=attempt2_dir,
        )

    assert len(called_seed_firsts) == 2
    assert attempt2_summary["completed_in_view_unit_ids"] == ["shard-02", "shard-03"]
    assert attempt2_summary["skipped_verified_unit_ids"] == ["shard-00", "shard-01"]
    assert attempt2_summary["status"] == "PASS"

    # Authoritative Mac-side aggregation across attempt-1 + attempt-2 verifies all 4 shards!
    final_summary = aggregate_attempt_dirs(
        [attempt1_dir, attempt2_dir],
        commit=commit,
        lane="main",
    )
    assert final_summary["status"] == "PASS"
    assert final_summary["complete_allocation_verified"] is True
    assert final_summary["completed_in_view_unit_ids"] == list(allocation_for_lane("main"))
    assert final_summary["num_envs_evaluated"] == 256
    assert final_summary["world_ticks"] == 256_000
    assert final_summary["snake_ticks"] == 1_536_000
    assert final_summary["unexplained_mismatches"] == 0

    # Duplicate unit across attempts must be refused.
    with pytest.raises(ValueError, match="duplicate verified unit"):
        aggregate_attempt_dirs([attempt1_dir, attempt1_dir], commit=commit, lane="main")


def test_foreign_ledger_namespace_and_allocation_refused(tmp_path: Path) -> None:
    commit = "d" * 40
    ledger_doc = empty_ledger(namespace=namespace_for(commit, "canary"), lane="canary")
    ledger_path = tmp_path / "canary-ledger.json"
    ledger_path.write_text(json.dumps(ledger_doc) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="foreign ledger namespace"):
        load_verified_ledger(
            ledger_path,
            expected_namespace=namespace_for(commit, "main"),
            expected_lane="main",
        )


def test_job_sh_device_contract_and_calendar_lint() -> None:
    pkg_dir = ROOT / "research" / "devicesim_v2_parity"
    files = sorted(p for p in pkg_dir.iterdir() if p.is_file())
    assert files
    for p in files:
        assert calendar_lint.calendar_literals(p) == []
    assert calendar_lint.lint(ROOT) == []

    sample_manifest = {
        "commit": "e" * 40,
        "script": "research/devicesim_v2_parity/job.sh",
        "device": "cuda",
        "env": {
            "DEVICESIM_V2_PARITY_DEVICE": "cuda",
        },
    }
    job_sh_text = (pkg_dir / "job.sh").read_text(encoding="utf-8")
    with patch.object(colab_canary, "_entrypoint", return_value=job_sh_text):
        assert colab_canary.job_device(sample_manifest, ROOT) == "cuda"

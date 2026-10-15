"""Contract tests for the resumable, filtered T3 Part-D collector."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.t3_rollout_labels_20261015 import bank_engine as bank
from research.t3_rollout_labels_20261015 import collect


def _job_plan() -> tuple[dict, dict]:
    job = {"arm": "b8_s0", "mix": "scripted", "cycle": 0, "index": 0}
    plan = {"seed": 1234, "mix": "scripted", "world_index": 0, "specs": [], "member_sha256s": []}
    return job, plan


def _receipt(job: dict, plan: dict, rows: Path, identity: dict) -> dict:
    return {
        "status": "PASS",
        "key": collect._key(job, plan),
        "arm": job["arm"],
        "mix": job["mix"],
        "cycle": job["cycle"],
        "seed": plan["seed"],
        "roster": plan,
        "run_identity": identity,
        "rows_sha256": bank.sha256(rows),
    }


def test_legacy_allowlist_retains_only_exact_receipt_and_shard(tmp_path: Path) -> None:
    job, plan = _job_plan()
    key = collect._key(job, plan)
    rows = tmp_path / "worlds" / f"{key}.npz"
    bank.write_world_rows(rows, [])
    old_identity, new_identity = {"source": "old"}, {"source": "new"}
    receipt_path = tmp_path / "worlds" / f"{key}.json"
    receipt_path.write_text(json.dumps(_receipt(job, plan, rows, old_identity)))
    allowlist = {
        key: {
            "receipt_sha256": bank.sha256(receipt_path),
            "rows_sha256": bank.sha256(rows),
            "run_identity": old_identity,
        }
    }

    trusted = collect._trusted_existing_receipt(job, plan, tmp_path, new_identity, allowlist)

    assert trusted is not None
    assert trusted["_resume_kind"] == "legacy"
    receipt_path.write_text(json.dumps(_receipt(job, plan, rows, {"source": "other"})))
    with pytest.raises(RuntimeError, match="allowlist mismatch"):
        collect._trusted_existing_receipt(job, plan, tmp_path, new_identity, allowlist)


def test_orphan_shard_is_quarantined_then_world_is_runnable(tmp_path: Path) -> None:
    job, plan = _job_plan()
    rows = tmp_path / "worlds" / f"{collect._key(job, plan)}.npz"
    bank.write_world_rows(rows, [])

    assert collect._trusted_existing_receipt(job, plan, tmp_path, {"new": True}, {}) is None
    assert not rows.exists()
    quarantined = list((tmp_path / "quarantine").rglob("*.npz"))
    assert len(quarantined) == 1


def test_prefilter_requires_bound_pass_coverage(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy-resume.json"
    legacy.write_text(json.dumps({"schema": "t3-legacy-resume/v1", "entries": {}}))
    calibration = tmp_path / "prefilter.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "t3-prefilter/v1",
                "status": "PASS",
                "threshold": 1.25,
                "coverage": 0.95,
                "legacy_resume_sha256": bank.sha256(legacy),
            }
        )
    )

    threshold, _ = collect._prefilter(calibration, legacy)

    assert threshold == 1.25
    calibration.write_text(calibration.read_text().replace("0.95", "0.949"))
    with pytest.raises(SystemExit, match="coverage"):
        collect._prefilter(calibration, legacy)
    calibration.write_text(
        json.dumps(
            {
                "schema": "t3-prefilter/v1",
                "status": "PASS",
                "threshold": 1.25,
                "coverage": float("nan"),
                "legacy_resume_sha256": bank.sha256(legacy),
            }
        )
    )
    with pytest.raises(SystemExit, match="coverage"):
        collect._prefilter(calibration, legacy)


def test_row_preserves_exact_float32_q_and_length() -> None:
    obs = {
        "local": np.zeros((1, 7, 31, 31), np.uint8),
        "global": np.zeros((1, 4, 37, 37), np.uint8),
        "scalars": np.zeros((1, 60), np.float32),
        "mask": np.ones((1, 6), bool),
    }
    row = collect._row(
        obs, 1, np.ones(6, bool), True, 2000, 2001, 712, np.arange(6, dtype=np.float32)
    )

    assert row["length"] == 712
    assert row["q"].dtype == np.float32
    np.testing.assert_array_equal(row["q"], np.arange(6, dtype=np.float32))


def test_twelve_workers_require_observed_ac_power(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        collect.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="Now drawing from 'AC Power'"),
    )
    collect._require_ac_for_twelve_workers(12)
    monkeypatch.setattr(
        collect.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="Now drawing from 'Battery Power'"
        ),
    )
    with pytest.raises(SystemExit, match="AC Power"):
        collect._require_ac_for_twelve_workers(12)


def test_progress_uses_completed_receipts_not_stale_merged_stats(tmp_path: Path) -> None:
    receipts = [
        {"key": "old", "_resume_kind": "legacy", "decisive_rows": 7},
        {"key": "new", "_resume_kind": "current", "decisive_rows": 9},
    ]

    collect._write_progress(tmp_path, receipts, {"decisive_rows": 0}, [0])

    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["decisive_rows"] == 16
    assert progress["keys_done"] == ["new", "old"]


def test_merge_rejects_duplicate_receipts_before_reading_shards(tmp_path: Path) -> None:
    receipt = {"key": "duplicate", "status": "PASS", "_resume_kind": "current"}

    with pytest.raises(RuntimeError, match="duplicate completed world receipts"):
        collect.merge_world_rows(tmp_path, [receipt, receipt], {"identity": True}, {})


def test_half_counterweight_ratio_keeps_all_decisive_and_is_deterministic() -> None:
    rows = [{"decisive": True, "id": i} for i in range(5)]
    rows += [{"decisive": False, "id": i + 5} for i in range(20)]
    selected = bank.counterweight_trim(rows, ratio=0.5)
    assert selected[:5] == rows[:5]
    assert len(selected) == 7
    assert selected == bank.counterweight_trim(rows, ratio=0.5)


def test_target_requires_the_half_counter_complement() -> None:
    assert not collect._target_met({"decisive_rows": 10000, "counterweight_rows": 4999}, 10000, 0.5)
    assert collect._target_met({"decisive_rows": 10001, "counterweight_rows": 5000}, 10000, 0.5)
    assert not collect._target_met({"decisive_rows": 9999, "counterweight_rows": 5000}, 10000, 0.5)


def test_death_window_validation_rejects_stale_code(tmp_path: Path, monkeypatch) -> None:
    repo, out = tmp_path / "repo", tmp_path / "bank"
    (out / "worlds").mkdir(parents=True)
    sources = {}
    for name in (
        "death_window",
        "bank_engine",
        "collect",
        "engine",
        "trigger",
        "validate_death_window",
    ):
        relative = f"research/t3_rollout_labels_20261015/{name}.py"
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source")
        sources[relative] = bank.sha256(path)
    for name in ("namespace", "prefilter"):
        (out / f"{name}.json").write_text("{}")
    keys = [f"world-{i}" for i in range(5)]
    bindings = {}
    for key in keys:
        bindings[key] = {}
        for suffix, field in (("json", "receipt_sha256"), ("npz", "rows_sha256")):
            path = out / "worlds" / f"{key}.{suffix}"
            path.write_text(key)
            bindings[key][field] = bank.sha256(path)
    value = {
        "schema": "t3-death-window-validation/v1",
        "status": "PASS",
        "keys": keys,
        "source_sha256": sources,
        "legacy_bindings": bindings,
        "namespace_sha256": bank.sha256(out / "namespace.json"),
        "prefilter_sha256": bank.sha256(out / "prefilter.json"),
        "selected": [{"key": key, "comparison": {"pass": True}} for key in keys],
    }
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(collect.census, "b8_path", lambda arm: checkpoint)
    value["checkpoint_sha256"] = {arm: bank.sha256(checkpoint) for arm in collect.census.B8_ARMS}
    receipt = out / "death-window-validation.json"
    receipt.write_text(json.dumps(value))
    assert collect._validate_death_window_receipt(receipt, out, repo)["status"] == "PASS"
    (repo / next(iter(sources))).write_text("changed")
    with pytest.raises(SystemExit, match="source changed"):
        collect._validate_death_window_receipt(receipt, out, repo)

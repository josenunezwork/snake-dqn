"""Focused no-provider contracts for P1 runtime durability evidence."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from research.bl_ppo.exports import export_torch
from research.bl_ppo.recovery import SCHEMA


def _snapshot(path, seed: int) -> None:
    """Write a minimal complete snapshot whose collector has worker RNG state."""
    export_torch(
        path,
        {
            "schema": SCHEMA,
            "identity": {"arm": "F"},
            "policy": {},
            "learner": {},
            "collector": {"world_rng": random.Random(seed)},
            "adapter": {},
            "counters": {"actor_transitions": 128, "collection_seconds": 1.0},
            "rng": {
                "torch_cpu": None,
                "torch_cuda_all": [],
                "numpy": (),
                "python": random.Random(3).getstate(),
            },
        },
    )


def test_snapshot_fingerprint_includes_worker_random_state(tmp_path) -> None:
    from research.bl_ppo.p1_runtime import _snapshot_fingerprint

    first, second = tmp_path / "first.pt", tmp_path / "second.pt"
    _snapshot(first, 1)
    _snapshot(second, 2)
    assert _snapshot_fingerprint(first) != _snapshot_fingerprint(second)


def test_assignment_endpoint_accepts_supervisor_uuid_generation(tmp_path, monkeypatch) -> None:
    from research.bl_ppo.p1_runtime import _assignment_endpoint

    (tmp_path / "spec.json").write_text(
        json.dumps({"release_endpoint": "endpoint-123", "generation": "60cf7f8d-abc"})
    )
    monkeypatch.setenv("COLAB_CLI_ROOT", str(tmp_path))
    assert _assignment_endpoint() == "endpoint-123#generation=60cf7f8d-abc"


def _control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **changes) -> Path:
    from research.bl_ppo.exports import digest, export_json
    from research.bl_ppo.p1_eval import materialize
    from research.bl_ppo.test_p1_eval import _audit

    bank = materialize(0)
    audit = tmp_path / "bank-audit.json"
    export_json(audit, _audit(bank))
    control = {
        "schema": "bl-p1-control/v1",
        "phase": "y3-source",
        "target": 100_000_000,
        "seed": 20261021,
        "arm": "F",
        "worlds": 128,
        "processes": 12,
        "rollout_steps": 128,
        "source_commit": "a" * 40,
        "scientific_admission": True,
        "registration_sha256": digest(Path(__file__).with_name("REGISTRATION_P1.md")),
        "evaluation_round": 0,
        "evaluation_bank_sha256": bank["allocation_sha256"],
        "evaluation_audit_sha256": digest(audit),
        "evaluation_excluded_seeds": sorted(row["world_seed"] for row in bank["rows"]),
    } | changes
    path = tmp_path / "p1-control.json"
    export_json(path, control)
    monkeypatch.setenv("BL_PPO_COMMIT", "a" * 40)
    return path


def test_real_shaped_source_control_accepts_frozen_bank(tmp_path, monkeypatch) -> None:
    from research.bl_ppo.p1_runtime import _load_control

    control = _load_control(_control(tmp_path, monkeypatch), "y3-source")
    assert control["worlds"] == 128 and len(control["evaluation_excluded_seeds"]) == 384


@pytest.mark.parametrize("changes", [{"processes": 6}, {"seed": 7}, {"worlds": 2}])
def test_source_control_refuses_recipe_substitution(tmp_path, monkeypatch, changes) -> None:
    from research.bl_ppo.p1_runtime import _load_control

    with pytest.raises(ValueError):
        _load_control(_control(tmp_path, monkeypatch, **changes), "y3-source")


def test_admission_boolean_cannot_substitute_for_y3_receipt(tmp_path, monkeypatch) -> None:
    from research.bl_ppo.p1_runtime import _load_control

    path = _control(tmp_path, monkeypatch, phase="qual-1m")
    with pytest.raises(ValueError, match="export component"):
        _load_control(path, "qual-1m")


@pytest.mark.parametrize("bad_counter,bad_digest", [(True, False), (False, True), (False, False)])
def test_tranche_requires_exact_previous_snapshot(tmp_path, bad_counter, bad_digest) -> None:
    from research.bl_ppo.exports import digest, export_json
    from research.bl_ppo.p1_runtime import _validate_resume

    control = {
        "phase": "tranche",
        "chunk_index": 2,
        "start_at": 20_000_000,
        "pause_at": 40_000_000,
        "source_commit": "a" * 40,
        "registration_sha256": "b" * 64,
        "evaluation_bank_sha256": "c" * 64,
    }
    snapshot = tmp_path / "previous.pt"
    export_torch(
        snapshot,
        {
            "schema": SCHEMA,
            "identity": {
                "p1_runtime": True,
                "arm": "F",
                "seed": 20261021,
                "target_transitions": 100_000_000,
                "source_commit": "a" * 40,
                "p1_registration_sha256": "b" * 64,
                "p1_evaluation_bank_sha256": "c" * 64,
            },
            "counters": {"actor_transitions": 19_999_872 if bad_counter else 20_000_000},
            "policy": {},
            "learner": {},
            "collector": {},
            "adapter": {},
            "rng": {},
        },
    )
    receipt = tmp_path / "previous.json"
    export_json(
        receipt,
        {
            "schema": "bl-p1-tranche/v1",
            "status": "PAUSED",
            "actor_transitions": 20_000_000,
            "source_commit": "a" * 40,
            "snapshot": {"sha256": "d" * 64 if bad_digest else digest(snapshot)},
        },
    )
    control["previous_receipt_sha256"] = digest(receipt)
    if bad_counter or bad_digest:
        with pytest.raises(ValueError, match="predecessor"):
            _validate_resume(control, snapshot, tmp_path)
    else:
        assert (
            _validate_resume(control, snapshot, tmp_path)["counters"]["actor_transitions"]
            == 20_000_000
        )

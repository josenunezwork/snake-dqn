"""Focused no-provider contracts for P1 runtime durability evidence."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from research.bl_ppo.exports import export_torch
from research.bl_ppo.recovery import SCHEMA


def test_production_pause_zero_and_1m_rounding_are_literal() -> None:
    from research.bl_ppo.p1_contract import SHAPE, whole_world_pause
    from research.bl_ppo.p1_runtime import _train_args

    assert whole_world_pause(0, 128) == 0
    assert SHAPE.qualification_pause == 1_000_064
    assert SHAPE.qualification_pause - SHAPE.qualification_requested == 64
    assert SHAPE.target == 100_000_000
    args = _train_args(
        {
            "phase": "y3-source",
            "pause_at": 0,
            "processes": 12,
            "rollout_steps": 128,
            "registration_sha256": "a" * 64,
            "evaluation_bank_sha256": "b" * 64,
            "evaluation_excluded_seeds": [],
        },
        output=Path("unused"),
        checkpoint=Path("unused"),
        resume=None,
        durability=False,
        snapshot_label="origin",
    )
    assert args.pause_at == args.requested_pause_at == 0
    assert args.transitions == SHAPE.target


@pytest.mark.parametrize("bad_snapshot", [False, True])
def test_pause_requires_actual_counter_and_snapshot_boundary(tmp_path, bad_snapshot) -> None:
    from research.bl_ppo.exports import digest
    from research.bl_ppo.p1_runtime import _require_pause

    path = tmp_path / "snapshot.pt"
    actor = 128 if not bad_snapshot else 0
    counters = {"actor_transitions": actor, "actor_updates": 0}
    export_torch(
        path,
        {
            "schema": SCHEMA,
            "identity": {},
            "policy": {},
            "learner": {},
            "collector": {},
            "adapter": {},
            "rng": {},
            "counters": counters,
            "boundary": {
                "requested_pause_at": 0,
                "pause_at": 0,
                "actual_pause_at": 128,
                "kind": "exact",
            },
        },
    )
    status = {
        "status": "PAUSED",
        "critic_qualified": True,
        "pause_at": 0,
        "requested_pause_at": 0,
        "actual_pause_at": actor,
        "counters": counters,
        "snapshot": {"path": path.name, "sha256": digest(path)},
        "_output": str(tmp_path),
    }
    with pytest.raises(ValueError, match="actor counter|snapshot boundary"):
        _require_pause(status, pause=0)


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


def test_snapshot_fingerprint_includes_ordinary_bank_numpy_rng(tmp_path) -> None:
    import numpy as np

    from research.bl_ppo.p1_runtime import _snapshot_fingerprint
    from research.bl_ppo.recovery import read_snapshot

    first, same, changed = (tmp_path / name for name in ("first.pt", "same.pt", "changed.pt"))
    _snapshot(first, 1)
    state = read_snapshot(first)
    state["adapter"]["ordinary_bank_rng"] = np.random.default_rng(91)
    export_torch(same, state)
    before = _snapshot_fingerprint(same)
    state["adapter"]["ordinary_bank_rng"].random()
    export_torch(changed, state)
    assert before == _snapshot_fingerprint(same)
    assert before != _snapshot_fingerprint(changed)


def test_snapshot_fingerprint_hashes_object_array_values_not_addresses(tmp_path) -> None:
    import numpy as np

    from research.bl_ppo.p1_runtime import _snapshot_fingerprint
    from research.bl_ppo.recovery import read_snapshot

    paths = [tmp_path / f"state-{index}.pt" for index in range(3)]
    _snapshot(tmp_path / "base.pt", 1)
    for path, kill_length in zip(paths, (700, 700, 900)):
        state = read_snapshot(tmp_path / "base.pt")
        state["collector"]["kill_metadata"] = np.array(
            [{"length": kill_length}, None], dtype=object
        )
        export_torch(path, state)
    assert _snapshot_fingerprint(paths[0]) == _snapshot_fingerprint(paths[1])
    assert _snapshot_fingerprint(paths[0]) != _snapshot_fingerprint(paths[2])


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
    ledger = tmp_path / "verified-ledger.json"
    export_json(
        ledger,
        {
            "schema": "bl-p1-verified-ledger/v1",
            "namespace": "test-y3",
            "allocation": "source",
            "units": [],
        },
    )
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
        "late_body_amendment_sha256": digest(
            Path(__file__).with_name("E8_LATE_BODY_AMENDMENT_02.md")
        ),
        "evaluation_round": 0,
        "evaluation_bank_sha256": bank["allocation_sha256"],
        "evaluation_audit_sha256": digest(audit),
        "evaluation_excluded_seeds": sorted(row["world_seed"] for row in bank["rows"]),
        "resume": {
            "unit_kind": "operational-snapshot",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
            "namespace": "test-y3",
            "allocation": "source",
        },
    } | changes
    path = tmp_path / "p1-control.json"
    export_json(path, control)
    monkeypatch.setenv("BL_PPO_COMMIT", "a" * 40)
    return path


def test_real_shaped_source_control_accepts_frozen_bank(tmp_path, monkeypatch) -> None:
    from research.bl_ppo.p1_runtime import _load_control

    control = _load_control(_control(tmp_path, monkeypatch), "y3-source")
    assert control["worlds"] == 128 and len(control["evaluation_excluded_seeds"]) == 384


def test_source_control_refuses_unbound_late_body_amendment(tmp_path, monkeypatch) -> None:
    from research.bl_ppo.p1_runtime import _load_control

    with pytest.raises(ValueError, match="late-body amendment"):
        _load_control(
            _control(tmp_path, monkeypatch, late_body_amendment_sha256="0" * 64), "y3-source"
        )


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
            "boundary": {
                "requested_pause_at": 20_000_000,
                "pause_at": 20_000_000,
                "actual_pause_at": 20_000_000,
                "kind": "exact",
            },
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


def test_boundary_only_y3_is_admitted_with_exact_pause_resume(tmp_path, monkeypatch) -> None:
    from research.bl_ppo.exports import digest, export_json
    from research.bl_ppo.p1_runtime import _load_control

    admission = tmp_path / "admission.json"
    export_json(
        admission,
        {"schema": "bl-p1-y3/v1", "status": "Y3_PASS_BOUNDARY_ONLY", "source_commit": "a" * 40},
    )
    control = _control(
        tmp_path, monkeypatch, phase="qual-1m", prerequisites={"y3": {"sha256": digest(admission)}}
    )
    assert _load_control(control, "qual-1m")["phase"] == "qual-1m"


def test_new_y3_cannot_disable_operational_replay(tmp_path, monkeypatch) -> None:
    from research.bl_ppo.exports import digest
    from research.bl_ppo.p1_runtime import _load_control

    control = _control(
        tmp_path,
        monkeypatch,
        operational_replay_required=False,
        resume_amendment_sha256=digest(Path(__file__).with_name("P1_Y3_RESUME_AMENDMENT_05.md")),
    )
    with pytest.raises(ValueError, match="both registered replays"):
        _load_control(control, "y3-source")

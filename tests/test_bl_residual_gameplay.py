"""Contracts for the F3 gameplay-development screen seam."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from research.bl_residual import gameplay


def _row(mix: str = "scripted", seed: int = 7, parent: str = "parent-7") -> dict:
    return {
        "world_id": f"world-{seed}",
        "parent_world_id": parent,
        "world_seed": seed,
        "mix": mix,
        "plan": {"seed": seed, "mix": mix, "specs": []},
    }


def _fixture_bank(row: dict | None = None) -> dict:
    return {
        "namespace": "f3-fixture/v1",
        "collision_receipt": {"status": "CLEAR", "namespace": "f3-fixture/v1"},
        "rows": [_row() if row is None else row],
    }


def test_fixture_runs_paired_arms_and_never_emits_a_decision(monkeypatch, tmp_path: Path) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    candidates = {}
    for dose in gameplay.DOSES:
        path = tmp_path / f"{dose}.pth"
        path.write_bytes(dose.encode())
        candidates[dose] = path
    monkeypatch.setattr(
        gameplay, "sha256_file", lambda path: gameplay.B8_SHA256 if path == b8 else "a" * 64
    )
    monkeypatch.setattr(gameplay, "_validate_materialized_rosters", lambda rows: None)
    monkeypatch.setattr(gameplay, "_root_digest", lambda plan, **kwargs: "d" * 64)
    monkeypatch.setattr(
        gameplay,
        "_build_exact_roots",
        lambda plan, paths: (
            object(),
            {name: type("Template", (), {"hero": name})() for name in paths},
        ),
    )
    monkeypatch.setattr(
        gameplay,
        "_run_exact_arm_from_root",
        lambda root, hero, **kwargs: {
            "mass_integral": float(kwargs["horizon"]),
            "alive_frame_fraction": 1.0,
            "no_death_episode": True,
            "decisions": kwargs["horizon"],
            "action_trace": [{"decision": 0, "hero_length": 7, "served_action": 4}],
            "boost_share": 1.0,
        },
    )
    receipt = gameplay.run_screen(_fixture_bank(), b8_path=b8, candidates=candidates, horizon=3)
    assert receipt["fixture"] is True
    assert receipt["decision"]["label"] == "NOT_A_DECISION"
    assert list(receipt["observations"][0]["arms"]) == list(gameplay.ARMS)
    assert receipt["observations"][0]["boost_share"] == {arm: 1.0 for arm in gameplay.ARMS}


def test_unratified_production_refuses_before_bank_execution(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gameplay, "_require_mac", lambda: None)
    with pytest.raises(RuntimeError, match="unratified"):
        gameplay.run_screen({}, b8_path=tmp_path / "none", candidates={}, horizon=gameplay.HORIZON)


@pytest.mark.parametrize(
    "row, message",
    [
        (_row(parent="train-parent"), "collides"),
        ({**_row(), "plan": {"seed": 8, "mix": "scripted"}}, "plan identity"),
    ],
)
def test_bank_rejects_parent_collision_and_plan_mismatch(row: dict, message: str) -> None:
    parents = {"train-parent"} if row["parent_world_id"] == "train-parent" else None
    with pytest.raises(ValueError, match=message):
        gameplay.validate_bank(_fixture_bank(row), training_parent_ids=parents, production=False)


def test_bank_requires_clear_collision_receipt() -> None:
    bank = _fixture_bank()
    bank["collision_receipt"]["status"] = "REJECTED"
    with pytest.raises(ValueError, match="CLEAR"):
        gameplay.validate_bank(bank, production=False)


def test_production_requires_parent_allocation_before_audit(monkeypatch) -> None:
    rows = []
    for mix, offset in zip(gameplay.MIXES, range(3)):
        rows.extend(
            _row(mix=mix, seed=10_000 * offset + index, parent=f"p-{mix}-{index}")
            for index in range(gameplay.COUNT_PER_MIX)
        )
    bank = {"namespace": "production/v1", "collision_receipt": {"status": "CLEAR"}, "rows": rows}
    with pytest.raises(ValueError, match="parent allocation"):
        gameplay.validate_bank(bank, production=True)


def test_production_evidence_requires_exact_f4_approval_json() -> None:
    with pytest.raises(RuntimeError, match="unratified"):
        gameplay._approved_successor(
            {
                "schema": "bl-residual-f4-judgment/v1",
                "decision": "APPROVE",
                "judgment_sha256": "a" * 64,
                "registration_sha256": gameplay.F4_REGISTRATION_SHA256,
                "recipe_sha256": "b" * 64,
            }
        )


def test_production_evidence_binds_exact_registration_and_recipe(monkeypatch) -> None:
    from research.bl_residual.fit import FitRecipe
    from research.bl_residual.protocol import ROUND_26_JUDGMENT_SHA, canonical_sha

    monkeypatch.setattr(gameplay, "sha256_file", lambda path: gameplay.F4_REGISTRATION_SHA256)
    evidence = {
        "schema": "bl-residual-f4-judgment/v1",
        "decision": "APPROVE",
        "judgment_sha256": ROUND_26_JUDGMENT_SHA,
        "registration_sha256": gameplay.F4_REGISTRATION_SHA256,
        "recipe_sha256": canonical_sha(asdict(FitRecipe())),
    }
    assert gameplay._approved_successor(evidence) == {
        "registration_sha256": evidence["registration_sha256"],
        "recipe_sha256": evidence["recipe_sha256"],
    }


def test_exact_arm_reports_delivered_trace_and_boost_share(monkeypatch) -> None:
    class Sim:
        alive = np.array([[True]])
        length = np.array([[10]])

    class World:
        sim = Sim()
        hero = object()

        def clone(self):
            return self

    steps = iter(
        (
            {"alive": 1, "hero_length": 10, "served_action": 1},
            {"alive": 1, "hero_length": 11, "served_action": 4},
        )
    )
    monkeypatch.setattr("research.t34_shadow.shadow.clone_world_exact", lambda world: world)
    monkeypatch.setattr(gameplay, "_step_with_trace", lambda *args, **kwargs: next(steps))
    result = gameplay._run_exact_arm_from_root(World(), object(), horizon=2)
    assert [row["served_action"] for row in result["action_trace"]] == [1, 4]
    assert result["boost_share"] == 0.5
    assert result["decisions"] == 2


def test_probe_bank_requires_exact_r4_mix_counts(monkeypatch) -> None:
    rows = []
    for mix, count in gameplay.PROBE_COUNT_PER_MIX.items():
        rows.extend(_row(mix, 1000 + len(rows), f"probe-{len(rows)}") for _ in range(count))
    bank = {"namespace": "probe/v1", "collision_receipt": {"status": "CLEAR"}, "rows": rows}
    monkeypatch.setattr(gameplay, "_validate_collision_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(gameplay, "_validate_materialized_rosters", lambda *args, **kwargs: None)
    assert len(gameplay.validate_probe_bank(bank)) == 32
    bank["rows"].pop()
    with pytest.raises(ValueError, match="11/11/10"):
        gameplay.validate_probe_bank(bank)


def test_collapse_tripwire_is_sliding_and_preserves_equal_90_percent() -> None:
    actions = [0] * 90_000 + [1] * 10_001
    observation = {
        "arms": {
            "B8": {"action_trace": [{"served_action": action} for action in actions]},
        }
    }
    outcome = gameplay._collapse_tripwire([observation], "B8")
    assert outcome["full_window_observed"] is True
    assert outcome["pass"] is True
    actions[90_000] = 0
    outcome = gameplay._collapse_tripwire(
        [{"arms": {"B8": {"action_trace": [{"served_action": action} for action in actions]}}}],
        "B8",
    )
    assert outcome["max_full_window_action_share"] > 0.90
    assert outcome["pass"] is False


def test_screen_known_controller_fault_is_kill_fit_not_incomplete(
    monkeypatch, tmp_path: Path
) -> None:
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"b8")
    candidates = {dose: tmp_path / f"{dose}.pth" for dose in gameplay.DOSES}
    for path in candidates.values():
        path.write_bytes(b"candidate")
    monkeypatch.setattr(gameplay, "sha256_file", lambda path: gameplay.B8_SHA256)
    monkeypatch.setattr(gameplay, "_validate_materialized_rosters", lambda rows: None)
    monkeypatch.setattr(gameplay, "_root_digest", lambda *args, **kwargs: "root")
    monkeypatch.setattr(
        gameplay,
        "_build_exact_roots",
        lambda *args: (
            object(),
            {name: type("T", (), {"hero": object()})() for name in gameplay.ARMS},
        ),
    )
    monkeypatch.setattr(
        gameplay,
        "_run_exact_arm_from_root",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            gameplay.ControllerFault("illegal_served_action")
        ),
    )
    receipt = gameplay.run_screen(_fixture_bank(), b8_path=b8, candidates=candidates, horizon=3)
    assert receipt["decision"]["label"] == "KILL_FIT"
    assert receipt["known_controller_fault"]["B8"]["fault"] == "illegal_served_action"

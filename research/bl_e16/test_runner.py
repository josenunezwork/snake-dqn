"""Focused runner guards; simulator execution is covered by the explicit smoke command."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_e16 import runner


def test_train_requires_external_validated_plan() -> None:
    with pytest.raises(runner.AdmissionRequired, match="validated external"):
        runner.require_plan(None, "A")


def test_stub_arms_are_not_executable() -> None:
    parser = runner.build_parser()
    arguments = parser.parse_args(
        [
            "smoke",
            "--arm",
            "A",
            "--bank",
            "bank",
            "--seed-audit",
            "audit",
            "--b8-checkpoint",
            "b8",
            "--parent",
            "parent",
            "--output",
            "out",
        ]
    )
    arguments.arm = "R"
    with pytest.raises(runner.AdmissionRequired, match="UNRANKED"):
        runner.run(arguments)


def test_audit_requires_complete_sidecars(tmp_path: Path) -> None:
    bank_path, audit_path = tmp_path / "BANK.json", tmp_path / "SEED_AUDIT.json"
    bank_path.write_text("{}")
    audit_path.write_text("{}")
    with pytest.raises(ValueError, match="missing or unsafe"):
        runner.load_audited_bank(bank_path, audit_path)


def test_snapshot_captures_explicit_collector_and_rng_state() -> None:
    # The schema contract is intentionally inspectable without launching a simulator.
    assert runner.SCHEMA == "bl-e16-runner/v1"
    assert runner.CRITIC_TRANSITIONS == 2_000_000
    assert runner.ACTOR_TRANSITIONS == 4_000_000

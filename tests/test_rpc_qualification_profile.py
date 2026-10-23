"""Frozen envelopes and explicit dispatch routing for the engineering profile."""

import json
from pathlib import Path

import pytest

from research.rpc_qualification.admission import envelope, validate_manifest
from tools.ops.dispatch import profile_launcher, require_resume_policy


def test_c1_envelopes_apply_owner_multiplier_exactly_once() -> None:
    l4, cpu, injection = (
        envelope("l4", "loop"),
        envelope("cpu", "loop"),
        envelope("cpu", "injection"),
    )
    assert l4["effective"] == {"unit_cap": 5.0, "reserve_units": 1.0, "timeout_minutes": 160}
    assert cpu["effective"] == {"unit_cap": 0.30, "reserve_units": 0.06, "timeout_minutes": 160}
    assert injection["effective"]["unit_cap"] == 0.12
    assert injection["effective"]["timeout_minutes"] == 30
    assert 3 * 5 + 3 * 0.30 + 0.12 == pytest.approx(16.02)
    assert 160 * 1.54 / 60 + 1 <= 5.2  # first unit cap may bind before wall
    with pytest.raises(ValueError):
        envelope("l4", "injection")


def test_only_exact_script_selects_explicit_adapter(tmp_path: Path) -> None:
    manifest = tmp_path / "job.json"
    manifest.write_text(json.dumps({"script": "research/rpc_qualification/colab.sh"}))
    assert profile_launcher(tmp_path, manifest) == tmp_path / "research/rpc_qualification/launch.sh"
    manifest.write_text(json.dumps({"script": "research/bl_labels/f1_colab.sh"}))
    assert profile_launcher(tmp_path, manifest) == tmp_path / "research/colab_cli/launch.sh"


def test_profile_cannot_enter_as_generic_family_or_resume_exemption(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="foreign"):
        validate_manifest({}, tmp_path)
    require_resume_policy(
        {
            "resume": {
                "unit_kind": "f1-root-repeat",
                "ledger_path": "inputs/verified-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            }
        }
    )

"""The P1 maintenance exception never changes ordinary dispatch priorities."""

import json
import subprocess

import pytest

from research.bl_ppo.p1_attempt2_pack import FINALISATION
from tools.ops import dispatch
from tools.ops.dispatch import P1_ATTEMPT2_RULING, Refused, persistent_shell, require_resume_policy


def manifest():
    return {
        "p1_finalisation_ruling_sha256": FINALISATION,
        "env": {"BL_P1_MODE": "y3-source"},
        "resume": {
            "unit_kind": "operational-snapshot",
            "ledger_path": "inputs/ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
        "p1_attempt2": {
            "ruling_sha256": P1_ATTEMPT2_RULING,
            "canary_attempts": 1,
            "repair_attempts": 1,
            "canary_retries": 0,
            "further_repair_retries": 0,
        },
    }


def test_attempt2_wrapper_declaration_does_not_grant_extra_attempts():
    value = manifest()
    require_resume_policy(value)
    value["p1_attempt2"]["further_repair_retries"] = 1
    with pytest.raises(Refused, match="no further"):
        require_resume_policy(value)


def test_legacy_p1_still_requires_existing_single_retry_declaration():
    value = manifest()
    value.pop("p1_attempt2")
    with pytest.raises(Refused, match="ruled integer 1"):
        require_resume_policy(value)
    value["resume"]["max_outcome_blind_retries"] = 1
    require_resume_policy(value)


def test_quiet_control_shell_registers_root_and_refuses_nonzero_nice():
    ordinary = persistent_shell("true")
    assert "/usr/bin/nice" in ordinary
    quiet = persistent_shell(
        "true",
        p1_window={
            "arm_path": "/tmp/arm",
            "owner_token": "test-owner",
            "root_record": "/tmp/root.json",
            "registration_evidence": "/tmp/registered.json",
        },
    )
    assert "/usr/bin/nice" not in quiet
    assert "p1_attempt2_control" in quiet
    assert "register-control" in quiet
    assert "absolute nice 0" in quiet
    assert "--root-pid" in quiet
    assert "set +e" in quiet


def test_no_other_assignment_is_admitted_under_allocation_lock(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest()))
    monkeypatch.setattr(
        dispatch,
        "command",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout="[other] endpoint-other | Hardware: L4\n"
        ),
    )
    with pytest.raises(Refused, match="zero other assignments"):
        dispatch.refresh_assignments(path, tmp_path / "inventory.json", 100)
    assert not (tmp_path / "inventory.json").exists()

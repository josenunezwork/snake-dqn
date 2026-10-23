"""Amendment-06 retry declarations stay narrower than generic resume policy."""

import pytest

from tools.ops.dispatch import Refused, require_resume_policy

RULING = "fe0584595cca6f6fcb28071692e52e4fe8460f3ab90fc29429c027203ff9ebfc"


def manifest(phase: str, retries: int) -> dict:
    """Build a minimal declaration; no operational entrypoint is called."""
    return {
        "p1_finalisation_ruling_sha256": RULING,
        "env": {"BL_P1_MODE": phase},
        "resume": {
            "unit_kind": "operational-snapshot",
            "ledger_path": "inputs/verified-ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": retries,
        },
    }


@pytest.mark.parametrize("phase,retries", [("y3-source", 1), ("y3-replacement", 1), ("qual-1m", 0)])
def test_exact_ruled_phase_maximum(phase, retries):
    require_resume_policy(manifest(phase, retries))


@pytest.mark.parametrize("phase", ["y3-source", "y3-replacement", "qual-1m"])
@pytest.mark.parametrize("retries", [2, True, 1.0, "1", -1, 3])
def test_ruled_family_refuses_wider_or_malformed_maximum(phase, retries):
    with pytest.raises(Refused, match="ruled integer"):
        require_resume_policy(manifest(phase, retries))


@pytest.mark.parametrize("change", ["wrong-ruling", "wrong-phase", "missing-ruling"])
def test_exception_does_not_apply_to_other_families(change):
    value = manifest("y3-source", 1)
    if change == "wrong-ruling":
        value["p1_finalisation_ruling_sha256"] = "0" * 64
    elif change == "wrong-phase":
        value["env"]["BL_P1_MODE"] = "unrelated"
    else:
        del value["p1_finalisation_ruling_sha256"]
    with pytest.raises(Refused, match="integer 2"):
        require_resume_policy(value)

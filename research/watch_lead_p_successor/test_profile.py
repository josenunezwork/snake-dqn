"""Outcome boundary, disjoint allocation, exact path and identity smoke helpers."""

import pytest

from research.watch_lead_p.protocol import FIXTURE_SEED, SEEDS
from research.watch_lead_p_successor.host import SCRIPT, require_profile
from research.watch_lead_p_successor.prepare import audit_bank, bank_record
from research.watch_lead_p_successor.profile import measurement


def test_fresh_bank_excludes_closed_P_and_fixture() -> None:
    bank = bank_record()
    assert not set(bank["seeds"]) & (set(SEEDS) | {FIXTURE_SEED})
    assert len(set(bank["seeds"])) == 4
    assert bank["successor_eligible"] is False
    assert audit_bank()["collisions"] == []


def test_export_allowlist_discards_outcomes() -> None:
    row = dict(
        seed=1,
        arm="P0",
        frames=64,
        trace_sha256="opaque",
        initial_state="opaque",
        deaths=19,
        mass=999,
        counts={"changed": 9},
        roots=[{"outcomes": "SECRET"}],
    )
    output = measurement(row, 8, 6, 4, 2)
    assert not set(output) & {"deaths", "mass", "counts", "roots"}
    assert "SECRET" not in repr(output)
    assert output["seconds_per_frame"] == pytest.approx(0.125)
    assert output["cpu_percent_one_core"] == 75


def test_prepare_allocation_is_deterministic() -> None:
    assert bank_record() == bank_record()
    assert bank_record()["namespace"] != "a100-dev/t34-watch-lead-1028/"


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"script": "foreign.sh"},
        {"timeout_minutes": 40},
        {"canary": {}},
        {"device": "cuda"},
        {"p_profile_source_delivery": "other"},
        {"sandbox_admission": {}},
    ],
)
def test_source_adapter_is_narrow(change: dict | None) -> None:
    from research.watch_lead_p_successor.prepare import ROOT

    manifest = {
        "p_profile_source_delivery": "background-sparse-v1",
        "script": SCRIPT,
        "device": "cpu",
        "timeout_minutes": 30,
        "requirements": {"allowed_gpus": ["L4"]},
        "canary": {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT},
        "resume_exemption": {"kind": "short-canary"},
        "env": {"SNAKE_COLAB_CANARY": "1"},
    }
    if change is None:
        require_profile(manifest, ROOT)
    else:
        with pytest.raises(ValueError):
            require_profile({**manifest, **change}, ROOT)

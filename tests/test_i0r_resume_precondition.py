"""I0R ruling §2 precondition against the real I0 canary receipt; zero allocations.

Runs the host-only probe through its wrapper (qualified runtime first on
PYTHONPATH, nice 15, single-threaded) without governed dry-runs. Skips when the
qualified checkout or the canary receipt is not on this machine.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import i0r_resume_precondition as probe

ROOT = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    not (probe.RUN.is_dir() and probe.RECEIPT.is_file() and probe.RULING.is_file()),
    reason="qualified I0 checkout, canary receipt or I0R ruling absent",
)


@pytest.fixture(scope="module")
def result() -> dict:
    run = subprocess.run(
        ["bash", str(ROOT / "tools/i0r-resume-precondition.sh"), "--no-dispatch"],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    return json.loads(run.stdout)


def test_probe_binds_ruling_receipt_and_makes_no_allocation(result: dict) -> None:
    assert result["bindings"]["ruling"]["sha256"] == probe.RULING_SHA
    assert result["bindings"]["canary_receipt"]["sha256"] == probe.RECEIPT_SHA
    assert result["allocations"] == 0 and result["provider_queries"] == 0
    assert "governed_dry_run" not in result["cases"]["attempt1-empty-ledger"]


def test_canary_proved_exact_metadata_archive(result: dict) -> None:
    proved = {row["dest"]: row for row in result["canary_proved_inputs"]}
    assert set(proved) == {"metadata.tar", "public-inputs.json"}
    assert proved["metadata.tar"]["sha256"] == probe.METADATA_SHA


def test_attempt1_shape_with_empty_ledger_is_accepted(result: dict) -> None:
    case = result["cases"]["attempt1-empty-ledger"]
    assert case["ledger_units"] == []
    assert case["metadata_sha256"] == probe.METADATA_SHA
    assert case["canary_check"] == case["supervisor_validate"] == "ACCEPT"
    assert result["attempt_parameters"] == {
        "INTEGRATION_BENCH_SECONDS": "1440",
        "est_minutes": 30,
        "timeout_minutes": 60,
        "unit_cap": 10.0,
        "rate_units_per_hour": 10.0,
    }


def test_probe_ledger_is_a_valid_nonempty_resume_input(result: dict) -> None:
    case = result["cases"]["attempt2-nonempty-ledger"]
    assert case["ledger_units"] == [probe.SYNTHETIC_UNIT]
    assert case["metadata_sha256"] != probe.METADATA_SHA
    assert case["metadata_size"] > 122880
    dests = [row["dest"] for row in case["input_inventory"]]
    assert dests == ["metadata.tar", "public-inputs.json"]  # no extra upload row


@pytest.mark.xfail(
    strict=True,
    reason="I0R HOLD: qualified matcher requires the canary-proved metadata.tar SHA; "
    "a non-empty ledger changes it (ruling §2: requalification, no exception)",
)
def test_canary_matcher_accepts_nonempty_resume_ledger(result: dict) -> None:
    case = result["cases"]["attempt2-nonempty-ledger"]
    assert case["canary_check"] == "ACCEPT"
    assert case["supervisor_validate"] == "ACCEPT"


def test_refusal_is_the_exact_input_matcher(result: dict) -> None:
    case = result["cases"]["attempt2-nonempty-ledger"]
    reason = "canary does not cover real upload bytes x file count/destinations"
    assert case["canary_check"] == f"REFUSE CanaryError: {reason}"
    assert case["supervisor_validate"] == f"REFUSE Refused: {reason}"
    assert result["status"] == probe.HOLD
